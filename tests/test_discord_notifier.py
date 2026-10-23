import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from uuid import uuid4

from tools.discord_notifier import (
    Delivery, DeliveryWorker, DiscordTransport, FileTailer, LogBuffer, Monitor,
    Sanitizer, TerminalLines, Tmux, main, units,
)


FAKE_WEBHOOK = 'https://discord.com/api/webhooks/123/not-a-real-token'


def wait_until(predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError('condition did not become true')


class DiscordNotifierTests(unittest.TestCase):
    def buffer(self, **kwargs):
        return LogBuffer('tmux:test', Sanitizer({}), **kwargs)

    def test_fragmented_terminal_output_and_redaction(self):
        lines = []
        sanitize = Sanitizer({'DISCORD_WEBHOOK_URL': FAKE_WEBHOOK, 'WANDB_API_KEY': 'private-key'})
        stream = TerminalLines(lambda line: lines.append(sanitize(line)))
        text = '\x1b[31m한글\x1b[0m\r\nold\rnew\n\x1b]0;hidden\x07visible\n'
        text += f'{FAKE_WEBHOOK} private-key\n'
        for byte in text.encode():
            stream.feed(bytes([byte]))
        self.assertEqual(lines[:3], ['한글', 'new', 'visible'])
        self.assertEqual(lines[3], '[REDACTED_WEBHOOK] [REDACTED]')
        self.assertNotIn('not-a-real-token', '\n'.join(lines))
        stream.feed(b'partial')
        stream.flush()
        self.assertEqual(lines[-1], 'partial')

    def test_overlong_lines_are_omitted_whole(self):
        lines = []
        stream = TerminalLines(lines.append, max_line=8)
        stream.feed(b'private-long-line\nokay\n')
        self.assertEqual(lines, ['[overlong log line omitted]', 'okay'])

    def test_batch_limit_deduplication_and_acknowledgements(self):
        buffer = self.buffer(max_lines=3)
        buffer.add('pane:%1', 'same', now=1)
        buffer.add('file:tee.log', 'same', now=1.1)
        self.assertEqual(buffer.sequence, 1)
        buffer.add('file:tee.log', 'same', now=1.2)  # Same-source repetitions survive.
        self.assertEqual(buffer.sequence, 2)
        buffer.add('pane:%1', 'same', now=1.3)
        self.assertEqual(buffer.sequence, 2)  # Pair the second copy from each source.
        buffer.add('pane:%1', 'same', now=1.4)
        self.assertEqual(buffer.sequence, 3)
        for index in range(8):
            buffer.add('file:train.log', f'epoch {index}', now=index + 10)
        batch = buffer.snapshot()
        self.assertIn('11 log lines (8 earlier lines omitted', batch.content)
        self.assertIn('epoch 7', batch.content)
        self.assertNotIn('epoch 0', batch.content)
        buffer.add('pane:%2', 'after snapshot', now=30)
        buffer.acknowledge(batch.sequence)
        self.assertIn('after snapshot', buffer.snapshot().content)
        buffer.acknowledge(buffer.sequence)
        self.assertIsNone(buffer.snapshot())
        buffer.add('pane:%2', '😀' * 4000, now=31)
        self.assertLessEqual(units(buffer.snapshot().content), 1800)
        self.assertLessEqual(len(buffer.recent), 3)

    def test_transport_payload_headers_and_rate_limits(self):
        response = Mock()
        response.headers = {'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset-After': '90.5'}
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        transport = DiscordTransport(FAKE_WEBHOOK + '?thread_id=7&wait=false', opener=opener)
        result = transport.post('hello @everyone')
        self.assertTrue(result.ok)
        self.assertEqual(result.delay, 90.5)
        req = opener.open.call_args.args[0]
        self.assertIn('thread_id=7', req.full_url)
        self.assertIn('wait=true', req.full_url)
        self.assertNotIn('wait=false', req.full_url)
        self.assertEqual(json.loads(req.data)['allowed_mentions'], {'parse': []})
        opener.open.side_effect = HTTPError(FAKE_WEBHOOK, 429, 'limited', {'Retry-After': '4'},
                                           io.BytesIO(b'{"retry_after": 6.25}'))
        result = transport.post('hello')
        self.assertEqual(result.delay, 6.25)
        self.assertFalse(result.disabled)
        opener.open.side_effect = HTTPError(FAKE_WEBHOOK, 429, 'limited', {}, io.BytesIO(b'bad json'))
        self.assertGreaterEqual(transport.post('hello').delay, 1)

    def test_transport_failure_types_do_not_expose_credentials(self):
        opener = Mock()
        transport = DiscordTransport(FAKE_WEBHOOK, opener=opener)
        for code in (400, 401, 403, 404):
            opener.open.side_effect = HTTPError(FAKE_WEBHOOK, code, FAKE_WEBHOOK, {}, io.BytesIO())
            result = transport.post('hello')
            self.assertTrue(result.disabled)
            self.assertNotIn(FAKE_WEBHOOK, result.reason)
        opener.open.side_effect = HTTPError(FAKE_WEBHOOK, 503, 'server', {}, io.BytesIO())
        self.assertFalse(transport.post('hello').disabled)
        opener.open.side_effect = URLError(FAKE_WEBHOOK)
        self.assertEqual(transport.post('hello').reason, 'URLError')
        for invalid in ('', 'http://discord.com/api/webhooks/1/token', 'https://example.com/api/webhooks/1/token'):
            with self.assertRaises(ValueError):
                DiscordTransport(invalid)

    def test_worker_backoff_cooldown_recovery_and_final_flush(self):
        buffer = self.buffer()
        buffer.add('pane:%1', 'one')
        transport = Mock(timeout=1)
        transport.post.side_effect = [Delivery(delay=100, reason='HTTP 429'), Delivery(ok=True),
                                      Delivery(disabled=True, reason='HTTP 404')]
        messages = []
        worker = DeliveryWorker(buffer, transport, interval=60, status=messages.append)
        with patch('tools.discord_notifier.time.monotonic', return_value=0):
            worker.due = 0
            worker.attempt(0)
            buffer.add('pane:%1', 'two')
            worker.attempt(99, final=True)
            self.assertEqual(transport.post.call_count, 1)
            worker.attempt(100)
            self.assertIn('two', transport.post.call_args.args[0])
            self.assertIsNone(buffer.snapshot())
            buffer.add('pane:%1', 'three')
            worker.attempt(101, final=True)
            self.assertEqual(transport.post.call_count, 2)
            worker.attempt(160)
            worker.attempt(500)
        self.assertTrue(worker.disabled)
        self.assertEqual(transport.post.call_count, 3)
        self.assertIn('recovered', ' '.join(messages))

    def test_network_backoff_is_bounded_and_does_not_drop_pending_logs(self):
        buffer = self.buffer()
        buffer.add('pane:%1', 'pending')
        transport = Mock(timeout=1)
        transport.post.return_value = Delivery(reason='TimeoutError')
        worker = DeliveryWorker(buffer, transport)
        worker.due = 0
        now = 0
        with patch('tools.discord_notifier.time.monotonic', return_value=0):
            for delay in (5, 10, 20, 40, 80, 160, 300, 300):
                worker.attempt(now)
                self.assertEqual(worker.due - now, delay)
                now = worker.due
        self.assertIn('pending', buffer.snapshot().content)

    def test_network_request_does_not_lock_ingestion(self):
        entered, release = threading.Event(), threading.Event()
        buffer = self.buffer()
        buffer.add('pane:%1', 'before')
        transport = Mock(timeout=1)
        def post(_):
            entered.set()
            release.wait(2)
            return Delivery(ok=True)
        transport.post.side_effect = post
        worker = DeliveryWorker(buffer, transport, interval=60)
        worker.due = 0
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            buffer.add('pane:%1', 'while request is blocked')
            self.assertIn('while request is blocked', buffer.snapshot().content)
        finally:
            release.set()
            worker.close()
        # Acknowledge only the batch that was actually sent.
        self.assertIn('while request is blocked', buffer.snapshot().content)

    def test_log_files_append_new_truncate_recreate_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / 'train.log'
            existing.write_text('old contents\n')
            captured = root / 'capture'
            captured.mkdir()
            output = []
            tailer = FileTailer([root, root / 'not-yet-created'], lambda source, line: output.append((source, line)), excluded=captured)
            tailer.poll()
            self.assertEqual(output, [])
            with existing.open('a') as stream:
                stream.write('new contents\n')
            tailer.poll()
            self.assertEqual(output[-1][1], 'new contents')
            existing.write_text('replacement longer than the complete original contents\n')
            tailer.poll()  # Truncated and regrown past the old offset in a single poll.
            self.assertTrue(output[-1][1].startswith('replacement'))
            existing.unlink()
            existing.write_text('recreated\n')
            tailer.poll()
            self.assertEqual(output[-1][1], 'recreated')
            (root / 'eval.log').write_text('fresh file\n')
            os.mkfifo(root / 'pipe.log')  # A directory scan must never block on a FIFO.
            (captured / 'panes.log').write_text('must not loop\n')
            (root / 'not-yet-created').mkdir()
            (root / 'not-yet-created' / 'later.log').write_text('new directory\n')
            tailer.poll()
            tailer.close()
            text = '\n'.join(line for _, line in output)
            self.assertIn('fresh file', text)
            self.assertIn('new directory', text)
            self.assertNotIn('must not loop', text)

    def test_cli_missing_webhook_is_quiet_and_does_not_start_tmux(self):
        with patch.dict(os.environ, {'DISCORD_WEBHOOK_URL': ''}), patch('sys.stderr', new_callable=io.StringIO) as stderr, patch.object(Tmux, 'panes') as panes:
            self.assertEqual(main(['--session', 'test']), 2)
            panes.assert_not_called()
            self.assertIn('DISCORD_WEBHOOK_URL', stderr.getvalue())

    def test_pane_disappearing_during_attach_does_not_stop_other_panes(self):
        tmux = Mock()
        tmux.panes.return_value = {'%1': '0', '%2': '0'}
        tmux.attach.side_effect = [subprocess.CalledProcessError(1, 'tmux'), None]
        monitor = Monitor(tmux, self.buffer(), Path('/unused'))
        monitor.discover('/unused/socket')
        self.assertEqual(monitor.requested, {'%2'})


@unittest.skipUnless(shutil.which('tmux'), 'tmux not installed')
class TmuxNotifierIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tmux = Tmux('notifier-test', ['tmux', '-L', 'discord-test-' + uuid4().hex[:12], '-f', '/dev/null'])
        self.tmux.call('new-session', '-d', '-s', 'notifier-test', '-n', 'self')
        self.self_pane = next(iter(self.tmux.panes()))
        self.tmux.call('new-window', '-d', '-t', '=notifier-test', '-n', 'output')
        self.output_pane = next(p for p in self.tmux.panes() if p != self.self_pane)
        self.stop = threading.Event()
        self.buffer = LogBuffer('test', Sanitizer({'DISCORD_WEBHOOK_URL': FAKE_WEBHOOK}))
        self.monitor = Monitor(self.tmux, self.buffer, self.root / 'capture', [self.root / 'logs'], self.self_pane)
        self.errors = []
        def run():
            try:
                self.monitor.run(self.stop)
            except BaseException as exc:
                self.errors.append(exc)
        self.thread = threading.Thread(target=run, daemon=True)

    def tearDown(self):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(8)
        try:
            self.tmux.call('kill-server')
        finally:
            self.temporary.cleanup()

    def print_in_pane(self, pane, text):
        # The shell echoes the command, so octal-encode the marker to distinguish
        # the program output from its command line.
        encoded = ''.join(f'\\{byte:03o}' for byte in text.encode())
        self.tmux.call('send-keys', '-t', pane, '-l', "printf '" + encoded + "\\n'")
        self.tmux.call('send-keys', '-t', pane, 'Enter')

    def capture_contains(self, text):
        path = self.root / 'capture' / 'panes.log'
        return path.exists() and text in path.read_text()

    def test_live_panes_files_foreign_pipe_and_cleanup(self):
        self.tmux.call('new-window', '-d', '-t', '=notifier-test', '-n', 'foreign')
        foreign = next(p for p in self.tmux.panes() if p not in (self.self_pane, self.output_pane))
        self.tmux.call('pipe-pane', '-O', '-t', foreign, 'cat >/dev/null')
        self.thread.start()
        wait_until(lambda: self.tmux.panes()[self.output_pane] == '1')
        self.assertEqual(self.tmux.panes()[self.self_pane], '0')
        self.print_in_pane(self.output_pane, 'live output marker')
        wait_until(lambda: self.capture_contains('live output marker'))
        self.print_in_pane(self.output_pane, FAKE_WEBHOOK)
        wait_until(lambda: self.capture_contains('[REDACTED_WEBHOOK]'))
        self.tmux.call('new-window', '-d', '-t', '=notifier-test', '-n', 'new')
        new_pane = next(p for p in self.tmux.panes() if p not in (self.self_pane, self.output_pane, foreign))
        wait_until(lambda: self.tmux.panes()[new_pane] == '1')
        self.print_in_pane(new_pane, 'new pane marker')
        wait_until(lambda: self.capture_contains('new pane marker'))
        logs = self.root / 'logs'
        logs.mkdir()
        (logs / 'train.log').write_text('file training marker\n')
        wait_until(lambda: 'file training marker' in (self.buffer.snapshot().content if self.buffer.snapshot() else ''))
        self.stop.set()
        self.thread.join(8)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.errors, [])
        wait_until(lambda: self.tmux.panes()[self.output_pane] == '0' and self.tmux.panes()[new_pane] == '0')
        self.assertEqual(self.tmux.panes()[foreign], '1')
        text = (self.root / 'capture' / 'panes.log').read_text()
        self.assertNotIn('not-a-real-token', text)

    def test_replaced_pipe_is_preserved_on_shutdown(self):
        self.thread.start()
        wait_until(lambda: any(state['pane'] == self.output_pane for state in self.monitor.connections.values()))
        self.tmux.call('pipe-pane', '-O', '-t', self.output_pane, 'cat >/dev/null')
        wait_until(lambda: not self.monitor.connections)
        self.stop.set()
        self.thread.join(8)
        self.assertEqual(self.errors, [])
        self.assertEqual(self.tmux.panes()[self.output_pane], '1')


if __name__ == '__main__':
    unittest.main()
