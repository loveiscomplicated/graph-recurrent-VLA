"""Forward selected tmux output and log files to a Discord incoming webhook.

Run on the same host as tmux. Uses only the Python standard library; it does
not create experiments, inspect models, or control the programs being watched.
"""
from __future__ import annotations

import argparse
import codecs
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib import error, parse, request
from uuid import uuid4


SECRET_NAME = re.compile(r'api.?key|token|password|secret|credential|webhook', re.I)
WEBHOOK = re.compile(r'https?://(?:(?:canary|ptb)\.)?discord(?:app)?\.com/api/(?:v\d+/)?webhooks/[^\s\x1b\'"`<>]+', re.I)


class Sanitizer:
    def __init__(self, environment=None):
        environment = os.environ if environment is None else environment
        self.secrets = sorted({v for k, v in environment.items()
                               if SECRET_NAME.search(k) and v}, key=len, reverse=True)

    def __call__(self, text):
        text = WEBHOOK.sub('[REDACTED_WEBHOOK]', text)
        for value in self.secrets:
            text = text.replace(value, '[REDACTED]')
        return text


class TerminalLines:
    """Decode fragmented UTF-8/ANSI, retaining the last carriage-return update."""
    def __init__(self, emit, max_line=65536):
        self.emit = emit
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.line = []
        self.state = 'text'
        self.carriage = False
        self.discard = False
        self.max_line = max_line
        self.updated = time.monotonic()

    def feed(self, data):
        self.updated = time.monotonic()
        for char in self.decoder.decode(data):
            if self.state == 'escape':
                self.state = ('csi' if char == '[' else 'string' if char in ']P^_'
                              else 'escape_intermediate' if ' ' <= char <= '/' else 'text')
                continue
            if self.state == 'escape_intermediate':
                if '0' <= char <= '~':
                    self.state = 'text'
                continue
            if self.state == 'csi':
                if '@' <= char <= '~':
                    self.state = 'text'
                continue
            if self.state == 'string':
                if char == '\x07':
                    self.state = 'text'
                elif char == '\x1b':
                    self.state = 'string_escape'
                continue
            if self.state == 'string_escape':
                self.state = 'text' if char == '\\' else 'string'
                continue
            if char == '\x1b':
                self.state = 'escape'
                continue
            if char == '\n':
                self.flush()
                continue
            if char == '\r':
                self.carriage = True
                continue
            if self.carriage:
                self.line.clear()
                self.discard = False
                self.carriage = False
            if char == '\b':
                if self.line:
                    self.line.pop()
            elif (char >= ' ' and char != '\x7f') or char == '\t':
                if not self.discard:
                    self.line.append(char)
                    if len(self.line) > self.max_line:
                        # Never emit fragments of a potentially secret oversized line.
                        self.line.clear()
                        self.discard = True

    def flush(self):
        if self.discard:
            self.emit('[overlong log line omitted]')
        elif self.line:
            self.emit(''.join(self.line))
        self.line.clear()
        self.carriage = self.discard = False

    def flush_idle(self, now):
        if now - self.updated >= 2:
            self.flush()


def units(text):
    """Conservative Discord character budget, including astral Unicode."""
    return len(text.encode('utf-16-le', errors='replace')) // 2


def tail_text(text, budget):
    return text.encode('utf-16-le', errors='replace')[-budget * 2:].decode('utf-16-le', errors='ignore')


@dataclass
class Batch:
    sequence: int
    content: str


class LogBuffer:
    """Bounded pending delivery; acknowledgements preserve concurrently added lines."""
    def __init__(self, label, sanitize, max_lines=1000):
        self.label = sanitize(label)[:80]
        self.sanitize = sanitize
        self.lines = deque(maxlen=max_lines)
        self.recent = OrderedDict()
        self.sequence = self.acknowledged = 0
        self.lock = threading.Lock()

    def add(self, source, line, now=None):
        source, line = self.sanitize(source), self.sanitize(line)
        now = time.monotonic() if now is None else now
        with self.lock:
            previous = self.recent.get(line)
            within_window = previous is not None and now - previous[0] <= 2
            counts = previous[1] if within_window else {}
            maximum = max(counts.values(), default=0)
            counts[source] = counts.get(source, 0) + 1
            self.recent[line] = (previous[0] if within_window else now, counts)
            self.recent.move_to_end(line)
            while len(self.recent) > self.lines.maxlen:
                self.recent.popitem(last=False)
            if counts[source] <= maximum:
                return
            self.sequence += 1
            self.lines.append((self.sequence, tail_text(source, 120), tail_text(line, 1800)))

    def snapshot(self):
        with self.lock:
            if self.sequence == self.acknowledged:
                return None
            # Reserve room for the header, omission count and code fences.
            selected, used = [], 0
            for _, source, line in reversed(self.lines):
                text = f'[{source}] {line}'.replace('```', "'''")
                text = tail_text(text, 1450)
                if used + units(text) + 1 > 1500:
                    break
                selected.append(text)
                used += units(text) + 1
            count = self.sequence - self.acknowledged
            omitted = count - len(selected)
            header = f'[{self.label}] {count} log lines'
            if omitted:
                header += f' ({omitted} earlier lines omitted; see local logs)'
            content = header + '\n```text\n' + '\n'.join(reversed(selected)) + '\n```'
            return Batch(self.sequence, content)

    def acknowledge(self, sequence):
        with self.lock:
            self.acknowledged = max(self.acknowledged, sequence)
            while self.lines and self.lines[0][0] <= sequence:
                self.lines.popleft()


@dataclass
class Delivery:
    ok: bool = False
    delay: float = 0
    disabled: bool = False
    reason: str = ''


def seconds(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def retry_delay(headers, body=b''):
    delays = [seconds(headers.get('Retry-After')), seconds(headers.get('X-RateLimit-Reset-After'))]
    try:
        delays.append(seconds(json.loads(body).get('retry_after')))
    except (ValueError, AttributeError, UnicodeError):
        pass
    try:
        date = parsedate_to_datetime(headers.get('Retry-After', ''))
        delays.append(max(0, date.timestamp() - time.time()))
    except (ValueError, TypeError, OverflowError):
        pass
    return max(delays)


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DiscordTransport:
    def __init__(self, url, opener=None, timeout=5):
        parsed = parse.urlsplit(url.strip())
        if (parsed.scheme != 'https' or parsed.netloc not in
                ('discord.com', 'discordapp.com', 'canary.discord.com', 'ptb.discord.com')
                or not re.fullmatch(r'/api/(?:v\d+/)?webhooks/\d+/[^/]+', parsed.path)
                or parsed.fragment):
            raise ValueError('DISCORD_WEBHOOK_URL must be a Discord incoming webhook HTTPS URL')
        query = [(k, v) for k, v in parse.parse_qsl(parsed.query) if k != 'wait']
        self.url = parse.urlunsplit(parsed._replace(query=parse.urlencode(query + [('wait', 'true')])))
        self.opener = opener or request.build_opener(NoRedirect())
        self.timeout = timeout

    def post(self, content):
        payload = json.dumps({'content': content, 'allowed_mentions': {'parse': []}}).encode()
        req = request.Request(self.url, data=payload, method='POST', headers={
            'Content-Type': 'application/json', 'User-Agent': 'graph-recurrent-vla-discord-notifier/1.0'})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                delay = (retry_delay(response.headers)
                         if response.headers.get('X-RateLimit-Remaining') == '0' else 0)
                return Delivery(ok=True, delay=delay)
        except error.HTTPError as exc:
            with exc:
                if exc.code == 429:
                    return Delivery(delay=max(1, retry_delay(exc.headers, exc.read(4096))), reason='HTTP 429')
                if 500 <= exc.code <= 599:
                    return Delivery(reason=f'HTTP {exc.code}')
                return Delivery(disabled=True, reason=f'HTTP {exc.code}')
        except Exception as exc:
            # Never render exception text: urllib exceptions can contain the URL.
            return Delivery(reason=type(exc).__name__)


class DeliveryWorker:
    def __init__(self, buffer, transport, interval=60, status=lambda text: None):
        self.buffer, self.transport = buffer, transport
        self.interval, self.status = interval, status
        self.due = time.monotonic() + interval
        self.allowed = 0
        self.failures = 0
        self.disabled = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='discord-delivery', daemon=True)

    def attempt(self, now, final=False):
        if self.disabled or now < self.allowed or (not final and now < self.due):
            return
        batch = self.buffer.snapshot()
        if batch is None:
            self.due = now + self.interval
            return
        result = self.transport.post(batch.content)
        now = max(now, time.monotonic())
        if result.ok:
            self.buffer.acknowledge(batch.sequence)
            if self.failures:
                self.status('Discord delivery recovered.')
            self.failures = 0
            self.allowed = self.due = now + max(self.interval, result.delay)
        elif result.disabled:
            self.disabled = True
            self.status(f'Discord delivery disabled ({result.reason}); local capture continues. Restart after fixing the webhook.')
        else:
            self.failures += 1
            delay = result.delay or min(300, 5 * 2 ** min(self.failures - 1, 6))
            self.allowed = self.due = now + delay
            if self.failures == 1 or self.failures % 10 == 0:
                self.status(f'Discord delivery deferred ({result.reason}); local capture continues.')

    def run(self):
        while not self.stop_event.wait(min(60, max(0, self.due - time.monotonic()))):
            self.attempt(time.monotonic())
            if self.disabled:
                self.stop_event.wait()
                break
        # A final batch may use the initial interval, never a rate-limit cooldown.
        self.attempt(time.monotonic(), final=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=self.transport.timeout + 1)


@dataclass
class TailFile:
    identity: tuple
    offset: int
    decoder: TerminalLines
    anchor: bytes


class FileTailer:
    def __init__(self, directories, emit, excluded=None, status=lambda text: None):
        self.directories = [Path(path).resolve() for path in directories]
        self.emit, self.status = emit, status
        self.excluded = Path(excluded).resolve() if excluded else None
        self.files = {}
        self.initial = True
        self.warned = set()

    def poll(self):
        found = set()
        for directory in self.directories:
            try:
                paths = list(directory.rglob('*.log'))
            except OSError:
                self.status('Unable to scan a log directory; retrying later.')
                continue
            for path in paths:
                if (path.is_symlink() or not path.is_file()
                        or (self.excluded and path.is_relative_to(self.excluded))):
                    continue
                found.add(path)
                try:
                    with path.open('rb') as stream:
                        stat = os.fstat(stream.fileno())
                        identity = (stat.st_dev, stat.st_ino)
                        previous = self.files.get(path)
                        changed = (previous is None or previous.identity != identity
                                   or stat.st_size < previous.offset)
                        if previous and not changed:
                            stream.seek(max(0, previous.offset - len(previous.anchor)))
                            changed = stream.read(len(previous.anchor)) != previous.anchor
                        if changed:
                            if previous:
                                previous.decoder.flush()
                            decoder = TerminalLines(lambda line, p=path: self.emit(f'file:{p}', line))
                            offset = stat.st_size if self.initial else 0
                        else:
                            offset, decoder = previous.offset, previous.decoder
                        stream.seek(offset)
                        data = stream.read(262144)
                        if data:
                            decoder.feed(data)
                        decoder.flush_idle(time.monotonic())
                        offset = stream.tell()
                        stream.seek(max(0, offset - 64))
                        anchor = stream.read(offset - stream.tell())
                        self.files[path] = TailFile(identity, offset, decoder, anchor)
                        self.warned.discard(path)
                except OSError as exc:
                    if path not in self.warned:
                        self.status(f'Unable to read a log file ({type(exc).__name__}); retrying later.')
                        self.warned.add(path)
        for path in set(self.files) - found:
            self.files.pop(path).decoder.flush()
        self.initial = False

    def close(self):
        for state in self.files.values():
            state.decoder.flush()


class Tmux:
    def __init__(self, session, command=None):
        self.session = session
        self.command = command or ['tmux']

    def call(self, *args):
        return subprocess.check_output([*self.command, *args], stderr=subprocess.DEVNULL, timeout=3).decode()

    def panes(self):
        result = self.call('list-panes', '-s', '-t', '=' + self.session, '-F', '#{pane_id}\t#{pane_pipe}')
        return dict(line.split('\t') for line in result.splitlines())

    def attach(self, pane, socket_path):
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), '_relay', str(socket_path), pane])
        # pipe-pane expands tmux formats AND strftime sequences after shell
        # quoting. In particular, literal pane IDs such as %1 must survive.
        command = command.replace('#', '##').replace('%', '%%')
        self.call('pipe-pane', '-O', '-o', '-t', pane, command + ' >/dev/null 2>&1')


def relay(socket_path, pane):
    """Private pipe-pane helper. Parent EOF stops the helper after a crash."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(socket_path)
            connection.sendall(pane.encode() + b'\n')
            with selectors.DefaultSelector() as selector:
                selector.register(connection, selectors.EVENT_READ)
                selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
                while True:
                    for key, _ in selector.select():
                        if key.fileobj is connection:
                            return 0  # Parent closed; don't leave a running relay.
                        data = os.read(sys.stdin.fileno(), 65536)
                        if not data:
                            return 0
                        connection.sendall(data)
    except OSError:
        return 0


class Monitor:
    def __init__(self, tmux, buffer, output, log_dirs=(), self_pane=None, status=lambda text: None):
        self.tmux, self.buffer, self.output = tmux, buffer, Path(output)
        self.self_pane, self.status = self_pane, status
        self.connections = {}
        self.requested = set()
        self.warned = set()
        self.tailer = FileTailer(log_dirs, self.emit, excluded=self.output, status=status)
        self.spool = None

    def emit(self, source, line):
        source, line = self.buffer.sanitize(source), self.buffer.sanitize(line)
        if source.startswith('pane:'):
            self.spool.write(f'[{source}] {line}\n')
            self.spool.flush()
        self.buffer.add(source, line)

    def discover(self, socket_path):
        panes = self.tmux.panes()
        for pane, piped in panes.items():
            if pane == self.self_pane:
                continue
            if piped == '1':
                if pane not in self.requested and pane not in self.warned:
                    self.status(f'Skipping {pane}: another tmux pipe is already connected.')
                    self.warned.add(pane)
                continue
            try:
                self.tmux.attach(pane, socket_path)
            except subprocess.CalledProcessError:
                continue  # A pane can disappear between discovery and attachment.
            self.requested.add(pane)

    def read_connection(self, selector, connection, state):
        try:
            data = connection.recv(65536)
        except BlockingIOError:
            return False
        except OSError:
            data = b''
        if not data:
            if state['decoder']:
                state['decoder'].flush()
            selector.unregister(connection)
            connection.close()
            self.connections.pop(connection, None)
            return False
        if state['decoder'] is None:
            state['hello'] += data
            if b'\n' not in state['hello']:
                if len(state['hello']) > 64:
                    connection.shutdown(socket.SHUT_RDWR)
                return True
            pane, data = state['hello'].split(b'\n', 1)
            pane = pane.decode(errors='replace')
            if pane not in self.requested:
                connection.shutdown(socket.SHUT_RDWR)
                return True
            state['hello'] = b''
            state['pane'] = pane
            state['decoder'] = TerminalLines(lambda line: self.emit(f'pane:{pane}', line))
        state['decoder'].feed(data)
        return True

    def run(self, stop):
        self.output.mkdir(parents=True, mode=0o700, exist_ok=False)
        fd = os.open(self.output / 'panes.log', os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as self.spool:
            # Keep Unix socket paths below macOS's short sockaddr_un limit.
            with tempfile.TemporaryDirectory(prefix='discord-notifier-', dir='/tmp') as temporary:
                socket_path = Path(temporary) / 'relay.sock'
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, selectors.DefaultSelector() as selector:
                    listener.bind(str(socket_path))
                    listener.listen()
                    listener.setblocking(False)
                    selector.register(listener, selectors.EVENT_READ)
                    self.tailer.poll()  # Existing files start at EOF, before attaching panes.
                    next_poll = 0
                    try:
                        while not stop.is_set():
                            now = time.monotonic()
                            if now >= next_poll:
                                try:
                                    self.discover(socket_path)
                                except subprocess.CalledProcessError:
                                    self.status('tmux session ended; stopping notifier.')
                                    break
                                self.tailer.poll()
                                for state in self.connections.values():
                                    if state['decoder']:
                                        state['decoder'].flush_idle(now)
                                next_poll = now + 1
                            for key, _ in selector.select(timeout=0.2):
                                if key.fileobj is listener:
                                    connection, _ = listener.accept()
                                    connection.setblocking(False)
                                    state = {'hello': b'', 'decoder': None, 'pane': None}
                                    self.connections[connection] = state
                                    selector.register(connection, selectors.EVENT_READ, state)
                                else:
                                    self.read_connection(selector, key.fileobj, key.data)
                    finally:
                        try:
                            self.tailer.poll()
                            self.tailer.close()
                            # Consume already buffered output before releasing the relays.
                            while True:
                                try:
                                    connection, _ = listener.accept()
                                except BlockingIOError:
                                    break
                                connection.setblocking(False)
                                state = {'hello': b'', 'decoder': None, 'pane': None}
                                self.connections[connection] = state
                                selector.register(connection, selectors.EVENT_READ, state)
                            for connection, state in list(self.connections.items()):
                                for _ in range(16):
                                    if not self.read_connection(selector, connection, state):
                                        break
                            for state in self.connections.values():
                                if state['decoder']:
                                    state['decoder'].flush()
                        finally:
                            # Only detach panes whose relay is still connected. EOF
                            # from a replaced/closed pipe removes it from this set.
                            for connection, state in list(self.connections.items()):
                                try:
                                    if state['pane']:
                                        self.tmux.call('pipe-pane', '-t', state['pane'])
                                except (OSError, subprocess.SubprocessError):
                                    pass
                                finally:
                                    connection.close()
                            self.connections.clear()


def positive(value):
    result = seconds(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive finite number')
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['_relay'] and len(argv) == 3:
        return relay(argv[1], argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='exact tmux session name to watch')
    parser.add_argument('--log-dir', action='append', type=Path, default=[],
                        help='also follow *.log recursively; repeat for multiple directories')
    parser.add_argument('--interval', type=positive, default=60, help='seconds between Discord batches (default: 60)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/discord_notifier'),
                        help='parent directory for local pane capture')
    args = parser.parse_args(argv)
    sanitize = Sanitizer()
    def status(text):
        print(sanitize(text), file=sys.stderr, flush=True)
    try:
        transport = DiscordTransport(os.environ.get('DISCORD_WEBHOOK_URL', ''))
    except ValueError as exc:
        status(str(exc))
        return 2
    tmux = Tmux(args.session)
    try:
        tmux.panes()
    except (OSError, subprocess.SubprocessError):
        status('Cannot find the tmux session. Install tmux and run on the same host/server as the target session.')
        return 2
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_dir.resolve() / f'{stamp}-{uuid4().hex[:8]}'
    buffer = LogBuffer('tmux:' + args.session, sanitize)
    monitor = Monitor(tmux, buffer, output, args.log_dir, os.environ.get('TMUX_PANE'), status)
    worker = DeliveryWorker(buffer, transport, args.interval, status)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    status(f'Watching tmux session {args.session}; pane capture: {output / "panes.log"}')
    worker.start()
    result = 0
    try:
        monitor.run(stop)
    except (OSError, subprocess.SubprocessError) as exc:
        status(f'Notifier stopped ({type(exc).__name__}); training processes remain independent.')
        result = 1
    finally:
        worker.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
