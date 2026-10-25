"""Task06-C local atomic records and adapter to the existing Discord transport."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from tools.discord_notifier import DiscordTransport, DeliveryWorker, LogBuffer, Sanitizer


REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value, *, immutable=False):
    """Flush file and directory; immutable records are linked without replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError('immutable record differs: '+str(path))
        return
    tmp = path.with_name(path.name+'.'+uuid4().hex+'.partial')
    try:
        with tmp.open('x') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if read(tmp) != value:
            raise ValueError('JSON readback mismatch')
        if immutable:
            os.link(tmp, path); tmp.unlink()
        else:
            os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def finalize(tmp, path):
    """No overwrite even if another process has produced the same destination."""
    with Path(tmp).open('rb') as f: os.fsync(f.fileno())
    os.link(tmp, path)
    Path(tmp).unlink()


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('another process holds '+str(path))
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def check_space(run, config):
    if shutil.disk_usage(run).free < config['minimum_free_bytes']:
        raise OSError('insufficient free disk space')


class StopRequested(Exception):
    """Normal cooperative stop at a recoverable boundary, never a fatal event."""


def stop_gate(run):
    if (Path(run)/'STOP').exists():
        raise StopRequested('user stop request')


class Events:
    """Only summaries enter the reused bounded, rate-limited delivery worker."""
    def __init__(self, run, *, required=False):
        self.run = Path(run); self.clean = Sanitizer()
        self.worker = None
        url = os.environ.get('DISCORD_WEBHOOK_URL', '')
        if url:
            transport = DiscordTransport(url)
            self.buffer = LogBuffer('Graph-Recurrent VLA | Task06-C | '+self.run.name, self.clean)
            self.worker = DeliveryWorker(self.buffer, transport, interval=5, status=self.delivery_status)
            self.worker.start()
        elif required:
            raise RuntimeError('BLOCKED_DISCORD_CONFIGURATION: DISCORD_WEBHOOK_URL is absent')

    def delivery_status(self, text):
        append(self.run/'discord_delivery.jsonl', dict(time=now(), message=self.clean(text)))

    def emit(self, kind, **values):
        record = json.loads(self.clean(json.dumps(dict(time=now(), event=kind, **values), allow_nan=False)))
        append(self.run/'events.jsonl', record)
        message = self.clean(json.dumps(record, ensure_ascii=False))
        print(message, flush=True)
        if self.worker:
            self.buffer.add(kind, message)

    def probe(self):
        if self.worker is None:
            return dict(status='MISSING', actual_delivery=False)
        result = self.worker.transport.post('[Graph-Recurrent VLA | Task06-C] Preflight '+self.run.name)
        # Respect even a preflight rate limit before worker dispatch.
        self.worker.allowed = time.monotonic()+max(5, result.delay)
        report = dict(status='PASS' if result.ok else 'FAILED', actual_delivery=result.ok,
                      reason=result.reason, retry_after=result.delay)
        atomic(self.run/'discord_verification.json', report)
        if not result.ok:
            raise RuntimeError('BLOCKED_DISCORD_VERIFICATION: '+result.reason)
        return report

    def close(self):
        if self.worker: self.worker.close()
