"""Local atomic writes, Linux writer locks, and owned child process cleanup."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def writer_lock(path, *, blocking=False):
    """Never unlink a lock file: live writers may still hold its inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another writer holds {path}") from exc
        # Closing our copy leaves inherited copies locked after SIGKILL.
        yield stream.fileno()


def inherited_fds():
    return tuple(int(x) for x in os.environ.get("RECURRENT_LOCK_FDS", "").split(",") if x)


def lock_environment(fd, **extra):
    return dict(os.environ, RECURRENT_LOCK_FDS=",".join(map(str, (*inherited_fds(), fd))), **extra)


class Children:
    """Reap children on failure/signals; optionally own their whole sessions."""
    def __init__(self, groups=True):
        self.groups = groups
        self.processes = []
        self.handlers = {}

    def __enter__(self):
        def stop(signum, frame):
            raise SystemExit(128 + signum)
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.handlers[signum] = signal.signal(signum, stop)
        return self

    def start(self, cmd, *, env=None, **kwargs):
        env = dict(os.environ) if env is None else env
        fds = tuple(int(x) for x in env.get("RECURRENT_LOCK_FDS", "").split(",") if x)
        proc = subprocess.Popen(cmd, env=env, start_new_session=self.groups, pass_fds=fds, **kwargs)
        self.processes.append(proc)
        return proc

    def run(self, cmd, **kwargs):
        proc = self.start(cmd, **kwargs)
        code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, cmd)

    def cleanup(self):
        def send(proc, sig):
            try:
                if self.groups:
                    os.killpg(proc.pid, sig)
                elif proc.poll() is None:
                    proc.send_signal(sig)
            except ProcessLookupError:
                pass
        for proc in self.processes:
            send(proc, signal.SIGTERM)
        deadline = time.monotonic() + 3
        for proc in self.processes:
            try:
                proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        # A session leader can have exited while its descendants are still alive.
        for proc in self.processes:
            send(proc, signal.SIGKILL)
        for proc in self.processes:
            proc.wait()

    def __exit__(self, *exc):
        # Ignore additional termination signals while reaping children.
        for signum in self.handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            self.cleanup()
        finally:
            for signum, handler in self.handlers.items():
                signal.signal(signum, handler)
