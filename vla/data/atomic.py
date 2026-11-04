import json
import os
from pathlib import Path
from uuid import uuid4

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
