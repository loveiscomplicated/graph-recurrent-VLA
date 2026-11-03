"""Diagnostic-only observation of JPEG file operations, without replacing bytes."""
from pathlib import Path
import builtins
import os
import time


class JPEGIO:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.rows = []
        self.preload_ms = None

    def __enter__(self):
        import sam2.sam2_video_predictor as video
        owner = self
        self.original_open = builtins.open
        self.video = video
        self.original_preload = video.load_video_frames

        class ObservedFile:
            def __init__(self, file, path):
                self.file, self.path = file, path

            def __getattr__(self, name):
                return getattr(self.file, name)

            def operation(self, name, *args, **kwargs):
                start = time.perf_counter()
                result = getattr(self.file, name)(*args, **kwargs)
                owner.rows.append(dict(path=self.path, operation=name,
                    ms=(time.perf_counter()-start)*1000,
                    bytes=len(result) if isinstance(result, bytes) else 0))
                return result

            def read(self, *a, **k): return self.operation('read', *a, **k)
            def seek(self, *a, **k): return self.operation('seek', *a, **k)
            def close(self): return self.operation('close')
            def __enter__(self): return self
            def __exit__(self, *args): self.close()

        def observed_open(file, *args, **kwargs):
            match = isinstance(file, (str, os.PathLike)) and Path(file).resolve().parent == owner.folder
            start = time.perf_counter()
            result = owner.original_open(file, *args, **kwargs)
            if not match: return result
            path = Path(file).name
            owner.rows.append(dict(path=path, operation='open', ms=(time.perf_counter()-start)*1000, bytes=0))
            return ObservedFile(result, path)

        def preload(*a, **k):
            start = time.perf_counter()
            result = owner.original_preload(*a, **k)
            owner.preload_ms = (time.perf_counter()-start)*1000
            return result

        builtins.open = observed_open
        video.load_video_frames = preload
        return self

    def __exit__(self, *args):
        builtins.open = self.original_open
        self.video.load_video_frames = self.original_preload

    def result(self):
        return dict(file_operations=self.rows, file_io_ms=sum(r['ms'] for r in self.rows),
                    file_read_bytes=sum(r['bytes'] for r in self.rows),
                    cpu_preload_decode_resize_normalize_ms=self.preload_ms,
                    scope='diagnostic pass; OS page cache retained; file timings exclude JPEG decode and resize')
