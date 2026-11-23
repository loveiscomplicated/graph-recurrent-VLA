"""Opt-in Objective-C++ RANSAC on the current PyTorch MPS stream."""

from contextlib import contextmanager
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
import torch

from . import metal_ops


_SOURCES = Path(__file__).with_name("native")
_ERROR_SIZE = 2048
_SAMPLE = ct.CFUNCTYPE(
    ct.c_int, ct.c_uint32, ct.c_uint32, ct.c_uint32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
)


def _build(build_dir):
    """Compile against the headers and dylibs of the active PyTorch installation."""
    source = _SOURCES / "ransac.mm"
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib = torch_root / "lib"
    abi = f"{torch.version.git_version}:{torch_root}"
    key = hashlib.sha256(source.read_bytes() + abi.encode()).hexdigest()[:16]
    build_dir = Path(build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / f"mps_ransac_{key}.dylib"
    if library.is_file():
        return library
    with TemporaryDirectory(dir=build_dir) as temporary:
        built = Path(temporary) / library.name
        subprocess.run(
            [
                "xcrun", "clang++", "-std=c++20", "-O3", "-fobjc-arc", "-fblocks",
                "-dynamiclib", "-framework", "Foundation", "-framework", "Metal",
                "-I", str(torch_root / "include"), "-L", str(torch_lib), f"-Wl,-rpath,{torch_lib}",
                "-ltorch_cpu", "-lc10", str(source), "-o", str(built),
            ],
            check=True,
        )
        os.replace(built, library)
    return library


class NativeRansac:
    """Own Metal pipelines and scratch storage for one benchmark worker."""

    def __init__(self, build_dir):
        if sys.platform != "darwin" or not torch.backends.mps.is_available():
            raise RuntimeError("Native RANSAC requires PyTorch MPS on macOS")
        self.library_path = _build(build_dir)
        self.library = ct.CDLL(str(self.library_path))
        self.library.corner_ransac_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t]
        self.library.corner_ransac_create.restype = ct.c_void_p
        self.library.corner_ransac_destroy.argtypes = [ct.c_void_p]
        self.library.corner_ransac_destroy.restype = None
        self.library.corner_ransac_fit.argtypes = [
            ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_uint32, ct.c_void_p,
            ct.c_int, ct.c_int, ct.c_int, ct.c_float, _SAMPLE, ct.c_char_p, ct.c_size_t,
        ]
        self.library.corner_ransac_fit.restype = ct.c_int
        shader = metal_ops.SOURCE + "\n" + (_SOURCES / "ransac.metal").read_text()
        error = ct.create_string_buffer(_ERROR_SIZE)
        self._handle = self.library.corner_ransac_create(shader.encode(), error, _ERROR_SIZE)
        if not self._handle:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        self._lock = Lock()
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Release the native workspace after pending synchronous calls return."""
        with self._lock:
            if self._handle:
                self.library.corner_ransac_destroy(self._handle)
                self._handle = None

    @contextmanager
    def activate(self):
        """Select this implementation for optimized MPS RANSAC in the current context."""
        token = metal_ops._native_ransac.set(self)
        try:
            yield self
        finally:
            metal_ops._native_ransac.reset(token)

    def fit_planes(self, points, parameters, seed, maximum):
        """Return the accepted planes as a view of native MPS output storage."""
        if points.device.type != "mps" or points.dtype != torch.float32 or points.shape[1:] != (3,):
            raise ValueError("Native RANSAC expects an MPS float32 tensor with shape [N, 3]")
        minimum = parameters.minimum_plane_pixels
        trials = parameters.ransac_trials
        if minimum <= 0 or trials <= 0 or maximum < 0:
            raise ValueError("Native RANSAC requires positive inlier and trial limits and a nonnegative plane limit")
        if maximum == 0 or len(points) < minimum:
            return points.new_empty((0, 4))
        cloud = points.contiguous()
        output = cloud.new_empty((maximum, 4))
        rng = np.random.default_rng(seed)
        callback_error = None

        @_SAMPLE
        def sample(size, samples, trials, sample_ptr, triple_ptr):
            nonlocal callback_error
            # Keeping NumPy's scheduler preserves its data-dependent RNG state.
            try:
                np.ctypeslib.as_array(sample_ptr, shape=(samples,))[:] = rng.choice(
                    size, samples, replace=False
                )
                np.ctypeslib.as_array(triple_ptr, shape=(trials, 3))[:] = rng.choice(
                    samples, (trials, 3), replace=True
                )
                return 1
            except BaseException as error:
                callback_error = error
                return 0

        # On the MPS allocator storage.data_ptr() is an id<MTLBuffer>.
        # The tensor owns the storage, including when cloud has a nonzero offset.
        error = ct.create_string_buffer(_ERROR_SIZE)
        with self._lock:
            if self._handle is None:
                raise RuntimeError("Native RANSAC workspace is closed")
            self.calls += 1
            count = self.library.corner_ransac_fit(
                self._handle,
                cloud.untyped_storage().data_ptr(),
                cloud.storage_offset() * cloud.element_size(),
                len(cloud),
                output.untyped_storage().data_ptr(),
                trials,
                minimum,
                maximum,
                parameters.plane_tolerance_m,
                sample,
                error,
                _ERROR_SIZE,
            )
        if callback_error is not None:
            raise callback_error
        if count < 0:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        return output.narrow(0, 0, count)
