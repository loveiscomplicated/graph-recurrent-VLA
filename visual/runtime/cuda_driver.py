"""Small experimental CUDA-driver bridge; no Torch/runtime package install.

Only used by the opt-in sampling probe. Compiles our kernels with the installed
nvcc and keeps allocations in one primary context. Production defaults do not
depend on this module.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


class CudaDriver:
    def __init__(self, source=None):
        self.lib = C.CDLL('libcuda.so.1')
        self._buffers = {}
        self._functions = {}
        def bind(name, args):
            f = getattr(self.lib, name); f.restype = C.c_int; f.argtypes = args
            return f
        self.init = bind('cuInit', [C.c_uint])
        self.device_get = bind('cuDeviceGet', [C.POINTER(C.c_int), C.c_int])
        self.attribute = bind('cuDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int])
        self.primary = bind('cuDevicePrimaryCtxRetain', [C.POINTER(C.c_void_p), C.c_int])
        self.current = bind('cuCtxSetCurrent', [C.c_void_p])
        self.sync_call = bind('cuCtxSynchronize', [])
        self.alloc = bind('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t])
        self.free = bind('cuMemFree_v2', [C.c_uint64])
        self.htod = bind('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t])
        self.dtoh = bind('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t])
        self.load = bind('cuModuleLoad', [C.POINTER(C.c_void_p), C.c_char_p])
        self.func = bind('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self.launch_call = bind('cuLaunchKernel', [C.c_void_p, *([C.c_uint]*7), C.c_void_p,
                                                   C.POINTER(C.c_void_p), C.c_void_p])
        self.check(self.init(0)); device = C.c_int(); self.check(self.device_get(C.byref(device), 0))
        major, minor = C.c_int(), C.c_int()
        self.check(self.attribute(C.byref(major), 75, device))
        self.check(self.attribute(C.byref(minor), 76, device))
        self.compute_capability = f'{major.value}{minor.value}'
        self.context = C.c_void_p(); self.check(self.primary(C.byref(self.context), device))
        self.check(self.current(self.context))
        self.module = C.c_void_p()
        if source is not None:
            source = Path(source)
            nvcc = os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc')
            version = subprocess.check_output([nvcc, '--version'])
            key = hashlib.sha256(source.read_bytes()+version+self.compute_capability.encode()).hexdigest()
            cache = Path(os.environ.get('VISION_CUDA_CACHE', Path(tempfile.gettempdir())/'vision-sampling-cuda'))
            cache.mkdir(parents=True, exist_ok=True)
            ptx = cache/(key+'.ptx')
            if not ptx.exists():
                partial = cache/(key+f'.{os.getpid()}.partial.ptx')
                subprocess.run([nvcc, '-O3', '-ptx', '-arch=compute_'+self.compute_capability,
                                str(source), '-o', str(partial)], check=True)
                partial.replace(ptx)
            self.check(self.load(C.byref(self.module), os.fsencode(ptx)))

    def check(self, code):
        if code:
            name = C.c_char_p()
            self.lib.cuGetErrorName(C.c_int(code), C.byref(name))
            raise RuntimeError(f'CUDA driver {code}: {name.value!r}')

    def sync(self):
        self.check(self.sync_call())

    def buffer(self, name, size):
        size = max(1, int(size))
        if name not in self._buffers or self._buffers[name][1] < size:
            if name in self._buffers: self.check(self.free(self._buffers[name][0]))
            p = C.c_uint64(); self.check(self.alloc(C.byref(p), size))
            self._buffers[name] = (p, size)
        return self._buffers[name][0]

    def upload(self, name, array):
        import numpy as np
        a = np.ascontiguousarray(array)
        p = self.buffer(name, a.nbytes)
        if a.nbytes: self.check(self.htod(p, C.c_void_p(a.ctypes.data), a.nbytes))
        return p

    def download(self, name, shape, dtype):
        import numpy as np
        a = np.empty(shape, dtype=dtype)
        p = self.buffer(name, a.nbytes)
        if a.nbytes: self.check(self.dtoh(C.c_void_p(a.ctypes.data), p, a.nbytes))
        return a

    def launch(self, name, grid, block, arguments):
        if name not in self._functions:
            f = C.c_void_p(); self.check(self.func(C.byref(f), self.module, name.encode()))
            self._functions[name] = f
        params = (C.c_void_p*len(arguments))(*[C.addressof(x) for x in arguments])
        self.check(self.launch_call(self._functions[name], grid, 1, 1, block, 1, 1, 0, None, params, None))

    def close(self):
        self.sync()
        for p, _ in self._buffers.values(): self.check(self.free(p))
        self._buffers.clear()
