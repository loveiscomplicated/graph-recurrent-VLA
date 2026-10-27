"""Opt-in FP32 PCA5/FPS experiment, returning existing CPU evidence types.

Local connected support, configured gates, source identity and budget semantics
remain. Float32 eigensystems and FPS may change normals/candidates/selected rows;
the probe reports those differences rather than requiring bitwise equality.
Transfers are part of the measured public call. No CPU default is switched.
"""
import ctypes as C
from pathlib import Path
from time import perf_counter

import numpy as np

from visual.runtime.cuda_driver import CudaDriver
from visual.runtime.geometry import GeometryFrame, GeometryParameters
from visual.runtime.pca import build_geometry, cropped_features
from visual.runtime.edge_evidence import pair_evidence
from visual.runtime.sampling import SamplingEvidence, _integer


class CudaSampling:
    def __init__(self):
        self.driver = CudaDriver(Path(__file__).with_suffix('.cu'))

    def normals(self, geometry, parameters=GeometryParameters()):
        d = self.driver; h, w = geometry.valid_depth.shape
        xyz = d.upload('xyz', geometry.xyz.astype(np.float32, copy=False))
        supports = np.stack([geometry.observed_mask(i) for i in (0, 1)]).astype(np.uint8)
        support = d.upload('support', supports)
        normal = d.buffer('normal', 2*h*w*3*4); valid = d.buffer('normal_valid', 2*h*w)
        d.launch('pca5', (2*h*w+127)//128, 128,
                 [xyz, support, normal, valid, C.c_int(h), C.c_int(w),
                  C.c_float(parameters.depth_jump_abs_m), C.c_float(parameters.depth_jump_rel)])
        values = d.download('normal', (2, h, w, 3), np.float32)
        available = d.download('normal_valid', (2, h, w), np.uint8).astype(bool)
        return {i: (values[i], available[i]) for i in (0, 1)}

    def evidence(self, frame, parameters=GeometryParameters()):
        start = perf_counter(); geometry = build_geometry(frame); a = perf_counter()
        normals = self.normals(geometry, parameters); b = perf_counter()
        features = {i: cropped_features(geometry, i, *normals[i], parameters) for i in (0, 1)}
        c = perf_counter()
        pairs = {i: pair_evidence(geometry, i, *normals[i], parameters) for i in (0, 1)}
        end = perf_counter()
        timing = dict(geometry=(a-start)*1000, normals=(b-a)*1000, features=(c-b)*1000,
                      pair_evidence=(end-c)*1000, extraction_total=(end-start)*1000)
        result = GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx, frame.timestamp,
                               geometry, features, timing)
        return SamplingEvidence(result, pairs, 'pca5', timing)

    def fps_indices(self, points, count, *, seed=0, initial=()):
        _integer(count, 'FPS count'); _integer(seed, 'FPS seed')
        xyz = np.asarray(points, np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('FPS requires finite [N,3] points')
        initial = np.asarray(initial, np.int64)
        if initial.ndim != 1 or len(np.unique(initial)) != len(initial) or ((initial<0)|(initial>=len(xyz))).any():
            raise ValueError('invalid/duplicate initial source rows')
        count = min(count, len(xyz))
        if len(initial)>count: raise ValueError('initial rows exceed FPS count')
        if not count: return np.empty(0, np.int64)
        if len(xyz)>np.iinfo(np.int32).max: raise ValueError('FPS int32 capacity exceeded')
        d = self.driver
        source = d.upload('fps_xyz', xyz)
        prefix = d.upload('fps_initial', initial.astype(np.int32))
        output = d.buffer('fps_output', count*4); distance = d.buffer('fps_distance', len(xyz)*4)
        first = int(np.random.default_rng(seed).integers(len(xyz))) if not len(initial) else int(initial[0])
        d.launch('fps', 1, 256, [source, distance, output, prefix, C.c_int(len(xyz)),
                 C.c_int(count), C.c_int(len(initial)), C.c_int(first)])
        rows = d.download('fps_output', (count,), np.int32).astype(np.int64)
        if len(np.unique(rows)) != len(rows) or not np.array_equal(rows[:len(initial)], initial):
            raise RuntimeError('GPU FPS violated source uniqueness/prefix')
        return rows

    def close(self):
        self.driver.close()
