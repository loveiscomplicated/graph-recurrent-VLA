"""Exact current-frame memoization of repeated, mask-independent robot classification."""
import hashlib
import copy
import json
import types
import numpy as np


class CurrentRobotMemo:
    def __init__(self,classifier):
        self.classifier=classifier;self.frame=None;self.cache={};self.hits=0;self.misses=0
    def __call__(self,camera,surfaces):
        frame=(camera.episode,camera.timestep,camera.timestamp)
        if frame!=self.frame:self.frame=frame;self.cache.clear()
        h=hashlib.sha256(camera.camera.encode())
        arrays=[camera.depth_raw_mm,camera.intrinsic,camera.extrinsic_cv,camera.cam2world_gl,camera.tcp_pose]
        for s in surfaces:
            h.update(s.link.encode());arrays.extend((s.local_vertices,s.world_from_link))
        for array in arrays:
            a=np.asarray(array);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
        key=h.digest()
        if key not in self.cache:
            self.misses+=1;self.cache[key]=self.classifier(camera,surfaces)
        else:self.hits+=1
        return self.cache[key]


def install(geometry):
    original=geometry.system.prepare.__func__
    namespace=original.__globals__.copy()
    memo=CurrentRobotMemo(namespace['classify_robot'])
    namespace['classify_robot']=memo
    # Preserve the ORIGINAL prepare bytecode; bind only its pure classifier name
    # in an instance-owned namespace. No shared-module monkeypatch or stale map.
    function=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    native=types.MethodType(function,geometry.system)
    prepared={};last_frame=[None]
    def prepare(cameras,data,masks,surfaces,*,clean_masks=None):
        frame=tuple((c.episode,c.timestep,c.camera,c.timestamp) for c in cameras)
        if frame!=last_frame[0]:last_frame[0]=frame;prepared.clear()
        h=hashlib.sha256(repr(frame).encode())
        h.update(json.dumps(dict(main=geometry.system.main,wrist=geometry.system.wrist,protocol=geometry.system.protocol,
            wrist_gate_policy=geometry.system.wrist_gate_backend,precleaned=clean_masks is not None),sort_keys=True).encode())
        arrays=[]
        for c in cameras:
            arrays.extend((c.rgb,c.depth_raw_mm,c.intrinsic,c.extrinsic_cv,c.cam2world_gl,c.tcp_pose,masks[c.camera]))
            if clean_masks is not None:arrays.append(clean_masks[c.camera])
        for s in surfaces:h.update(s.link.encode());arrays.extend((s.local_vertices,s.world_from_link))
        for k in sorted(data):h.update(k.encode());arrays.append(data[k])
        for array in arrays:
            a=np.asarray(array);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
        key=h.digest()
        if key in prepared:return copy.deepcopy(prepared[key])
        result=native(cameras,data,masks,surfaces,clean_masks=clean_masks)
        if len(prepared)>=16:prepared.clear()
        prepared[key]=copy.deepcopy(result)
        return result
    geometry.system.prepare=prepare
    return memo
