"""Explicit CPU compatibility boundary AFTER the GPU robot filter.

CurrentGeometry.build accepts raw CUDA masks inside this scope. Filtering returns
clean CUDA masks first. Only then are raw+clean masks copied for the unchanged
PlanA gate/selection/Graph consumers. Raw masks are retained so gate fractions
are not silently recomputed with already-filtered masks. No second filtering.
Process-local patches; no concurrent builders in one process.
"""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class _AlreadyFiltered:
    camera: str


class GpuMaskBoundary:
    def __init__(self,robot_filter=None):
        if robot_filter is None:
            from .gpu_filter import CudaRobotFilter
            robot_filter=CudaRobotFilter()
        self.robot_filter=robot_filter;self.clean_cuda={};self._active=None;self._restore=[]

    def __enter__(self):
        if self._restore:raise RuntimeError('boundary already active')
        import graph.plan_a_geometry.core as core
        original_prepare=core.PlanA.prepare
        original_classifier=core.classify_robot;original_filter=core.filtered_masks
        ctx=self
        def prepare(system,cameras,data,masks,surfaces):
            ctx.clean_cuda={}
            if not any(isinstance(m,torch.Tensor) and m.is_cuda for m in masks.values()):
                return original_prepare(system,cameras,data,masks,surfaces)
            if not all(isinstance(m,torch.Tensor) and m.is_cuda for m in masks.values()):
                raise ValueError('mixed CPU/CUDA camera masks are not supported')
            core.synchronized(cameras)
            if {c.camera for c in cameras}!=set(masks):raise ValueError('camera identity')
            # There is no raw-mask host copy before these filter calls.
            ctx.clean_cuda={c.camera:ctx.robot_filter(c,surfaces,masks[c.camera]) for c in cameras}
            raw={name:value.detach().cpu().numpy() for name,value in masks.items()}
            clean={name:value.detach().cpu().numpy() for name,value in ctx.clean_cuda.items()}
            ctx._active=clean
            try:return original_prepare(system,cameras,data,raw,surfaces)
            finally:ctx._active=None
        def classify(camera,surfaces,**kwargs):
            if ctx._active is None:return original_classifier(camera,surfaces,**kwargs)
            return _AlreadyFiltered(camera.camera)
        def filtered(raw,result):
            if not isinstance(result,_AlreadyFiltered):return original_filter(raw,result)
            clean=ctx._active[result.camera]
            return clean,np.any(np.asarray(raw,bool)&~clean,axis=0)
        for owner,name,value in ((core.PlanA,'prepare',prepare),(core,'classify_robot',classify),(core,'filtered_masks',filtered)):
            self._restore.append((owner,name,getattr(owner,name)));setattr(owner,name,value)
        return self

    def __exit__(self,*args):
        for owner,name,value in reversed(self._restore):setattr(owner,name,value)
        self._restore.clear();self._active=None
