"""GPU image boundary for the existing, unchanged robot-filter API.

Only snapshot ownership and resident depth arguments are adapted here. Hull
preparation and filtering are delegated to their existing implementations.
"""
import numpy as np
import torch
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation import RobotFilterPreparation,PendingPreparation
from graph.persistent_geometry.tensor_camera import TensorCameraObservation,tensor_camera
from graph.persistent_geometry.tensor_fusion import depth_from_mm
from graph.persistent_geometry.contracts import synchronized
from graph.persistent_geometry.self_filter import LinkSurface


class TensorRobotPreparation(RobotFilterPreparation):
    def submit(self,cameras,surfaces):
        cameras=tuple(cameras)
        if not any(isinstance(c,TensorCameraObservation) for c in cameras):
            job=super().submit(cameras,surfaces)
        else:
            if self._closed:raise RuntimeError('preparation is closed')
            if self._pending is not None:raise RuntimeError('previous prepared frame has not been consumed')
            if not all(isinstance(c,TensorCameraObservation) for c in cameras):raise TypeError('mixed image devices')
            synchronized(cameras)
            snapshots=tuple(tensor_camera(c) for c in cameras)
            shapes=tuple(LinkSurface(s.link,np.array(s.local_vertices,copy=True),np.array(s.world_from_link,copy=True)) for s in surfaces)
            future=self._executor.submit(self.robot_filter.prepare_surfaces,shapes,snapshots)
            job=PendingPreparation(snapshots,shapes,future);self._pending=job
        self.depths={c.camera:depth_from_mm(torch.as_tensor(c.depth_raw_mm,device='cuda')) for c in job.cameras}
        return job

    def filter(self,job,masks,**kwargs):
        tensors={c.camera:dict(depth_cuda=self.depths[c.camera],world_cuda=torch.as_tensor(c.world_from_camera.astype(np.float64),device='cuda')) for c in job.cameras}
        return super().filter(job,masks,camera_tensors=tensors,**kwargs)
