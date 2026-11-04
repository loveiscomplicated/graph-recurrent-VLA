"""Current-frame CPU preparation that can overlap main-thread SAM2 inference.

The worker runs only NumPy/SciPy geometry. CUDA/model calls stay on the caller's
thread and stream. One outstanding snapshot bounds the queue and prevents a
prepared frame from being silently reused for a later observation.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import numpy as np
from graph.geometry.contracts import CameraObservation, synchronized
from graph.geometry.self_filter import LinkSurface


@dataclass(frozen=True)
class PendingPreparation:
    cameras: tuple
    surfaces: tuple
    future: object


class RobotFilterPreparation:
    def __init__(self,robot_filter):
        self.robot_filter=robot_filter
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='robot-prepare')
        self._pending=None;self._closed=False

    def submit(self,cameras,surfaces):
        if self._closed:raise RuntimeError('preparation is closed')
        if self._pending is not None:raise RuntimeError('previous prepared frame has not been consumed')
        cameras=tuple(cameras);synchronized(cameras)
        if any(type(c) is not CameraObservation for c in cameras):
            raise TypeError('async preparation requires CameraObservation snapshots')
        # Calibration, raw depth and FK matrices may be overwritten by the next
        # simulator step. Snapshot before starting either concurrent branch.
        # RGB is not read by preparation/filtering; SAM2 receives its own input.
        copies=tuple(replace(c,depth_raw_mm=c.depth_raw_mm.copy(),intrinsic=c.intrinsic.copy(),
            extrinsic_cv=c.extrinsic_cv.copy(),cam2world_gl=c.cam2world_gl.copy(),tcp_pose=c.tcp_pose.copy()) for c in cameras)
        shapes=tuple(LinkSurface(s.link,np.array(s.local_vertices,copy=True),np.array(s.world_from_link,copy=True)) for s in surfaces)
        future=self._executor.submit(self.robot_filter.prepare_surfaces,shapes,copies)
        job=PendingPreparation(copies,shapes,future);self._pending=job
        return job

    def filter(self,job,masks,**kwargs):
        if job is not self._pending:raise ValueError('foreign or already consumed preparation')
        try:
            prepared=job.future.result()
            return self.robot_filter.filter_cameras(job.cameras,job.surfaces,masks,prepared=prepared,**kwargs)
        finally:self._pending=None

    def discard(self,job):
        if job is not self._pending:return
        try:
            if not job.future.cancel():
                try:job.future.result()
                except Exception:pass # preserve an original SAM2/caller exception
        finally:self._pending=None

    def close(self):
        if self._closed:return
        self._closed=True
        self._executor.shutdown(wait=True,cancel_futures=True);self._pending=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
