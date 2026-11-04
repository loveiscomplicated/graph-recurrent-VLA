"""Snapshot CPU simulator observations; upload raw payload once per frame."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from graph.geometry.contracts import synchronized
from graph.perception.vision_packet import pose_matrix, camera_to_ee


@dataclass
class CameraTensors:
    camera: object
    rgb: torch.Tensor
    raw_depth: torch.Tensor
    depth64: torch.Tensor
    depth32: torch.Tensor
    intrinsic: torch.Tensor
    world: torch.Tensor
    inverse: torch.Tensor
    ee: torch.Tensor
    world_ee: torch.Tensor

    def world_points(self,pixels):
        h,w=self.depth64.shape;k=self.intrinsic
        z=self.depth64.flatten()[pixels];u=(pixels%w).to(torch.float64);v=(pixels//w).to(torch.float64)
        xyz=torch.stack(((u+.5-k[0,2])*z/k[0,0],(v+.5-k[1,2])*z/k[1,1],z),1)
        return xyz@self.world[:3,:3].T+self.world[:3,3]


@dataclass
class StagedFrame:
    cameras: tuple
    tensors: dict
    data: dict
    data_cuda: dict
    surfaces: tuple
    prepared: object


class TensorIngress:
    def __init__(self,robot_filter,device='cuda',*,hull_workers=4):
        self.robot_filter=robot_filter;self.device=torch.device(device)
        self.executor=ThreadPoolExecutor(hull_workers) if hull_workers else None

    def close(self):
        if self.executor is not None:self.executor.shutdown();self.executor=None

    @torch.inference_mode()
    def stage(self,cameras,data,surfaces):
        synchronized(cameras)
        allowed={'robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world','timestep','timestamp','episode'}
        allowed.update(c.camera+'/'+k for c in cameras for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
        if set(data)-allowed:raise ValueError('forbidden non-sensor input')
        if len(cameras)!=2 or {c.camera for c in cameras}!={'base_camera','hand_camera'}:raise ValueError('camera identity')
        if str(data['episode'])!=cameras[0].episode or int(data['timestep'])!=cameras[0].timestep:raise ValueError('sensor identity join')
        snapshots=[];chunks=[];layout={};offset=0
        def add(key,array):
            nonlocal offset
            a=np.ascontiguousarray(array);pad=(-offset)%max(8,a.dtype.itemsize)
            if pad:chunks.append(np.zeros(pad,np.uint8));offset+=pad
            layout[key]=(offset,a.nbytes,a.dtype,a.shape);chunks.append(a.view(np.uint8).reshape(-1));offset+=a.nbytes
        for c in cameras:
            # Current calibration and signed sensor depth have independent
            # snapshots even if the simulator reuses its next-step CPU buffers.
            snap=replace(c,depth_raw_mm=c.depth_raw_mm.copy(),intrinsic=c.intrinsic.copy(),
                extrinsic_cv=c.extrinsic_cv.copy(),cam2world_gl=c.cam2world_gl.copy(),tcp_pose=c.tcp_pose.copy())
            snap.validate();snapshots.append(snap);name=c.camera
            world=snap.world_from_camera
            add((name,'rgb'),snap.rgb);add((name,'raw_depth'),snap.depth_raw_mm)
            add((name,'intrinsic'),np.asarray(snap.intrinsic,np.float64));add((name,'world'),world.astype(np.float64))
            add((name,'inverse'),np.linalg.inv(world).astype(np.float64))
            add((name,'ee'),camera_to_ee(SimpleNamespace(tcp_pose=snap.tcp_pose,cam2world=world)))
            add((name,'world_ee'),pose_matrix(snap.tcp_pose))
        copied={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in data.items()}
        for k,v in copied.items():
            if isinstance(v,np.ndarray) and v.dtype.kind in 'fiu' and v.ndim>0:add(('robot',k),v)
        arena=torch.as_tensor(np.concatenate(chunks),device=self.device)
        dtypes={np.dtype('uint8'):torch.uint8,np.dtype('int16'):torch.int16,np.dtype('float32'):torch.float32,
            np.dtype('float64'):torch.float64,np.dtype('int64'):torch.int64,np.dtype('int32'):torch.int32}
        def get(key):
            a,n,dtype,shape=layout[key]
            return arena[a:a+n].view(dtypes[dtype]).reshape(shape)
        tensors={}
        for c in snapshots:
            name=c.camera;raw=get((name,'raw_depth'));depth=self.robot_filter.depth_tensor(raw)
            tensors[name]=CameraTensors(c,get((name,'rgb')),raw,depth,depth.to(torch.float32),
                *[get((name,k)) for k in ('intrinsic','world','inverse','ee','world_ee')])
        robot={k:get(('robot',k)) for k in copied if ('robot',k) in layout}
        surfaces=tuple(surfaces)
        # Exact current world-Qhull, never cached by episode/time or pose. The
        # raw-depth CUDA work has already been submitted while this CPU runs.
        prepared=self.robot_filter.prepare_surfaces(surfaces,snapshots,executor=self.executor)
        return StagedFrame(tuple(snapshots),tensors,copied,robot,surfaces,prepared)

    def filter(self,frame,masks):
        current={name:dict(depth_cuda=t.depth64,world_cuda=t.world) for name,t in frame.tensors.items()}
        return self.robot_filter.filter_cameras(frame.cameras,frame.surfaces,masks,
            camera_tensors=current,prepared=frame.prepared)
