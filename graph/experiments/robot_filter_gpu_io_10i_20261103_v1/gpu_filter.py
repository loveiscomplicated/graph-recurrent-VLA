"""CUDA mask -> CUDA clean mask. No mask readback or scalar synchronization.

Collision preparation remains CPU; ray/numerator arithmetic matches Stage 9.
Full-screen grouped arithmetic is retained for <=128². Larger images clip only
raw mask-union lanes inside the existing link bounds. Diagnostics explicitly
carry evaluated_pixels; values elsewhere are not a full-screen classification.
"""
import numpy as np
from dataclasses import dataclass, field
from collections import OrderedDict
import torch
import triton
import triton.language as tl
from scipy.spatial import ConvexHull
from visual.runtime.vision_packet import validate_transform
from graph.persistent_geometry.contracts import CameraObservation
from graph.experiments.graph_latency_10i_20261103_v2.robot import _links, _reduce


@triton.jit
def _depth_mm(Raw,Table,Depth,N:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    mm=tl.load(Raw+i,ok,0).to(tl.int32)
    # Finite int16 domain: exact NumPy float64 values, no per-frame division.
    value=tl.load(Table+mm,ok&(mm>0),float('nan'))
    tl.store(Depth+i,value,ok)


@triton.jit
def _clip_grouped(D,Mask,Normals,Num,Depth,Meta,Z,Interior,Entries,
                  W:tl.constexpr,H:tl.constexpr,
                  Planes:tl.constexpr,DIAG:tl.constexpr,P:tl.constexpr=32):
    link=tl.program_id(1)
    start=tl.load(Meta+8*link);F=tl.load(Meta+8*link+1)
    X=tl.load(Meta+8*link+2);Y=tl.load(Meta+8*link+3)
    BW=tl.load(Meta+8*link+4);N=tl.load(Meta+8*link+5)
    idx=tl.program_id(0)*P+tl.arange(0,P)
    row=(Y+idx//BW)*W+X+idx%BW
    active=(idx<N)&(tl.load(Mask+row,idx<N,0)|tl.load(Mask+H*W+row,idx<N,0))
    if tl.sum(active.to(tl.int32),0)>0:
        dx=tl.load(D+3*row,active,0);dy=tl.load(D+3*row+1,active,0);dz=tl.load(D+3*row+2,active,0)
        face=tl.arange(0,Planes);offset=start+face
        nx=tl.load(Normals+3*offset,face<F,0);ny=tl.load(Normals+3*offset+1,face<F,0);nz=tl.load(Normals+3*offset+2,face<F,0)
        num=tl.load(Num+offset,face<F,0)
        denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
        parallel=tl.abs(denom)<1e-12;crossing=tl.where(parallel,0,num[None,:]/denom)
        lower=tl.max(tl.where((face[None,:]<F)&(denom < -1e-12),crossing,-float('inf')),1)
        upper=tl.min(tl.where((face[None,:]<F)&(denom > 1e-12),crossing,float('inf')),1)
        outside=tl.sum(((face[None,:]<F)&parallel&(num[None,:]<0)).to(tl.int32),1)>0
        entry=tl.maximum(lower,0);hit=(upper>=entry)&(upper>0)&~outside&active
        measured=tl.load(Depth+row,active,float('nan'))
        interior=hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005)
        tl.atomic_or(Interior+row,interior.to(tl.int32),active,sem='relaxed')
        tl.atomic_min(Z+row,entry,hit,sem='relaxed')
        if DIAG:
            es=tl.load(Meta+8*link+7)
            tl.store(Entries+es+idx,tl.where(hit,entry,float('inf')),active)


@triton.jit
def _compact_rows(Mask,Meta,Counts,Rows,W:tl.constexpr,H:tl.constexpr,B:tl.constexpr=256):
    link=tl.program_id(1);idx=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(Meta+8*link+2);y=tl.load(Meta+8*link+3)
    width=tl.load(Meta+8*link+4);n=tl.load(Meta+8*link+5)
    original=tl.load(Meta+8*link+6);base=tl.load(Meta+8*link+7)
    row=(y+idx//width)*W+x+idx%width
    active=(idx<n)&(tl.load(Mask+row,idx<n,0)|tl.load(Mask+H*W+row,idx<n,0))
    total=tl.sum(active.to(tl.int32),0)
    if total>0:
        start=tl.atomic_add(Counts+original,total,sem='relaxed')
        offset=tl.cumsum(active.to(tl.int32),0)-1
        tl.store(Rows+base+start+offset,row,active)


@triton.jit
def _clip_compact(D,Normals,Num,Depth,Meta,Counts,Rows,Z,Interior,Entries,
                  Planes:tl.constexpr,DIAG:tl.constexpr,P:tl.constexpr=32):
    link=tl.program_id(1);idx=tl.program_id(0)*P+tl.arange(0,P)
    original=tl.load(Meta+8*link+6);n=tl.load(Counts+original)
    if tl.program_id(0)*P<n:
        base=tl.load(Meta+8*link+7);start=tl.load(Meta+8*link);faces=tl.load(Meta+8*link+1)
        active=idx<n;row=tl.load(Rows+base+idx,active,0)
        dx=tl.load(D+3*row,active,0);dy=tl.load(D+3*row+1,active,0);dz=tl.load(D+3*row+2,active,0)
        f=tl.arange(0,Planes);face=start+f
        nx=tl.load(Normals+3*face,f<faces,0);ny=tl.load(Normals+3*face+1,f<faces,0);nz=tl.load(Normals+3*face+2,f<faces,0)
        num=tl.load(Num+face,f<faces,0)
        denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
        parallel=tl.abs(denom)<1e-12;crossing=tl.where(parallel,0,num[None,:]/denom)
        lower=tl.max(tl.where((f[None,:]<faces)&(denom < -1e-12),crossing,-float('inf')),1)
        upper=tl.min(tl.where((f[None,:]<faces)&(denom > 1e-12),crossing,float('inf')),1)
        outside=tl.sum(((f[None,:]<faces)&parallel&(num[None,:]<0)).to(tl.int32),1)>0
        entry=tl.maximum(lower,0);hit=(upper>=entry)&(upper>0)&~outside&active
        measured=tl.load(Depth+row,active,float('nan'))
        interior=hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005)
        tl.atomic_or(Interior+row,interior.to(tl.int32),active,sem='relaxed')
        tl.atomic_min(Z+row,entry,hit,sem='relaxed')
        if DIAG:tl.store(Entries+base+idx,tl.where(hit,entry,float('inf')),active)


@triton.jit
def _compact_owners(Meta,Counts,Rows,Entries,Z,Owner,P:tl.constexpr=256):
    link=tl.program_id(1);idx=tl.program_id(0)*P+tl.arange(0,P)
    original=tl.load(Meta+8*link+6);n=tl.load(Counts+original);base=tl.load(Meta+8*link+7)
    if tl.program_id(0)*P<n:
        active=idx<n;row=tl.load(Rows+base+idx,active,0)
        entry=tl.load(Entries+base+idx,active,float('inf'));nearest=tl.load(Z+row,active,float('inf'))
        tl.atomic_min(Owner+row,original,active&(entry==nearest)&(nearest<float('inf')),sem='relaxed')


@triton.jit
def _owners(Mask,Meta,Entries,Z,Owner,W:tl.constexpr,H:tl.constexpr,P:tl.constexpr=256):
    link=tl.program_id(1);idx=tl.program_id(0)*P+tl.arange(0,P)
    X=tl.load(Meta+8*link+2);Y=tl.load(Meta+8*link+3)
    BW=tl.load(Meta+8*link+4);N=tl.load(Meta+8*link+5)
    original=tl.load(Meta+8*link+6);es=tl.load(Meta+8*link+7)
    row=(Y+idx//BW)*W+X+idx%BW
    active=(idx<N)&(tl.load(Mask+row,idx<N,0)|tl.load(Mask+H*W+row,idx<N,0))
    entry=tl.load(Entries+es+idx,active,float('inf'));nearest=tl.load(Z+row,active,float('inf'))
    tl.atomic_min(Owner+row,original,active&(entry==nearest)&(nearest<float('inf')),sem='relaxed')


@triton.jit
def _finish_owner(Owner,N:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(Owner+i,i<N,2147483647)
    tl.store(Owner+i,tl.where(value==2147483647,-1,value),i<N)


@triton.jit
def _clean(Mask,Depth,Z,Inside,Clean,Flags,N:tl.constexpr,
           TOL:tl.constexpr,BAND:tl.constexpr,DIAG:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    a=tl.load(Mask+i,ok,0);b=tl.load(Mask+N+i,ok,0)
    d=tl.load(Depth+i,ok,float('nan'));z=tl.load(Z+i,ok,float('inf'));inside=tl.load(Inside+i,ok,0)
    valid=(d==d)&(tl.abs(d)<float('inf'));delta=tl.abs(d-z)
    if N>128*128:
        # High-resolution Stage 9 uses Torch float64 comparisons. Triton's
        # implicit float literal would otherwise round the limits to float32.
        tol=tl.full((),TOL,tl.float64);band=tl.full((),BAND,tl.float64)
    else:
        # Preserve the existing small grouped classifier's arithmetic.
        tol=TOL;band=BAND
    confirmed=valid&(((z<float('inf'))&(delta<=tol))|inside)
    uncertain=valid&(z<float('inf'))&~confirmed&(delta<=band)
    rejected=(a&b)|confirmed|uncertain
    tl.store(Clean+i,a&~rejected,ok);tl.store(Clean+N+i,b&~rejected,ok)
    if DIAG:
        tl.store(Flags+i,confirmed,ok);tl.store(Flags+N+i,uncertain,ok)
        tl.store(Flags+2*N+i,valid&~confirmed&~uncertain,ok)


def bounds(camera,vertices,camera_from_world=None):
    matrix=np.linalg.inv(camera.world_from_camera) if camera_from_world is None else camera_from_world
    return _bounds(camera,vertices,validate_transform(np.asarray(matrix,dtype=float)))


def _bounds(camera,vertices,camera_from_world):
    # Same transform operations as contracts.transform; validate the shared
    # camera transform once per frame/camera instead of once per link.
    h,w=camera.depth_raw_mm.shape
    vc=np.asarray(vertices,dtype=float)@camera_from_world[:3,:3].T+camera_from_world[:3,3]
    if np.all(vc[:,2]<=0):return (0,0,-1,-1)
    if not np.all(vc[:,2]>0):return (0,0,w-1,h-1)
    k=camera.intrinsic;uv=vc[:,:2]/vc[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]-.5
    lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w-1,h-1])
    return (0,0,-1,-1) if np.any(hi<lo) else (*map(int,lo),*map(int,hi))


@dataclass(frozen=True)
class _PreparedSurfaces:
    hulls: tuple
    starts: np.ndarray
    equations: np.ndarray
    device_equations: dict = field(default_factory=dict)
    camera_bounds: dict = field(default_factory=dict)
    packed: np.ndarray | None = None
    camera_groups: dict = field(default_factory=dict)


class CudaRobotFilter:
    """One call = one current camera/FK snapshot; no cross-frame pose cache.

    Input masks: CUDA bool [2,H,W], object order Peg/Box. Output has identical
    shape/device/dtype and never aliases or modifies the raw SAM2 masks.
    Optional depth_cuda is float64 CV-z metres on the same device, including NaN
    for invalid depth. CPU sensor calibration/FK/hulls may still upload to CUDA.
    """
    def __init__(self,*,compact=True,clip_pixels=32,depth_from_mm=True):
        if clip_pixels not in (8,16,32,64):raise ValueError('clip_pixels must be 8/16/32/64')
        # Calibration-only cache; current camera pose, depth, hulls and decisions
        # are never stored here. Content keys detect in-place intrinsic edits.
        self._rays=OrderedDict()
        self._compact=compact
        self._clip_pixels=clip_pixels
        self._depth_from_mm=depth_from_mm
        self._depth_tables={}

    def _depth_table(self,device):
        """256KiB per used device, universal mm mapping; no frame state."""
        if device not in self._depth_tables:
            values=np.arange(32768,dtype=np.float64)/1000
            values[0]=np.nan
            self._depth_tables[device]=torch.as_tensor(values,device=device)
        return self._depth_tables[device]

    def _camera_rays(self,camera,device):
        h,w=camera.depth_raw_mm.shape;k_cpu=np.asarray(camera.intrinsic,np.float64)
        key=(h,w,k_cpu.tobytes(),device)
        rays=self._rays.pop(key,None)
        if rays is None:
            k=torch.as_tensor(np.ascontiguousarray(k_cpu),device=device)
            v,u=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float64),torch.arange(w,device=device,dtype=torch.float64),indexing='ij')
            rays=torch.stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],torch.ones_like(u)),2).reshape(-1,3)
        self._rays[key]=rays
        while len(self._rays)>8 or sum(v.numel()*v.element_size() for v in self._rays.values())>64*1024**2:
            self._rays.popitem(last=False)
        return rays

    @staticmethod
    def prepare_surfaces(surfaces,cameras=()):
        """Explicit current-frame snapshot, shared only by its camera calls."""
        vertices=[s.world_vertices for s in surfaces]
        camera_bounds={}
        for camera in cameras:
            matrix=validate_transform(np.linalg.inv(camera.world_from_camera))
            camera_bounds[id(camera)]=[_bounds(camera,v,matrix) for v in vertices]
        visible=[any(bb[i][2]>=bb[i][0] and bb[i][3]>=bb[i][1] for bb in camera_bounds.values())
                 for i in range(len(vertices))] if camera_bounds else [True]*len(vertices)
        # Retain every original link slot, including invisible shapes. This
        # preserves owner indices and the small-resolution grouped reduction.
        hulls=[(v,ConvexHull(v).equations if visible[i] else np.empty((0,4),np.float64))
               for i,v in enumerate(vertices)]
        starts=np.r_[0,np.cumsum([len(eq) for _,eq in hulls])].astype(np.int64)
        equations=np.concatenate([eq for _,eq in hulls]) if hulls else np.empty((0,4),np.float64)
        # One frame-local upload contains equations, contiguous normals and all
        # camera/group metadata. Integer fields retain their int64 bit pattern.
        chunks=[equations.ravel(),equations[:,:3].ravel()];camera_groups={};cursor=0
        for key,bb in camera_bounds.items():
            sizes=[max(0,x1-x0+1)*max(0,y1-y0+1) for x0,y0,x1,y1 in bb]
            offsets=np.r_[0,np.cumsum(sizes)].astype(np.int64);groups={}
            for i,(_,eq) in enumerate(hulls):
                if sizes[i]:groups.setdefault(triton.next_power_of_2(len(eq)),[]).append(i)
            records=[]
            for planes,ids in groups.items():
                meta=np.array([[starts[i],len(hulls[i][1]),bb[i][0],bb[i][1],
                    bb[i][2]-bb[i][0]+1,sizes[i],i,offsets[i]] for i in ids],np.int64)
                records.append((planes,max(sizes[i] for i in ids),len(ids),cursor))
                chunks.append(meta.ravel().view(np.float64));cursor+=meta.size
            camera_groups[key]=records
        packed=np.concatenate(chunks) if camera_bounds else None
        return _PreparedSurfaces(tuple(hulls),starts,equations,camera_bounds=camera_bounds,
            packed=packed,camera_groups=camera_groups)

    def filter_cameras(self,cameras,surfaces,masks,**kwargs):
        """Two-camera/live-frame entrypoint; preparation stays inside this call.

        No pose state survives the call. Camera depth/calibration and masks stay
        independent. Single-camera __call__ remains compatible and uncached.
        """
        cameras=tuple(cameras)
        names=[c.camera for c in cameras]
        if len(set(names))!=len(names) or set(names)!=set(masks):raise ValueError('camera identity')
        prepared=self.prepare_surfaces(surfaces,cameras)
        return {c.camera:self(c,prepared,masks[c.camera],**kwargs) for c in cameras}

    @torch.inference_mode()
    def __call__(self,camera,surfaces,masks,*,depth_cuda=None,diagnostics=False,
                 tolerance_m=.0025,uncertainty_band_m=.006):
        h,w=camera.depth_raw_mm.shape
        if not isinstance(masks,torch.Tensor) or not masks.is_cuda or masks.dtype!=torch.bool or tuple(masks.shape)!=(2,h,w):
            raise ValueError('CUDA bool masks [2,H,W] required')
        device=masks.device
        if depth_cuda is not None and (not isinstance(depth_cuda,torch.Tensor) or depth_cuda.device!=device
                or depth_cuda.dtype!=torch.float64 or tuple(depth_cuda.shape)!=(h,w)):
            raise ValueError('depth_cuda must be float64 metres [H,W] on the mask device')
        with torch.cuda.device(device):
            def gpu(a):return torch.as_tensor(np.ascontiguousarray(a),device=device)
            masks=masks.contiguous()
            if depth_cuda is not None:depth=depth_cuda.contiguous()
            elif self._depth_from_mm and type(camera) is CameraObservation and camera.depth_raw_mm.dtype==np.int16:
                raw_depth=gpu(camera.depth_raw_mm)
                depth=torch.empty((h,w),device=device,dtype=torch.float64)
                _depth_mm[(triton.cdiv(h*w,256),)](raw_depth,self._depth_table(device),depth,h*w)
            else:depth=gpu(camera.depth)
            world=gpu(camera.world_from_camera.astype(np.float64))
            rays=self._camera_rays(camera,device)
            direction=rays@world[:3,:3].T
            prepared=surfaces if isinstance(surfaces,_PreparedSurfaces) else self.prepare_surfaces(surfaces,(camera,))
            hulls=prepared.hulls
            if prepared.camera_bounds:
                if id(camera) not in prepared.camera_bounds:raise ValueError('camera outside prepared frame')
                bb=prepared.camera_bounds[id(camera)]
            else:
                matrix=validate_transform(np.linalg.inv(camera.world_from_camera))
                bb=[_bounds(camera,v,matrix) for v,_ in hulls]
            z=torch.full((h*w,),torch.inf,device=device,dtype=torch.float64)
            inside=torch.zeros(h*w,device=device,dtype=torch.bool if h*w<=128*128 else torch.int32)
            owner=torch.full((h*w if diagnostics else 0,),-1,device=device,dtype=torch.int32)
            flags=torch.empty((3,h,w) if diagnostics else (0,),device=device,dtype=torch.bool)
            if h*w<=128*128:
                # Retain the exact Stage 9 grouped numerator arithmetic/ties.
                sizes=[max(0,x1-x0+1)*max(0,y1-y0+1) for x0,y0,x1,y1 in bb]
                starts=np.r_[0,np.cumsum(sizes)].astype(np.int64)
                entry=torch.empty(int(starts[-1]),device=device,dtype=torch.float64)
                upper=torch.empty_like(entry);iv=torch.empty_like(entry,dtype=torch.bool)
                groups={}
                for i,(_,eq) in enumerate(hulls):
                    if sizes[i]:groups.setdefault(triton.next_power_of_2(len(eq)),[]).append(i)
                for f,links in groups.items():
                    host=np.zeros((len(links),f,4));host[:,:,3]=np.inf
                    for j,i in enumerate(links):host[j,:len(hulls[i][1])]=hulls[i][1]
                    eq=gpu(host);num=-(eq[:,:,:3]@gpu(camera.world_from_camera[:3,3].astype(np.float64))+eq[:,:,3])
                    _links[(triton.cdiv(max(sizes[i] for i in links),32),len(links))](direction,depth,eq,num,
                        gpu(np.array(bb,np.int64)[links]),gpu(np.array(sizes,np.int64)[links]),gpu(starts[links]),entry,upper,iv,w,f,enable_fp_fusion=False)
                full_owner=owner if diagnostics else torch.empty(h*w,device=device,dtype=torch.int32)
                full_flags=flags if diagnostics else torch.empty((3,h,w),device=device,dtype=torch.bool)
                _reduce[(triton.cdiv(h*w,256),)](entry,iv,gpu(starts),gpu(np.asarray(bb,np.int64).reshape(-1,4)),depth,z,full_owner,inside,full_flags,h,w,len(hulls),tolerance_m,uncertainty_band_m,enable_fp_fusion=False)
                evaluated=torch.ones((h,w),device=device,dtype=torch.bool) if diagnostics else None
            else:
                sizes=[max(0,x1-x0+1)*max(0,y1-y0+1) for x0,y0,x1,y1 in bb]
                offsets=np.r_[0,np.cumsum(sizes)].astype(np.int64)
                entries=torch.empty(int(offsets[-1]) if diagnostics else 0,device=device,dtype=torch.float64)
                if device not in prepared.device_equations:
                    if prepared.packed is not None:
                        arena=gpu(prepared.packed);nf=len(prepared.equations)
                        prepared.device_equations[device]=(arena[:4*nf].view(nf,4),
                            arena[4*nf:7*nf].view(nf,3),arena[7*nf:].view(torch.int64))
                    else:
                        equations=gpu(prepared.equations)
                        prepared.device_equations[device]=(equations,equations[:,:3].contiguous(),None)
                equations,normals,all_meta=prepared.device_equations[device]
                # Keep each original strided per-link GEMV, including its shape
                # and rounding. A single combined GEMV changes exact owner ties.
                nums=[-(equations[a:b,:3]@world[:3,3]+equations[a:b,3]) if sizes[i] else
                      torch.empty(int(b-a),device=device,dtype=torch.float64)
                      for i,(a,b) in enumerate(zip(prepared.starts[:-1],prepared.starts[1:]))]
                numerator=torch.cat(nums) if nums else torch.empty(0,device=device,dtype=torch.float64)
                if diagnostics:owner.fill_(np.iinfo(np.int32).max)
                metadata=[]
                if all_meta is not None:
                    group_meta=[(planes,count,nlinks,all_meta[start:start+8*nlinks])
                        for planes,count,nlinks,start in prepared.camera_groups[id(camera)]]
                else:
                    groups={}
                    for i,(_,eq) in enumerate(hulls):
                        if sizes[i]:groups.setdefault(triton.next_power_of_2(len(eq)),[]).append(i)
                    group_meta=[]
                    for planes,ids in groups.items():
                        meta=gpu(np.array([[prepared.starts[i],len(hulls[i][1]),bb[i][0],bb[i][1],
                            bb[i][2]-bb[i][0]+1,sizes[i],i,offsets[i]] for i in ids],np.int64))
                        group_meta.append((planes,max(sizes[i] for i in ids),len(ids),meta))
                for planes,count,nlinks,meta in group_meta:
                    if self._compact:
                        # A fixed-capacity GPU list avoids host count readback.
                        # Only independent pixel order changes; plane arithmetic
                        # and deterministic depth/owner reduction stay identical.
                        if not metadata:
                            counts=torch.zeros(len(hulls),device=device,dtype=torch.int32)
                            rows=torch.empty(int(offsets[-1]),device=device,dtype=torch.int32)
                        _compact_rows[(triton.cdiv(count,256),nlinks)](masks,meta,counts,rows,w,h)
                        _clip_compact[(triton.cdiv(count,self._clip_pixels),nlinks)](direction,normals,numerator,depth,meta,counts,rows,z,inside,entries,
                            planes,diagnostics,P=self._clip_pixels,enable_fp_fusion=False)
                    else:
                        _clip_grouped[(triton.cdiv(count,self._clip_pixels),nlinks)](direction,masks,normals,numerator,depth,meta,z,inside,entries,
                            w,h,planes,diagnostics,P=self._clip_pixels,enable_fp_fusion=False)
                    metadata.append((meta,count,nlinks))
                if diagnostics:
                    for meta,count,nlinks in metadata:
                        if self._compact:
                            _compact_owners[(triton.cdiv(count,256),nlinks)](meta,counts,rows,entries,z,owner)
                        else:_owners[(triton.cdiv(count,256),nlinks)](masks,meta,entries,z,owner,w,h)
                    _finish_owner[(triton.cdiv(h*w,256),)](owner,h*w)
                evaluated=masks.any(0) if diagnostics else None
            clean=torch.empty_like(masks)
            _clean[(triton.cdiv(h*w,256),)](masks,depth,z,inside,clean,flags,h*w,tolerance_m,uncertainty_band_m,diagnostics,enable_fp_fusion=False)
            if not diagnostics:return clean
            return dict(clean_masks=clean,confirmed_robot=flags[0],uncertain=flags[1],candidate=flags[2],
                predicted_robot_depth=z.reshape(h,w),link_owner=owner.reshape(h,w),inside_known_link_volume=inside.reshape(h,w).bool(),evaluated_pixels=evaluated)
