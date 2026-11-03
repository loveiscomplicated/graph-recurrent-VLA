"""CUDA mask -> CUDA clean mask. No mask readback or scalar synchronization.

Collision preparation remains CPU; ray/numerator arithmetic matches Stage 9.
Full-screen grouped arithmetic is retained for <=128². Larger images clip only
raw mask-union lanes inside the existing link bounds. Diagnostics explicitly
carry evaluated_pixels; values elsewhere are not a full-screen classification.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from scipy.spatial import ConvexHull
from graph.persistent_geometry.contracts import transform
from graph.experiments.graph_latency_10i_20261103_v2.robot import _links, _reduce


@triton.jit(do_not_specialize=["X","Y","BW","N","F","Link"],
            do_not_specialize_on_alignment=["X","Y","BW","N","F","Link"])
def _clip_masked(D,Mask,Normals,Num,Depth,Z,Owner,Interior,
                 W:tl.constexpr,H:tl.constexpr,X,Y,
                 BW,N,F,Link,
                 Planes:tl.constexpr,DIAG:tl.constexpr,P:tl.constexpr=32):
    idx=tl.program_id(0)*P+tl.arange(0,P)
    row=(Y+idx//BW)*W+X+idx%BW
    active=(idx<N)&(tl.load(Mask+row,idx<N,0)|tl.load(Mask+H*W+row,idx<N,0))
    if tl.sum(active.to(tl.int32),0)>0:
        dx=tl.load(D+3*row,active,0);dy=tl.load(D+3*row+1,active,0);dz=tl.load(D+3*row+2,active,0)
        face=tl.arange(0,Planes)
        nx=tl.load(Normals+3*face,face<F,0);ny=tl.load(Normals+3*face+1,face<F,0);nz=tl.load(Normals+3*face+2,face<F,0)
        num=tl.load(Num+face,face<F,0)
        denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
        parallel=tl.abs(denom)<1e-12;crossing=tl.where(parallel,0,num[None,:]/denom)
        lower=tl.max(tl.where((face[None,:]<F)&(denom < -1e-12),crossing,-float('inf')),1)
        upper=tl.min(tl.where((face[None,:]<F)&(denom > 1e-12),crossing,float('inf')),1)
        outside=tl.sum(((face[None,:]<F)&parallel&(num[None,:]<0)).to(tl.int32),1)>0
        entry=tl.maximum(lower,0);hit=(upper>=entry)&(upper>0)&~outside&active
        measured=tl.load(Depth+row,active,float('nan'));prior=tl.load(Interior+row,active,False)
        interior=prior|(hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005))
        tl.store(Interior+row,interior,active)
        nearest=tl.load(Z+row,active,float('inf'));change=hit&(entry<nearest)
        tl.store(Z+row,entry,change)
        if DIAG:tl.store(Owner+row,Link,change)


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


def bounds(camera,vertices):
    h,w=camera.depth_raw_mm.shape;vc=transform(vertices,np.linalg.inv(camera.world_from_camera))
    if np.all(vc[:,2]<=0):return (0,0,-1,-1)
    if not np.all(vc[:,2]>0):return (0,0,w-1,h-1)
    k=camera.intrinsic;uv=vc[:,:2]/vc[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]-.5
    lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w-1,h-1])
    return (0,0,-1,-1) if np.any(hi<lo) else (*map(int,lo),*map(int,hi))


class CudaRobotFilter:
    """One call = one current camera/FK snapshot; no cross-frame pose cache.

    Input masks: CUDA bool [2,H,W], object order Peg/Box. Output has identical
    shape/device/dtype and never aliases or modifies the raw SAM2 masks.
    Optional depth_cuda is float64 CV-z metres on the same device, including NaN
    for invalid depth. CPU sensor calibration/FK/hulls may still upload to CUDA.
    """
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
            depth=gpu(camera.depth) if depth_cuda is None else depth_cuda.contiguous()
            world=gpu(camera.world_from_camera.astype(np.float64));k=gpu(np.asarray(camera.intrinsic,np.float64))
            v,u=torch.meshgrid(torch.arange(h,device=device,dtype=torch.float64),torch.arange(w,device=device,dtype=torch.float64),indexing='ij')
            rays=torch.stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],torch.ones_like(u)),2).reshape(-1,3)
            direction=rays@world[:3,:3].T
            hulls=[]
            for s in surfaces:
                vertices=s.world_vertices;hulls.append((vertices,ConvexHull(vertices).equations))
            bb=[bounds(camera,v) for v,_ in hulls]
            z=torch.full((h*w,),torch.inf,device=device,dtype=torch.float64)
            inside=torch.zeros(h*w,device=device,dtype=torch.bool)
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
                for i,((_,eq),(x0,y0,x1,y1)) in enumerate(zip(hulls,bb)):
                    if x1<x0 or y1<y0:continue
                    equations=gpu(eq);normals=equations[:,:3];numerator=-(normals@world[:3,3]+equations[:,3])
                    width=x1-x0+1;n=width*(y1-y0+1)
                    _clip_masked[(triton.cdiv(n,32),)](direction,masks,normals.contiguous(),numerator,depth,z,owner,inside,
                        w,h,x0,y0,width,n,len(eq),i,triton.next_power_of_2(len(eq)),diagnostics,enable_fp_fusion=False)
                evaluated=masks.any(0) if diagnostics else None
            clean=torch.empty_like(masks)
            _clean[(triton.cdiv(h*w,256),)](masks,depth,z,inside,clean,flags,h*w,tolerance_m,uncertainty_band_m,diagnostics,enable_fp_fusion=False)
            if not diagnostics:return clean
            return dict(clean_masks=clean,confirmed_robot=flags[0],uncertain=flags[1],candidate=flags[2],
                predicted_robot_depth=z.reshape(h,w),link_owner=owner.reshape(h,w),inside_known_link_volume=inside.reshape(h,w),evaluated_pixels=evaluated)
