"""Batch link clipping into two GPU kernels, retaining host hulls and exact link order."""
import numpy as np
import torch
import triton
import triton.language as tl
from scipy.spatial import ConvexHull
from graph.persistent_geometry.contracts import transform


@triton.jit
def _links(D,DEPTH,EQ,NUM,BOUNDS,SIZE,START,ENTRY,UPPER,INTERIOR,
           W:tl.constexpr,F:tl.constexpr,P:tl.constexpr=32):
    link=tl.program_id(1);i=tl.program_id(0)*P+tl.arange(0,P)
    width=tl.load(BOUNDS+link*4+2)-tl.load(BOUNDS+link*4)+1
    n=tl.load(SIZE+link)
    if tl.program_id(0)*P<n:
        row=(tl.load(BOUNDS+link*4+1)+i//width)*W+tl.load(BOUNDS+link*4)+i%width
        dx=tl.load(D+3*row,i<n,0);dy=tl.load(D+3*row+1,i<n,0);dz=tl.load(D+3*row+2,i<n,0)
        face=tl.arange(0,F);nx=tl.load(EQ+link*F*4+face*4);ny=tl.load(EQ+link*F*4+face*4+1);nz=tl.load(EQ+link*F*4+face*4+2)
        valid_face=tl.load(EQ+link*F*4+face*4+3)!=float('inf')
        num=tl.load(NUM+link*F+face)
        denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
        parallel=tl.abs(denom)<1e-12;crossing=tl.where(parallel,0,num[None,:]/denom)
        lower=tl.max(tl.where(valid_face[None,:]&(denom < -1e-12),crossing,-float('inf')),1)
        upper=tl.min(tl.where(valid_face[None,:]&(denom > 1e-12),crossing,float('inf')),1)
        outside=tl.sum((valid_face[None,:]&parallel&(num[None,:]<0)).to(tl.int32),1)>0
        entry=tl.maximum(lower,0);hit=(upper>=entry)&(upper>0)&~outside&(i<n)
        measured=tl.load(DEPTH+row,i<n,float('nan'))
        interior=hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005)
        out=tl.load(START+link)+i
        tl.store(ENTRY+out,tl.where(hit,entry,float('inf')),i<n)
        tl.store(UPPER+out,upper,i<n);tl.store(INTERIOR+out,interior,i<n)


@triton.jit
def _reduce(ENTRY,INTERIOR,START,BOUNDS,DEPTH,Z,OWNER,INSIDE,FLAGS,
            H:tl.constexpr,W:tl.constexpr,L:tl.constexpr,TOL:tl.constexpr,BAND:tl.constexpr,B:tl.constexpr=256):
    row=tl.program_id(0)*B+tl.arange(0,B);ok=row<H*W;v=row//W;u=row%W
    z=tl.full((B,),float('inf'),tl.float64);owner=tl.full((B,),-1,tl.int32);interior=tl.full((B,),False,tl.int1)
    for link in range(L):
        x0=tl.load(BOUNDS+4*link);y0=tl.load(BOUNDS+4*link+1);x1=tl.load(BOUNDS+4*link+2);y1=tl.load(BOUNDS+4*link+3)
        inside=ok&(u>=x0)&(u<=x1)&(v>=y0)&(v<=y1)
        i=tl.load(START+link)+(v-y0)*(x1-x0+1)+u-x0
        value=tl.load(ENTRY+i,inside,float('inf'));iv=tl.load(INTERIOR+i,inside,0)
        change=value<z;z=tl.where(change,value,z);owner=tl.where(change,link,owner);interior=interior|iv
    depth=tl.load(DEPTH+row,ok,float('nan'));valid=depth==depth
    difference=tl.abs(depth-z);finite=z<float('inf')
    confirmed=valid&((finite&(difference<=TOL))|interior)
    uncertain=valid&finite&~confirmed&(difference<=BAND)
    tl.store(Z+row,z,ok);tl.store(OWNER+row,owner,ok);tl.store(INSIDE+row,interior,ok)
    tl.store(FLAGS+row,confirmed,ok);tl.store(FLAGS+H*W+row,uncertain,ok);tl.store(FLAGS+2*H*W+row,valid&~confirmed&~uncertain,ok)


def classify(context,camera,surfaces,tolerance_m=.0025,uncertainty_band_m=.006):
    h,w=camera.depth_raw_mm.shape;wc=camera.world_from_camera;inverse=np.linalg.inv(wc)
    cache_key=('hulls',tuple(map(id,surfaces)))
    if cache_key not in context.cache:context.cache[cache_key]=[(s.world_vertices,ConvexHull(s.world_vertices).equations) for s in surfaces]
    hulls=context.cache[cache_key]
    bounds=[];sizes=[]
    for i,(vertices,eq) in enumerate(hulls):
        vc=transform(vertices,inverse)
        lo=np.array([0,0]);hi=np.array([w-1,h-1])
        if np.all(vc[:,2]<=0):lo=np.array([0,0]);hi=np.array([-1,-1])
        elif np.all(vc[:,2]>0):
            k=camera.intrinsic;uv=vc[:,:2]/vc[:,2,None]*[k[0,0],k[1,1]]+k[:2,2]-.5
            lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w-1,h-1])
        if np.any(hi<lo):lo=np.array([0,0]);hi=np.array([-1,-1])
        bounds.append([*lo,*hi]);sizes.append(max(0,int(np.prod(hi-lo+1))) if np.all(hi>=lo) else 0)
    starts=np.r_[0,np.cumsum(sizes)].astype(np.int64)
    def gpu(a):return torch.as_tensor(np.ascontiguousarray(a),device='cuda')
    bd=gpu(np.array(bounds,np.int64));st=gpu(starts)
    v,u=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float64),torch.arange(w,device='cuda',dtype=torch.float64),indexing='ij')
    cv=context.camera_values(camera);k=cv['k']
    rays=torch.stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],torch.ones_like(u)),2).reshape(-1,3)
    direction=rays@cv['world'][:3,:3].T
    entry=torch.empty(int(starts[-1]),device='cuda',dtype=torch.float64);upper=torch.empty_like(entry);interior=torch.empty_like(entry,dtype=torch.bool)
    groups={}
    for i,(_,eq) in enumerate(hulls):
        if sizes[i]:groups.setdefault(triton.next_power_of_2(len(eq)),[]).append(i)
    for f,links in groups.items():
        equations=np.zeros((len(links),f,4));equations[:,:,3]=np.inf
        for j,i in enumerate(links):equations[j,:len(hulls[i][1])]=hulls[i][1]
        eq=gpu(equations);num=-(eq[:,:,:3]@gpu(wc[:3,3].astype(np.float64))+eq[:,:,3])
        gb=gpu(np.array(bounds,np.int64)[links]);sz=gpu(np.array(sizes,np.int64)[links]);gs=gpu(starts[links])
        _links[(triton.cdiv(max(sizes[i] for i in links),32),len(links))](direction,cv['depth'],eq,num,gb,sz,gs,entry,upper,interior,w,f,enable_fp_fusion=False)
    z=torch.empty((h,w),device='cuda',dtype=torch.float64);owner=torch.empty((h,w),device='cuda',dtype=torch.int32)
    inside=torch.empty((h,w),device='cuda',dtype=torch.bool);flags=torch.empty((3,h,w),device='cuda',dtype=torch.bool)
    _reduce[(triton.cdiv(h*w,256),)](entry,interior,st,bd,cv['depth'],z,owner,inside,flags,h,w,len(surfaces),tolerance_m,uncertainty_band_m,enable_fp_fusion=False)
    masks=torch.cat([flags,inside[None]]).cpu().numpy()
    return dict(confirmed_robot=masks[0],uncertain=masks[1],candidate=masks[2],
        predicted_robot_depth=z.cpu().numpy(),link_owner=owner.cpu().numpy(),inside_known_link_volume=masks[3],
        method='per-shape collision CV-depth agreement or measured point inside actual link collision volume v2',
        tolerance_m=tolerance_m,uncertainty_band_m=uncertainty_band_m)
