"""One-block deterministic small-cloud FPS; larger clouds retain the prior kernel."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _small_fps(X,INITIAL,OUT,N:tl.constexpr,COUNT:tl.constexpr,USED:tl.constexpr,
               FIRST:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);valid=i<N
    x=tl.load(X+3*i,valid,0);y=tl.load(X+3*i+1,valid,0);z=tl.load(X+3*i+2,valid,0)
    distance=tl.full((B,),float('inf'),tl.float64)
    for k in range(USED):
        j=tl.load(INITIAL+k);tl.store(OUT+k,j)
        dx=x-tl.load(X+3*j);dy=y-tl.load(X+3*j+1);dz=z-tl.load(X+3*j+2)
        distance=tl.minimum(distance,(dx*dx+dy*dy)+dz*dz)
    selected=tl.full((B,),False,tl.int1)
    for k in range(USED):selected=selected|(i==tl.load(INITIAL+k))
    if USED==0:
        tl.store(OUT,FIRST);selected=selected|(i==FIRST)
        dx=x-tl.load(X+3*FIRST);dy=y-tl.load(X+3*FIRST+1);dz=z-tl.load(X+3*FIRST+2)
        distance=tl.minimum(distance,(dx*dx+dy*dy)+dz*dz)
    for k in range(tl.maximum(USED,1),COUNT):
        d=tl.where(valid&~selected,distance,-float('inf'));best=tl.max(d,0)
        j=tl.min(tl.where((d==best)&valid&~selected,i,2147483647),0)
        tl.store(OUT+k,j);selected=selected|(i==j)
        dx=x-tl.load(X+3*j);dy=y-tl.load(X+3*j+1);dz=z-tl.load(X+3*j+2)
        distance=tl.minimum(distance,(dx*dx+dy*dy)+dz*dz)


def fps(points,count,*,seed=0,initial=()):
    from visual.runtime.sampling import _integer
    from gpu_graph import fps_indices_cuda
    _integer(count,'FPS count');_integer(seed,'FPS seed')
    xyz=np.ascontiguousarray(points,dtype=np.float64);initial=np.ascontiguousarray(initial,dtype=np.int64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('finite XYZ required')
    if initial.ndim!=1 or len(np.unique(initial))!=len(initial) or ((initial<0)|(initial>=len(xyz))).any():raise ValueError('invalid FPS initial rows')
    count=min(count,len(xyz))
    if len(initial)>count:raise ValueError('initial exceeds count')
    if not count:return np.empty(0,np.int64)
    if len(xyz)>8192:return fps_indices_cuda(points,count,seed=seed,initial=initial,fused=True)
    coords=torch.as_tensor(xyz,device='cuda');init=torch.as_tensor(initial,device='cuda');out=torch.empty(count,device='cuda',dtype=torch.int64)
    first=int(np.random.default_rng(seed).integers(len(xyz))) if not len(initial) else 0
    _small_fps[(1,)](coords,init,out,len(xyz),count,len(initial),first,triton.next_power_of_2(len(xyz)),num_warps=8,enable_fp_fusion=False)
    return out.cpu().numpy()
