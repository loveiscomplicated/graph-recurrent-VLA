"""CUDA radius connectivity shared by the wrist gate and resident backend."""
from itertools import product
import numpy as np
import torch
import triton
import triton.language as tl

def _tensor(a, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device='cuda')

def _numpy(a):
    return a.detach().cpu().numpy()

def _radius_kernel():
    @triton.jit
    def exists(X, Starts, A, B, Out, R2: tl.constexpr, P: tl.constexpr=16, Q: tl.constexpr=32):
        edge=tl.program_id(0)
        a=tl.load(A+edge);b=tl.load(B+edge)
        lo_a=tl.load(Starts+a);hi_a=tl.load(Starts+a+1)
        lo_b=tl.load(Starts+b);hi_b=tl.load(Starts+b+1)
        found=tl.full((),False,tl.int1)
        ia=lo_a
        while (ia<hi_a)&(~found):
            ra=ia+tl.arange(0,P)
            ax=tl.load(X+3*ra,ra<hi_a,other=0)
            ay=tl.load(X+3*ra+1,ra<hi_a,other=0)
            az=tl.load(X+3*ra+2,ra<hi_a,other=0)
            ib=lo_b
            while (ib<hi_b)&(~found):
                rb=ib+tl.arange(0,Q)
                bx=tl.load(X+3*rb,rb<hi_b,other=0)
                by=tl.load(X+3*rb+1,rb<hi_b,other=0)
                bz=tl.load(X+3*rb+2,rb<hi_b,other=0)
                dx=ax[:,None]-bx[None,:];dy=ay[:,None]-by[None,:];dz=az[:,None]-bz[None,:]
                good=((dx*dx+dy*dy)+dz*dz<=R2)&(ra[:,None]<hi_a)&(rb[None,:]<hi_b)
                found=tl.sum(tl.sum(good.to(tl.int32),0),0)>0
                ib+=Q
            ia+=P
        tl.store(Out+edge,found)
    return exists

_EXISTS = None

def radius_components(xyz, radius, *, support_callback=None):
    """Exact-radius cell-clique connectivity; CUDA state never contains E edges."""
    global _EXISTS
    x=_tensor(xyz);n=len(x)
    if not n:
        if support_callback is not None:support_callback(np.empty(0,bool),radius)
        return np.empty(0,np.int64)
    cell=torch.floor((x-x.amin(0))/(radius/2)).long()
    dimensions=(cell.amax(0)+1).tolist()
    stride=torch.tensor([dimensions[1]*dimensions[2],dimensions[2],1],device='cuda')
    if dimensions[0]*dimensions[1]*dimensions[2]>=2**62:raise ValueError('cell key overflow')
    code=(cell*stride).sum(1)
    keys,inverse,counts=torch.unique(code,sorted=True,return_inverse=True,return_counts=True)
    nc=len(keys)
    if support_callback is not None:support_callback(_numpy(counts[inverse]>=3),radius)
    if nc==1:return np.zeros(n,np.int64)
    order=torch.argsort(code,stable=True);sorted_x=x[order].contiguous()
    starts=torch.cat((torch.zeros(1,dtype=torch.long,device='cuda'),counts.cumsum(0)))
    coordinates=torch.stack((keys//stride[0],(keys//stride[1])%dimensions[1],keys%dimensions[2]),1)
    offsets=torch.tensor([d for d in product(range(-3,4),repeat=3) if d>(0,0,0)],device='cuda')
    neighbor=coordinates[:,None,:]+offsets[None,:,:]
    inside=((neighbor>=0)&(neighbor<torch.tensor(dimensions,device='cuda'))).all(2)
    target=(neighbor*stride).sum(2)
    j=torch.searchsorted(keys,target).clamp_max(nc-1)
    valid=inside&(keys[j]==target)
    a=torch.arange(nc,device='cuda')[:,None].expand_as(j)[valid];b=j[valid]
    lower=torch.full((nc,3),torch.inf,dtype=x.dtype,device='cuda')
    upper=torch.full((nc,3),-torch.inf,dtype=x.dtype,device='cuda')
    ix=inverse[:,None].expand(-1,3)
    lower.scatter_reduce_(0,ix,x,reduce='amin',include_self=True)
    upper.scatter_reduce_(0,ix,x,reduce='amax',include_self=True)
    gap=torch.maximum(lower[a]-upper[b],lower[b]-upper[a]).clamp_min_(0)
    possible=((gap[:,0]**2+gap[:,1]**2)+gap[:,2]**2)<=radius*radius
    a=a[possible].contiguous();b=b[possible].contiguous()
    if len(a):
        if _EXISTS is None:_EXISTS=_radius_kernel()
        keep=torch.empty(len(a),dtype=torch.bool,device='cuda')
        _EXISTS[(len(a),)](sorted_x,starts,a,b,keep,radius*radius,enable_fp_fusion=False)
        a=a[keep];b=b[keep]
    labels=torch.arange(nc,device='cuda')
    for _ in range(nc):
        old=labels.clone()
        roots=torch.minimum(labels[a],labels[b])
        labels.scatter_reduce_(0,a,roots,reduce='amin',include_self=True)
        labels.scatter_reduce_(0,b,roots,reduce='amin',include_self=True)
        labels=labels[labels]
        if torch.equal(old,labels):break
    else:raise AssertionError('cell connectivity did not converge')
    roots=labels[inverse]
    first=torch.full((nc,),n,dtype=torch.long,device='cuda')
    first.scatter_reduce_(0,roots,torch.arange(n,device='cuda'),reduce='amin',include_self=True)
    _,result=torch.unique(first[roots],sorted=True,return_inverse=True)
    return _numpy(result)
