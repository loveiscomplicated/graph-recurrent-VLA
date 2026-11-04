"""Bounded-memory exact inter-cell radius witnesses."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def masked_world_points(Depth, Pixels, K, M, Out, N, W: tl.constexpr,
                        KS0: tl.constexpr, KS1: tl.constexpr,
                        MS0: tl.constexpr, MS1: tl.constexpr,
                        PS: tl.constexpr, BLOCK: tl.constexpr=256):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=row<N
    pixel=tl.load(Pixels+row*PS,valid,other=0)
    z=tl.load(Depth+pixel,valid,other=0)
    u=(pixel%W).to(tl.float64);v=(pixel//W).to(tl.float64)
    x=libdevice.div_rn(((u+0.5)-tl.load(K+2*KS1))*z,tl.load(K))
    y=libdevice.div_rn(((v+0.5)-tl.load(K+KS0+2*KS1))*z,tl.load(K+KS0+KS1))
    for axis in tl.static_range(3):
        value=((x*tl.load(M+axis*MS0)+y*tl.load(M+axis*MS0+MS1))
               +z*tl.load(M+axis*MS0+2*MS1))+tl.load(M+axis*MS0+3*MS1)
        tl.store(Out+row*3+axis,value,valid)


@triton.jit
def label_edges(Labels, A, B, Merged, E, BLOCK: tl.constexpr=256):
    edge=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=edge<E
    a=tl.load(A+edge,valid,other=0);b=tl.load(B+edge,valid,other=0)
    root=tl.minimum(tl.load(Labels+a,valid,other=0),tl.load(Labels+b,valid,other=0))
    tl.atomic_min(Merged+a,root,valid,sem='relaxed')
    tl.atomic_min(Merged+b,root,valid,sem='relaxed')


@triton.jit
def label_jump(Merged, Old, Out, Changed, N, CHECK: tl.constexpr,
               BLOCK: tl.constexpr=256):
    row=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=row<N
    root=tl.load(Merged+row,valid,other=0)
    value=tl.load(Merged+root,valid,other=0)
    tl.store(Out+row,value,valid)
    if CHECK:
        old=tl.load(Old+row,valid,other=0)
        tl.store(Changed+tl.program_id(0),tl.sum((valid&(value!=old)).to(tl.int32),0))

@triton.jit
def radius_exists(X, Starts, A, B, Out, R2: tl.constexpr, P: tl.constexpr=16, Q: tl.constexpr=32):
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
