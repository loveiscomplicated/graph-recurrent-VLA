"""Bounded-memory exact inter-cell radius witnesses."""
import triton
import triton.language as tl

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
