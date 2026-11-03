"""Bounded (<=128 rows) CUDA KD queries for the current graph budget.

Compact median splits and strict neighbor-heap insertion follow SciPy 1.18's
cKDTree policy. Median partition uses GNU introselect's ordinary path; a bounded
heap-select fallback handles its depth limit. No host point/neighbor arrays.
"""
import triton as tr
import triton.language as tl


@tr.jit
def _value(X, I, row, axis):
    return tl.load(X + tl.load(I + tl.maximum(row, 0)) * 3 + axis)


@tr.jit
def _swap(I, a, b):
    va = tl.load(I+a); vb = tl.load(I+b)
    tl.store(I+a, vb); tl.store(I+b, va)


@tr.jit
def _push_index(X, I, start, hole, top, value, axis):
    parent = (hole-1)//2
    while (hole > top) & (_value(X, I, start+parent, axis) < tl.load(X+value*3+axis)):
        tl.store(I+start+hole, tl.load(I+start+parent))
        hole = parent; parent = (hole-1)//2
    tl.store(I+start+hole, value)


@tr.jit
def _adjust_index(X, I, start, hole, length, value, axis):
    hole = hole + tl.full((),0,tl.int32)
    top = hole; child = 2*(hole+1)
    while child < length:
        if _value(X,I,start+child,axis) < _value(X,I,start+child-1,axis):
            child -= 1
        tl.store(I+start+hole, tl.load(I+start+child))
        hole=child; child=2*(child+1)
    if child == length:
        tl.store(I+start+hole,tl.load(I+start+child-1)); hole=child-1
    _push_index(X,I,start,hole,top,value,axis)


@tr.jit
def _nth(X, I, lo, hi, nth, axis):
    length=hi-lo; power=length; depth=0
    while power>1:
        power=power//2; depth+=2
    fallback=False
    while (hi-lo>3) & (~fallback):
        if depth == 0:
            length=nth-lo+1; parent=(length-2)//2
            while parent>=0:
                _adjust_index(X,I,lo,parent,length,tl.load(I+lo+parent),axis); parent-=1
            j=nth+1
            while j<hi:
                if _value(X,I,j,axis)<_value(X,I,lo,axis):
                    value=tl.load(I+j); tl.store(I+j,tl.load(I+lo))
                    _adjust_index(X,I,lo,0,length,value,axis)
                j+=1
            _swap(I,lo,nth); fallback=True
        else:
            depth-=1; a=lo+1; b=lo+(hi-lo)//2; c=hi-1
            av=_value(X,I,a,axis); bv=_value(X,I,b,axis); cv=_value(X,I,c,axis)
            median=a
            if av<bv:
                if bv<cv: median=b
                elif av<cv: median=c
            else:
                if av<cv: median=a
                elif bv<cv: median=c
                else: median=b
            _swap(I,lo,median)
            pivot=_value(X,I,lo,axis); left=lo+1; right=hi; done=False
            while ~done:
                while _value(X,I,left,axis)<pivot: left+=1
                right-=1
                while pivot<_value(X,I,right,axis): right-=1
                if left>=right: done=True
                else:
                    _swap(I,left,right); left+=1
            if left<=nth: lo=left
            else: hi=left
    if ~fallback:
        j=lo+1
        while j<hi:
            value=tl.load(I+j); key=tl.load(X+value*3+axis); hole=j
            while (hole>lo) & (key<_value(X,I,hole-1,axis)):
                tl.store(I+hole,tl.load(I+hole-1)); hole-=1
            tl.store(I+hole,value); j+=1


@tr.jit
def _partition(X,I,lo,hi,axis,split):
    done=False
    while ~done:
        while (lo<hi) & (_value(X,I,lo,axis)<split): lo+=1
        if lo==hi: done=True
        else:
            hi-=1
            while (lo<hi) & (~(_value(X,I,hi,axis)<split)): hi-=1
            if lo==hi: done=True
            else:
                _swap(I,lo,hi); lo+=1
    return lo


@tr.jit
def build(X,I,T,S,N:tl.constexpr,B:tl.constexpr):
    r=tl.arange(0,B); tl.store(I+r,r,r<N); tl.store(I+N,0)
    tl.store(T,0); tl.store(T+1,N)
    node=tl.full((),0,tl.int32); total=tl.full((),1,tl.int32)
    while node<total:
        tl.debug_barrier()
        lo=tl.load(T+node*5); hi=tl.load(T+node*5+1)
        idx=tl.load(I+r, r<N, 0); active=(r>=lo)&(r<hi)
        x=tl.load(X+idx*3,active,0); y=tl.load(X+idx*3+1,active,0); z=tl.load(X+idx*3+2,active,0)
        xlo=tl.min(tl.where(active,x,float('inf')),0); xhi=tl.max(tl.where(active,x,-float('inf')),0)
        ylo=tl.min(tl.where(active,y,float('inf')),0); yhi=tl.max(tl.where(active,y,-float('inf')),0)
        zlo=tl.min(tl.where(active,z,float('inf')),0); zhi=tl.max(tl.where(active,z,-float('inf')),0)
        axis=0; spread=xhi-xlo; minimum=xlo
        if yhi-ylo>spread: axis=1; spread=yhi-ylo; minimum=ylo
        if zhi-zlo>spread: axis=2; spread=zhi-zlo; minimum=zlo
        if (hi-lo<=16) | (spread==0): tl.store(T+node*5+2,-1)
        else:
            mid=lo+(hi-lo)//2
            _nth(X,I,lo,hi,mid,axis)
            split=_value(X,I,mid,axis)
            cut=_partition(X,I,lo,mid,axis,split)
            if cut==lo:
                bits=minimum.to(tl.int64,bitcast=True)
                bits=tl.where(minimum==0,1,tl.where(minimum>0,bits+1,bits-1))
                split=bits.to(tl.float64,bitcast=True)
                cut=_partition(X,I,lo,hi,axis,split)
            tl.store(T+node*5+2,axis); tl.store(S+node,split)
            tl.store(T+node*5+3,total); tl.store(T+node*5+4,total+1)
            tl.store(T+total*5,lo); tl.store(T+total*5+1,cut)
            tl.store(T+(total+1)*5,cut); tl.store(T+(total+1)*5+1,hi)
            total+=2
        node+=1


@tr.jit
def _hcopy(H,dst,src):
    for k in tl.static_range(5): tl.store(H+dst*5+k,tl.load(H+src*5+k))


@tr.jit
def _hpush(H,n,p,v,x,y,z):
    hole=n; done=False
    while (hole>0) & (~done):
        parent=(hole-1)//2
        if p<tl.load(H+parent*5):
            _hcopy(H,hole,parent); hole=parent
        else: done=True
    tl.store(H+hole*5,p); tl.store(H+hole*5+1,v)
    tl.store(H+hole*5+2,x); tl.store(H+hole*5+3,y); tl.store(H+hole*5+4,z)
    return n+1


@tr.jit
def _hpop(H,n):
    p=tl.load(H); v=tl.load(H+1); x=tl.load(H+2); y=tl.load(H+3); z=tl.load(H+4)
    n-=1
    if n>0:
        _hcopy(H,0,n); hole=0; done=False
        while ~done:
            left=hole*2+1; right=left+1; child=hole
            if left<n:
                child=left
                if right<n:
                    if tl.load(H+left*5)>tl.load(H+right*5): child=right
                if tl.load(H+hole*5)>tl.load(H+child*5):
                    for k in tl.static_range(5):
                        a=tl.load(H+hole*5+k); b=tl.load(H+child*5+k)
                        tl.store(H+hole*5+k,b); tl.store(H+child*5+k,a)
                    hole=child
                else: done=True
            else: done=True
    return n,p,v,x,y,z


@tr.jit
def query(X,I,T,S,H,Q,OUT,N:tl.constexpr,K:tl.constexpr,CAP:tl.constexpr):
    row=tl.program_id(0); heap=H+row*CAP*5; near=Q+row*K*5
    qx=tl.load(X+row*3); qy=tl.load(X+row*3+1); qz=tl.load(X+row*3+2)
    node=tl.full((),0,tl.int32); mind=0.; sx=0.; sy=0.; sz=0.; ub=float('inf'); nh=tl.full((),0,tl.int32); nn=tl.full((),0,tl.int32); active=True
    # Float64 accumulators, including empty/root distances.
    mind=mind.to(tl.float64); sx=sx.to(tl.float64); sy=sy.to(tl.float64); sz=sz.to(tl.float64); ub=ub.to(tl.float64)
    while active:
        axis=tl.load(T+node*5+2)
        if axis==-1:
            j=tl.load(T+node*5); end=tl.load(T+node*5+1)
            while j<end:
                idx=tl.load(I+j); dx=tl.load(X+idx*3)-qx; dy=tl.load(X+idx*3+1)-qy; dz=tl.load(X+idx*3+2)-qz
                d=(dx*dx+dy*dy)+dz*dz
                if d<ub:
                    if nn==K: nn,_,_,_,_,_=_hpop(near,nn)
                    nn=_hpush(near,nn,-d,idx.to(tl.float64),0.,0.,0.)
                    if nn==K: ub=-tl.load(near)
                j+=1
            if nh>0:
                nh,mind,v,sx,sy,sz=_hpop(heap,nh); node=v.to(tl.int32)
            else: active=False
        elif mind>ub: active=False
        else:
            split=tl.load(S+node); less=tl.load(T+node*5+3); greater=tl.load(T+node*5+4)
            q=tl.where(axis==0,qx,tl.where(axis==1,qy,qz))
            nd=tl.where(q<split,less,greater); fd=tl.where(q<split,greater,less)
            side=(split-q)*(split-q); old=tl.where(axis==0,sx,tl.where(axis==1,sy,sz))
            fm=mind+(side-old); fx=tl.where(axis==0,side,sx); fy=tl.where(axis==1,side,sy); fz=tl.where(axis==2,side,sz)
            if fm<mind:
                if mind<=ub: nh=_hpush(heap,nh,mind,nd.to(tl.float64),sx,sy,sz)
                node=fd; mind=fm; sx=fx; sy=fy; sz=fz
            else:
                if fm<=ub: nh=_hpush(heap,nh,fm,fd.to(tl.float64),fx,fy,fz)
                node=nd
    for j in range(K):
        idx=tl.load(near+j*5+1).to(tl.int32)
        tl.store(OUT+row*N+idx,1)
