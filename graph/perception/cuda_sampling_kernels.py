"""CUDA-only primitives for ordered external contours and exact FP64 FPS."""
import triton
import triton.language as tl


@triton.jit
def mask_bounds(MASK,BOUNDS,H:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    keep=tl.load(MASK+i,i<H*W,other=0)
    v=i//W;u=i%W
    tl.atomic_min(BOUNDS,tl.min(tl.where(keep,v,H),0),sem='relaxed')
    tl.atomic_max(BOUNDS+1,tl.max(tl.where(keep,v+1,0),0),sem='relaxed')
    tl.atomic_min(BOUNDS+2,tl.min(tl.where(keep,u,W),0),sem='relaxed')
    tl.atomic_max(BOUNDS+3,tl.max(tl.where(keep,u+1,0),0),sem='relaxed')


@triton.jit
def voxel_choices(POINTS,PERM,STARTS,LENGTHS,CENTROIDS,FLAGS,B:tl.constexpr):
    group=tl.program_id(0);start=tl.load(STARTS+group);length=tl.load(LENGTHS+group)
    cx=tl.load(CENTROIDS+group*3);cy=tl.load(CENTROIDS+group*3+1);cz=tl.load(CENTROIDS+group*3+2)
    lane=tl.arange(0,B);best=tl.full((),float('inf'),tl.float64)
    best_row=tl.full((),2147483647,tl.int64)
    for chunk in range(tl.cdiv(length,B)):
        pos=chunk*B+lane;valid=pos<length
        row=tl.load(PERM+start+pos,valid,other=0)
        dx=tl.load(POINTS+row*3,valid,other=0)-cx
        dy=tl.load(POINTS+row*3+1,valid,other=0)-cy
        dz=tl.load(POINTS+row*3+2,valid,other=0)-cz
        d=tl.where(valid,(dx*dx+dy*dy)+dz*dz,float('inf'))
        local=tl.min(d,0)
        chosen=tl.min(tl.where(valid&(d==local),row,2147483647),0)
        best_row=tl.where(local<best,chosen,tl.where(local==best,tl.minimum(best_row,chosen),best_row))
        best=tl.minimum(best,local)
    tl.store(FLAGS+best_row,True)


@triton.jit
def mark_membership(A,B,C,D,ROWS,FLAGS,NA,NB,NC,ND,SA,SB,SC,SD,K:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    for field in tl.static_range(4):
        if field==0:ptr=A;length=NA;stride=SA
        elif field==1:ptr=B;length=NB;stride=SB
        elif field==2:ptr=C;length=NC;stride=SC
        else:ptr=D;length=ND;stride=SD
        valid=i<length;source=tl.load(ptr+i*stride,valid,other=0)
        row=tl.load(ROWS+source,valid,other=-1)
        tl.atomic_or(FLAGS+row,1<<field,valid&(row>=0),sem='relaxed')


@triton.jit
def gather_membership(FLAGS,ROWS,CANDIDATE,CHOSEN,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    row=tl.load(ROWS+i,valid,other=0);bits=tl.load(FLAGS+row,valid,other=0)
    tl.store(CANDIDATE+i*2,(bits&1)!=0,valid);tl.store(CANDIDATE+i*2+1,(bits&2)!=0,valid)
    tl.store(CHOSEN+i*2,(bits&4)!=0,valid);tl.store(CHOSEN+i*2+1,(bits&8)!=0,valid)


@triton.jit
def candidate_sides(PIXELS,OBJECTS,DEPTH,NEAR,FAR,N,H:tl.constexpr,W:tl.constexpr,
                    ABS:tl.constexpr,REL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);inside=i<N
    source=tl.load(PIXELS+i,inside,other=0);v=source//W;u=source%W
    z=tl.load(DEPTH+source,inside,other=0)
    near=tl.full((B,),False,tl.int1);far=near
    for direction in tl.static_range(4):
        off=tl.where(direction==0,-1,tl.where(direction==1,1,tl.where(direction==2,-W,W)))
        ok=inside&tl.where(direction==0,u>0,tl.where(direction==1,u+1<W,tl.where(direction==2,v>0,v+1<H)))
        q=tl.load(DEPTH+source+off,ok,other=0)
        valid=ok&(q>0)&(q<float('inf'))
        jump=valid&(tl.abs(z-q)>ABS+REL*tl.minimum(z,q))
        near=near|(jump&(z<q));far=far|(jump&(z>q))
    tl.store(NEAR+i,near,inside);tl.store(FAR+i,far,inside)


@triton.jit
def compress(PARENT, ROOTS, CHANGED, N, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);inside=i<N
    r=tl.load(PARENT+i,inside,other=0)
    q=tl.load(PARENT+r,inside,other=0)
    while tl.sum((inside&(q!=r)).to(tl.int32),0)>0:
        r=q;q=tl.load(PARENT+r,inside,other=0)
    tl.store(ROOTS+i,r,inside)
    # No union runs concurrently: every store shortens an existing path to
    # the same stable root. Other lanes may observe old or shortened paths.
    tl.store(PARENT+i,r,inside)
    if tl.program_id(0)==0:tl.store(CHANGED,0)


@triton.jit
def hook(MASK, ROOTS, PARENT, CHANGED, H, W, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);inside=i<H*W
    v=i//W;u=i%W;m=tl.load(MASK+i,inside,other=0)
    r=tl.load(ROOTS+i,inside,other=0)
    for direction in tl.static_range(4):
        off=tl.where(direction==0,1,tl.where(direction==1,W,tl.where(direction==2,W-1,W+1)))
        ok=inside&tl.where(direction==0,u+1<W,tl.where(direction==1,v+1<H,
            tl.where(direction==2,(v+1<H)&(u>0)&m,(v+1<H)&(u+1<W)&m)))
        qm=tl.load(MASK+i+off,ok,other=0)
        qr=tl.load(ROOTS+i+off,ok,other=0)
        connect=ok&(qm==m)&(r!=qr)
        tl.atomic_min(PARENT+tl.maximum(r,qr),tl.minimum(r,qr),connect,sem='relaxed')
        if tl.sum(connect.to(tl.int32),0)>0:
            tl.atomic_or(CHANGED,1,sem='relaxed')


@triton.jit
def _dv(d):
    return tl.where((d>=1)&(d<=3),1,tl.where((d>=5)&(d<=7),-1,0))


@triton.jit
def _du(d):
    return tl.where((d==0)|(d==1)|(d==7),-1,tl.where((d>=3)&(d<=5),1,0))


@triton.jit
def trace(MASK, STARTS, OFFSETS, WALK, LENGTHS, ERROR, H, W):
    """Moore 8-neighbour walk; Jacob stopping rule, including thin revisits."""
    group=tl.program_id(0)
    start=tl.load(STARTS+group).to(tl.int32);offset=tl.load(OFFSETS+group)
    limit=tl.load(OFFSETS+group+1)-offset
    current=start;back=0;first=-1;length=0;running=True
    lane=tl.arange(0,8)
    while running&(length<limit):
        dirs=(back+lane)%8
        neighbour=current+_dv(dirs)*W+_du(dirs)
        occupied=tl.load(MASK+neighbour)
        step=tl.min(tl.where(occupied,lane,8),0)
        found=step<8;direction=(back+step)%8
        nxt=current+_dv(direction)*W+_du(direction)
        close=(length>0)&(current==start)&(nxt==first)
        if ~close:
            source=(current//W-1)*(W-2)+current%W-1
            tl.store(WALK+offset+length,source);length+=1
        if first<0:first=nxt
        prev=(direction+7)%8
        by=_dv(prev)-_dv(direction);bx=_du(prev)-_du(direction)
        back=tl.min(tl.where((_dv(lane)==by)&(_du(lane)==bx),lane,8),0)
        current=nxt
        running=found&~close
    tl.store(LENGTHS+group,length)
    if running:tl.atomic_or(ERROR,1,sem='relaxed')


@triton.jit
def canonical(WALK, OFFSETS, LENGTHS, CANON, B:tl.constexpr):
    """Lexicographically smallest minimum-source rotation in either direction."""
    group=tl.program_id(0);offset=tl.load(OFFSETS+group);n=tl.load(LENGTHS+group).to(tl.int32)
    # Trace starts at the component's unique minimum pixel. Thin walks can
    # revisit it, so all occurrences and both directions participate.
    minimum=tl.load(WALK+offset);best=0;sign=1
    lane=tl.arange(0,B)
    for chunk in range(tl.cdiv(n,B)):
        pos=chunk*B+lane;inside=pos<n
        values=tl.load(WALK+offset+pos,inside,other=-1)
        matches=inside&(values==minimum)
        while tl.sum(matches.to(tl.int32),0)>0:
            at=tl.min(tl.where(matches,pos,n),0)
            for direction in range(2):
                ds=1-2*direction
                # A sequence cannot improve on itself. The old path compared
                # every element of the initial walk to the identical walk.
                if (at!=best)|(ds!=sign):
                    j=0;equal=True;less=False
                    while (j<n)&equal:
                        a=tl.load(WALK+offset+(at+ds*j+n)%n)
                        b=tl.load(WALK+offset+(best+sign*j+n)%n)
                        less=a<b;equal=a==b;j+=1
                    if less:best=at;sign=ds
            matches=matches&(pos!=at)
    for chunk in range(tl.cdiv(n,B)):
        j=chunk*B+lane;inside=j<n
        value=tl.load(WALK+offset+(best+sign*j+n)%n,inside,other=-1)
        tl.store(CANON+offset+j,value,inside)


@triton.jit
def fps_seed(POINTS, INITIAL, USED, DISTANCE, BEST_D, BEST_I,
             N:tl.constexpr, S:tl.constexpr, B:tl.constexpr, T:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    x=tl.load(POINTS+i*3,ok,other=0).to(tl.float64)
    y=tl.load(POINTS+i*3+1,ok,other=0).to(tl.float64)
    z=tl.load(POINTS+i*3+2,ok,other=0).to(tl.float64)
    distance=tl.full((B,),float('inf'),tl.float64)
    for start in range(tl.cdiv(S,T)):
        j=start*T+tl.arange(0,T);valid=j<S
        row=tl.load(INITIAL+j,valid,other=0)
        qx=tl.load(POINTS+row*3,valid,other=0).to(tl.float64)
        qy=tl.load(POINTS+row*3+1,valid,other=0).to(tl.float64)
        qz=tl.load(POINTS+row*3+2,valid,other=0).to(tl.float64)
        dx=x[:,None]-qx[None,:];dy=y[:,None]-qy[None,:];dz=z[:,None]-qz[None,:]
        d=(dx*dx+dy*dy)+dz*dz
        distance=tl.minimum(distance,tl.min(tl.where(valid[None,:],d,float('inf')),1))
    used=tl.load(USED+i,ok,other=1)
    distance=tl.where(ok&~used,distance,-float('inf'))
    tl.store(DISTANCE+i,distance,ok)
    maximum=tl.max(distance,0)
    row=tl.min(tl.where(ok&(distance==maximum),i,2147483647),0)
    tl.store(BEST_D+tl.program_id(0),maximum);tl.store(BEST_I+tl.program_id(0),row)


@triton.jit
def fps_seed_pruned(POINTS,INITIAL,USED,DISTANCE,BEST_D,BEST_I,COUNTER,
                    N,S,B:tl.constexpr,T:tl.constexpr,TRACK:tl.constexpr=False):
    """Exact tile-AABB lower bounds skip seed tiles that cannot improve a block."""
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    x=tl.load(POINTS+i*3,ok,other=0).to(tl.float64)
    y=tl.load(POINTS+i*3+1,ok,other=0).to(tl.float64)
    z=tl.load(POINTS+i*3+2,ok,other=0).to(tl.float64)
    used=tl.load(USED+i,ok,other=1)
    distance=tl.where(ok&~used,float('inf'),-float('inf')).to(tl.float64)
    for start in range(tl.cdiv(S,T)):
        j=start*T+tl.arange(0,T);valid=j<S
        row=tl.load(INITIAL+j,valid,other=-1);valid=valid&(row>=0)
        qx=tl.load(POINTS+row*3,valid,other=0).to(tl.float64)
        qy=tl.load(POINTS+row*3+1,valid,other=0).to(tl.float64)
        qz=tl.load(POINTS+row*3+2,valid,other=0).to(tl.float64)
        xmin=tl.min(tl.where(valid,qx,float('inf')),0);xmax=tl.max(tl.where(valid,qx,-float('inf')),0)
        ymin=tl.min(tl.where(valid,qy,float('inf')),0);ymax=tl.max(tl.where(valid,qy,-float('inf')),0)
        zmin=tl.min(tl.where(valid,qz,float('inf')),0);zmax=tl.max(tl.where(valid,qz,-float('inf')),0)
        bx=tl.maximum(tl.maximum(xmin-x,x-xmax),0.)
        by=tl.maximum(tl.maximum(ymin-y,y-ymax),0.)
        bz=tl.maximum(tl.maximum(zmin-z,z-zmax),0.)
        lower=(bx*bx+by*by)+bz*bz
        needs=ok&~used&(lower<distance)
        if tl.sum(needs.to(tl.int32),0)>0:
            px=x[:,None]-qx[None,:];py=y[:,None]-qy[None,:];pz=z[:,None]-qz[None,:]
            d=(px*px+py*py)+pz*pz
            distance=tl.minimum(distance,tl.min(tl.where(valid[None,:],d,float('inf')),1))
            if TRACK:tl.atomic_add(COUNTER,tl.sum(ok.to(tl.int32),0).to(tl.int64)*tl.sum(valid.to(tl.int32),0),sem='relaxed')
    tl.store(DISTANCE+i,distance,ok)
    maximum=tl.max(distance,0);row=tl.min(tl.where(ok&(distance==maximum),i,2147483647),0)
    tl.store(BEST_D+tl.program_id(0),maximum);tl.store(BEST_I+tl.program_id(0),row)


@triton.jit
def seed_tile_bounds(POINTS,INITIAL,BOUNDS,S,T:tl.constexpr):
    tile=tl.program_id(0);j=tile*T+tl.arange(0,T);valid=j<S
    row=tl.load(INITIAL+j,valid,other=-1);valid=valid&(row>=0)
    for axis in tl.static_range(3):
        value=tl.load(POINTS+row*3+axis,valid,other=0).to(tl.float64)
        lo=tl.min(tl.where(valid,value,float('inf')),0)
        hi=tl.max(tl.where(valid,value,-float('inf')),0)
        tl.store(BOUNDS+tile*6+axis*2,lo);tl.store(BOUNDS+tile*6+axis*2+1,hi)


@triton.jit
def fps_seed_bounded(POINTS,INITIAL,USED,BOUNDS,DISTANCE,BEST_D,BEST_I,COUNTER,
                     N,S,B:tl.constexpr,T:tl.constexpr,TRACK:tl.constexpr=False):
    """Reuse each exact seed-tile bound instead of reducing it per point block."""
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    x=tl.load(POINTS+i*3,ok,other=0).to(tl.float64)
    y=tl.load(POINTS+i*3+1,ok,other=0).to(tl.float64)
    z=tl.load(POINTS+i*3+2,ok,other=0).to(tl.float64)
    used=tl.load(USED+i,ok,other=1);active=ok&~used
    distance=tl.where(active,float('inf'),-float('inf')).to(tl.float64)
    if tl.sum(active.to(tl.int32),0)>0:
        for tile in range(tl.cdiv(S,T)):
            xmin=tl.load(BOUNDS+tile*6).to(tl.float64);xmax=tl.load(BOUNDS+tile*6+1).to(tl.float64)
            ymin=tl.load(BOUNDS+tile*6+2).to(tl.float64);ymax=tl.load(BOUNDS+tile*6+3).to(tl.float64)
            zmin=tl.load(BOUNDS+tile*6+4).to(tl.float64);zmax=tl.load(BOUNDS+tile*6+5).to(tl.float64)
            bx=tl.maximum(tl.maximum(xmin-x,x-xmax),0.)
            by=tl.maximum(tl.maximum(ymin-y,y-ymax),0.)
            bz=tl.maximum(tl.maximum(zmin-z,z-zmax),0.)
            lower=(bx*bx+by*by)+bz*bz
            needs=active&(lower<distance)
            if tl.sum(needs.to(tl.int32),0)>0:
                j=tile*T+tl.arange(0,T);valid=j<S
                row=tl.load(INITIAL+j,valid,other=-1);valid=valid&(row>=0)
                qx=tl.load(POINTS+row*3,valid,other=0).to(tl.float64)
                qy=tl.load(POINTS+row*3+1,valid,other=0).to(tl.float64)
                qz=tl.load(POINTS+row*3+2,valid,other=0).to(tl.float64)
                dx=x[:,None]-qx[None,:];dy=y[:,None]-qy[None,:];dz=z[:,None]-qz[None,:]
                d=(dx*dx+dy*dy)+dz*dz
                distance=tl.minimum(distance,tl.min(tl.where(valid[None,:],d,float('inf')),1))
                if TRACK:tl.atomic_add(COUNTER,tl.sum(ok.to(tl.int32),0).to(tl.int64)*tl.sum(valid.to(tl.int32),0),sem='relaxed')
    tl.store(DISTANCE+i,distance,ok)
    maximum=tl.max(distance,0);row=tl.min(tl.where(ok&(distance==maximum),i,2147483647),0)
    tl.store(BEST_D+tl.program_id(0),maximum);tl.store(BEST_I+tl.program_id(0),row)


@triton.jit
def fps_choose(BEST_D,BEST_I,OUTPUT,K,G,B:tl.constexpr):
    i=tl.arange(0,B);d=tl.load(BEST_D+i,i<G,other=-float('inf'))
    maximum=tl.max(d,0);rows=tl.load(BEST_I+i,i<G,other=2147483647)
    row=tl.min(tl.where(d==maximum,rows,2147483647),0)
    tl.store(OUTPUT+K,row)


@triton.jit
def fps_update(POINTS,DISTANCE,OUTPUT,BEST_D,BEST_I,
               K,N,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    chosen=tl.load(OUTPUT+K)
    x=tl.load(POINTS+i*3,ok,other=0).to(tl.float64)-tl.load(POINTS+chosen*3).to(tl.float64)
    y=tl.load(POINTS+i*3+1,ok,other=0).to(tl.float64)-tl.load(POINTS+chosen*3+1).to(tl.float64)
    z=tl.load(POINTS+i*3+2,ok,other=0).to(tl.float64)-tl.load(POINTS+chosen*3+2).to(tl.float64)
    old=tl.load(DISTANCE+i,ok,other=-float('inf'))
    d=tl.minimum(old,(x*x+y*y)+z*z)
    d=tl.where(ok&(i!=chosen),d,-float('inf'))
    tl.store(DISTANCE+i,d,ok)
    maximum=tl.max(d,0);row=tl.min(tl.where(ok&(d==maximum),i,2147483647),0)
    tl.store(BEST_D+tl.program_id(0),maximum);tl.store(BEST_I+tl.program_id(0),row)
