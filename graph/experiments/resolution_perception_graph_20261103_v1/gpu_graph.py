"""Scoped CUDA acceleration of the frozen builder, preserving its input policy.

No shared source is edited. Host contour/Qhull/selection remain explicitly host
operations; dense robot clipping, radius connectivity and edge gates use CUDA.
Per-build read-only camera conversions are reused, never across builds.
"""
from itertools import product
import numpy as np
import torch
import triton
import triton.language as tl
from scipy.spatial import ConvexHull, cKDTree


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


@triton.jit
def _clip_robot(D, Rows, Normals, Numerator, Depth, Z, Owner, Interior,
                N, F, Link,
                Planes: tl.constexpr, Pixels: tl.constexpr=32):
    idx=tl.program_id(0)*Pixels+tl.arange(0,Pixels)
    row=tl.load(Rows+idx,idx<N,other=0)
    dx=tl.load(D+3*row,idx<N,other=0);dy=tl.load(D+3*row+1,idx<N,other=0);dz=tl.load(D+3*row+2,idx<N,other=0)
    face=tl.arange(0,Planes)
    nx=tl.load(Normals+3*face,face<F,other=0);ny=tl.load(Normals+3*face+1,face<F,other=0);nz=tl.load(Normals+3*face+2,face<F,other=0)
    num=tl.load(Numerator+face,face<F,other=0)
    denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
    parallel=tl.abs(denom)<1e-12
    crossing=tl.where(parallel,0,num[None,:]/denom)
    lower=tl.max(tl.where((face[None,:]<F)&(denom < -1e-12),crossing,-float('inf')),1)
    upper=tl.min(tl.where((face[None,:]<F)&(denom > 1e-12),crossing,float('inf')),1)
    outside=tl.sum(((face[None,:]<F)&parallel&(num[None,:]<0)).to(tl.int32),1)>0
    entry=tl.maximum(lower,0)
    hit=(upper>=entry)&(upper>0)&~outside&(idx<N)
    measured=tl.load(Depth+row,idx<N,other=float('nan'))
    prior=tl.load(Interior+row,idx<N,other=False)
    interior=prior|(hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005))
    tl.store(Interior+row,interior,idx<N)
    nearest=tl.load(Z+row,idx<N,other=float('inf'))
    change=hit&(entry<nearest)
    tl.store(Z+row,entry,change);tl.store(Owner+row,Link,change)


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


@triton.jit
def _fps_step(X,Distance,Chosen,BlockMax,BlockIndex,N,First,Last,SEED:tl.constexpr,B:tl.constexpr=512):
    block=tl.program_id(0);i=block*B+tl.arange(0,B);valid=i<N
    x=tl.load(X+3*i,valid,other=0);y=tl.load(X+3*i+1,valid,other=0);z=tl.load(X+3*i+2,valid,other=0)
    if SEED:d=tl.full((B,),float('inf'),tl.float64)
    else:d=tl.load(Distance+i,valid,other=-float('inf'))
    for k in range(First,Last):
        j=tl.load(Chosen+k)
        dx=x-tl.load(X+3*j);dy=y-tl.load(X+3*j+1);dz=z-tl.load(X+3*j+2)
        d=tl.minimum(d,(dx*dx+dy*dy)+dz*dz)
        d=tl.where(i==j,-float('inf'),d)
    tl.store(Distance+i,d,valid)
    d=tl.where(valid,d,-float('inf'));best=tl.max(d,0)
    first=tl.min(tl.where((d==best)&valid,i,2147483647),0)
    tl.store(BlockMax+block,best);tl.store(BlockIndex+block,first)


@triton.jit
def _fps_pick(BlockMax,BlockIndex,Chosen,Blocks,Position,P:tl.constexpr):
    i=tl.arange(0,P);v=tl.load(BlockMax+i,i<Blocks,other=-float('inf'))
    best=tl.max(v,0);idx=tl.load(BlockIndex+i,i<Blocks,other=2147483647)
    chosen=tl.min(tl.where((v==best)&(i<Blocks),idx,2147483647),0)
    tl.store(Chosen+Position,chosen)


def fps_indices_cuda(points, count, *, seed=0, initial=(), fused=False):
    """Original deterministic float64 FPS, without host round trips per point."""
    from visual.runtime.sampling import _integer
    _integer(count,'FPS count');_integer(seed,'FPS seed')
    xyz=np.asarray(points,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('FPS requires finite [N,3] observed coordinates')
    initial=np.asarray(initial,dtype=np.int64)
    if initial.ndim!=1 or len(np.unique(initial))!=len(initial) or ((initial<0)|(initial>=len(xyz))).any():
        raise ValueError('initial FPS rows must be distinct valid indices')
    count=min(count,len(xyz))
    if len(initial)>count:raise ValueError('initial rows exceed FPS count')
    if not count:return np.empty(0,np.int64)
    coords=_tensor(xyz);distance=torch.full((len(xyz),),torch.inf,dtype=torch.float64,device='cuda')
    chosen=torch.empty(count,dtype=torch.long,device='cuda');chosen[:len(initial)]=_tensor(initial,torch.long)
    if fused:
        used=len(initial)
        if not used:
            chosen[0]=int(np.random.default_rng(seed).integers(len(xyz)));used=1
        if used==count:return _numpy(chosen)
        blocks=triton.cdiv(len(xyz),512)
        maxima=torch.empty(blocks,dtype=torch.float64,device='cuda')
        indices=torch.empty(blocks,dtype=torch.int64,device='cuda')
        _fps_step[(blocks,)](coords,distance,chosen,maxima,indices,len(xyz),0,used,True,enable_fp_fusion=False)
        for i in range(used,count):
            _fps_pick[(1,)](maxima,indices,chosen,blocks,i,triton.next_power_of_2(blocks))
            if i+1<count:
                _fps_step[(blocks,)](coords,distance,chosen,maxima,indices,len(xyz),i,i+1,False,enable_fp_fusion=False)
        return _numpy(chosen)
    def update(j):
        dx=coords[:,0]-coords[j,0];dy=coords[:,1]-coords[j,1];dz=coords[:,2]-coords[j,2]
        value=(dx*dx+dy*dy)+dz*dz
        torch.minimum(distance,value,out=distance);distance[j]=-torch.inf
    for j in initial:update(int(j))
    used=len(initial)
    if not used:
        first=int(np.random.default_rng(seed).integers(len(xyz)))
        chosen[0]=first;update(first);used=1
    for i in range(used,count):
        j=torch.argmax(distance);chosen[i]=j;update(j)
    return _numpy(chosen)


class OptimizedGraph:
    def __init__(self, backend='cuda'):
        self.fused=backend.startswith('cuda_fused')
        self.fps_cuda=backend in ('cuda_fused_v3','cuda_fused_v4','cuda_fused_v5')
        self.fps_fused=backend in ('cuda_fused_v4','cuda_fused_v5')
        self.fast_support=backend=='cuda_fused_v5';self.support=None
        self.backend='cuda' if backend.startswith('cuda') else backend
        self.restore=[];self.cache={};self.reference={};self.reference_checks=0

    def load_reference(self,root):
        """Compare previously completed actual receipts; never inspect GT."""
        import common as C
        from pathlib import Path
        root=Path(root)
        for path in (root/'graphs'/'actual').glob('*/*/*/receipt.json'):
            receipt=C.read(path);outcome=receipt.get('outcome',{})
            if outcome.get('status')!='COMPLETE' or not outcome.get('mask_hash'):continue
            folder=root/outcome['reused_from'] if outcome.get('reused_from') else path.parent
            expected=C.read(folder/'result.json')['hashes']
            key=(path.parent.parent.parent.name,int(path.parent.parent.name),outcome['mask_hash'])
            if key in self.reference:assert self.reference[key]==expected,'inconsistent original graph hashes'
            self.reference[key]=expected

    def patch(self,owner,name,value):
        self.restore.append((owner,name,getattr(owner,name)));setattr(owner,name,value)

    def __enter__(self):
        import graph.plan_a_geometry.core as core
        from graph.plan_a_bc.geometry import CurrentGeometry
        from graph.persistent_geometry.contracts import CameraObservation
        from radius_components import components
        context=self
        original_build=CurrentGeometry.build
        def build(geo,*args,**kwargs):
            context.cache={};context.support=None
            result=original_build(geo,*args,**kwargs)
            if context.reference:
                import common as C
                cameras,data,_,masks=args
                mask_hash=C.digest({c:C.array_sha(a) for c,a in masks.items()})
                key=(str(data['episode']),cameras[0].depth.shape[0],mask_hash)
                if key in context.reference:
                    graph=result[0]
                    got={name:C.array_sha(getattr(graph,name).numpy()) for name in C.FIELDS}
                    assert got==context.reference[key],('CUDA graph differs from original',key,got,context.reference[key])
                    context.reference_checks+=1
            return result
        self.patch(CurrentGeometry,'build',build)
        for name in ['depth','world_from_camera']:
            getter=getattr(CameraObservation,name).fget
            def cached(camera,getter=getter,name=name):
                key=(id(camera),name)
                if key not in context.cache:context.cache[key]=getter(camera)
                return context.cache[key]
            self.patch(CameraObservation,name,property(cached))
        if self.backend=='cpu_cells':self.patch(core,'components',components)
        elif self.fast_support:
            def gpu_components(xyz,radius):
                def save(known,base_radius):context.support=(id(xyz),known,base_radius)
                return radius_components(xyz,radius,support_callback=save)
            self.patch(core,'components',gpu_components)
        else:self.patch(core,'components',radius_components)
        # Only >=3 support is consumed. k=3 avoids counting every dense neighbor.
        original_tree=core.cKDTree
        class GateTree:
            def __init__(self,xyz):self.xyz=xyz
            def query_ball_point(self,xyz,radius,return_length=False):
                if not return_length:raise ValueError('gate only consumes support counts')
                known=np.zeros(len(xyz),bool)
                if context.fast_support and context.support is not None:
                    identity,clique,base_radius=context.support
                    if identity==id(self.xyz)==id(xyz) and np.sqrt(3)*base_radius/2<radius:known=clique
                result=np.full(len(xyz),3,dtype=int);rare=np.flatnonzero(~known)
                if len(rare):
                    d,_=original_tree(self.xyz).query(xyz[rare],k=3,distance_upper_bound=np.nextafter(radius,np.inf))
                    result[rare]=(np.asarray(d)<=radius).sum(1)
                return result
        self.patch(core,'cKDTree',GateTree)
        if self.backend=='cuda':
            self.patch(core,'classify_robot',self.classify_robot)
            import graph.persistent_geometry.graph as graphmod
            self.patch(graphmod,'surface_edges',self.surface_edges)
            if self.fps_cuda:
                from visual.runtime.test import run_variable_rate_shape_first as shape_first
                if self.fps_fused:
                    def fps(*args,**kwargs):return fps_indices_cuda(*args,**kwargs,fused=True)
                    self.patch(shape_first,'fps_indices',fps)
                else:self.patch(shape_first,'fps_indices',fps_indices_cuda)
        return self

    def __exit__(self,*_):
        for owner,name,value in reversed(self.restore):setattr(owner,name,value)
        self.cache={}

    def camera_values(self,camera):
        key=(id(camera),'cuda')
        if key not in self.cache:
            self.cache[key]=dict(depth=_tensor(camera.depth),world=_tensor(camera.world_from_camera),
                inverse=_tensor(np.linalg.inv(camera.world_from_camera)),k=_tensor(camera.intrinsic))
        return self.cache[key]

    def classify_robot(self,camera,surfaces,tolerance_m=.0025,uncertainty_band_m=.006):
        from graph.persistent_geometry.contracts import transform
        h,w=camera.depth_raw_mm.shape;v,u=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float64),torch.arange(w,device='cuda',dtype=torch.float64),indexing='ij')
        cv=self.camera_values(camera);k=cv['k'];wc=cv['world']
        rays=torch.stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],torch.ones((h,w),device='cuda')),2).reshape(-1,3)
        direction=rays@wc[:3,:3].T;origin=wc[:3,3]
        zbuffer=torch.full((h*w,),torch.inf,dtype=torch.float64,device='cuda')
        owner=torch.full((h*w,),-1,dtype=torch.int32,device='cuda');interior=torch.zeros(h*w,dtype=torch.bool,device='cuda')
        measured=cv['depth'].flatten();wc_host=camera.world_from_camera
        for i,surface in enumerate(surfaces):
            vertices=surface.world_vertices;vc=transform(vertices,np.linalg.inv(wc_host))
            if np.all(vc[:,2]<=0):continue
            if np.all(vc[:,2]>0):
                kh=camera.intrinsic;uv=vc[:,:2]/vc[:,2,None]*np.array([kh[0,0],kh[1,1]])+kh[:2,2]-.5
                lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w-1,h-1])
                if np.any(hi<lo):continue
                vv,uu=torch.meshgrid(torch.arange(int(lo[1]),int(hi[1])+1,device='cuda'),torch.arange(int(lo[0]),int(hi[0])+1,device='cuda'),indexing='ij')
                rows=(vv*w+uu).flatten()
            else:rows=torch.arange(h*w,device='cuda')
            equations=_tensor(ConvexHull(vertices).equations);normals,offset=equations[:,:3],equations[:,3]
            numerator=-(normals@origin+offset)
            if self.fused:
                _clip_robot[(triton.cdiv(len(rows),32),)](direction,rows,normals.contiguous(),numerator,
                    measured,zbuffer,owner,interior,len(rows),len(normals),i,
                    triton.next_power_of_2(len(normals)),enable_fp_fusion=False)
                continue
            denominator=direction[rows]@normals.T
            parallel=denominator.abs()<1e-12;crossing=torch.where(parallel,0,numerator/denominator)
            lower=torch.where(denominator < -1e-12,crossing,-torch.inf).amax(1)
            upper=torch.where(denominator > 1e-12,crossing,torch.inf).amin(1)
            outside=(parallel&(numerator<0)).any(1);entry=lower.clamp_min(0)
            hit=(upper>=entry)&(upper>0)&~outside
            interior[rows]|=hit&torch.isfinite(measured[rows])&(measured[rows]>entry+.0005)&(measured[rows]<upper-.0005)
            nearer=hit&(entry<zbuffer[rows]);rr=rows[nearer]
            zbuffer[rr]=entry[nearer];owner[rr]=i
        predicted=zbuffer.reshape(h,w);interior=interior.reshape(h,w);valid=torch.isfinite(cv['depth'])
        difference=(cv['depth']-predicted).abs()
        confirmed=valid&((torch.isfinite(predicted)&(difference<=tolerance_m))|interior)
        uncertain=valid&torch.isfinite(predicted)&~confirmed&(difference<=uncertainty_band_m)
        return dict(confirmed_robot=_numpy(confirmed),uncertain=_numpy(uncertain),candidate=_numpy(valid&~confirmed&~uncertain),
            predicted_robot_depth=_numpy(predicted),link_owner=_numpy(owner.reshape(h,w)),inside_known_link_volume=_numpy(interior),
            method='per-shape collision CV-depth agreement or measured point inside actual link collision volume v2',
            tolerance_m=tolerance_m,uncertainty_band_m=uncertainty_band_m)

    def surface_edges(self,xyz,normals,normal_valid,object_id,reference,cameras=(),k=8,radius=.02):
        from graph.integration_baseline.graph import bidirectional
        # Preserve scipy's existing nearest-neighbor/tie order on <=256 selected nodes.
        candidates=set();visited=set()
        for oid in (0,1):
            rows=np.flatnonzero(object_id==oid)
            if len(rows)<2:continue
            _,near=cKDTree(xyz[rows]).query(xyz[rows],k=min(k+1,len(rows)))
            for ii,indices in enumerate(near):
                for jj in np.atleast_1d(indices):
                    a,b=sorted((int(rows[ii]),int(rows[jj])))
                    if a!=b and (a,b) not in visited:visited.add((a,b));candidates.add((a,b))
        pairs=np.array(sorted(candidates),dtype=np.int64).reshape(-1,2)
        audit=dict(candidates=0,rejected_normal=0,rejected_witness=0,rejected_free_space=0)
        if not len(pairs):return bidirectional(pairs),audit
        x=_tensor(xyz);normal=_tensor(normals);nv=_tensor(normal_valid,torch.bool);ij=_tensor(pairs,torch.long)
        a,b=ij.T;delta=x[b]-x[a];length=torch.linalg.vector_norm(delta,dim=1)
        eligible=(length<=radius)&(length>=1e-9)
        bad_normal=nv[a]&nv[b]&(((normal[a]*normal[b]).sum(1)<.5)|torch.maximum((normal[a]*delta).sum(1).abs(),(normal[b]*delta).sum(1).abs()).gt(.002+length*.25))
        samples=x[a,None,:]+torch.tensor([.25,.5,.75],device='cuda',dtype=x.dtype)[None,:,None]*delta[:,None,:]
        witness_bad=torch.zeros(len(pairs),dtype=torch.bool,device='cuda')
        for oid in (0,1):
            rows=np.flatnonzero(object_id[pairs[:,0]]==oid)
            if not len(rows):continue
            refs=_tensor(reference[oid]);q=samples[rows]
            distances=((q[:,:,None,:]-refs[None,None,:,:])**2).sum(3).sqrt().amin(2)
            witness_bad[rows]=(distances>.004).any(1)
        free=torch.zeros(len(pairs)*3,dtype=torch.bool,device='cuda');q=samples.reshape(-1,3)
        for camera in cameras:
            cv=self.camera_values(camera);inv=cv['inverse'];p=q@inv[:3,:3].T+inv[:3,3];kh=cv['k']
            uv=torch.floor(p[:,:2]/p[:,2,None].clamp_min(1e-12)*torch.stack((kh[0,0],kh[1,1]))+kh[:2,2]).long()
            h,w=cv['depth'].shape;u,v=uv.T
            inside=(p[:,2]>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
            ids=torch.nonzero(inside).flatten();z=cv['depth'][v[ids],u[ids]]
            free[ids]|=torch.isfinite(z)&(p[ids,2]<z-.006)
        free_bad=free.reshape(-1,3).any(1)
        e=_numpy(eligible);bn=_numpy(bad_normal);bw=_numpy(witness_bad);bf=_numpy(free_bad)
        audit.update(candidates=int(e.sum()),rejected_normal=int((e&bn).sum()),
            rejected_witness=int((e&~bn&bw).sum()),rejected_free_space=int((e&~bn&~bw&bf).sum()))
        return bidirectional(pairs[e&~bn&~bw&~bf]),audit


if __name__=='__main__':
    import runpy
    import sys
    from pathlib import Path
    import common as C
    protocol=C.assert_source()
    assert protocol['graph_execution_backend']=='cuda_fused_v5'
    target=sys.argv[1];sys.argv=sys.argv[1:]
    context=OptimizedGraph(protocol['graph_execution_backend'])
    if target=='graph_worker.py' and 'quality' in sys.argv and 'actual' in sys.argv:
        context.load_reference(protocol['optimization_reference_root'])
    with context:
        runpy.run_path(str(Path(__file__).with_name(target)),run_name='__main__')
    stage='_'.join(sys.argv).replace('/','_').replace('--','')
    C.write(C.ROOT/('optimization_validation_'+stage+'.json'),
        dict(status='PASS',reference_graphs=len(context.reference),matched_original_graph_builds=context.reference_checks,
             reference='completed pre-optimization actual Graph receipts; no GT',frozen_utc=C.now()))
