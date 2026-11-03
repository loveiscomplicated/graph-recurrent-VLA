"""Private Graph-only sparse robot decisions. Public full-screen API stays intact.

Each decision is valid ONLY at `pixels`, the raw two-object mask union. No dense
candidate/depth/owner array is fabricated. Float64, link order and strict ties
match Stage 9. No frame-dependent result survives a build.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import torch
import triton
import triton.language as tl
from scipy.spatial import ConvexHull
from graph.persistent_geometry.contracts import transform
from gpu_graph import _clip_robot


@dataclass
class SparseDecision:
    shape: tuple
    pixels: np.ndarray
    confirmed: np.ndarray
    uncertain: np.ndarray
    diagnostics: dict | None = None


def filtered_masks(masks, result):
    if not isinstance(result, SparseDecision):
        from graph.persistent_geometry.fusion import filtered_masks as full
        return full(masks, result)
    masks=np.asarray(masks,bool).copy()
    if masks.shape != (2,*result.shape):raise ValueError('ROI mask alignment')
    if not np.array_equal(np.flatnonzero(masks.any(0)),result.pixels):
        raise ValueError('sparse decision used outside its original mask union')
    rejected=masks[0]&masks[1]
    rejected.ravel()[result.pixels] |= result.confirmed | result.uncertain
    masks[:,rejected]=False
    return masks,rejected


def gpu(a):
    return torch.as_tensor(np.ascontiguousarray(a),device='cuda')


def projected_bounds(camera, vertices):
    h,w=camera.depth_raw_mm.shape
    vc=transform(vertices,np.linalg.inv(camera.world_from_camera))
    if np.all(vc[:,2]<=0):return (0,0,-1,-1)
    if not np.all(vc[:,2]>0):return (0,0,w-1,h-1)
    k=camera.intrinsic
    uv=vc[:,:2]/vc[:,2,None]*np.array([k[0,0],k[1,1]])+k[:2,2]-.5
    lo=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0])
    hi=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[w-1,h-1])
    if np.any(hi<lo):return (0,0,-1,-1)
    return (int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1]))


class RayCache:
    """At most 8 O(H+W) calibration vectors; content keys detect in-place edits.

    Safe across frames/cameras, including moving wrist extrinsics: only CV rays
    are cached. World rotation, origin, depth and surfaces are always current.
    Creation, key validation and eviction happen inside the timed build.
    """
    def __init__(self):self.entries=OrderedDict();self.hits=0;self.misses=0
    def get(self,camera):
        h,w=camera.depth_raw_mm.shape
        k=np.asarray(camera.intrinsic,dtype=np.float64)
        key=(h,w,k.tobytes())
        if key in self.entries:
            self.hits+=1;value=self.entries.pop(key)
        else:
            self.misses+=1
            x=(torch.arange(w,device='cuda',dtype=torch.float64)+.5-float(k[0,2]))/float(k[0,0])
            y=(torch.arange(h,device='cuda',dtype=torch.float64)+.5-float(k[1,2]))/float(k[1,1])
            value=(x,y)
        self.entries[key]=value
        while len(self.entries)>8:self.entries.popitem(last=False)
        return value
    @property
    def bytes(self):return sum(t.numel()*t.element_size() for pair in self.entries.values() for t in pair)


@triton.jit
def _flags(Depth,Z,Interior,Flags,N:tl.constexpr,TOL:tl.constexpr,BAND:tl.constexpr,B:tl.constexpr=256):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    d=tl.load(Depth+i,ok,float('nan'));z=tl.load(Z+i,ok,float('inf'))
    interior=tl.load(Interior+i,ok,0);valid=d==d;diff=tl.abs(d-z)
    confirmed=valid&(((z<float('inf'))&(diff<=TOL))|interior)
    uncertain=valid&(z<float('inf'))&~confirmed&(diff<=BAND)
    tl.store(Flags+i,confirmed,ok);tl.store(Flags+N+i,uncertain,ok)


class RobotFilter:
    def __init__(self, variant='strict', rays=None):
        if variant not in ('roi','hulls','packed','rays','atomic','strict'):raise ValueError(variant)
        self.variant=variant;self.rays=rays or RayCache();self.hulls=None;self.surface_ids=None
        self.last_metrics=[]

    def hull_values(self,surfaces):
        # Per-build only, with strong references. Existing grouped Stage 3 already
        # reuses hulls; applying that idea to sparse high-resolution is not novel.
        if self.variant=='roi' or self.hulls is None:
            hulls=[(s.world_vertices,None) for s in surfaces]
            hulls=[(v,ConvexHull(v).equations) for v,_ in hulls]
            if self.variant!='roi':self.hulls=hulls;self.surface_ids=tuple(surfaces)
        else:
            if len(surfaces)!=len(self.surface_ids) or any(a is not b for a,b in zip(surfaces,self.surface_ids)):
                raise ValueError('surfaces changed within one private build')
            hulls=self.hulls
        return hulls

    def classify(self,camera,surfaces,masks,*,diagnostics=False,tolerance_m=.0025,uncertainty_band_m=.006):
        h,w=camera.depth_raw_mm.shape;union=np.asarray(masks,bool).any(0)
        pixels=np.flatnonzero(union);n=len(pixels)
        if not n:
            extra=dict(predicted_robot_depth=np.empty(0),link_owner=np.empty(0,np.int32),
                inside_known_link_volume=np.empty(0,bool),candidate=np.empty(0,bool)) if diagnostics else None
            self.last_metrics.append(dict(camera=camera.camera,mask_pixels=0,image_pixels=h*w,link_pixels=[],bbox_pixels=[],selected_intermediate_bytes=0))
            return SparseDecision((h,w),pixels,np.empty(0,bool),np.empty(0,bool),extra)
        if self.variant=='strict' and h*w<=128*128:
            # Stage 9 already has a separately grouped arithmetic layout here.
            # Keep that implementation, including its exact owner ties.
            from gpu_graph import OptimizedGraph
            import robot
            out=robot.classify(OptimizedGraph('cuda_fused_v5'),camera,surfaces,
                tolerance_m=tolerance_m,uncertainty_band_m=uncertainty_band_m)
            extra={k:out[k].ravel()[pixels] for k in ('predicted_robot_depth','link_owner','inside_known_link_volume','candidate')} if diagnostics else None
            self.last_metrics.append(dict(camera=camera.camera,mask_pixels=n,image_pixels=h*w,dispatch='Stage9 small full-screen fallback'))
            return SparseDecision((h,w),pixels,out['confirmed_robot'].ravel()[pixels],out['uncertain'].ravel()[pixels],extra)
        hulls=self.hull_values(surfaces);bounds=[projected_bounds(camera,v) for v,_ in hulls]
        v,u=np.divmod(pixels,w)
        if self.variant in ('atomic','strict'):
            # The union IDs are sorted. Restrict the host scan vertically first;
            # this changes neither projected bounds nor the ordered row set.
            rows=[]
            for x0,y0,x1,y1 in bounds:
                if x1<x0 or y1<y0:rows.append(np.empty(0,np.int64));continue
                lo=int(np.searchsorted(pixels,y0*w+x0))
                hi=int(np.searchsorted(pixels,y1*w+x1,side='right'))
                rows.append(lo+np.flatnonzero((u[lo:hi]>=x0)&(u[lo:hi]<=x1)))
        else:
            rows=[np.flatnonzero((u>=x0)&(u<=x1)&(v>=y0)&(v<=y1)) for x0,y0,x1,y1 in bounds]
        depth=gpu(camera.depth.ravel()[pixels]);world=gpu(camera.world_from_camera.astype(np.float64))
        if self.variant in ('roi','hulls','strict'):
            # First candidate isolates ROI clipping from full-screen ray work.
            vv,uu=torch.meshgrid(torch.arange(h,device='cuda',dtype=torch.float64),torch.arange(w,device='cuda',dtype=torch.float64),indexing='ij')
            k=gpu(np.asarray(camera.intrinsic,np.float64))
            rays=torch.stack(((uu+.5-k[0,2])/k[0,0],(vv+.5-k[1,2])/k[1,1],torch.ones_like(uu)),2).reshape(-1,3)
            direction=(rays@world[:3,:3].T)[gpu(pixels)]
        else:
            if self.variant=='rays':
                x,y=self.rays.get(camera);uv=gpu(np.stack((u,v),1))
                rays=torch.stack((x[uv[:,0]],y[uv[:,1]],torch.ones(n,device='cuda',dtype=torch.float64)),1)
            else:
                # Identical subtract/divide in float64, no altered world planes.
                k=np.asarray(camera.intrinsic,np.float64)
                rays=gpu(np.column_stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],np.ones(n))))
            direction=rays@world[:3,:3].T
        z=torch.full((n,),torch.inf,device='cuda',dtype=torch.float64)
        owner=torch.full((n,),-1,device='cuda',dtype=torch.int32)
        interior=torch.zeros(n,device='cuda',dtype=torch.int32 if self.variant in ('atomic','strict') else torch.bool)
        if self.variant in ('packed','rays','atomic','strict'):
            starts=np.r_[0,np.cumsum([len(eq) for _,eq in hulls])]
            if len(hulls):
                equations=gpu(np.concatenate([eq for _,eq in hulls]))
                normals=equations[:,:3].contiguous()
                if self.variant=='strict':
                    numerator=torch.cat([-(equations[a:b,:3]@world[:3,3]+equations[a:b,3]) for a,b in zip(starts[:-1],starts[1:])])
                else:numerator=-(normals@world[:3,3]+equations[:,3])
            row_starts=np.r_[0,np.cumsum([len(r) for r in rows])]
            all_rows=gpu(np.concatenate(rows)) if rows else gpu(np.empty(0,np.int64))
        if self.variant in ('atomic','strict'):
            groups={}
            for i,(_,eq) in enumerate(hulls):
                if len(rows[i]):groups.setdefault(triton.next_power_of_2(len(eq)),[]).append(i)
            entries=torch.empty(int(row_starts[-1]) if diagnostics else 0,device='cuda',dtype=torch.float64)
            if diagnostics:owner.fill_(np.iinfo(np.int32).max)
            for f,ids in groups.items():
                meta=gpu(np.array([[starts[i],len(hulls[i][1]),row_starts[i],len(rows[i])] for i in ids],np.int64))
                grid=(triton.cdiv(max(len(rows[i]) for i in ids),32),len(ids))
                _atomic_links[grid](direction,depth,normals,numerator,all_rows,meta,z,interior,entries,f,diagnostics,enable_fp_fusion=False)
                if diagnostics:
                    # Run owner resolution after ALL depth updates (below).
                    groups[f]=(ids,meta)
            if diagnostics:
                for f,(ids,meta) in groups.items():
                    _atomic_owners[(triton.cdiv(max(len(rows[i]) for i in ids),256),len(ids))](all_rows,meta,gpu(np.array(ids,np.int32)),entries,z,owner)
                owner=torch.where(owner==np.iinfo(np.int32).max,-1,owner)
        for i,((_,eq),rr) in enumerate(zip(hulls,rows)):
            if self.variant in ('atomic','strict'):break
            if not len(rr):continue
            if self.variant in ('packed','rays','atomic','strict'):
                a,b=starts[i:i+2];normal=normals[a:b];num=numerator[a:b]
                r=all_rows[row_starts[i]:row_starts[i+1]]
            else:
                equations=gpu(eq);normal=equations[:,:3].contiguous()
                num=-(equations[:,:3]@world[:3,3]+equations[:,3]);r=gpu(rr)
            _clip_robot[(triton.cdiv(len(rr),32),)](direction,r,normal,num,depth,z,owner,interior,
                len(rr),len(eq),i,triton.next_power_of_2(len(eq)),enable_fp_fusion=False)
        flags=torch.empty((2,n),device='cuda',dtype=torch.bool)
        _flags[(triton.cdiv(n,256),)](depth,z,interior,flags,n,tolerance_m,uncertainty_band_m,enable_fp_fusion=False)
        host=flags.cpu().numpy()
        extra=None
        if diagnostics:
            zz=z.cpu().numpy();inside=interior.cpu().numpy().astype(bool)
            extra=dict(predicted_robot_depth=zz,link_owner=owner.cpu().numpy(),
                inside_known_link_volume=inside,candidate=np.isfinite(camera.depth.ravel()[pixels])&~host[0]&~host[1])
        self.last_metrics.append(dict(camera=camera.camera,mask_pixels=n,image_pixels=h*w,
            link_pixels=[len(r) for r in rows],bbox_pixels=[max(0,x1-x0+1)*max(0,y1-y0+1) for x0,y0,x1,y1 in bounds],
            selected_intermediate_bytes=int(direction.numel()*8+n*(8+8+4+(4 if self.variant in ('atomic','strict') else 1)+2)+sum(len(r)*8 for r in rows)),
            array_shapes=dict(rays=list(rays.shape),direction=list(direction.shape),depth=[n],nearest=[n],owner=[n],interior=[n],flags=[2,n],link_rows=[len(r) for r in rows]),
            actual_pixel_plane_pairs=sum(len(r)*len(eq) for r,(_,eq) in zip(rows,hulls)),
            padded_pixel_plane_pairs=sum(len(r)*triton.next_power_of_2(len(eq)) for r,(_,eq) in zip(rows,hulls)),
            full_bbox_pixel_plane_pairs=sum(max(0,x1-x0+1)*max(0,y1-y0+1)*len(eq) for (x0,y0,x1,y1),(_,eq) in zip(bounds,hulls)),
            decision_readback_bytes=2*n,full_diagnostic_readback_bytes=16*h*w,
            ray_cache_bytes=self.rays.bytes))
        return SparseDecision((h,w),pixels,host[0],host[1],extra)


@triton.jit
def _atomic_links(D,Depth,Normals,Num,Rows,Meta,Z,Inside,Entries,
                  F:tl.constexpr,DIAGNOSTIC:tl.constexpr,P:tl.constexpr=32):
    link=tl.program_id(1);i=tl.program_id(0)*P+tl.arange(0,P)
    start=tl.load(Meta+4*link);faces=tl.load(Meta+4*link+1)
    rs=tl.load(Meta+4*link+2);count=tl.load(Meta+4*link+3)
    if tl.program_id(0)*P<count:
        row=tl.load(Rows+rs+i,i<count,0)
        dx=tl.load(D+3*row,i<count,0);dy=tl.load(D+3*row+1,i<count,0);dz=tl.load(D+3*row+2,i<count,0)
        f=tl.arange(0,F);face=start+f;valid=f<faces
        nx=tl.load(Normals+3*face,valid,0);ny=tl.load(Normals+3*face+1,valid,0);nz=tl.load(Normals+3*face+2,valid,0)
        num=tl.load(Num+face,valid,0)
        denom=(dx[:,None]*nx[None,:]+dy[:,None]*ny[None,:])+dz[:,None]*nz[None,:]
        parallel=tl.abs(denom)<1e-12;crossing=tl.where(parallel,0,num[None,:]/denom)
        lower=tl.max(tl.where(valid[None,:]&(denom < -1e-12),crossing,-float('inf')),1)
        upper=tl.min(tl.where(valid[None,:]&(denom > 1e-12),crossing,float('inf')),1)
        outside=tl.sum((valid[None,:]&parallel&(num[None,:]<0)).to(tl.int32),1)>0
        entry=tl.maximum(lower,0);hit=(upper>=entry)&(upper>0)&~outside&(i<count)
        measured=tl.load(Depth+row,i<count,float('nan'))
        interior=hit&(measured==measured)&(measured>entry+.0005)&(measured<upper-.0005)
        tl.atomic_min(Z+row,entry,hit,sem='relaxed')
        tl.atomic_or(Inside+row,interior.to(tl.int32),i<count,sem='relaxed')
        if DIAGNOSTIC:tl.store(Entries+rs+i,tl.where(hit,entry,float('inf')),i<count)


@triton.jit
def _atomic_owners(Rows,Meta,LinkIds,Entries,Z,Owner,P:tl.constexpr=256):
    link=tl.program_id(1);i=tl.program_id(0)*P+tl.arange(0,P)
    rs=tl.load(Meta+4*link+2);count=tl.load(Meta+4*link+3)
    if tl.program_id(0)*P<count:
        row=tl.load(Rows+rs+i,i<count,0);entry=tl.load(Entries+rs+i,i<count,float('inf'))
        z=tl.load(Z+row,i<count,float('inf'));original=tl.load(LinkIds+link)
        tl.atomic_min(Owner+row,original,(i<count)&(entry==z)&(z<float('inf')),sem='relaxed')
