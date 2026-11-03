"""GPU Shape-First: evidence, canonical outer contours, voxels and FP64 FPS.

No NumPy/OpenCV operation or geometry download occurs before selection. Python
controls launches and reads convergence/cardinality scalars; all pixel/point
decisions and their arrays remain CUDA tensors. sample_cuda also returns CUDA
tensors by default; CPU output requires device='cpu'. Historical NumPy
implementations and the explicit CPU packet adapter remain reference APIs.
"""
import math
from collections import OrderedDict
import torch
import torch.nn.functional as functional
import triton

from visual.runtime.geometry import GeometryParameters
from visual.runtime import cuda_sampling_kernels as kernels

_FPS_PLANS=OrderedDict()
_FPS_PLAN_LIMIT=8


class _FPSPlan:
    """Shape-bucketed, stream-local CUDA graph; current data is copied per call."""
    def __init__(self,n,seeds,extra,points):
        self.points=torch.zeros((n,3),dtype=points.dtype,device=points.device)
        self.initial=torch.full((seeds,),-1,dtype=torch.int64,device=points.device)
        self.used=torch.ones(n,dtype=torch.bool,device=points.device)
        self.bounds=torch.empty((triton.cdiv(seeds,32),6),dtype=points.dtype,device=points.device)
        self.distance=torch.empty(n,dtype=torch.float64,device=points.device)
        self.groups=triton.cdiv(n,128)
        self.best_d=torch.empty(self.groups,dtype=torch.float64,device=points.device)
        self.best_i=torch.empty(self.groups,dtype=torch.int32,device=points.device)
        self.output=torch.empty(extra,dtype=torch.int64,device=points.device)
        def work():
            kernels.seed_tile_bounds[(triton.cdiv(seeds,32),)](self.points,self.initial,self.bounds,seeds,32,num_warps=1)
            kernels.fps_seed_bounded[(self.groups,)](self.points,self.initial,self.used,self.bounds,self.distance,
                self.best_d,self.best_i,self.used,n,seeds,128,32,num_warps=4,enable_fp_fusion=False)
            for k in range(extra):
                kernels.fps_choose[(1,)](self.best_d,self.best_i,self.output,k,self.groups,
                    triton.next_power_of_2(self.groups),num_warps=4)
                if k+1<extra:
                    kernels.fps_update[(self.groups,)](self.points,self.distance,self.output,
                        self.best_d,self.best_i,k,n,128,num_warps=4,enable_fp_fusion=False)
        # Compile and finish warmup outside capture. Captures contain no host
        # cardinality decisions and replay on the caller's stream.
        work();work();torch.cuda.synchronize(points.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):work()

    def run(self,points,initial):
        self.points[:len(points)].copy_(points)
        self.initial.fill_(-1);self.initial[:len(initial)].copy_(initial)
        self.used.fill_(True);self.used[:len(points)].fill_(False);self.used[initial]=True
        self.graph.replay()
        # Materialize the result before a later replay overwrites work buffers.
        return torch.cat((initial,self.output))


def external_contour_sources(mask, observed):
    """RETR_EXTERNAL/CHAIN_APPROX_NONE candidate order without CPU contours."""
    mask=mask.contiguous();h,w=mask.shape
    bounds=torch.tensor([h,0,w,0],dtype=torch.int32,device=mask.device)
    kernels.mask_bounds[(triton.cdiv(mask.numel(),256),)](mask,bounds,h,w,256)
    # Only four shape scalars cross the host boundary; pixels stay on CUDA.
    v0,v1,u0,u1=bounds.tolist()
    if v1<=v0 or u1<=u0:return torch.empty(0,dtype=torch.int64,device=mask.device)
    source=_external_contour_sources(mask[v0:v1,u0:u1],observed[v0:v1,u0:u1].contiguous())
    return (source//(u1-u0)+v0)*w+source%(u1-u0)+u0


def _external_contour_sources(mask, observed):
    """Local raster walk. One background pad preserves image-edge topology."""
    padded=functional.pad(mask,(1,1,1,1),value=False).contiguous()
    h,w=padded.shape;n=padded.numel();device=mask.device
    parent=torch.arange(n,dtype=torch.int64,device=device)
    roots=torch.empty_like(parent);changed=torch.zeros((),dtype=torch.int32,device=device)
    for _ in range((n+1)//2):
        # Compress also resets the next union's flag; only every second
        # completed round needs a host convergence check.
        for _round in range(2):
            kernels.compress[(triton.cdiv(n,256),)](parent,roots,changed,n,256)
            kernels.hook[(triton.cdiv(n,256),)](padded,roots,parent,changed,h,w,256)
        if not bool(changed):break
    else:raise RuntimeError('CUDA contour components did not converge')
    index=torch.arange(n,device=device)
    # A foreground component's minimum pixel has background on its left.
    # Padded exterior background has root 0; nested islands are excluded.
    starts=torch.nonzero(padded.view(-1)&(roots==index)&(roots[(index-1).clamp_min(0)]==0)).flatten()
    if not len(starts):return torch.empty(0,dtype=torch.int64,device=device)
    counts=torch.bincount(roots[padded.view(-1)],minlength=n)[starts]
    offsets=torch.cat((torch.zeros(1,dtype=torch.int64,device=device),(counts*8).cumsum(0)))
    size=int(offsets[-1])
    walk=torch.empty(size,dtype=torch.int64,device=device)
    canonical=torch.full_like(walk,-1);lengths=torch.empty_like(starts)
    error=torch.zeros((),dtype=torch.int32,device=device)
    kernels.trace[(len(starts),)](padded,starts,offsets,walk,lengths,error,h,w,num_warps=1)
    if bool(error):raise RuntimeError('CUDA contour walk exceeded its component reservation')
    kernels.canonical[(len(starts),)](walk,offsets,lengths,canonical,256,num_warps=4)
    positions=torch.nonzero(canonical>=0).flatten();source=canonical[positions]
    good=observed.view(-1)[source];source=source[good];positions=positions[good]
    first=torch.full((mask.numel(),),size,dtype=torch.int64,device=device)
    first.scatter_reduce_(0,source,positions,reduce='amin',include_self=True)
    return source[positions==first[source]]


def voxel_representatives(source,points,a,*,source_sorted=False):
    """Source-ascending centroid-nearest original point; first-source ties."""
    if not len(source):return source.clone()
    order=None if source_sorted else torch.argsort(source,stable=True)
    asc=source if source_sorted else source[order]
    xyz=(points if source_sorted else points[order]).to(torch.float64)
    keys=torch.floor(xyz/a).to(torch.int64)
    permutation=torch.arange(len(asc),device=source.device)
    for axis in (2,1,0):
        permutation=permutation[torch.argsort(keys[permutation,axis],stable=True)]
    sorted_keys=keys[permutation]
    first=torch.ones(len(asc),dtype=torch.bool,device=source.device)
    first[1:]=(sorted_keys[1:]!=sorted_keys[:-1]).any(1)
    starts=torch.nonzero(first).flatten()
    lengths=torch.diff(torch.cat((starts,starts.new_tensor([len(asc)]))))
    ordered=xyz[permutation]
    centroid=torch.segment_reduce(ordered,'mean',lengths=lengths)
    flags=torch.zeros(len(source),dtype=torch.bool,device=source.device)
    kernels.voxel_choices[(len(starts),)](xyz,permutation,starts,lengths,centroid,flags,32,
        num_warps=1,enable_fp_fusion=False)
    if source_sorted:return source[flags]
    # Preserve canonical contour order without sorting/set-membership again.
    in_input_order=torch.empty_like(flags);in_input_order[order]=flags
    return source[in_input_order]


def surface_fps(points,initial,additional):
    """Original deterministic source-row order and FP64 squared distances."""
    n=len(points);requested=min(additional,n-len(initial))
    count=len(initial)+requested
    if not count:return initial.clone()
    if not len(initial):initial=torch.zeros(1,dtype=torch.int64,device=points.device)
    extra=count-len(initial)
    if not extra:return initial.clone()
    if n>=4096 and extra>=16:
        with torch.cuda.device(points.device):
            key=(points.device,points.dtype,torch.cuda.current_stream().cuda_stream,
                triton.next_power_of_2(n),triton.next_power_of_2(len(initial)),extra)
            plan=_FPS_PLANS.pop(key,None)
            if plan is None:plan=_FPSPlan(key[3],key[4],extra,points)
            _FPS_PLANS[key]=plan
            if len(_FPS_PLANS)>_FPS_PLAN_LIMIT:_FPS_PLANS.popitem(last=False)
            return plan.run(points,initial)
    distance=torch.empty(n,dtype=torch.float64,device=points.device)
    used=torch.zeros(n,dtype=torch.bool,device=points.device);used[initial]=True
    groups=triton.cdiv(n,128)
    best_d=torch.empty(groups,dtype=torch.float64,device=points.device)
    best_i=torch.empty(groups,dtype=torch.int32,device=points.device)
    output=torch.empty(extra,dtype=torch.int64,device=points.device)
    kernels.fps_seed_pruned[(groups,)](points,initial,used,distance,best_d,best_i,used,n,len(initial),128,32,
        num_warps=4,enable_fp_fusion=False)
    for k in range(extra):
        kernels.fps_choose[(1,)](best_d,best_i,output,k,groups,triton.next_power_of_2(groups),num_warps=4)
        if k+1<extra:
            kernels.fps_update[(groups,)](points,distance,output,best_d,best_i,k,n,128,
                num_warps=4,enable_fp_fusion=False)
    return torch.cat((initial,output))


def build_sampling_info_cuda(projected,features,depth,masks,intrinsic,parameters=GeometryParameters()):
    """Compact S3 evidence on CUDA, including directed depth-boundary causes."""
    if depth.device.type!='cuda':raise ValueError('CUDA Shape-First requires CUDA inputs')
    n=len(projected['pixels']);device=depth.device
    near=torch.empty(n,dtype=torch.bool,device=device);far=torch.empty_like(near)
    if n:kernels.candidate_sides[(triton.cdiv(n,256),)](projected['pixels'],projected['object_id'],depth,
        near,far,n,*depth.shape,parameters.depth_jump_abs_m,parameters.depth_jump_rel,256,
        enable_fp_fusion=False)
    orientation=torch.rad2deg(features['normal_variation_rad'])>20
    metadata=dict(features)
    metadata.update(object_mask=torch.ones(n,dtype=torch.bool,device=device),
        valid_points=torch.ones(n,dtype=torch.bool,device=device),
        orientation_evidence=orientation,depth_evidence=features['depth_discontinuity'],
        occluding_evidence=near,occluded_evidence=far,
        invalid_evidence=features['depth_validity_boundary'],mask_evidence=features['mask_boundary'],
        orientation_score_deg=torch.rad2deg(features['normal_variation_rad']),
        orientation_candidate=orientation,depth_only_candidate=features['depth_discontinuity']&~orientation,
        structural_candidate=orientation)
    return dict(projected=projected,metadata=metadata,orientation=orientation,
        depth=depth,masks=masks,intrinsic=intrinsic)


def nominal_scale_cuda(xyz,intrinsic):
    """Retain full sort: dual kthvalue regressed on the measured depth input."""
    z=xyz[:,2].to(torch.float64).sort().values;n=len(z)
    median=(z[(n-1)//2]+z[n//2])*.5
    return median*torch.hypot(2/intrinsic[0,0].to(torch.float64),2/intrinsic[1,1].to(torch.float64))/math.sqrt(3)


def shape_first_cuda(info):
    """>20deg candidates, canonical silhouette, .5/phase0, surface 97/95."""
    projected=info['projected'];metadata=info['metadata'];orientation=info['orientation']
    depth=info['depth'];masks=info['masks'];intrinsic=info['intrinsic']
    prep={};selected={};selected_rows=[]
    for oid in (0,1):
        rows=torch.nonzero(projected['object_id']==oid).flatten()
        x=projected['pixels'][rows];xyz=projected['points'][rows]
        pca=x[orientation[rows]]
        support=masks[oid]&torch.isfinite(depth)&(depth>0)&~(masks[0]&masks[1])
        sil=external_contour_sources(masks[oid],support) if len(rows) else x.clone()
        c=dict(pca=pca,sil=sil)
        if len(x):
            a=nominal_scale_cuda(xyz,intrinsic)
            reps={name:voxel_representatives(s,xyz[torch.searchsorted(x,s)],a,source_sorted=name=='pca') for name,s in c.items()}
        else:reps={name:s.clone() for name,s in c.items()}
        thin={name:r[::2] for name,r in reps.items()}
        priority=torch.cat((thin['pca'],thin['sil'][~torch.isin(thin['sil'],thin['pca'])]))
        initial=torch.searchsorted(x,priority)
        chosen=surface_fps(xyz,initial,97 if oid==0 else 95)
        final=x[chosen]
        prep[oid]=dict(C=c,R=reps)
        selected[oid]=dict(F=final,P=priority,S=final[len(priority):],xyz=xyz[chosen],T=thin)
        selected_rows.append(rows[chosen])
    gather=torch.cat(selected_rows)
    source_rows=projected.get('rows')
    if source_rows is None:
        from visual.runtime.test.pca_triton import compact_rows
        source_rows=compact_rows(projected['pixels'],depth.shape)
    return dict(prepared=prep,selected=selected,rows=gather,
        source_rows=source_rows,point_count=len(projected['points']),
        metadata={name:value[gather] for name,value in metadata.items()},
        valid=torch.stack([(projected['object_id']==i).any() for i in (0,1)]))


def select_cuda(projected,features,depth,masks,intrinsic,parameters=GeometryParameters()):
    return shape_first_cuda(build_sampling_info_cuda(projected,features,depth,masks,intrinsic,parameters))


def _map_tensors(tree, function):
    if isinstance(tree,dict):return {key:_map_tensors(value,function) for key,value in tree.items()}
    return function(tree)


def selection_output(result,*,device=None,diagnostic=False):
    """Complete sampling output; default retains the sampler's CUDA device.

    Coordinates, selected features, source IDs and selection provenance share
    the requested device. No NumPy conversion or implicit CPU copy occurs.
    """
    target=result['rows'].device if device is None else torch.device(device)
    if target.type not in ('cpu','cuda'):raise ValueError('sampling output device must be cpu or cuda')
    selected={oid:{name:value for name,value in obj.items() if name!='T'}
        for oid,obj in result['selected'].items()}
    output=dict(selected=selected,metadata=dict(result['metadata']),valid=result['valid'])
    # Keep selection provenance on CUDA as well, even after the last FPS row.
    if 'source_rows' in result:
        flags=torch.zeros(result['point_count'],dtype=torch.int32,device=result['rows'].device)
        for oid in (0,1):
            arrays=[result['prepared'][oid]['C'][name] for name in ('pca','sil')]
            arrays.extend(result['selected'][oid]['T'][name] for name in ('pca','sil'))
            maximum=max(map(len,arrays))
            if maximum:kernels.mark_membership[(triton.cdiv(maximum,128),)](*arrays,result['source_rows'],flags,
                *map(len,arrays),*[a.stride(0) for a in arrays],128)
        n=len(result['rows']);candidate=torch.empty((n,2),dtype=torch.bool,device=flags.device)
        chosen=torch.empty_like(candidate)
        if n:kernels.gather_membership[(triton.cdiv(n,128),)](flags,result['rows'],candidate,chosen,n,128)
        output['candidate_evidence']=candidate;output['selected_by_structural_source']=chosen
    else:
        # Compatibility with raw selection records produced before row maps.
        candidate=[];chosen=[]
        for oid in (0,1):
            final=result['selected'][oid]['F']
            candidate.append(torch.stack([torch.isin(final,result['prepared'][oid]['C'][name]) for name in ('pca','sil')],1))
            chosen.append(torch.stack([torch.isin(final,result['selected'][oid]['T'][name]) for name in ('pca','sil')],1))
        output['candidate_evidence']=torch.cat(candidate)
        output['selected_by_structural_source']=torch.cat(chosen)
    if diagnostic:
        output['prepared']={oid:{kind:dict(obj[kind]) for kind in ('C','R')}
            for oid,obj in result['prepared'].items()}
    return _map_tensors(output,lambda t:t.to(device=target))


def sample_cuda(projected,features,depth,masks,intrinsic,parameters=GeometryParameters(),*,device=None,diagnostic=False):
    """GPU sampling through its final output; device='cpu' opts into download."""
    if device is not None and torch.device(device).type not in ('cpu','cuda'):
        raise ValueError('sampling output device must be cpu or cuda')
    return selection_output(select_cuda(projected,features,depth,masks,intrinsic,parameters),
        device=device,diagnostic=diagnostic)


def numpy_selection(output):
    """Legacy CPU packet adapter; refuses GPU tensors rather than copying them."""
    def convert(t):
        if t.device.type!='cpu':raise ValueError("request device='cpu' before NumPy packet assembly")
        return t.numpy()
    return _map_tensors(output,convert)


def to_cpu_selection(result,*,diagnostic=False):
    """Compatibility API explicitly requesting CPU output for NumPy consumers."""
    return numpy_selection(selection_output(result,device='cpu',diagnostic=diagnostic))
