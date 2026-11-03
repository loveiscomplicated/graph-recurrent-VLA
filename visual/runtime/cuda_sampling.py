"""GPU Shape-First: evidence, canonical outer contours, voxels and FP64 FPS.

No NumPy/OpenCV operation or geometry download occurs before selection. Python
controls launches and reads convergence/cardinality scalars; all pixel/point
decisions and their arrays remain CUDA tensors. sample_cuda also returns CUDA
tensors by default; CPU output requires device='cpu'. Historical NumPy
implementations and the explicit CPU packet adapter remain reference APIs.
"""
import math
import torch
import torch.nn.functional as functional
import triton

from visual.runtime.geometry import GeometryParameters
from visual.runtime import cuda_sampling_kernels as kernels


def external_contour_sources(mask, observed):
    """RETR_EXTERNAL/CHAIN_APPROX_NONE candidate order without CPU contours."""
    padded=functional.pad(mask,(1,1,1,1),value=False).contiguous()
    h,w=padded.shape;n=padded.numel();device=mask.device
    parent=torch.arange(n,dtype=torch.int64,device=device)
    roots=torch.empty_like(parent);changed=torch.zeros((),dtype=torch.int32,device=device)
    for _ in range(n):
        kernels.compress[(triton.cdiv(n,256),)](parent,roots,n,256)
        parent.copy_(roots);changed.zero_()
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
    kernels.canonical[(len(starts),)](walk,offsets,lengths,canonical,num_warps=1)
    positions=torch.nonzero(canonical>=0).flatten();source=canonical[positions]
    good=observed.view(-1)[source];source=source[good];positions=positions[good]
    first=torch.full((mask.numel(),),size,dtype=torch.int64,device=device)
    first.scatter_reduce_(0,source,positions,reduce='amin',include_self=True)
    return source[positions==first[source]]


def voxel_representatives(source,points,a):
    """Source-ascending centroid-nearest original point; first-source ties."""
    if not len(source):return source.clone()
    order=torch.argsort(source,stable=True)
    asc=source[order];xyz=points[order].to(torch.float64)
    keys=torch.floor(xyz/a).to(torch.int64)
    permutation=torch.arange(len(asc),device=source.device)
    for axis in (2,1,0):
        permutation=permutation[torch.argsort(keys[permutation,axis],stable=True)]
    sorted_keys=keys[permutation]
    first=torch.ones(len(asc),dtype=torch.bool,device=source.device)
    first[1:]=(sorted_keys[1:]!=sorted_keys[:-1]).any(1)
    group=first.cumsum(0)-1
    starts=torch.nonzero(first).flatten()
    lengths=torch.diff(torch.cat((starts,starts.new_tensor([len(asc)]))))
    ordered=xyz[permutation]
    centroid=torch.segment_reduce(ordered,'mean',lengths=lengths)
    delta=ordered-centroid[group]
    distance=(delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2]
    minimum=torch.full((len(starts),),float('inf'),dtype=torch.float64,device=source.device)
    minimum.scatter_reduce_(0,group,distance,reduce='amin',include_self=True)
    rows=torch.full((len(starts),),len(asc),dtype=torch.int64,device=source.device)
    rows.scatter_reduce_(0,group,torch.where(distance==minimum[group],permutation,len(asc)),reduce='amin',include_self=True)
    representatives=asc[rows]
    # Restore the input order (ascending for PCA, canonical walk for silhouette).
    return source[torch.isin(source,representatives)]


def surface_fps(points,initial,additional):
    """Original deterministic source-row order and FP64 squared distances."""
    n=len(points);requested=min(additional,n-len(initial))
    count=len(initial)+requested
    if not count:return initial.clone()
    if not len(initial):initial=torch.zeros(1,dtype=torch.int64,device=points.device)
    extra=count-len(initial)
    if not extra:return initial.clone()
    distance=torch.empty(n,dtype=torch.float64,device=points.device)
    used=torch.zeros(n,dtype=torch.bool,device=points.device);used[initial]=True
    groups=triton.cdiv(n,128)
    best_d=torch.empty(groups,dtype=torch.float64,device=points.device)
    best_i=torch.empty(groups,dtype=torch.int32,device=points.device)
    output=torch.empty(extra,dtype=torch.int64,device=points.device)
    kernels.fps_seed[(groups,)](points,initial,used,distance,best_d,best_i,n,len(initial),128,32,
        num_warps=8,enable_fp_fusion=False)
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
            z=torch.sort(xyz[:,2].to(torch.float64)).values
            median=(z[(len(z)-1)//2]+z[len(z)//2])*.5
            a=median*torch.hypot(2/intrinsic[0,0].to(torch.float64),2/intrinsic[1,1].to(torch.float64))/math.sqrt(3)
            reps={name:voxel_representatives(s,xyz[torch.searchsorted(x,s)],a) for name,s in c.items()}
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
    return dict(prepared=prep,selected=selected,rows=gather,
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
