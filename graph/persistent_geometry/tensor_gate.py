"""Exact radius components and current-camera gate on resident CUDA images."""
from itertools import product
from dataclasses import replace
import math
import torch
from .tensor_fusion import prepare_tensor_views,consistency_keep
from . import tensor_gate_kernels as kernels

def components(x,radius):
    n=len(x)
    if not n:return torch.empty(0,device=x.device,dtype=torch.int64),torch.empty(0,device=x.device,dtype=torch.bool)
    cell=torch.floor((x-x.amin(0))/(radius/2)).long();dims=(cell.amax(0)+1).tolist()
    if math.prod(dims)>=2**62:raise ValueError('cell key overflow')
    stride=torch.tensor([dims[1]*dims[2],dims[2],1],device=x.device);code=(cell*stride).sum(1)
    keys,inverse,counts=torch.unique(code,sorted=True,return_inverse=True,return_counts=True);nc=len(keys)
    clique=counts[inverse]>=3
    if nc==1:return torch.zeros(n,device=x.device,dtype=torch.int64),clique
    order=torch.argsort(code,stable=True);sx=x[order].contiguous();starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    coords=torch.stack((keys//stride[0],keys//stride[1]%dims[1],keys%dims[2]),1)
    offsets=torch.tensor([d for d in product(range(-3,4),repeat=3) if d>(0,0,0)],device=x.device)
    neighbor=coords[:,None]+offsets;inside=((neighbor>=0)&(neighbor<torch.tensor(dims,device=x.device))).all(2)
    target=(neighbor*stride).sum(2);j=torch.searchsorted(keys,target).clamp_max(nc-1);valid=inside&(keys[j]==target)
    a=torch.arange(nc,device=x.device)[:,None].expand_as(j)[valid];b=j[valid]
    low=torch.full((nc,3),torch.inf,device=x.device,dtype=x.dtype);high=torch.full_like(low,-torch.inf)
    index=inverse[:,None].expand(-1,3);low.scatter_reduce_(0,index,x,reduce='amin');high.scatter_reduce_(0,index,x,reduce='amax')
    gap=torch.maximum(low[a]-high[b],low[b]-high[a]).clamp_min_(0);possible=(gap[:,0]**2+gap[:,1]**2)+gap[:,2]**2<=radius*radius
    a=a[possible].contiguous();b=b[possible].contiguous()
    if len(a):
        keep=torch.empty(len(a),device=x.device,dtype=torch.bool)
        kernels.radius_exists[(len(a),)](sx,starts,a,b,keep,radius*radius,enable_fp_fusion=False);a=a[keep];b=b[keep]
    labels=torch.arange(nc,device=x.device)
    for _ in range(nc):
        old=labels.clone();roots=torch.minimum(labels[a],labels[b]);labels.scatter_reduce_(0,a,roots,reduce='amin');labels.scatter_reduce_(0,b,roots,reduce='amin');labels=labels[labels]
        if torch.equal(old,labels):break
    else:raise ValueError('component convergence')
    roots=labels[inverse];first=torch.full((nc,),n,device=x.device,dtype=torch.int64)
    first.scatter_reduce_(0,roots,torch.arange(n,device=x.device),reduce='amin')
    return torch.unique(first[roots],sorted=True,return_inverse=True)[1],clique



def masked_world_points(camera,view,pixels):
    z=view.depth.flatten()[pixels];w=view.depth.shape[1];k=view.intrinsic
    u=(pixels%w).to(torch.float64);v=(pixels//w).to(torch.float64)
    xyz=torch.stack(((u+.5-k[0,2])*z/k[0,0],(v+.5-k[1,2])*z/k[1,1],z),1)
    m=torch.as_tensor(camera.world_from_camera.astype('float64'),device=z.device)
    return (xyz[:,0,None]*m[:3,0]+xyz[:,1,None]*m[:3,1])+xyz[:,2,None]*m[:3,2]+m[:3,3]


@torch.inference_mode()
def wrist_gate_tensor(cameras,raw,clean,cfg,*,views=None,wrist='hand_camera'):
    views=prepare_tensor_views(cameras,clean,coordinate_frame='world',
        depths=None if views is None else {name:t.depth for name,t in views.by_name.items()})
    c=next(c for c in cameras if c.camera==wrist);t=views.by_name[wrist];depth=t.depth
    support=clean[wrist][0]&torch.isfinite(depth);pix=torch.nonzero(support.flatten()).flatten()
    xyz=masked_world_points(c,t,pix);n=len(pix)
    labels,clique=components(xyz,cfg['component_radius_m'])
    counts=torch.bincount(labels) if n else labels;largest=labels==counts.argmax() if n else support.new_empty(0)
    local=clique.clone()
    if math.sqrt(3)*cfg['component_radius_m']/2>=cfg['local_neighbor_radius_m']:local[:]=False
    rare=torch.nonzero(~local).flatten()
    for start in range(0,len(rare),128):
        rows=rare[start:start+128];d=xyz[rows,None]-xyz[None]
        distance=torch.sqrt((d[:,:,0]*d[:,:,0]+d[:,:,1]*d[:,:,1])+d[:,:,2]*d[:,:,2])
        local[rows]=(distance<=cfg['local_neighbor_radius_m']).sum(1)>=3
    allowed=consistency_keep(xyz,0,wrist,views)
    summary=torch.stack((raw[wrist][0].sum(),(raw[wrist][0]&torch.isfinite(depth)).sum(),largest.sum(),local.sum(),(~allowed).sum())).tolist()
    raw_n,valid_n,large_n,local_n,conflicts=map(int,summary);fraction=lambda a,b:float(a/b) if b else 0.
    diameter=0.
    if large_n:
        extent=xyz[largest].amax(0)-xyz[largest].amin(0)
        diameter=float(torch.sqrt((extent[0]*extent[0]+extent[1]*extent[1])+extent[2]*extent[2]))
    trace=dict(camera=wrist,episode=c.episode,timestep=c.timestep,raw_pixels=raw_n,valid_depth=valid_n,clean_points=n,
        valid_depth_fraction=fraction(valid_n,raw_n),clean_fraction=fraction(n,valid_n),largest_component_fraction=fraction(large_n,n),
        local_support_fraction=fraction(local_n,n),component_diameter_m=diameter,tracker_confidence_available=False,main_required=False,
        identity_conflict_fraction=fraction(conflicts,n))
    reasons=[]
    for reject,name in ((n<cfg['minimum_valid_points'],'insufficient_clean_depth'),
        (trace['valid_depth_fraction']<cfg['minimum_valid_depth_fraction'],'invalid_depth_support'),
        (trace['clean_fraction']<cfg['minimum_clean_fraction'],'robot_or_overlap_dominated'),
        (trace['largest_component_fraction']<cfg['minimum_component_fraction'],'fragmented_3d_support'),
        (trace['local_support_fraction']<cfg['minimum_local_support_fraction'],'isolated_surface'),
        (diameter>cfg['maximum_component_diameter_m'],'implausible_extent'),
        (trace['identity_conflict_fraction']>cfg['maximum_identity_conflict_fraction'],'cross_camera_identity_conflict')):
        if reject:reasons.append(name)
    accepted=torch.zeros_like(support)
    if not reasons:accepted.flatten()[pix[largest&allowed]]=True
    trace.update(accepted=not reasons,reasons=reasons,accepted_points=int(accepted.sum()),
        association=[dict(camera=wrist,object_id=0,count=n,unknown_conflict=conflicts)],GT_used=False,future_used=False)
    return accepted,trace


@torch.inference_mode()
def prepare_tensor_gate(cameras,data,raw,clean,protocol,*,coordinate_frame,depths=None):
    from .contracts import synchronized
    from visual.runtime.coordinate_frames import camera_from_frame,camera_to_frame
    from types import SimpleNamespace
    synchronized(cameras)
    names={c.camera for c in cameras}
    if names!={'base_camera','hand_camera'} or set(raw)!=names or set(clean)!=names:raise ValueError('camera identity')
    if str(data['episode'])!=cameras[0].episode or int(data['timestep'])!=cameras[0].timestep:raise ValueError('sensor identity join')
    allowed={'robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world','timestep','timestamp','episode'}
    allowed.update(c.camera+'/'+k for c in cameras for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
    if set(data)-allowed:raise ValueError('forbidden non-sensor input')
    raw={name:torch.as_tensor(value,device='cuda') for name,value in raw.items()}
    clean={name:torch.as_tensor(value,device=raw[name].device) for name,value in clean.items()}
    for c in cameras:
        r=raw[c.camera];v=clean[c.camera]
        if r.dtype!=torch.bool or v.dtype!=torch.bool or r.shape!=(2,*c.depth_raw_mm.shape) or v.shape!=r.shape:raise ValueError('bool aligned raw/clean masks required')
        if bool((v&~r).any()|v[:,r[0]&r[1]].any()):raise ValueError('clean mask subset/overlap')
    views=prepare_tensor_views(cameras,clean,coordinate_frame=coordinate_frame,depths=depths)
    accepted,gate=wrist_gate_tensor(cameras,raw,clean,protocol['gate'],views=views)
    cm={name:value.clone() for name,value in clean.items()};cm['hand_camera'][1]=False;cm['hand_camera'][0]=accepted
    by_name={name:replace(value,mask=cm[name]) for name,value in views.by_name.items()}
    return dict(conditions={'C3':cm},gate=gate,wrist_gate_backend='torch_cuda'),replace(views,by_name=by_name)
