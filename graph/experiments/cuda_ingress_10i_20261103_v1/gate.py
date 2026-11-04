"""Current-camera gate on resident depth/masks, preserving existing rules."""
from itertools import product
import numpy as np
import torch
import triton
from graph.experiments.resolution_perception_graph_20261103_v1.gpu_graph import _radius_kernel

_EXISTS=None


def components(x,radius):
    global _EXISTS
    n=len(x)
    if not n:return torch.empty(0,device=x.device,dtype=torch.int64),torch.empty(0,device=x.device,dtype=torch.bool)
    cell=torch.floor((x-x.amin(0))/(radius/2)).long();dims=(cell.amax(0)+1).tolist()
    if np.prod(dims,dtype=object)>=2**62:raise ValueError('cell key overflow')
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
        if _EXISTS is None:_EXISTS=_radius_kernel()
        keep=torch.empty(len(a),device=x.device,dtype=torch.bool)
        _EXISTS[(len(a),)](sx,starts,a,b,keep,radius*radius,enable_fp_fusion=False);a=a[keep];b=b[keep]
    labels=torch.arange(nc,device=x.device)
    for _ in range(nc):
        old=labels.clone();roots=torch.minimum(labels[a],labels[b]);labels.scatter_reduce_(0,a,roots,reduce='amin');labels.scatter_reduce_(0,b,roots,reduce='amin');labels=labels[labels]
        if torch.equal(old,labels):break
    else:raise ValueError('component convergence')
    roots=labels[inverse];first=torch.full((nc,),n,device=x.device,dtype=torch.int64)
    first.scatter_reduce_(0,roots,torch.arange(n,device=x.device),reduce='amin')
    return torch.unique(first[roots],sorted=True,return_inverse=True)[1],clique


def consistent(world,oid,camera,frames,masks,*,coordinate_frame='world'):
    from visual.runtime.coordinate_frames import check_frame
    check_frame(coordinate_frame)
    keep=torch.ones(len(world),device=world.device,dtype=torch.bool)
    for name,t in frames.items():
        if name==camera or not len(world):continue
        inverse=t.inverse if coordinate_frame=='world' else t.inverse@t.world_ee
        xyz=world@inverse[:3,:3].T+inverse[:3,3];k=t.intrinsic;h,w=t.depth64.shape
        uv=torch.floor(xyz[:,:2]/xyz[:,2,None].clamp_min(1e-12)*torch.stack((k[0,0],k[1,1]))+k[:2,2]).long()
        inside=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        rows=torch.where(inside)[0];u,v=uv[rows].unbind(1)
        agree=(t.depth64[v,u]-xyz[rows,2]).abs()<.004;other=masks[name][1-oid,v,u]&~masks[name][oid,v,u]
        keep[rows[agree&other]]=False
    return keep


def wrist_gate(frame,raw,clean,cfg,wrist='hand_camera'):
    t=frame.tensors[wrist];c=t.camera;depth=t.depth64;support=clean[wrist][0]&torch.isfinite(depth)
    pix=torch.where(support.flatten())[0];xyz=t.world_points(pix);n=len(pix)
    labels,clique=components(xyz,cfg['component_radius_m'])
    counts=torch.bincount(labels) if n else labels;largest=labels==counts.argmax() if n else support.new_empty(0)
    local=clique.clone()
    if np.sqrt(3)*cfg['component_radius_m']/2>=cfg['local_neighbor_radius_m']:local[:]=False
    rare=torch.where(~local)[0]
    for start in range(0,len(rare),128):
        rows=rare[start:start+128];delta=xyz[rows,None]-xyz[None]
        d=torch.sqrt((delta[:,:,0]**2+delta[:,:,1]**2)+delta[:,:,2]**2)
        local[rows]=(d<=cfg['local_neighbor_radius_m']).sum(1)>=3
    allowed=consistent(xyz,0,wrist,frame.tensors,clean)
    # Only trace scalars leave the device; full raw/clean images and XYZ do not.
    summary=torch.stack((raw[wrist][0].sum(),(raw[wrist][0]&torch.isfinite(depth)).sum(),largest.sum(),local.sum(),(~allowed).sum())).tolist()
    raw_n,valid_n,large_n,local_n,conflicts=map(int,summary);fraction=lambda a,b:float(a/b) if b else 0.
    diameter=0.
    if n and large_n:
        extent=(xyz[largest].amax(0)-xyz[largest].amin(0)).cpu().numpy()
        diameter=float(np.linalg.norm(extent))
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
