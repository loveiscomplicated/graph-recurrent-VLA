"""Current-only CUDA graph from final FPS tensors; explicit archival readback.

Point identity stays encoded as camera/pixel tensors in packet metadata. Call
numpy_graph only at a CPU inspection or file-writing boundary. The normal graph
path never materializes per-point Python dictionaries or strings.
"""
from copy import deepcopy
import numpy as np
import torch

from graph.integration_baseline.graph import Graph
from graph.integration_baseline.robot import geometry_in_ee
from visual.runtime.vision_packet import pose_matrix
from visual.runtime.coordinate_frames import FRAME_NAMES
from . import SCHEMA_ID, EDGE_POLICY
from .graph import EXTRA_FIELDS
from .tensor_records import TensorRecords, _identity_ranks
from .tensor_knn import neighbor_membership

FIELDS=('node_input','node_type','edge_index','edge_attr','edge_type')


def _dot(a,b):
    return (a[...,0]*b[...,0]+a[...,1]*b[...,1])+a[...,2]*b[...,2]


def _norm(a):
    return torch.sqrt(_dot(a,a))


def _both(pairs):
    return torch.cat((pairs,pairs.flip(1)),0).T.contiguous()


@torch.inference_mode()
def free_space_tensor(samples,views,tolerance=.006):
    """Batched sparse depth gathers, retaining strict six-mm comparison."""
    bad=torch.zeros(samples.shape[:-1],device=samples.device,dtype=torch.bool)
    for camera in views.by_name.values():
        m=camera.camera_from_frame
        # Explicit double operations avoid float32/TF32 point transforms.
        p=(samples[...,0,None]*m[:3,0]+samples[...,1,None]*m[:3,1])+samples[...,2,None]*m[:3,2]+m[:3,3]
        uv=torch.floor(p[...,:2]/p[...,2,None].clamp_min(1e-12)*camera.intrinsic.diag()[:2]+camera.intrinsic[:2,2])
        h,w=camera.depth.shape
        inside=(p[...,2]>0)&(uv[...,0]>=0)&(uv[...,0]<w)&(uv[...,1]>=0)&(uv[...,1]<h)
        u=uv[...,0].clamp(0,w-1).to(torch.int64);v=uv[...,1].clamp(0,h-1).to(torch.int64)
        z=camera.depth[v,u]
        bad |= inside & torch.isfinite(z) & (p[...,2]<z-tolerance)
    return bad


@torch.inference_mode()
def surface_edges_tensor(xyz,normals,normal_valid,objects,views):
    pairs=[]
    for oid in (0,1):
        rows=torch.nonzero(objects==oid).flatten()
        if len(rows)<2: continue
        near=neighbor_membership(xyz[rows].contiguous())
        local=torch.nonzero(torch.triu(near|near.T,diagonal=1))
        pairs.append(rows[local])
    device=xyz.device
    pairs=torch.cat(pairs) if pairs else torch.empty((0,2),device=device,dtype=torch.int64)
    a,b=pairs.T; delta=xyz[b]-xyz[a]; length=_norm(delta)
    candidate=(length<=.02)&(length>=1e-9)
    normal_bad=normal_valid[a]&normal_valid[b]&((_dot(normals[a],normals[b])<.5)|
        (torch.maximum(_dot(normals[a],delta).abs(),_dot(normals[b],delta).abs())>.002+length*.25))
    normal_bad &= candidate
    survive=candidate&~normal_bad
    # Compact before witness/projection work; all three witnesses in one batch.
    rows=torch.nonzero(survive).flatten(); chosen=pairs[rows]
    samples=xyz[chosen[:,0],None,:]+torch.tensor([.25,.5,.75],device=device,dtype=torch.float64)[None,:,None]*delta[rows,None,:]
    distance=_norm(samples[:,:,None,:]-xyz[None,None,:,:])
    distance=torch.where(objects[None,None,:]==objects[chosen[:,0],None,None],distance,torch.inf)
    witness_bad=(distance.amin(-1)>.004).any(-1) if len(xyz) else torch.zeros(len(chosen),device=device,dtype=torch.bool)
    free_bad=free_space_tensor(samples,views).any(-1)&~witness_bad
    kept=chosen[~witness_bad&~free_bad]
    # CPU sorts unordered pairs globally, then all forward/all reverse edges.
    if len(kept): kept=kept[torch.argsort(kept[:,0]*max(len(xyz),1)+kept[:,1],stable=True)]
    audit=dict(candidates=candidate.sum(),rejected_normal=normal_bad.sum(),
        rejected_witness=witness_bad.sum(),rejected_free_space=free_bad.sum())
    return _both(kept),audit


@torch.inference_mode()
def build_graph_tensor(records,robot_world,tcp_pose,timestep,*,views):
    """Final <=128/object FPS -> CUDA Graph21/Edge4, with current-only data."""
    if not isinstance(records,TensorRecords) or records.device.type!='cuda':
        raise ValueError('resident CUDA TensorRecords required')
    frame=records.coordinate_frame
    if (records.episode,records.timestep)!=(views.identity[0],timestep) or timestep!=views.identity[1]:
        raise ValueError('Graph/record/view identity mismatch')
    if views.coordinate_frame!=frame or views.device!=records.device or tuple(tcp_pose)!=views.identity[3]:
        raise ValueError('Graph/view frame/device/EE reference mismatch')
    if frame=='ee' and records.reference_ee_pose!=tuple(tcp_pose):
        raise ValueError('Graph/record EE reference mismatch')
    if any(int((records.object_id==oid).sum())>128 for oid in (0,1)) or bool(((records.object_id<0)|(records.object_id>1)).any()):
        raise ValueError('final Graph requires <=128 points/object')
    order=torch.argsort(_identity_ranks(records),stable=True)
    order=order[torch.argsort(records.object_id[order],stable=True)]
    xyz=records.xyz[order]; normals=records.normals[order]; normal_valid=records.normal_valid[order]
    objects=records.object_id[order]; valid=records.feature_valid[order]
    # Three host FK positions are small calibration metadata, exactly matching
    # the existing validated FK convention; all selected-point math is CUDA.
    if frame=='world':
        robots=np.asarray(robot_world)
        if robots.shape!=(3,3) or not np.isfinite(robots).all() or not np.allclose(robots[0],pose_matrix(tcp_pose)[:3,3],rtol=0,atol=1e-7):
            raise ValueError('same-time World robot geometry mismatch')
        robots=robots.astype(np.float32)
    else: robots=geometry_in_ee(robot_world,tcp_pose)
    device=records.device;n=len(records)
    positions=torch.cat((torch.as_tensor(robots,device=device),xyz.to(torch.float32)))
    inputs=torch.zeros((n+3,21),device=device,dtype=torch.float32);inputs[:,:3]=positions
    f5=records.f5[order]
    inputs[3:,3:8]=torch.where(valid[:,[0,0,0,1,2]],f5,0).to(torch.float32)
    inputs[3:,8:11]=1;inputs[3:,11:14]=valid
    intra,audit=surface_edges_tensor(xyz,normals,normal_valid,objects,views)
    robot_pairs=[]
    for robot in range(3):
        for oid in (0,1):
            rows=torch.nonzero(objects==oid).flatten()
            delta=positions[rows+3].to(torch.float64)-positions[robot].to(torch.float64)
            chosen=rows[torch.argsort(_dot(delta,delta),stable=True)[:4]]
            robot_pairs.append(torch.stack((torch.full_like(chosen,robot),chosen+3),1))
    ro=_both(torch.cat(robot_pairs))
    peg,box=(torch.nonzero(objects==oid).flatten() for oid in (0,1))
    points=positions[3:].to(torch.float64)
    d=points[peg,None,:]-points[box][None,:,:]
    chosen=torch.argsort(_dot(d,d).flatten(),stable=True)[:16]
    if len(box): pb=torch.stack((peg[chosen//len(box)],box[chosen%len(box)]),1)+3
    else: pb=torch.empty((0,2),device=device,dtype=torch.int64)
    pb=_both(pb)
    edges=torch.cat((intra+3,ro,pb),1)
    types=torch.cat([torch.full((e.shape[1],),i,device=device,dtype=torch.int64) for i,e in enumerate((intra,ro,pb))])
    delta=positions[edges[1]].to(torch.float64)-positions[edges[0]].to(torch.float64);distance=_norm(delta)
    direction=torch.where(distance[:,None]>0,delta/distance[:,None],0)
    attrs=torch.cat((distance[:,None],direction),1).to(torch.float32)
    metadata=dict(schema_id=SCHEMA_ID,edge_policy=EDGE_POLICY,feature_fields=list(EXTRA_FIELDS),
        node_ids_encoded=dict(episode=records.episode,timestep=timestep,camera_names=records.camera_names,
            camera_id=records.camera_id[order],pixels=records.pixels[order]),
        surface_edge_audit=audit,timestep=timestep,point_cap_per_object=None,
        point_budget_policy='stable_identity_seeded_farthest_v1',status_features=False,
        coordinate_frame=FRAME_NAMES[frame],normal_frame=frame,graph_backend='torch_cuda')
    if frame=='ee':metadata['reference_ee_pose']=[float(v) for v in records.reference_ee_pose]
    return Graph(inputs,torch.cat((torch.arange(3,device=device),objects+3)),edges,attrs,types,metadata,{})


def numpy_graph(graph):
    """Explicit device -> CPU Graph for archival and legacy inspection only."""
    if 'node_ids_encoded' not in graph.packet and graph.node_input.device.type=='cpu':return graph
    from visual.runtime.tensor_packet import numpy_packet
    packed={name:getattr(graph,name) for name in FIELDS}
    packet={k:deepcopy(v) for k,v in graph.packet.items() if k not in ('node_ids_encoded','surface_edge_audit')}
    ids=graph.packet.get('node_ids_encoded')
    if ids is not None:
        packed.update(id_camera=ids['camera_id'],id_pixel=ids['pixels'])
    for k,v in graph.packet['surface_edge_audit'].items():
        packed['audit_'+k]=v if isinstance(v,torch.Tensor) else torch.tensor(v,device=graph.node_input.device)
    arrays=numpy_packet(packed)
    if ids is not None:
        packet['node_ids']=[f"{ids['episode']}:current{ids['timestep']}:{ids['camera_names'][int(c)]}:{int(p)}" for c,p in zip(arrays['id_camera'],arrays['id_pixel'])]
    packet['surface_edge_audit']={k:int(arrays['audit_'+k]) for k in graph.packet['surface_edge_audit']}
    return Graph(*(torch.from_numpy(arrays[name]) for name in FIELDS),packet,deepcopy(graph.timings_ms))
