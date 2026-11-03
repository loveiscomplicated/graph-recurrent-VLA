"""Tensor form of the existing selected v2 packet; no selection policy here.

Large arrays remain on their producer device. Small sensor identity/transform
metadata stays explicit. numpy_packet is an opt-in, single packed readback for
legacy consumers, never an implicit part of CUDA packet construction.
"""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from .vision_packet import FEATURE_FIELDS, DIAGNOSTIC_DTYPES, NUMERIC_CONTRACT, camera_to_ee,validate_transform,pose_matrix
from .object_groups import object_slices,object_slices_match
from .coordinate_frames import check_frame, camera_to_frame, packet_frame, feature_fields, FRAME_NAMES

_NUMPY={torch.float32:np.dtype('float32'),torch.float64:np.dtype('float64'),
    torch.int64:np.dtype('int64'),torch.int32:np.dtype('int32'),
    torch.uint8:np.dtype('uint8'),torch.bool:np.dtype('bool')}


def map_tree(value,fn):
    if isinstance(value,torch.Tensor):return fn(value)
    if isinstance(value,dict):return {k:map_tree(v,fn) for k,v in value.items()}
    if isinstance(value,list):return [map_tree(v,fn) for v in value]
    if isinstance(value,tuple):return tuple(map_tree(v,fn) for v in value)
    return deepcopy(value)


def numpy_packet(packet):
    """One packed array readback, preserving every leaf's dtype/shape/value."""
    leaves=[]
    def remember(t):leaves.append(t);return len(leaves)-1
    map_tree(packet,remember)
    if not leaves:return deepcopy(packet)
    device=leaves[0].device
    if any(t.device!=device for t in leaves):raise ValueError('mixed packet devices')
    offsets=np.r_[0,np.cumsum([t.numel()*t.element_size() for t in leaves])]
    packed=torch.cat([t.contiguous().reshape(-1).view(torch.uint8) for t in leaves])
    host=packed.detach().cpu().numpy()
    arrays=[host[a:b].view(_NUMPY[t.dtype]).reshape(tuple(t.shape)).copy()
        for t,a,b in zip(leaves,offsets[:-1],offsets[1:])]
    # Layout integers are ambiguous with scalar metadata, so rebuild using the
    # original tree and a leaf iterator rather than interpreting scalar values.
    it=iter(arrays)
    return map_tree(packet,lambda _:next(it))


@torch.inference_mode()
def selected_packet(camera,result,depth,masks,intrinsic,*,transform_cuda=None,coordinate_frame='ee'):
    check_frame(coordinate_frame)
    objects=result['selected'];meta=result['metadata'];device=depth.device
    source=torch.cat([objects[i]['F'] for i in (0,1)])
    xyz=torch.cat([objects[i]['xyz'] for i in (0,1)])
    oid=torch.cat([torch.full_like(objects[i]['F'],i) for i in (0,1)])
    surface=torch.cat([torch.arange(len(objects[i]['F']),device=device)>=len(objects[i]['P']) for i in (0,1)])
    n=len(source)
    frame=SimpleNamespace(tcp_pose=camera.tcp_pose,cam2world=camera.world_from_camera)
    transform_np=camera_to_frame(frame,coordinate_frame)
    if transform_cuda is None:transform_cuda=torch.as_tensor(transform_np,device=device)
    transform=transform_cuda.to(torch.float64)
    values={f['name']:meta[f['name']].clone() for f in FEATURE_FIELDS}
    diagnostics={k:meta[k].clone() for k in DIAGNOSTIC_DTYPES}
    nv=diagnostics['normal_valid']
    rotated=(values['normals'].to(torch.float64)@transform[:3,:3].T).to(torch.float32)
    values['normals']=torch.where(nv[:,None],rotated,values['normals'])
    validity=torch.stack([diagnostics[k] for k in ('normal_valid','normal_variation_valid','depth_pair_valid')],1)
    available=torch.ones((n,3),device=device,dtype=torch.bool)
    fields=[dict(name=name,shape=list(a.shape[1:]),dtype=_NUMPY[a.dtype].str) for name,a in sorted(meta.items())]
    artifact_available=torch.ones((n,len(fields)),device=device,dtype=torch.bool)
    provenance=dict(source_index=source.clone(),object_id=oid.clone(),candidate_evidence=result['candidate_evidence'].clone(),
        both_evidence=result['candidate_evidence'].all(1),selected_by_structural_source=result['selected_by_structural_source'].clone(),
        selection_stage=surface.to(torch.uint8),surface_selected=surface,
        feature_available=artifact_available.all(1),feature_field_available=artifact_available)
    packet=dict(schema_version=2,feature_mode='continuous',episode=camera.episode,camera=camera.camera,
        source_frame=int(camera.timestep),frame_idx=int(camera.timestep),timestamp=camera.timestamp,coordinate_frame=FRAME_NAMES[coordinate_frame],
        points=(xyz.to(torch.float64)@transform[:3,:3].T+transform[:3,3]).to(torch.float32),
        camera_points=xyz.clone(),object_id=oid,object_slices=object_slices(len(objects[0]['F']),len(objects[1]['F'])),
        geometric_features=torch.cat([values['normals'],values['normal_variation_rad'][:,None],values['depth_jump_m'][:,None]],1).to(torch.float64),
        feature_fields=feature_fields(coordinate_frame),compact_features={f['name']:dict(positions=torch.arange(n,device=device),values=values[f['name']].clone()) for f in FEATURE_FIELDS},
        feature_available=available.all(1),feature_field_available=available,feature_valid=validity.all(1),feature_field_valid=validity,
        feature_diagnostics=diagnostics,provenance=provenance,source_artifact_feature_fields=fields,
        T_frame_from_camera=transform_np,observation_ee_pose=np.asarray(camera.tcp_pose,np.float64).copy(),
        camera_to_world=np.asarray(camera.world_from_camera,np.float64).copy(),camera_intrinsic=np.asarray(camera.intrinsic,np.float64).copy(),
        image_shape=np.asarray(depth.shape,np.int64),valid=result['valid'].clone(),
        edge_status='not_implemented',intra_object_edge_index=torch.empty((2,0),device=device,dtype=torch.int64))
    if coordinate_frame=='ee':packet['T_EE_from_camera']=transform_np.copy()
    # One aggregate device decision, never a selected-array host copy.
    v,u=source//depth.shape[1],source%depth.shape[1];z=depth.flatten()[source].to(torch.float64);k=intrinsic.to(torch.float64)
    original=torch.stack(((u.to(torch.float64)+.5-k[0,2])*z/k[0,0],(v.to(torch.float64)+.5-k[1,2])*z/k[1,1],z),1).to(torch.float32)
    good=(torch.isfinite(z)&(z>0)&masks[oid,v,u]&~masks[1-oid,v,u]).all()&(original==xyz).all()
    if not bool(good):raise ValueError('selected tensor source/mask/depth mismatch')
    validate_packet(packet)
    return packet


def validate_packet(packet):
    """v2 measured-feature tensor contract, including source and invalidity."""
    p=packet;xyz=p['points'];device=xyz.device;n=len(xyz)
    frame=packet_frame(p)
    if p.get('schema_version')!=2 or p.get('feature_mode')!='continuous':raise ValueError('tensor packet requires continuous v2')
    for k in ('episode','camera'):
        if not isinstance(p.get(k),str) or not p[k]:raise ValueError('packet identity')
    if type(p.get('source_frame')) is not int or p['source_frame']<0:raise ValueError('source frame')
    def check(a,dtype,shape):
        if not isinstance(a,torch.Tensor) or a.device!=device or a.dtype!=dtype or tuple(a.shape)!=shape:raise ValueError('tensor packet shape/dtype/device')
    check(xyz,torch.float32,(n,3));check(p['object_id'],torch.int64,(n,));q=p['provenance'];source=q['source_index']
    check(source,torch.int64,(n,));check(p['camera_points'],torch.float32,(n,3))
    check(p['geometric_features'],torch.float64,(n,5));check(p['feature_field_valid'],torch.bool,(n,3));check(p['feature_field_available'],torch.bool,(n,3))
    check(p['intra_object_edge_index'],torch.int64,(2,0));check(p['valid'],torch.bool,(2,))
    for name in ('feature_available','feature_valid'):check(p[name],torch.bool,(n,))
    for name in ('candidate_evidence','selected_by_structural_source'):check(q[name],torch.bool,(n,2))
    for name in ('both_evidence','surface_selected'):check(q[name],torch.bool,(n,))
    check(q['object_id'],torch.int64,(n,));check(q['selection_stage'],torch.uint8,(n,))
    if p['edge_status']!='not_implemented' or p['feature_fields']!=feature_fields(frame):raise ValueError('tensor packet feature/edge contract')
    image=np.asarray(p['image_shape'])
    if image.dtype!=np.int64 or image.shape!=(2,) or (image<=0).any():raise ValueError('source image shape')
    oid=p['object_id'];valid=p['feature_field_valid'];fa=p['feature_field_available'];features=p['geometric_features']
    # This adapter represents measured selected rows only. Partial-availability
    # legacy packets must use the existing NumPy receiver instead.
    if not bool(fa.all()):raise ValueError('tensor selected packet requires measured feature availability')
    good=torch.isfinite(xyz).all()&(source>=0).all()&(source<int(image.prod())).all()&((oid==0)|(oid==1)).all()
    good&=object_slices_match(p)
    if n:good&=torch.unique(oid*(int(image.prod())+1)+source).numel()==n
    good&=(p['feature_available']==fa.all(1)).all()&(p['feature_valid']==valid.all(1)).all()&~(valid&~fa).any()
    offset=0
    for f in FEATURE_FIELDS:
        name=f['name'];a=p['compact_features'][name]['values'];dtype=torch.float32
        check(a,dtype,(n,*f['shape']));j=[x['name'] for x in FEATURE_FIELDS].index(name)
        vv=valid[:,j].reshape((n,)+(1,)*len(f['shape'])).expand_as(a)
        good&=torch.where(vv,torch.isfinite(a),torch.isnan(a)).all()
        positions=p['compact_features'][name]['positions'];check(positions,torch.int64,(n,))
        good&=(positions==torch.arange(n,device=device)).all()
        width=3 if name=='normals' else 1;view=features[:,offset:offset+width];aa=a.reshape(n,width).to(torch.float64)
        good&=((view==aa)|(torch.isnan(view)&torch.isnan(aa))).all();offset+=width
    diag=p['feature_diagnostics']
    for name,dtype in DIAGNOSTIC_DTYPES.items():check(diag[name],torch.uint8 if np.dtype(dtype)==np.uint8 else torch.bool,(n,))
    good&=diag['valid_points'].all()
    for j,name in enumerate(('normal_valid','normal_variation_valid','depth_pair_valid')):good&=(valid[:,j]==diag[name]).all()
    good&=(diag['normal_variation_valid']==(diag['normal_valid']&(diag['normal_neighbor_count']>0))).all()&(diag['normal_neighbor_count']<=4).all()
    normal_norm=torch.linalg.vector_norm(torch.nan_to_num(features[:,:3]),dim=1)
    good&=(~valid[:,0]|((normal_norm-1).abs()<=NUMERIC_CONTRACT['normal_norm_atol'])).all()
    good&=(~valid[:,1]|((features[:,3]>=0)&(features[:,3]<=np.pi+1e-6))).all()
    good&=(~valid[:,2]|(features[:,4]>=0)).all()
    good&=p['valid'][oid].all()&torch.isfinite(p['camera_points']).all()&(p['camera_points'][:,2]>0).all()
    good&=(q['object_id']==oid).all()&(q['both_evidence']==q['candidate_evidence'].all(1)).all()
    good&=~(q['selected_by_structural_source']&~q['candidate_evidence']).any()
    good&=(q['surface_selected']==(q['selection_stage']==1)).all()
    good&=((q['selection_stage']==0)==q['selected_by_structural_source'].any(1)).all()
    good&=((q['selection_stage']==0)|(q['selection_stage']==1)).all()
    artifact=p['source_artifact_feature_fields'];check(q['feature_field_available'],torch.bool,(n,len(artifact)));check(q['feature_available'],torch.bool,(n,))
    good&=(q['feature_available']==q['feature_field_available'].all(1)).all()
    # Small immutable calibration metadata was already on CPU at ingress. Keep
    # its original rigid-transform/pose checks without downloading any arrays.
    def array(name,shape):
        a=np.asarray(p[name])
        if a.shape!=shape or a.dtype!=np.float64:raise ValueError('calibration shape/dtype')
        return a
    key='T_frame_from_camera' if 'T_frame_from_camera' in p else 'T_EE_from_camera'
    t=validate_transform(array(key,(4,4)));world=validate_transform(array('camera_to_world',(4,4)))
    pose=pose_matrix(array('observation_ee_pose',(7,)))
    expected=np.linalg.inv(pose)@world if frame=='ee' else world
    eps=NUMERIC_CONTRACT['reference_eps_factor']*np.finfo(np.float64).eps
    if not np.allclose(t,expected,rtol=eps,atol=eps):raise ValueError('source transform mismatch')
    if 'T_EE_from_camera' in p and (frame!='ee' or not np.array_equal(t,p['T_EE_from_camera'])):raise ValueError('EE transform alias/frame mismatch')
    k=array('camera_intrinsic',(3,3))
    if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or k[0,1]!=0 or k[1,0]!=0 or not np.array_equal(k[2],[0,0,1]):raise ValueError('camera intrinsic')
    tt=torch.as_tensor(t,device=device);reference=p['camera_points'].to(torch.float64)@tt[:3,:3].T+tt[:3,3]
    af=reference.abs().to(torch.float32);ulp=(torch.nextafter(af,torch.full_like(af,torch.inf))-af).to(torch.float64)
    tolerance=torch.maximum(4*ulp,torch.full_like(ulp,NUMERIC_CONTRACT['float32_point_abs_floor_m']))
    good&=((xyz.to(torch.float64)-reference).abs()<=tolerance).all()
    if not bool(good):raise ValueError('tensor packet validity/provenance mismatch')


def receive_packet(packet,*,robot_positions=None):
    from graph.vision_input import VisionInput
    validate_packet(packet);saved=map_tree(packet,lambda t:t.clone());x=saved['points'];device=x.device;n=len(x)
    offset=0 if robot_positions is None else 3
    if offset:
        if robot_positions.shape!=(3,3) or robot_positions.dtype!=torch.float32 or robot_positions.device!=device:raise ValueError('robot tensor alignment')
        if not bool(torch.isfinite(robot_positions).all()):raise ValueError('nonfinite robot tensor')
        x=torch.cat((robot_positions,x))
    kinds=torch.cat((torch.arange(1,offset+1,device=device,dtype=torch.uint8),torch.zeros(n,device=device,dtype=torch.uint8)))
    return VisionInput(saved,x,torch.arange(n,device=device)+offset,kinds,
        torch.cat((torch.full((offset,),-1,device=device,dtype=torch.int64),saved['provenance']['source_index'])),
        torch.cat((torch.zeros((offset,2),device=device,dtype=torch.bool),saved['provenance']['candidate_evidence'])),
        kinds!=0,saved['intra_object_edge_index'].clone()+offset)
