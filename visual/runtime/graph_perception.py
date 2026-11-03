"""Graph default: compact CUDA projection followed by triton_fused PCA5.

Both stages retain compact XYZ on CUDA. The explicit CPU handoff after feature
calculation adapts to existing Shape-First/packet consumers without rebuilding
dense XYZ. This module does not perform robot filtering or tracking.
"""
from time import perf_counter
import numpy as np

from visual.runtime.geometry import (CompactOrganizedGeometry, GeometryFrame,
    GeometryParameters, GeometricFeatures)
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import SamplingEvidence

DEFAULT_BACKEND = 'triton_fused'


def prepare_cuda(depth, masks, intrinsic):
    from visual.runtime.torch_projection import project
    from visual.runtime.test.pca_triton import compact_rows
    if depth.device.type != 'cuda':
        raise ValueError('Graph triton_fused backend requires CUDA tensors')
    result=project(depth,masks,intrinsic)
    result['rows']=compact_rows(result['pixels'],depth.shape)
    return result


def features_cuda(prepared, depth, masks, parameters=GeometryParameters()):
    from visual.runtime.test.pca_triton import fused_normals_compact,fused_features_compact
    p=prepared
    normal,valid=fused_normals_compact(p['points'],p['pixels'],p['object_id'],p['rows'],parameters)
    return fused_features_compact(p['points'],p['pixels'],p['object_id'],p['rows'],
        depth,masks,normal,valid,parameters)


def process_cuda(depth, masks, intrinsic, parameters=GeometryParameters()):
    prepared=prepare_cuda(depth,masks,intrinsic)
    return prepared,features_cuda(prepared,depth,masks,parameters)


def _mask_boundary(mask):
    result=np.zeros_like(mask)
    for a,b in (((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))):
        result[a]|=mask[a]&~mask[b];result[b]|=mask[b]&~mask[a]
    return result


def to_cpu_evidence(frame, prepared, features, parameters=GeometryParameters()):
    """Compatibility boundary for CPU selection, after both GPU stages."""
    from visual.runtime.edge_evidence import pair_evidence
    points=prepared['points'].cpu().numpy()
    pixels=prepared['pixels'].cpu().numpy()
    ids=prepared['object_id'].cpu().numpy()
    values={name:t.cpu().numpy() for name,t in features.items()}
    valid=np.isfinite(frame.depth)&(frame.depth>0)
    masks={i:np.asarray(frame.masks.get(i,np.zeros_like(valid)),bool).copy() for i in (0,1)}
    order=np.argsort(pixels,kind='stable')
    geometry=CompactOrganizedGeometry(None,valid,masks,masks[0]&masks[1],
        pixels[order],points[order],np.where(valid,frame.depth,np.nan))
    objects={};pairs={}
    for oid in (0,1):
        take=ids==oid;source=pixels[take]
        arrays=dict(object_mask=masks[oid],valid_points=geometry.observed_mask(oid))
        for name,value in values.items():
            shape=(*valid.shape,3) if name=='normals' else valid.shape
            a=np.full(shape,np.nan if value.dtype.kind=='f' else 0,dtype=value.dtype)
            a.reshape((-1,3) if name=='normals' else (-1,))[source]=value[take]
            arrays[name]=a
        # Raw-mask boundaries can include invalid/overlap pixels absent from
        # compact support. Preserve these old diagnostic fields as well.
        arrays['mask_boundary']=_mask_boundary(masks[oid])
        edge=np.zeros_like(valid);edge[[0,-1],:]=masks[oid][[0,-1],:]
        edge[:,[0,-1]]=masks[oid][:,[0,-1]];arrays['image_boundary']=edge
        objects[oid]=GeometricFeatures(**arrays)
        pairs[oid]=pair_evidence(geometry,oid,arrays['normals'],arrays['normal_valid'],parameters)
    gf=GeometryFrame(frame.source_frame,frame.camera,frame.frame_idx,frame.timestamp,geometry,objects,{})
    return SamplingEvidence(gf,pairs,'pca5',{})


def extract(frame, estimator='pca5', parameters=GeometryParameters()):
    import torch
    if estimator!='pca5':raise ValueError('Graph CUDA path requires PCA5')
    if not torch.cuda.is_available():
        raise RuntimeError("Graph default requires CUDA; request perception_backend='cpu' explicitly for CPU diagnostics")
    depth=torch.as_tensor(np.ascontiguousarray(frame.depth),device='cuda')
    masks=torch.as_tensor(np.stack([frame.masks[i] for i in (0,1)]),device='cuda')
    k=torch.as_tensor(np.ascontiguousarray(frame.intrinsic),device='cuda')
    prepared,values=process_cuda(depth,masks,k,parameters)
    return to_cpu_evidence(frame,prepared,values,parameters)


def make_packet(camera, masks, *, backend=DEFAULT_BACKEND):
    """Existing Shape-First/F5 packet semantics with the new Graph default."""
    from graph.integration_baseline import vision
    if backend not in ('cpu','triton_fused'):raise ValueError('unknown Graph perception backend')
    frame=TrackedRGBDFrame(camera.timestep,camera.camera,camera.rgb,camera.depth.astype(np.float32),
        camera.intrinsic,camera.world_from_camera,camera.tcp_pose,
        {oid:np.asarray(masks[oid],bool) for oid in (0,1)},camera.timestep,camera.timestamp)
    tick=perf_counter()
    evidence=(extract(frame) if backend=='triton_fused' else vision.extract_sampling_evidence(frame,estimator='pca5'))
    after_evidence=perf_counter()
    candidates=vision.extract_candidates(evidence,vision.SamplingConfig(),method='S3')
    prepared={};selected={};cases={};g=evidence.frame.geometry
    for oid in (0,1):
        obj=candidates[oid];orientation=evidence.pairs[oid].candidates(20)['orientation']
        pca=obj.flat_index[orientation.ravel()[obj.flat_index]]
        prepared[oid]=vision.shape_first.prepare_object(obj.flat_index,obj.points,pca,frame.masks[oid],
            g.valid_depth,g.overlap_mask,frame.intrinsic)
        selected[oid]=vision.shape_first.select_object(prepared[oid],.5,0,vision.shape_first.SURFACE[oid])
        cases[oid]=dict(saved_flat=obj.flat_index,saved_features=obj.metadata)
    selection,_=vision.shape_first.make_packet(frame.source_frame,prepared,selected,cases)
    selection.update(episode=camera.episode,camera=frame.camera)
    packet=vision.complete_vision_packet(selection,evidence,frame)
    vision.receive_vision_packet(packet)
    return packet,dict(pca5_evidence_ms=(after_evidence-tick)*1000,
        total_offline_graph_ms=(perf_counter()-tick)*1000,
        projection_backend='torch_cuda_masked' if backend=='triton_fused' else 'numpy_full',
        geometry_backend=backend,compact_xyz=backend=='triton_fused')
