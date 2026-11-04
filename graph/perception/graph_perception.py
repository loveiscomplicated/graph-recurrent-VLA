"""Graph default: compact CUDA projection followed by triton_fused PCA5.

Compact XYZ/features remain CUDA-resident through GPU Shape-First. Only selected
rows cross the CPU packet boundary. The old evidence adapter remains available
for diagnostics. This module does not perform robot filtering or tracking.
"""
from time import perf_counter
import numpy as np

from graph.perception.geometry import (CompactOrganizedGeometry,
    GeometryFrame,
    GeometryParameters,
    GeometricFeatures)
from vla.perception.rgbd import TrackedRGBDFrame
from graph.perception.sampling import SamplingEvidence
from graph.perception.object_groups import object_slices
from graph.perception.coordinate_frames import check_frame

DEFAULT_BACKEND = 'triton_fused'


def make_packet_cuda(camera,masks,*,depth_cuda=None,intrinsic_cuda=None,transform_cuda=None,coordinate_frame='ee'):
    """Existing PCA5/Shape-First semantics through a CUDA v2 packet return."""
    check_frame(coordinate_frame)
    import torch
    from graph.perception.cuda_sampling import sample_cuda
    from graph.perception.tensor_packet import selected_packet
    if not isinstance(masks,torch.Tensor) or not masks.is_cuda:raise ValueError('CUDA masks required')
    device=masks.device
    depth=(torch.as_tensor(camera.depth.astype(np.float32),device=device) if depth_cuda is None else depth_cuda.to(torch.float32))
    k=torch.as_tensor(camera.intrinsic,device=device) if intrinsic_cuda is None else intrinsic_cuda
    projected,values=process_cuda(depth,masks,k)
    result=sample_cuda(projected,values,depth,masks,k)
    return selected_packet(camera,result,depth,masks,k,transform_cuda=transform_cuda,coordinate_frame=coordinate_frame)


def prepare_cuda(depth, masks, intrinsic):
    from graph.perception.torch_projection import project
    from graph.perception.pca_triton import compact_rows
    if depth.device.type != 'cuda':
        raise ValueError('Graph triton_fused backend requires CUDA tensors')
    result=project(depth,masks,intrinsic)
    result['rows']=compact_rows(result['pixels'],depth.shape)
    return result


def features_cuda(prepared, depth, masks, parameters=GeometryParameters()):
    from graph.perception.pca_triton import fused_normals_compact, fused_features_compact
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
    from graph.perception.edge_evidence import pair_evidence
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
    """Historical CPU-evidence adapter; Graph default uses GPU selection below."""
    prepared,values=_process_frame(frame,estimator,parameters)
    return to_cpu_evidence(frame,prepared,values,parameters)


def _process_frame(frame,estimator='pca5',parameters=GeometryParameters()):
    import torch
    if estimator!='pca5':raise ValueError('Graph CUDA path requires PCA5')
    if not torch.cuda.is_available():
        raise RuntimeError("Graph default requires CUDA; request perception_backend='cpu' explicitly for CPU diagnostics")
    depth=torch.as_tensor(np.ascontiguousarray(frame.depth),device='cuda')
    masks=torch.as_tensor(np.stack([frame.masks[i] for i in (0,1)]),device='cuda')
    k=torch.as_tensor(np.ascontiguousarray(frame.intrinsic),device='cuda')
    return process_cuda(depth,masks,k,parameters)


def _selection_packet(frame,episode,result):
    """CPU packet packaging from selected rows only; selection already finished."""
    objects=result['selected'];metadata=result['metadata']
    source=np.concatenate([objects[i]['F'] for i in (0,1)])
    xyz=np.concatenate([objects[i]['xyz'] for i in (0,1)])
    oid=np.concatenate([np.full(len(objects[i]['F']),i,np.int64) for i in (0,1)])
    surface=np.concatenate([np.arange(len(objects[i]['F']))>=len(objects[i]['P']) for i in (0,1)])
    fields=[dict(name=name,shape=list(a.shape[1:]),dtype=a.dtype.str) for name,a in sorted(metadata.items())]
    n=len(source)
    feature_view=np.concatenate([metadata[f['name']].reshape(n,int(np.prod(f['shape'])) if f['shape'] else 1)
        for f in fields],axis=1).astype(np.float64)
    available=np.ones((n,len(fields)),bool)
    evidence=result['candidate_evidence']
    return dict(episode=episode,camera=frame.camera,source_frame=int(frame.source_frame),
        coordinate_frame='CV_camera_metres',points=xyz,object_id=oid,
        object_slices=object_slices(len(objects[0]['F']),len(objects[1]['F'])),
        feature_fields=fields,geometric_features=feature_view,
        compact_features={name:dict(positions=np.arange(n,dtype=np.int64),values=a) for name,a in metadata.items()},
        edge_status='not_implemented',intra_object_edge_index=np.empty((2,0),np.int64),
        provenance=dict(source_index=source,object_id=oid.copy(),candidate_evidence=evidence,
            both_evidence=evidence.all(1),selected_by_structural_source=result['selected_by_structural_source'],
            selection_stage=surface.astype(np.uint8),surface_selected=surface,
            feature_available=available.all(1),feature_field_available=available))


def make_packet(camera, masks, *, backend=DEFAULT_BACKEND, coordinate_frame='ee'):
    """Existing Shape-First/F5 packet semantics with the new Graph default."""
    import graph.perception.offline as vision
    check_frame(coordinate_frame)
    if backend not in ('cpu','triton_fused'):raise ValueError('unknown Graph perception backend')
    frame=TrackedRGBDFrame(camera.timestep,camera.camera,camera.rgb,camera.depth.astype(np.float32),
        camera.intrinsic,camera.world_from_camera,camera.tcp_pose,
        {oid:np.asarray(masks[oid],bool) for oid in (0,1)},camera.timestep,camera.timestamp)
    tick=perf_counter()
    if backend=='triton_fused':
        import torch
        from graph.perception.cuda_sampling import sample_cuda, numpy_selection
        from graph.perception.vision_packet import complete_selected_vision_packet
        if not torch.cuda.is_available():raise RuntimeError('Graph default requires CUDA')
        depth=torch.as_tensor(np.ascontiguousarray(frame.depth),device='cuda')
        tm=torch.as_tensor(np.stack([frame.masks[i] for i in (0,1)]),device='cuda')
        k=torch.as_tensor(np.ascontiguousarray(frame.intrinsic),device='cuda')
        projected,values=process_cuda(depth,tm,k)
        after_features=perf_counter()
        # This legacy Graph consumer is explicitly CPU/NumPy. The public
        # sample_cuda API keeps its result on CUDA unless CPU is requested.
        result=numpy_selection(sample_cuda(projected,values,depth,tm,k,device='cpu'))
        selection=_selection_packet(frame,camera.episode,result)
        packet=complete_selected_vision_packet(selection,frame,result['metadata'],result['valid'],coordinate_frame=coordinate_frame)
        vision.receive_vision_packet(packet)
        return packet,dict(pca5_evidence_ms=(after_features-tick)*1000,
            total_offline_graph_ms=(perf_counter()-tick)*1000,
            projection_backend='torch_cuda_masked',geometry_backend=backend,
            sampling_backend='torch_triton_cuda',compact_xyz=True)
    evidence=vision.extract_sampling_evidence(frame,estimator='pca5')
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
    selection['object_slices']=object_slices(len(selected[0]['F']),len(selected[1]['F']))
    selection.update(episode=camera.episode,camera=frame.camera)
    packet=vision.complete_vision_packet(selection,evidence,frame,coordinate_frame=coordinate_frame)
    vision.receive_vision_packet(packet)
    return packet,dict(pca5_evidence_ms=(after_evidence-tick)*1000,
        total_offline_graph_ms=(perf_counter()-tick)*1000,
        projection_backend='torch_cuda_masked' if backend=='triton_fused' else 'numpy_full',
        geometry_backend=backend,compact_xyz=backend=='triton_fused')
