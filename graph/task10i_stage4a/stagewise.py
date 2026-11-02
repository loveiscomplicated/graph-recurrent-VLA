"""Task-owned stage traces; baseline operations and source pixels are retained."""
from pathlib import Path
import time
import numpy as np
from graph.plan_a_geometry.core import budget
from graph.persistent_geometry.fusion import observations_from_packet,consistent_views
from .instrumentation import support


def records_mask(records,camera,oid,shape):
    out=np.zeros(shape,bool)
    for r in records:
        if r['object_id']!=oid:continue
        for _,_,c,p in r['source_observation_ids']:
            if c==camera:out.ravel()[p]=True
    return out


def trace_geometry(geometry,cameras,data,surfaces,masks,references,object_ids,robot_ids,output):
    """references are evaluation-only and never passed to geometry methods.

    Point-source support counts include attached deduplicated observations;
    representative point purity must be measured separately from source support.
    """
    start=time.perf_counter();prepared=geometry.system.prepare(cameras,data,masks,surfaces)
    filter_gate_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter();pools,merge,_,packets=geometry.system.represent(cameras,prepared,'C3',('SF',))
    native=pools['SF'];selected=budget(native,128)
    sampling_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter();graph=geometry.system.graph(selected,data,cameras)
    graph_ms=1000*(time.perf_counter()-start)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);rows=[];arrays={}
    for camera in cameras:
        c=camera.camera;shape=camera.depth.shape;raw=np.asarray(masks[c],bool);valid=np.isfinite(camera.depth)
        filtered=prepared['filters'][c];overlap=raw[0]&raw[1]
        # Post-packet cross-view association is preserved as an explicit stage.
        packet=packets[c];before=observations_from_packet(packet,camera)
        after,association=consistent_views(before,cameras,prepared['conditions']['C3'])
        for oid in (0,1):
            sampled=np.zeros(shape,bool);sampled.ravel()[before[oid].pixels]=True
            associated=np.zeros(shape,bool);associated.ravel()[after[oid].pixels]=True
            stages=dict(sam2=raw[oid],depth_valid=raw[oid]&valid,
                robot_confirmed_removed=raw[oid]&valid&~filtered['confirmed_robot'],
                robot_uncertain_removed=raw[oid]&valid&~filtered['confirmed_robot']&~filtered['uncertain'],
                overlap_removed=prepared['clean'][c][oid]&valid,
                gate_accepted=prepared['conditions']['C3'][c][oid]&valid,
                shape_first_sampled=sampled,identity_associated=associated,
                deduplicated_sources=records_mask(native,c,oid,shape),final_graph_sources=records_mask(selected,c,oid,shape))
            previous=int(raw[oid].sum())
            for stage,mask in stages.items():
                key=c+'__'+str(oid)+'__'+stage;arrays[key]=np.flatnonzero(mask).astype(np.int32)
                metrics=support(mask,references[c],oid,robot_ids,object_ids) if references.get(c) is not None else None
                rows.append(dict(episode=str(data['episode']),timestep=int(data['timestep']),timestamp=float(data['timestamp']),
                    camera=c,object_id=oid,stage=stage,input_support=previous,output_support=int(mask.sum()),
                    reference_kind='EXACT_GT' if metrics is not None else 'GT_UNSUPPORTED',metrics=metrics,
                    gate=prepared['gate'] if c=='hand_camera' and oid==0 else None,
                    main_only_box_excluded=(c=='hand_camera' and oid==1),association=association[oid] if stage=='identity_associated' else None))
                previous=int(mask.sum())
        arrays[c+'__confirmed_robot']=np.flatnonzero(filtered['confirmed_robot']).astype(np.int32)
        arrays[c+'__uncertain_robot']=np.flatnonzero(filtered['uncertain']).astype(np.int32)
        arrays[c+'__predicted_robot_depth']=filtered['predicted_robot_depth']
        arrays[c+'__robot_link_owner']=filtered['link_owner']
    arrays['point_world']=np.asarray([r['world_xyz'] for r in selected]).reshape(-1,3)
    arrays['point_object_id']=np.asarray([r['object_id'] for r in selected],np.int8)
    arrays['point_f5']=np.asarray([r['last_f5'] for r in selected]).reshape(-1,5)
    arrays['point_valid']=np.asarray([r['last_feature_valid'] for r in selected]).reshape(-1,3)
    for k in ('node_input','node_type','edge_index','edge_attr','edge_type'):arrays[k]=getattr(graph,k).numpy()
    np.savez_compressed(output/'support.npz',**arrays)
    from .run_pipeline import write
    write(output/'trace.json',dict(rows=rows,gate=prepared['gate'],merge=merge,
        timing_ms=dict(filter_and_gate=filter_gate_ms,representation_sampling=sampling_ms,graph=graph_ms),
        actual_stage_order='Depth/robot+overlap -> wrist gate -> shape-first sampling -> cross-view identity -> dedup -> budget -> graph',
        representative_sources=[dict(point=i,object_id=r['object_id'],sources=r['source_observation_ids'],representative=r['source_observation_ids'][0]) for i,r in enumerate(selected)],
        interpretation='Final source support is distinct from spatial visible-surface coverage. Actual source order is audited, not assumed from conceptual pipeline.'))
    return graph,selected,rows
