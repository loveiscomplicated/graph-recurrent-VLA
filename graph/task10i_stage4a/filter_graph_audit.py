"""Independent GT source/filter and existing graph/aperture audit reuse."""
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from .run_pipeline import read,write,STAGE3,CAMERAS

def ratio(a,b):return float(a/b) if b else None

def run(root,baseline,corrected):
    from graph.persistent_geometry.metrics import sample_meshes,visible_reference
    from graph.persistent_geometry.contracts import CameraObservation
    from graph.persistent_geometry.geometry_audit import aperture_reference,graph_audit
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()};filters=[];gates=[];graphs=[];coordinates=[];features=[]
    for receipt in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(receipt);eid=str(cr['episode_id']);e=entries[eid];rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
        assert all(len(rr['object_ids'][name])==1 for name in ('peg','box'))
        with np.load(rrpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        with np.load(baseline/'references'/eid/'geometry_reference.npz') as z:poses=z['actor_pose'].copy()
        meshes=sample_meshes(baseline/'references'/eid/'meshes.npz',samples_per_object=10000)
        with h5py.File(e['raw_path']) as f,np.load(e['gt_path']) as labels:
            for t in range(e['actions']+1):
                if not exact[t]:raise AssertionError('must retain and explicitly handle unsupported graph-reference frame')
                cameras=[CameraObservation(eid,t,c,float(f['timestamp'][t]),f[c+'/rgb'][t],f[c+'/depth_raw_mm'][t],f[c+'/intrinsic'][t],f[c+'/extrinsic_cv'][t],f[c+'/cam2world_gl'][t],f['tcp_pose'][t]) for c in CAMERAS]
                for c in cameras:c.validate()
                gt=dict(actor_ids=np.array([rr['object_ids']['peg'][0],rr['object_ids']['box'][0]]),actor_pose=poses[t],hole_pose=labels['raw_hole_pose'][t],hole_radius=labels['raw_hole_radius'][t],robot_ids=rr['robot_ids'])
                gt.update({c+'/segmentation':seg[t,i] for i,c in enumerate(CAMERAS)})
                folder=baseline/'traces'/eid/f't{t:04d}';trace=read(folder/'trace.json')
                with np.load(folder/'support.npz') as z:
                    arrays={k:z[k].copy() for k in z.files}
                def indices_mask(key):
                    out=np.zeros((128,128),bool);out.ravel()[arrays[key]]=True;return out
                for c in cameras:
                    confirmed=indices_mask(c.camera+'__confirmed_robot');uncertain=indices_mask(c.camera+'__uncertain_robot');world=c.world_points()
                    robot=np.isin(gt[c.camera+'/segmentation'],rr['robot_ids'])&np.isfinite(c.depth)
                    for oid in (0,1):
                        obj=(gt[c.camera+'/segmentation']==gt['actor_ids'][oid])&np.isfinite(c.depth)
                        near=np.zeros_like(obj)
                        if obj.any():near[obj]=np.linalg.norm(world[obj,None]-f['robot_world_geometry'][t,1:][None],axis=2).min(1)<.02
                        filters.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,camera=c.camera,object_id=oid,gt_object_pixels=int(obj.sum()),
                            gt_preserved_pixels=int((obj&~confirmed&~uncertain).sum()),confirmed_false_removed=int((obj&confirmed).sum()),uncertain_false_removed=int((obj&uncertain).sum()),
                            near_finger_pixels=int(near.sum()),near_finger_confirmed_false_removed=int((near&confirmed).sum()),near_finger_uncertain_false_removed=int((near&uncertain).sum()),
                            gt_robot_pixels=int(robot.sum()),confirmed_robot_removed=int((robot&confirmed).sum()),uncertain_robot_removed=int((robot&uncertain).sum()),
                            confirmed_tolerance_m=.0025,uncertainty_band_m=.006,near_finger_definition='within20mm of observed finger-pad centres, not contact-force ground truth'))
                    wristraw=indices_mask('hand_camera__0__sam2') if c.camera=='hand_camera' else None
                    if wristraw is not None:
                        correct=wristraw&(gt[c.camera+'/segmentation']==gt['actor_ids'][0]);precision=ratio(correct.sum(),wristraw.sum())
                        quality='good_2d' if correct.sum()>=6 and precision is not None and precision>=.9 else 'bad_2d' if wristraw.any() and precision is not None and precision<=.1 else 'missing_upstream' if not wristraw.any() else 'ambiguous_2d'
                        gates.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,quality=quality,raw_correct_pixels=int(correct.sum()),raw_precision=precision,accepted=trace['gate']['accepted'],reject_reasons=trace['gate']['reasons'],trace=trace['gate']))
                by={c.camera:c for c in cameras};errors=[];foreign=0
                for i,r in enumerate(trace['representative_sources']):
                    if r['point']!=i:raise AssertionError('representative index order')
                    ep,ts,c,p=r['representative'];foreign+=ep!=eid or ts!=t
                    errors.append(float(np.linalg.norm(arrays['point_world'][i]-by[c].world_points().reshape(-1,3)[p])))
                coordinates.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,point_count=len(errors),backprojection_error_max_m=max(errors) if errors else 0.,foreign_or_history_sources=int(foreign),camera_contract_validated=True))
                if foreign or (errors and max(errors)>3e-6):raise AssertionError('point/source coordinate mismatch')
                for oid in (0,1):
                    chosen=arrays['point_object_id']==oid;valid=arrays['point_valid'][chosen]
                    features.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,object_id=oid,points=int(chosen.sum()),normal_valid=int(valid[:,0].sum()),variation_valid=int(valid[:,1].sum()),jump_valid=int(valid[:,2].sum()),
                        node_count=int((arrays['node_type']==oid+3).sum()),finite_graph_input=bool(np.isfinite(arrays['node_input']).all()),finite_graph_edge=bool(np.isfinite(arrays['edge_attr']).all())))
                visible=visible_reference(cameras,gt);visible[1]=by['base_camera'].world_points()[(gt['base_camera/segmentation']==gt['actor_ids'][1])&np.isfinite(by['base_camera'].depth)]
                for row in graph_audit(arrays,cameras,gt,meshes,aperture_reference(meshes,gt),visible):row.update(episode=eid,parent_id=e['parent_id'],timestep=t,box_visible_denominator='Main only',reference_kind='EXACT_GT');graphs.append(row)
        print('filter/graph audit',eid,flush=True)
    write(root/'robot_filter_frame_metrics.json',filters);write(root/'wrist_gate_frame_metrics.json',gates);write(root/'graph_aperture_frame_metrics.json',graphs);write(root/'coordinate_source_checks.json',coordinates);write(root/'feature_validity_frame_metrics.json',features)
    write(root/'filter_graph_scope.json',dict(status='INDEPENDENT_FILTER_GATE_GRAPH_AUDIT_FINISHED',model_forwards=0,GT_used_only_by_evaluation=True,
        good_2d_definition='>=6 correct pixels, precision>=.9; bad_2d precision<=.1; ambiguous and missing upstream separate',
        graph_aperture_api='existing graph.persistent_geometry.geometry_audit.aperture_reference and graph_audit',
        distance_reference='10000 deterministic external collision surface samples; approximation floor is not an exact analytic mesh distance',
        coordinates='all real representatives backprojected from actual camera source; no new points or geometry modifications'))
