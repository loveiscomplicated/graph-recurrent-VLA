"""Re-evaluate frozen masks/points with validated repaired GT; zero model forwards."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from .run_pipeline import CAMERAS,STAGE3,read,write,sha
from .instrumentation import support


def updated_points(arrays,trace,refs,rr,cameras,meshes,poses,t,e):
    from visual.runtime.vision_packet import pose_matrix
    from graph.persistent_geometry.contracts import transform
    rows=[];source={r['point']:r['representative'] for r in trace['representative_sources']}
    for oid in (0,1):
        indices=np.flatnonzero(arrays['point_object_id']==oid);xyz=arrays['point_world'][indices];labels=[]
        for idx in indices:
            _,_,c,p=source[int(idx)];actor=int(refs[c].ravel()[p])
            labels.append(0 if actor in rr['object_ids'][('peg','box')[oid]] else 1 if actor in rr['robot_ids'] else 2 if actor in rr['object_ids'][('box','peg')[oid]] else 3)
        local=transform(xyz,np.linalg.inv(pose_matrix(poses[t,oid])));distance=meshes[oid]['tree'].query(local)[0]
        visible=[];boundary=[]
        for camera in cameras:
            if oid==1 and camera.camera!='base_camera':continue
            mask=np.isin(refs[camera.camera],rr['object_ids'][('peg','box')[oid]])&np.isfinite(camera.depth)
            world=camera.world_points();visible.append(world[mask]);boundary.append(world[mask&~binary_erosion(mask)])
        vv=np.concatenate(visible).reshape(-1,3);bb=np.concatenate(boundary).reshape(-1,3);tree=cKDTree(xyz) if len(xyz) else None
        rows.append(dict(episode=str(e['episode_id']),parent_id=e['parent_id'],timestep=t,object_id=oid,points=len(indices),
            correct=sum(x==0 for x in labels),robot=sum(x==1 for x in labels),wrong_object=sum(x==2 for x in labels),background=sum(x==3 for x in labels),reference_kind='EXACT_GT',
            surface_distance_mean_mm=float(distance.mean()*1000) if len(distance) else None,surface_distance_p95_mm=float(np.percentile(distance,95)*1000) if len(distance) else None,
            visible_reference_points=len(vv),visible_coverage_4mm=float((tree.query(vv)[0]<.004).mean()) if tree and len(vv) else 0. if len(vv) else None,
            projected_boundary_reference_points=len(bb),projected_boundary_coverage_4mm=float((tree.query(bb)[0]<.004).mean()) if tree and len(bb) else 0. if len(bb) else None,
            surface_reference='10000 deterministic area-weighted external collision-union samples; approximate mesh distance, not exact analytic distance',normal_valid_fraction=float(arrays['point_valid'][indices,0].mean()) if len(indices) else None))
    return rows

def run(root,baseline,repaired):
    import os
    import fcntl
    from utils.experiment_logger import ExperimentLogger
    from graph.persistent_geometry.metrics import sample_meshes
    from graph.persistent_geometry.contracts import CameraObservation
    root.mkdir(parents=True,exist_ok=False)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_exact_reference_reevaluation',seed=42,root=root/'experiments',job_type='evaluation',config=dict(baseline_root=str(baseline),repaired_reference_root=str(repaired),model_forward_calls=0)) as log:
            write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(__file__)))
            write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_BASELINE_AND_REPAIRED_REFERENCE',run_id=log.run_id,pid=os.getpid()))
            try:
                while not ((baseline/'development_perception_summary.json').exists() and (repaired/'reference_repair_summary.json').exists()):
                    for prior in (baseline,repaired):
                        if read(prior/'pipeline_state.json').get('status')=='FAILED':raise RuntimeError('prerequisite failed: '+str(prior))
                    time.sleep(10)
                entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()};summary=[]
                for receipt in sorted((baseline/'perception').glob('*/receipt.json')):
                    br=read(receipt);eid=str(br['episode_id']);e=entries[eid];out=root/'perception'/eid;out.mkdir(parents=True,exist_ok=False)
                    old=receipt.parent;ref=repaired/'references'/eid
                    if not (ref/'receipt.json').exists():ref=baseline/'references'/eid
                    rr=read(ref/'receipt.json');assert rr['raw_sha256']==e['raw_sha256']
                    assert sha(old/'masks.npz')==br['masks_sha256'] and sha(old/'stage_rows.json')==br['rows_sha256']
                    rows=read(old/'stage_rows.json');rep=read(old/'representative_points.json');changes=0
                    if ref.parent.parent==repaired:
                        assert sha(ref/'segmentation.npz')==rr['segmentation_sha256']
                        with np.load(ref/'segmentation.npz') as z:segs=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                        meshes=sample_meshes(baseline/'references'/eid/'meshes.npz',samples_per_object=10000)
                        with np.load(baseline/'references'/eid/'geometry_reference.npz') as z:poses=z['actor_pose'].copy()
                        for t in range(e['actions']+1):
                            if not exact[t]:continue
                            refs=dict(zip(CAMERAS,segs[t]));folder=baseline/'traces'/eid/f't{t:04d}';trace=read(folder/'trace.json')
                            with np.load(folder/'support.npz') as arrays:
                                for row in (r for r in rows if r['timestep']==t):
                                    pixels=arrays[row['camera']+'__'+str(row['object_id'])+'__'+row['stage']]
                                    mask=np.zeros((128,128),bool);mask.ravel()[pixels]=True
                                    m=support(mask,refs[row['camera']],row['object_id'],rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']])
                                    if row['metrics'] is not None:assert row['metrics']==m,'Previously EXACT_GT changed'
                                    else:changes+=1
                                    row['metrics']=m;row['reference_kind']='EXACT_GT'
                                # Reuse already validated spatial metrics on old exact
                                # frames; only fill formerly unsupported observations.
                                if not all(r['reference_kind']=='EXACT_GT' for r in rep if r['timestep']==t):
                                    with h5py.File(e['raw_path']) as f:
                                        cameras=[CameraObservation(eid,t,c,float(f['timestamp'][t]),f[c+'/rgb'][t],f[c+'/depth_raw_mm'][t],f[c+'/intrinsic'][t],f[c+'/extrinsic_cv'][t],f[c+'/cam2world_gl'][t],f['tcp_pose'][t]) for c in CAMERAS]
                                    new=updated_points(arrays,trace,refs,rr,cameras,meshes,poses,t,e)
                                    rep=[r for r in rep if r['timestep']!=t]+new
                    rep.sort(key=lambda r:(r['timestep'],r['object_id']))
                    write(out/'stage_rows.json',rows);write(out/'representative_points.json',rep)
                    write(out/'receipt.json',dict(**br,original_baseline_receipt=str(receipt),original_baseline_receipt_sha256=sha(receipt),
                        corrected_reference=str(ref/'receipt.json'),corrected_reference_sha256=sha(ref/'receipt.json'),filled_stage_rows=changes,
                        corrected_rows_sha256=sha(out/'stage_rows.json'),model_forward_calls=0,original_masks_and_graph_unchanged=True))
                    summary.append(dict(episode_id=eid,frames=e['actions']+1,exact_frames=rr['exact_frames'],filled_stage_rows=changes))
                    write(root/'progress.json',summary);print('reference re-evaluation',len(summary),eid,changes,flush=True)
                write(root/'protocol.json',read(baseline/'protocol.json'))
                from .aggregate_development import run as aggregate
                aggregate(root)
                write(root/'reevaluation_summary.json',dict(status='CORRECTED_DEVELOPMENT_REFERENCE_AUDIT_FINISHED',episodes=len(summary),frames=sum(r['frames'] for r in summary),exact_frames=sum(r['exact_frames'] for r in summary),model_forward_calls=0,original_masks_and_graph_unchanged=True,rows=summary))
                log.link_output(root/'reevaluation_summary.json',kind='metrics');log.link_output(root/'development_parent_macro.json',kind='metrics')
                write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='CORRECTED_DEVELOPMENT_AUDIT_FINISHED',completed_goal=False,run_id=log.run_id))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);p.add_argument('repaired',type=Path);a=p.parse_args();run(a.root,a.baseline,a.repaired)
