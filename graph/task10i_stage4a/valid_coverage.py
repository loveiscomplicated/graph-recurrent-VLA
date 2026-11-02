"""Post-inference coverage from correct-actor representatives only, no FP surface credit."""
import argparse
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(output,baseline,corrected,candidate):
    from graph.persistent_geometry.contracts import CameraObservation
    from scipy.ndimage import binary_erosion
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    rows=[]
    for p in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(p);eid=str(cr['episode_id']);e=entries[eid];rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
        with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:seg,exact=z['actor_segmentation'].copy(),z['exact_frame'].copy()
        with h5py.File(e['raw_path'],'r') as f:
            for t in range(e['actions']+1):
                if not exact[t]:
                    rows.extend(dict(episode=eid,parent_id=e['parent_id'],timestep=t,object_id=oid,reference_kind='GT_UNSUPPORTED',baseline=None,candidate=None) for oid in (0,1));continue
                camera=[CameraObservation(eid,t,c,float(f['timestamp'][t]),f[c+'/rgb'][t],f[c+'/depth_raw_mm'][t],f[c+'/intrinsic'][t],f[c+'/extrinsic_cv'][t],f[c+'/cam2world_gl'][t],f['tcp_pose'][t]) for c in CAMERAS]
                worlds=[c.world_points() for c in camera];per_condition={}
                for condition,root in [('baseline',baseline),('candidate',candidate)]:
                    folder=root/'traces'/eid/f't{t:04d}';trace=read(folder/'trace.json')
                    with np.load(folder/'support.npz',allow_pickle=False) as z:
                        xyz=z['point_world'].copy();oids=z['point_object_id'].copy()
                    correct=np.zeros(len(xyz),bool)
                    for r in trace['representative_sources']:
                        i=r['point'];_,_,c,pixel=r['representative'];actor=int(seg[t,CAMERAS.index(c)].ravel()[pixel])
                        correct[i]=actor in rr['object_ids'][('peg','box')[oids[i]]]
                    per_condition[condition]={oid:xyz[(oids==oid)&correct] for oid in (0,1)}
                for oid in (0,1):
                    visible=[];boundary=[];raw=0
                    for ci,c in enumerate(camera):
                        if oid==1 and ci==1:continue
                        gt=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]]);raw+=int(gt.sum())
                        valid=gt&np.isfinite(c.depth);visible.append(worlds[ci][valid]);boundary.append(worlds[ci][valid&~binary_erosion(valid)])
                    vv=np.concatenate(visible).reshape(-1,3);bb=np.concatenate(boundary).reshape(-1,3)
                    measures={}
                    for condition in per_condition:
                        xyz=per_condition[condition][oid];tree=cKDTree(xyz) if len(xyz) else None
                        measures[condition]=dict(correct_representatives=len(xyz),
                            valid_source_coverage_4mm=float((tree.query(vv)[0]<.004).mean()) if tree and len(vv) else 0. if len(vv) else None,
                            valid_source_boundary_coverage_4mm=float((tree.query(bb)[0]<.004).mean()) if tree and len(bb) else 0. if len(bb) else None,
                            raw_visible_object_recall=float(len(xyz)>0) if raw>=8 else None)
                    rows.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,object_id=oid,
                        raw_gt_visible_pixels=raw,valid_depth_gt_samples=len(vv),reference_kind='EXACT_GT',**measures))
        print('valid-source coverage',eid,flush=True)
    groups=defaultdict(list)
    for r in rows:groups[r['episode'],r['parent_id'],r['object_id']].append(r)
    episode=[]
    for (eid,pid,oid),rr in groups.items():
        for metric in ('valid_source_coverage_4mm','valid_source_boundary_coverage_4mm','raw_visible_object_recall'):
            eligible=[r for r in rr if r['reference_kind']=='EXACT_GT' and (r['raw_gt_visible_pixels']>=8 if metric=='raw_visible_object_recall' else r['valid_depth_gt_samples']>=8)]
            values={c:float(np.mean([r[c][metric] for r in eligible])) if eligible else None for c in ('baseline','candidate')}
            episode.append(dict(episode=eid,parent_id=pid,object_id=oid,metric=metric,eligible_frames=len(eligible),total_frames=len(rr),**values))
    parents=defaultdict(list)
    for r in episode:parents[r['parent_id'],r['object_id'],r['metric']].append(r)
    parent=[]
    for (pid,oid,metric),rr in parents.items():
        valid=[r for r in rr if r['baseline'] is not None and r['candidate'] is not None]
        values={c:float(np.mean([r[c] for r in valid])) if valid else None for c in ('baseline','candidate')}
        parent.append(dict(parent_id=pid,object_id=oid,metric=metric,eligible_episodes=len(valid),total_episodes=len(rr),
            **values,delta=values['candidate']-values['baseline'] if valid else None))
    output.mkdir(parents=True,exist_ok=False);write(output/'frame_metrics.json',rows);write(output/'episode_metrics.json',episode);write(output/'parent_metrics.json',parent)
    write(output/'scope.json',dict(status='CORRECT_SOURCE_ONLY_COVERAGE_FINISHED',no_model_forwards=True,
        no_FP_node_coverage_credit=True,object_recall_uses_raw_GT_visibility_not_depth_visibility=True,
        surface_coverage_uses_valid_depth_GT_samples=True,box_reference='Main only',peg_reference='Main and Wrist current pixel-derived surface samples',
        metric_not_used_to_retune_frozen_candidate=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();run(a.output,a.baseline,a.corrected,a.candidate)
