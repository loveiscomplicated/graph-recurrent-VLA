"""Post-forward current-observed-surface geometry residuals; no Graph rebuild.

Centroid residuals depend on visible surface coverage and are not an object-pose
estimator. They supplement source purity/coverage and never replace the frozen
adoption criteria. No inference model or simulator is constructed here.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree
from graph.persistent_geometry.contracts import CameraObservation,synchronized
from .dinov2_matching_audit import csv_write
from .evaluation_metrics import hierarchical_paired
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def residuals(points,correct,reference,previous=None,t=0):
    points=np.asarray(points,float).reshape(-1,3);reference=np.asarray(reference,float).reshape(-1,3)
    correct=np.asarray(correct,bool);assert correct.shape==(len(points),)
    assert np.isfinite(points).all() and np.isfinite(reference).all()
    result=dict(geometry_points=len(points),visible_reference_points=len(reference),
        all_point_surface_mean_m=None,all_point_surface_p95_m=None,all_point_surface_max_m=None,
        correct_point_surface_mean_m=None,visible_centroid_error_m=None,visible_centroid_step_residual_m=None)
    if not len(points) or not len(reference):return result,None
    distances=cKDTree(reference).query(points)[0]
    pc=points.mean(0);rc=reference.mean(0)
    result.update(all_point_surface_mean_m=float(distances.mean()),all_point_surface_p95_m=float(np.percentile(distances,95)),
        all_point_surface_max_m=float(distances.max()),correct_point_surface_mean_m=float(distances[correct].mean()) if correct.any() else None,
        visible_centroid_error_m=float(np.linalg.norm(pc-rc)))
    if previous is not None and previous[0]==t-1:
        result['visible_centroid_step_residual_m']=float(np.linalg.norm((pc-previous[1])-(rc-previous[2])))
    return result,(t,pc,rc)


def cameras_from_raw(raw,t,episode):
    tcp=raw['tcp_pose'][t]
    cameras=[CameraObservation(str(episode),t,c,float(raw['timestamp'][t]),raw[c+'/rgb'][t],
        raw[c+'/depth_raw_mm'][t],raw[c+'/intrinsic'][t],raw[c+'/extrinsic_cv'][t],raw[c+'/cam2world_gl'][t],tcp.copy()) for c in CAMERAS]
    synchronized(cameras);return cameras


def run(root,paired,references,expected,criteria,candidate):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    assert read(paired/'pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),criteria_sha256=sha(criteria),candidate=candidate,
        posthoc_descriptive=True,changes_adoption_rules=False,reference='current GT-visible actor RGB-D surface, Peg Main+Wrist and Box Main only',
        centroid_limitation='visible surface/coverage changes affect centroids; not a physical actor-pose drift estimate')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_observed_geometry_drift_audit',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        rows=[];inputs=[]
        for e in read(expected)['entries']:
            eid=str(e['episode_id']);folder=paired/'episodes'/eid;pr=read(folder/'receipt.json')
            assert pr['candidate']==candidate and pr['raw_sha256']==e['raw_sha256']
            ref=references/'references'/eid;rr=read(ref/'receipt.json')
            assert rr['raw_sha256']==e['raw_sha256'] and sha(e['raw_path'])==e['raw_sha256']
            assert sha(ref/'segmentation.npz')==rr['files']['segmentation.npz']
            with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            previous={};inputs.append(dict(paired_receipt=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
                reference_receipt=str(ref/'receipt.json'),reference_sha256=sha(ref/'receipt.json')))
            with h5py.File(e['raw_path']) as raw:
                for t in range(e['actions']+1):
                    cameras=cameras_from_raw(raw,t,eid);world=[c.world_points() for c in cameras]
                    reference={oid:np.concatenate([world[ci][np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])&np.isfinite(c.depth)]
                        for ci,c in enumerate(cameras) if not(oid==1 and ci==1)]).reshape(-1,3) for oid in (0,1)} if exact[t] else {}
                    for condition in ('B0',candidate):
                        frame=folder/condition/f't{t:04d}'
                        for name in ('graph.npz','provenance.json'):assert sha(frame/name)==pr['files'][str((frame/name).relative_to(folder))]
                        with np.load(frame/'graph.npz') as z:points=z['world_points'].copy();objects=z['object_id'].copy()
                        provenance=read(frame/'provenance.json');assert provenance['current_only'] and provenance['box_main_only']
                        assert len(points)==len(objects)==len(provenance['sources'])
                        for oid in (0,1):
                            selected=np.flatnonzero(objects==oid);correct=[]
                            for index in selected:
                                pe,pt,c,pixel=provenance['sources'][index]['sources'][0]
                                assert str(pe)==eid and pt==t and not(oid==1 and c!='base_camera')
                                correct.append(int(seg[t,CAMERAS.index(c)].ravel()[pixel]) in rr['object_ids'][('peg','box')[oid]] if exact[t] else False)
                            key=condition,oid
                            row=dict(episode=eid,parent_id=e['parent_id'],timestep=t,condition=condition,object_id=oid,
                                stratum=e.get('stratum','internal'),reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                            if exact[t]:metrics,next_value=residuals(points[selected],correct,reference[oid],previous.get(key),t);row.update(metrics);previous[key]=next_value
                            else:previous[key]=None
                            rows.append(row)
        rules=read(criteria)['bootstrap'];summary=[]
        for metric in ('all_point_surface_mean_m','all_point_surface_p95_m','all_point_surface_max_m',
                       'correct_point_surface_mean_m','visible_centroid_error_m','visible_centroid_step_residual_m'):
            summary.extend(hierarchical_paired(rows,metric,('stratum','object_id'),candidate=candidate,
                bootstrap=rules['replicates'],seed=rules['seed']))
        csv_write(root/'geometry_drift_metrics.csv',rows)
        write(root/'paired_geometry_drift_metrics.json',summary)
        write(root/'scope.json',dict(status='POST_FORWARD_OBSERVED_GEOMETRY_DRIFT_AUDIT_COMPLETE',run_id=logger.run_id,
            **config,inputs=inputs,frames=read(expected)['frames'],model_forwards=0,Graph_rebuilt=False,
            empty_or_no_visible_reference_residual_is_undefined=True,unsupported_interrupts_temporal_residual=True,
            whole_goal_complete=False,files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','paired','references','expected','criteria'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);run(**vars(p.parse_args()))
