"""Post-forward exact-GT mask reliability, durations and reappearance events."""
import argparse
from pathlib import Path
import numpy as np
from .evaluation_metrics import pixel_metrics
from .registered_evaluation import statistics
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def runs(flags):
    intervals=[];start=None
    for t,value in enumerate(list(flags)+[False]):
        if value and start is None:start=t
        elif not value and start is not None:intervals.append(dict(start=start,end=t-1,frames=t-start));start=None
    return intervals


def temporal(rows):
    """Unknown references/visibility interrupt runs; no compressed clock."""
    rows=sorted(rows,key=lambda r:r['timestep'])
    assert [r['timestep'] for r in rows]==list(range(len(rows)))
    eligible=[r.get('reference_kind')=='EXACT_GT' and r['visible_pixels']>=8 for r in rows]
    invisible=[r.get('reference_kind')=='EXACT_GT' and r['visible_pixels']==0 for r in rows]
    good=[eligible[t] and r['correct_mask_availability']==1 for t,r in enumerate(rows)]
    wrong=[eligible[t] and r['predicted_pixels']>0 and r['precision']<.2 and
           max(r['robot_pixels'],r['wrong_object_pixels'])/r['predicted_pixels']>=.8 for t,r in enumerate(rows)]
    empty=runs([eligible[t] and r['predicted_pixels']==0 for t,r in enumerate(rows)])
    ghosts=runs([invisible[t] and r['predicted_pixels']>0 for t,r in enumerate(rows)])
    identity=runs(wrong);drifts=[r for r in identity if any(good[:r['start']])]
    fragments=sum(good[t-1] and eligible[t] and not good[t] for t in range(1,len(rows)))
    recoveries=[]
    for interval in runs(invisible):
        t=interval['end']+1
        if t<len(rows) and eligible[t] and any(eligible[:interval['start']]):
            end=t
            while end<len(rows) and eligible[end] and not good[end]:end+=1
            recovered=end<len(rows) and eligible[end] and good[end]
            recoveries.append(dict(reappearance_timestep=t,preceding_invisible_frames=interval['frames'],
                recovered=recovered,delay_frames=end-t if recovered else None,
                right_censored_or_reference_interrupted=not recovered))
    first=next((t for t,v in enumerate(eligible) if v),None)
    init=next((t for t,v in enumerate(good) if v and first is not None and t>=first),None)
    return dict(exact_frames=sum(r['reference_kind']=='EXACT_GT' for r in rows),visible_frames=sum(eligible),
        invisible_frames=sum(invisible),correct_frames=sum(good),empty_visible_frames=sum(r['frames'] for r in empty),
        ghost_frames=sum(r['frames'] for r in ghosts),wrong_identity_frames=sum(wrong),
        empty_visible_runs=empty,ghost_runs=ghosts,wrong_identity_runs=identity,
        wrong_identity_onsets_after_earlier_correct_mask=drifts,initial_wrong_identity=bool(wrong[0]),
        temporal_fragmentation=fragments,occlusion_reappearance_events=recoveries,
        first_eligible_timestep=first,first_correct_mask_timestep=init,
        initialization_delay_from_first_visible=init-first if init is not None else None,
        never_correct_mask=init is None,semantic_drift_is_GT_posthoc_heuristic_not_SAM_confidence=True)


def run(root,inference,references,expected,candidate,criteria):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),candidate=candidate,
        criteria_sha256=sha(criteria),GT_posthoc_only=True,practical_threshold_changes=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_temporal_mask_diagnostic',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        pixels=[];temporal_rows=[];inputs=[]
        for e in read(expected)['entries']:
            eid=str(e['episode_id']);folder=inference/'perception'/eid;ir=read(folder/'receipt.json')
            assert all(sha(folder/p)==h for p,h in ir['files'].items())
            ref=references/'references'/eid;rr=read(ref/'receipt.json')
            assert all(sha(ref/p)==h for p,h in rr['files'].items())
            assert ir['raw_sha256']==rr['raw_sha256']==e['raw_sha256']
            with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            for condition in ('B0',candidate):
                with np.load(folder/(condition+'.npz')) as z:pred=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
                for ci,camera in enumerate(CAMERAS):
                    for oid in (0,1):
                        rows=[]
                        for t in range(e['actions']+1):
                            row=dict(episode=eid,parent_id=e['parent_id'],condition=condition,camera=camera,object_id=oid,
                                timestep=t,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                            if exact[t]:row.update(pixel_metrics(pred[t,ci,oid],np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]]),
                                np.isin(seg[t,ci],rr['robot_ids']),np.isin(seg[t,ci],rr['object_ids'][('box','peg')[oid]])))
                            rows.append(row)
                        pixels.extend(rows)
                        temporal_rows.append(dict(episode=eid,parent_id=e['parent_id'],condition=condition,camera=camera,
                            object_id=oid,graph_eligible=not(ci==1 and oid==1),**temporal(rows)))
            inputs.append(dict(inference=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
                               reference=str(ref/'receipt.json'),reference_sha256=sha(ref/'receipt.json')))
        summaries,primary,_,counts=statistics(pixels,[],candidate,read(criteria))
        write(root/'registered_primary_mask_metrics.json',primary)
        write(root/'paired_mask_parent_metrics.json',summaries)
        write(root/'sam2_causal_recovery_metrics.json',temporal_rows)
        csv_write(root/'sam2_temporal_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in temporal_rows])
        csv_write(root/'pixel_metrics.csv',pixels);csv_write(root/'false_positive_safety_audit.csv',counts)
        write(root/'scope.json',dict(status='POST_FORWARD_TEMPORAL_MASK_DIAGNOSTICS_COMPLETE',run_id=logger.run_id,
            **config,inputs=inputs,frames=read(expected)['frames'],parents=read(expected)['parents'],
            no_future_inference=True,future_used_only_to_measure_recovery_delay=True,
            Graph_evaluation='separate complete paired Graph pipeline required',candidate_selected=False,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','inference','references','expected','criteria'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args();run(a.root,a.inference,a.references,a.expected,a.candidate,a.criteria)
