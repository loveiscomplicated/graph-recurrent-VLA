"""GT-labelled posthoc recovery delays and trigger misses; no inference.

GT bad-mask intervals are diagnostic opportunities, not proof that a fresh
prompt would succeed. Actual Re-ID acceptance is distinct from mask recovery.
"""
import argparse
from pathlib import Path
from .aggregate_evaluation import csv_rows
from .temporal_mask_diagnostics import runs
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def first_correct(rows,start,stop=None):
    return next((r['timestep'] for r in rows[start:stop] if r['reference_kind']=='EXACT_GT' and r.get('correct_mask_availability')==1),None)


def run(root,final,expected,candidate):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    assert read(final/'paired/pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),candidate=candidate,
        posthoc_only=True,GT_never_practical_trigger=True,new_threshold_fitting=False,
        meaning='observed mask recovery and actual reference-matched accepted events; not an incremental DINOv2 causal effect',
        interval_limitation='GT-bad intervals without requests are missed opportunities, not an oracle guarantee of correct reanchor')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_reanchor_trigger_and_recovery_delay_audit',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        onsets=[];bad_runs=[];anchors=[];inputs=[]
        for e in read(expected)['entries']:
            eid=str(e['episode_id']);folder=final/'candidate/perception'/eid;receipt=read(folder/'receipt.json')
            assert receipt['raw_sha256']==e['raw_sha256']
            for name in ('events.json','baseline_handoff_events.json'):assert sha(folder/name)==receipt['files'][name]
            pixelpath=final/'paired/episodes'/eid/'pixel_metrics.csv';gr=read(pixelpath.parent/'receipt.json')
            assert sha(pixelpath)==gr['files']['pixel_metrics.csv']
            pixels=csv_rows(pixelpath);events=read(folder/'events.json');hand=read(folder/'baseline_handoff_events.json')
            inputs.append(dict(candidate_receipt=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
                pixel_metrics=str(pixelpath),pixel_sha256=sha(pixelpath)))
            for camera,oid in [('base_camera',0),('hand_camera',0),('base_camera',1)]:
                pe=sorted([r for r in events if r['camera']==camera and r['object_id']==oid],key=lambda r:r['timestep'])
                b=sorted([r for r in pixels if r['condition']=='B0' and r['camera']==camera and r['object_id']==oid],key=lambda r:r['timestep'])
                c=sorted([r for r in pixels if r['condition']==candidate and r['camera']==camera and r['object_id']==oid],key=lambda r:r['timestep'])
                assert len(pe)==len(b)==len(c)==e['actions']+1
                assert all([r['timestep'] for r in rows]==list(range(len(rows))) for rows in (pe,b,c))
                accepted=[dict(r,event_kind='fresh_memory_reanchor') for r in pe if r['accepted']]
                accepted.extend(dict(r,event_kind='later_original_initialization_handoff') for r in hand if r['camera']==camera and r['object_id']==oid and r['accepted'])
                reid_correct=[r['timestep'] for r in accepted if r['reference_available'] and c[r['timestep']].get('correct_mask_availability')==1]
                base=dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),camera=camera,object_id=oid)
                for interval in runs([not r['currently_valid'] for r in pe]):
                    requests=[r for r in pe[interval['start']:interval['end']+1] if r['detection_requested']]
                    if not requests:continue
                    t=requests[0]['timestep'];stop=min(interval['end']+2,len(c));recovered=first_correct(c,t,stop)
                    reid=next((v for v in sorted(reid_correct) if t<=v<stop),None)
                    visible=next((i for i in range(t,stop) if c[i]['reference_kind']=='EXACT_GT' and c[i].get('visible_pixels',0)>=8),None)
                    onsets.append(dict(**base,model_uncertain_start=interval['start'],model_uncertain_end=interval['end'],
                        first_request_t=t,requests_in_uncertain_interval=len(requests),reference_available_at_request=requests[0]['reference_available'],
                        first_correct_mask_t=recovered,mask_recovery_delay_frames=recovered-t if recovered is not None else None,
                        first_actual_reid_accepted_correct_t=reid,actual_reid_recovery_delay_frames=reid-t if reid is not None else None,
                        first_visible_after_request=visible,
                        actual_reid_delay_after_first_visible=reid-visible if reid is not None and visible is not None else None,
                        mask_recovery_right_censored=recovered is None,actual_reid_recovery_right_censored=reid is None,
                        attribution_window_last_frame=stop-1,no_incremental_matching_causality_claim=True))
                eligible_bad=[r['reference_kind']=='EXACT_GT' and r.get('visible_pixels',0)>=8 and r.get('correct_mask_availability')!=1 for r in c]
                for interval in runs(eligible_bad):
                    chunk=pe[interval['start']:interval['end']+1]
                    bad_runs.append(dict(**base,**interval,redetection_requests=sum(r['detection_requested'] for r in chunk),
                        no_request_during_GT_bad_run=not any(r['detection_requested'] for r in chunk),
                        support_confident_frames=sum(r['currently_valid'] for r in chunk),
                        frames_with_reference=sum(r['reference_available'] for r in chunk),
                        correction_scope_enabled=all(r['scope_enabled'] for r in chunk),
                        requests_due_but_scope_disabled=sum(r['redetection_due_but_scope_disabled'] for r in chunk)))
                for event in accepted:
                    t=event['timestep'];window=range(t,min(t+6,len(c)));bm=b[t];cm=c[t]
                    exact=cm['reference_kind']=='EXACT_GT'
                    anchors.append(dict(**base,event_kind=event['event_kind'],timestep=t,reference_available=event['reference_available'],
                        similarity=event.get('similarity',event.get('chosen',{}).get('similarity')),
                        reference_kind=cm['reference_kind'],B0_available=bm.get('correct_mask_availability'),candidate_available=cm.get('correct_mask_availability'),
                        correct_recovery_now=bool(exact and bm.get('correct_mask_availability')==0 and cm.get('correct_mask_availability')==1),
                        immediate_healthy_damage=bool(exact and bm.get('correct_mask_availability')==1 and cm.get('correct_mask_availability')!=1),
                        exact_frames_in_fixed_6frame_window=sum(c[i]['reference_kind']=='EXACT_GT' for i in window),
                        correct_frames_in_fixed_6frame_window=sum(c[i].get('correct_mask_availability')==1 for i in window),
                        healthy_B0_damage_frames_in_fixed_6frame_window=sum(c[i]['reference_kind']=='EXACT_GT' and b[i].get('correct_mask_availability')==1 and c[i].get('correct_mask_availability')!=1 for i in window),
                        GT_failure_time_not_practical_trigger=True))
        csv_write(root/'sam2_causal_recovery_event_delays.csv',onsets)
        csv_write(root/'sam2_GT_bad_intervals_and_trigger_requests.csv',bad_runs)
        csv_write(root/'sam2_reanchor_success_and_fixed_window_safety.csv',anchors)
        write(root/'scope.json',dict(status='POST_FORWARD_TRIGGER_REID_AND_REANCHOR_RECOVERY_AUDIT_COMPLETE',run_id=logger.run_id,**config,
            inputs=inputs,request_uncertain_intervals=len(onsets),GT_bad_intervals=len(bad_runs),accepted_events=len(anchors),
            accepted_actual_reference_matched_events=sum(r['reference_available'] for r in anchors),
            frames_not_wall_time=True,nominal_sensor_control_clock_Hz=20,
            right_censored_delays_not_imputed=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','final','expected'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);run(**vars(p.parse_args()))
