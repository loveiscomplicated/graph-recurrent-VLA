"""Post-forward actual Re-ID/reference/reanchor/consumer audit on Development.

GT sidecars are opened only after complete immutable inference receipts. This
does not fit rules, modify masks or claim similarity curves from absent calls.
"""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from .evaluation_metrics import pixel_metrics
from .dinov2_matching_audit import csv_write
from .prompt_audit import bbox
from .reference_pair_audit import curves
from visual.runtime.initialization import box_iou
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def mask_row(mask,seg,rr,oid):
    target=np.isin(seg,rr['object_ids'][('peg','box')[oid]])
    return pixel_metrics(mask,target,np.isin(seg,rr['robot_ids']),np.isin(seg,rr['object_ids'][('box','peg')[oid]]))


def identity_label(metrics):
    n=metrics['predicted_pixels']
    if not n:return 'EMPTY'
    if metrics['tp']>=6 and metrics['precision']>=.8:return 'TARGET_ACTOR'
    for name,key in [('ROBOT','robot_pixels'),('WRONG_OBJECT','wrong_object_pixels'),('BACKGROUND','background_pixels')]:
        if metrics[key]/n>=.8:return name
    return 'MIXED_UNKNOWN'


def run(root,inference,references,expected):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    assert read(inference/'pipeline_state.json')['status']=='DEVELOPMENT_C_BC_KEEP_INFERENCE_COMPLETE'
    specification=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),GT_posthoc_only=True,
        inference_identity_sha256=sha(inference/'source_config_frozen.json'),thresholds_changed=False,
        label='dominant80% predicted-mask GT actor; target requires6 correct pixels; mixed remains unknown',
        reanchor_future_window='fixed accepted t through min(t+5, episode end); diagnostic only, never an inference trigger')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_actual_matching_correction_audit',seed=42,
            root=root/'experiments',job_type='analysis',config=specification) as logger:
        reference_rows=[];matching=[];anchors=[];drops=[];calls=[];memories=[];inputs=[]
        differences=0;totals={c:Counter() for c in ('C_KEEP','BC_KEEP')}
        manifest=read(expected)
        for entry in manifest['entries']:
            eid=str(entry['episode_id']);folder=inference/'perception'/eid;receipt=read(folder/'receipt.json')
            assert all(sha(folder/p)==h for p,h in receipt['files'].items())
            ref=references/'references'/eid;rr=read(ref/'receipt.json')
            assert all(sha(ref/p)==h for p,h in rr['files'].items())
            assert rr['raw_sha256']==receipt['raw_sha256']==entry['raw_sha256']
            inputs.append(dict(inference_receipt=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
                reference_receipt=str(ref/'receipt.json'),reference_sha256=sha(ref/'receipt.json')))
            with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            predictions={}
            for condition in ('B0','C_KEEP','BC_KEEP'):
                with np.load(folder/(condition+'.npz')) as z:predictions[condition]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            differences+=int(np.any(predictions['C_KEEP']!=predictions['BC_KEEP'],axis=(1,2,3,4)).sum())
            for condition in ('C_KEEP','BC_KEEP'):
                path=folder/condition;cr=read(path/'receipt.json');total=totals[condition]
                for key in ('accepted','actual_dinov2_forwards','actual_reid_comparisons','actual_sam_frame_inferences'):total[key]+=cr[key]
                refs=read(path/'reference_provenance.json');ref_lookup={(r['camera'],r['object_id']):r for r in refs}
                feature=read(path/'feature_calls.json')
                calls.append(dict(episode=eid,parent_id=entry['parent_id'],condition=condition,
                    embedding_requests=len(feature),actual_forwards=sum(r['model_forward'] for r in feature),
                    patch_support_rejections=sum(bool(r.get('rejection')) for r in feature),
                    feature_trace=str(path/'feature_calls.json'),feature_trace_sha256=sha(path/'feature_calls.json'),
                    actual_similarity_comparisons=cr['actual_reid_comparisons']))
                for r in refs:
                    t=r['timestep'];ci=CAMERAS.index(r['camera']);oid=r['object_id']
                    embedding=np.asarray(r['embedding']);assert embedding.shape==(384,) and abs(np.linalg.norm(embedding)-1)<1e-4
                    record={k:v for k,v in r.items() if k not in ('mask_bits','embedding')}
                    record.update(parent_id=entry['parent_id'],condition=condition,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                        original_provenance=str(path/'reference_provenance.json'),provenance_sha256=sha(path/'reference_provenance.json'),
                        embedding_sha256=hashlib.sha256(embedding.tobytes()).hexdigest(),reference_updated=False)
                    if exact[t]:
                        mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                        m=mask_row(mask,seg[t,ci],rr,oid)
                        record.update(metrics=m,actual_actor=identity_label(m),contaminated=m['precision']<.8)
                    reference_rows.append(record)
                events=read(path/'events.json');event_lookup={(e['timestep'],e['camera'],e['object_id']):e for e in events}
                candidates=read(path/'candidate_masks.json')
                cm={(r['timestep'],r['camera'],r['object_id'],r['post_rank']):r for r in candidates}
                for event in events:
                    t=event['timestep'];camera=event['camera'];ci=CAMERAS.index(camera);oid=event['object_id']
                    if event['detection_requested']:
                        total['redetection_events']+=1
                        if not event['reference_available']:total['redetection_no_reference_guard']+=1
                    for prop in event['proposals']:
                        if prop['similarity'] is None:
                            total['proposal_no_reference' if not prop['reference_available'] else 'proposal_reference_but_no_usable_embedding']+=1
                        else:
                            bank=ref_lookup[camera,oid];assert bank['timestep']<t
                            proposal=cm[t,camera,oid,prop['post_rank']]
                            record=dict(episode=eid,parent_id=entry['parent_id'],condition=condition,timestep=t,camera=camera,object_id=oid,
                                reference_timestep=bank['timestep'],reference_origin=bank['origin'],**prop,
                                reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                                accepted_event=event['accepted'],chosen_this_proposal=event.get('chosen',{}).get('post_rank')==prop['post_rank'])
                            if exact[t]:
                                mask=np.unpackbits(np.asarray(proposal['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                                metrics=mask_row(mask,seg[t,ci],rr,oid);label=identity_label(metrics)
                                record.update(metrics=metrics,actual_actor=label,
                                    intended_positive=True if label=='TARGET_ACTOR' else False if label in ('ROBOT','WRONG_OBJECT','BACKGROUND') else None)
                            matching.append(record)
                    if event['accepted']:
                        record=dict(episode=eid,parent_id=entry['parent_id'],condition=condition,timestep=t,camera=camera,object_id=oid,
                            prediction_only_trigger=True,reference_available=event['reference_available'],
                            chosen=event['chosen'],trigger_evidence=event['evidence'],
                            reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                        if exact[t]:
                            b=mask_row(predictions['B0'][t,ci,oid],seg[t,ci],rr,oid)
                            c=mask_row(predictions[condition][t,ci,oid],seg[t,ci],rr,oid)
                            window=[]
                            for u in range(t,min(t+6,entry['actions']+1)):
                                if exact[u]:window.append(dict(timestep=u,
                                    B0=mask_row(predictions['B0'][u,ci,oid],seg[u,ci],rr,oid),
                                    candidate=mask_row(predictions[condition][u,ci,oid],seg[u,ci],rr,oid)))
                            record.update(B0=b,candidate=c,window_t_to_t5=window,
                                immediate_healthy_damage=b['correct_mask_availability']==1 and c['correct_mask_availability']!=1,
                                immediate_correct_recovery=b['correct_mask_availability']==0 and c['correct_mask_availability']==1)
                        anchors.append(record)
                    if not event['detection_requested']:continue
                    rawpath=event.get('raw_query_artifact')
                    if not rawpath:total['redetection_missing_raw_link']+=1;continue
                    with np.load(rawpath) as z:boxes=z['selected_boxes'].copy();scores=z['selected_scores'].copy()
                    peer=event_lookup.get((t,camera,1-oid))
                    unknown_peer=oid==0 and camera=='base_camera' and peer and peer['accepted']
                    peer_bbox=bbox(predictions[condition][t,ci,1-oid])
                    ranks={p['post_rank'] for p in event['proposals']};processed=0
                    target_bbox=bbox(np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])) if exact[t] else None
                    for rank,(box,score) in enumerate(zip(boxes,scores)):
                        clipped=np.clip(box,[0,0,0,0],[128,128,128,128]);area=np.prod(clipped[2:]-clipped[:2])
                        if area<=0 or area>=.25*128*128:reason='AREA'
                        elif processed>=5:reason='MAX5_SAM_CANDIDATES'
                        elif score<.2:reason='SCORE_BELOW_0_2'
                        elif unknown_peer:
                            if rank in ranks:reason='SAM_PROMPTED';processed+=1
                            else:reason='PEER_BEFORE_LATER_CORRECTION_NOT_SAVED'
                        elif peer_bbox is not None and box_iou(clipped,peer_bbox)>=.3:reason='PEER_BBOX_EXCLUSION'
                        else:
                            assert rank in ranks,('unexplained consumer drop',eid,condition,t,camera,oid,rank)
                            reason='SAM_PROMPTED';processed+=1
                        if rank in ranks:assert reason=='SAM_PROMPTED',('consumer trace mismatch',eid,t,rank,reason)
                        drops.append(dict(episode=eid,parent_id=entry['parent_id'],condition=condition,timestep=t,camera=camera,
                            object_id=oid,post_rank=rank,score=float(score),first_consumer_reason=reason,
                            gt_bbox_iou=box_iou(clipped,target_bbox) if target_bbox else None,
                            localized=bool(target_bbox and box_iou(clipped,target_bbox)>=.5),
                            reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                            peer_is_non_graph_Wrist_Box=camera=='hand_camera' and oid==0,
                            box_localization_is_not_semantic_identity=True))
                memories.append(dict(episode=eid,condition=condition,sam_trace=str(path/'sam_calls.json'),
                    sha256=sha(path/'sam_calls.json'),module_trace=str(path/'sam_modules.json'),
                    module_sha256=sha(path/'sam_modules.json'),actual_single_frame_calls=cr['actual_sam_frame_inferences'],
                    exposed_memory_keys_are_at_inference_return_before_caller_commit=True))
        write(root/'dinov2_call_trace.json',dict(episodes=calls,totals={k:dict(v) for k,v in totals.items()},
            baseline_actual_forwards=0,reference_missing_guard_is_explicit=True))
        write(root/'dinov2_reference_provenance.json',reference_rows)
        write(root/'dinov2_reference_contamination_audit.json',dict(references=reference_rows,
            supported_references=sum(r['reference_kind']=='EXACT_GT' for r in reference_rows),
            contaminated_supported_references=sum(r.get('contaminated',False) for r in reference_rows),
            recovery_or_causal_contamination_effect='not inferred from marginal reference purity alone'))
        write(root/'actual_reid_comparisons.json',matching)
        csv_write(root/'dinov2_similarity_distributions.csv',[{k:v for k,v in r.items() if k not in ('evidence','metrics')} for r in matching])
        labelled=[r for r in matching if r.get('intended_positive') is not None]
        csv_write(root/'dinov2_reidentification_metrics.csv',curves(labelled,['camera','object_id']))
        write(root/'sam2_reanchor_event_details.json',anchors)
        csv_write(root/'sam2_reanchor_event_log.csv',[{k:v for k,v in r.items() if k not in
            ('B0','candidate','window_t_to_t5','chosen','trigger_evidence')} for r in anchors])
        csv_write(root/'redetection_consumer_first_drop.csv',drops)
        write(root/'sam2_memory_state_trace.json',memories)
        write(root/'scope.json',dict(status='POST_FORWARD_ACTUAL_CORRECTION_AUDIT_COMPLETE',run_id=logger.run_id,
            **specification,episodes=manifest['episodes'],parents=manifest['parents'],frames=manifest['frames'],
            C_BC_different_mask_frames=differences,actual_matching_comparisons=len(matching),
            matching_sample_cannot_establish_general_ROC=len(labelled)<20,reference_rows=len(reference_rows),inputs=inputs,
            whole_goal_complete=False,files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','inference','references','expected'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.inference,a.references,a.expected)
