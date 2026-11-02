"""Actual frozen live Re-ID and reference outcomes, labelled after inference."""
import argparse
from pathlib import Path
import numpy as np
from .evaluation_metrics import pixel_metrics
from .correction_diagnostics import identity_label
from .reference_pair_audit import curves
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def evaluate(mask,seg,rr,oid):
    return pixel_metrics(mask,np.isin(seg,rr['object_ids'][('peg','box')[oid]]),
        np.isin(seg,rr['robot_ids']),np.isin(seg,rr['object_ids'][('box','peg')[oid]]))


def run(root,expected,baseline,challenger,references,candidate):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    for source in (baseline,challenger):assert read(source/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),candidate=candidate,
        GT_labels_after_complete_inference=True,threshold_selected=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_frozen_actual_ReID_outcomes',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        refs=[];pairs=[];events=[];call_rows=[];handoffs=[];inputs=[]
        for e in read(expected)['entries']:
            eid=str(e['episode_id']);folder=challenger/'perception'/eid;br=baseline/'perception'/eid
            receipt=read(folder/'receipt.json');base=read(br/'receipt.json')
            assert all(sha(folder/p)==h for p,h in receipt['files'].items())
            assert all(sha(br/p)==h for p,h in base['files'].items())
            with np.load(folder/'internal_B0.npz') as a,np.load(br/'B0.npz') as b:
                assert np.array_equal(a['mask_bits'],b['mask_bits']);b0=np.unpackbits(b['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            with np.load(folder/(candidate+'.npz')) as z:prediction=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            ref=references/'references'/eid;rr=read(ref/'receipt.json')
            assert all(sha(ref/p)==h for p,h in rr['files'].items())
            with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            assert rr['raw_sha256']==e['raw_sha256']==receipt['raw_sha256']==base['raw_sha256']
            bank=read(folder/'reference_provenance.json');lookup={(r['camera'],r['object_id']):r for r in bank}
            for r in bank:
                t=r['timestep'];oid=r['object_id'];ci=CAMERAS.index(r['camera'])
                vector=np.asarray(r['embedding'],np.float32);assert vector.shape==(384,) and abs(np.linalg.norm(vector)-1)<1e-4
                record={k:v for k,v in r.items() if k not in ('embedding','mask_bits')}
                record.update(parent_id=e['parent_id'],stratum=e.get('stratum','internal'),reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                    source_provenance=str(folder/'reference_provenance.json'),source_sha256=sha(folder/'reference_provenance.json'),
                    embedding_dim=384,embedding_normalized=True)
                if exact[t]:
                    mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                    m=evaluate(mask,seg[t,ci],rr,oid)
                    record.update(metrics=m,actual_actor=identity_label(m),contaminated=m['precision']<.8,
                        incomplete_reference_recall_below_half=m['recall'] is not None and m['recall']<.5)
                refs.append(record)
            candidates={(r['timestep'],r['camera'],r['object_id'],r['post_rank']):r for r in read(folder/'candidate_masks.json')}
            def match_record(event,prop,mask,kind):
                t=event['timestep'];c=event['camera'];oid=event['object_id'];reference=lookup[c,oid]
                assert reference['timestep']<t,'noncausal reference comparison'
                chosen=event['accepted'] and (kind=='later_original_initialization' or event.get('chosen',{}).get('post_rank')==prop.get('post_rank'))
                row=dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),kind=kind,
                    timestep=t,camera=c,object_id=oid,reference_timestep=reference['timestep'],
                    similarity=prop['similarity'],negative_similarity=prop['negative_similarity'],reid_pass=prop['reid_pass'],
                    accepted=chosen,event_accepted=event['accepted'],reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                if exact[t]:
                    m=evaluate(mask,seg[t,CAMERAS.index(c)],rr,oid);label=identity_label(m)
                    row.update(metrics=m,actual_actor=label,
                        intended_positive=True if label=='TARGET_ACTOR' else False if label in ('ROBOT','WRONG_OBJECT','BACKGROUND') else None)
                pairs.append(row)
            policy=read(folder/'events.json');hand=read(folder/'baseline_handoff_events.json')
            for event in policy:
                t=event['timestep'];camera=event['camera'];ci=CAMERAS.index(camera);oid=event['object_id']
                if event['detection_requested']:
                    r=dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),
                        **{k:event[k] for k in ('timestep','camera','object_id','ever_found','currently_valid','accepted','reference_available')},
                        proposals=len(event['proposals']),actual_comparisons=sum(p['similarity'] is not None for p in event['proposals']),
                        source_event_file=str(folder/'events.json'),source_sha256=sha(folder/'events.json'))
                    if exact[t]:
                        bm=evaluate(b0[t,ci,oid],seg[t,ci],rr,oid);cm=evaluate(prediction[t,ci,oid],seg[t,ci],rr,oid)
                        r.update(B0_availability=bm['correct_mask_availability'],candidate_availability=cm['correct_mask_availability'],
                            immediate_healthy_damage=bm['correct_mask_availability']==1 and cm['correct_mask_availability']!=1)
                    events.append(r)
                for prop in event['proposals']:
                    if prop['similarity'] is None:continue
                    raw=candidates[t,camera,oid,prop['post_rank']]
                    mask=np.unpackbits(np.asarray(raw['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                    match_record(event,prop,mask,'redetection_candidate')
            for event in hand:
                t=event['timestep'];ci=CAMERAS.index(event['camera']);oid=event['object_id']
                r=dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),**{k:v for k,v in event.items() if k!='episode'})
                if exact[t]:r['candidate_GT_metrics']=evaluate(b0[t,ci,oid],seg[t,ci],rr,oid)
                handoffs.append(r)
                if event['similarity'] is not None:match_record(event,event,b0[t,ci,oid],'later_original_initialization')
            feature=read(folder/'feature_calls.json');api=read(folder/'feature_total_timing.json')
            call_rows.append(dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),
                actual_model_forwards=sum(r['model_forward'] for r in feature),feature_API_requests=len(api),
                empty_or_patch_support_rejections=sum(not r['valid_embedding'] for r in api),
                actual_cosine_comparisons=sum(r['episode']==eid for r in pairs),
                actual_trace=str(folder/'feature_calls.json'),trace_sha256=sha(folder/'feature_calls.json')))
            inputs.append(dict(candidate_receipt=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
                reference_receipt=str(ref/'receipt.json'),reference_sha256=sha(ref/'receipt.json')))
        write(root/'dinov2_call_trace.json',call_rows)
        write(root/'dinov2_reference_provenance.json',refs)
        write(root/'dinov2_reference_contamination_audit.json',dict(references=refs,
            contaminated_exact=sum(r.get('contaminated',False) for r in refs),
            exact_supported=sum(r['reference_kind']=='EXACT_GT' for r in refs),
            partial_pure_references_are_not_wrong_actor_contamination=True))
        write(root/'dinov2_actual_positive_negative_pairs.json',pairs)
        csv_write(root/'dinov2_positive_negative_pairs.csv',[{k:v for k,v in r.items() if k!='metrics'} for r in pairs])
        csv_write(root/'dinov2_reidentification_metrics.csv',curves([r for r in pairs if r.get('intended_positive') is not None],['stratum','camera','object_id','kind']))
        csv_write(root/'sam2_reanchor_event_log.csv',events);write(root/'baseline_handoff_events.json',handoffs)
        write(root/'scope.json',dict(status='FROZEN_LIVE_MATCHING_OUTCOMES_LABELLED',run_id=logger.run_id,**config,inputs=inputs,
            actual_comparisons=len(pairs),references=len(refs),GT_never_used_to_select_a_practical_event=True,
            sparse_matching_groups_not_population_ROC=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','expected','baseline','challenger','references'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args();run(a.root,a.expected,a.baseline,a.challenger,a.references,a.candidate)
