"""Recompute frozen Final detector queries, checking original postprocessed bytes.

The observer-light latency run did not serialize original900-query tensors.
These are explicitly recomputed diagnostics, not invented original raw traces.
No candidate, mask, source, threshold or adoption criterion is changed.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
import csv
import h5py
import numpy as np
import torch
from .dino_proposal_audit import pixel_boxes
from .prompt_audit import bbox
from .evaluation_metrics import pixel_metrics
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS
from visual.runtime.initialization import assign,box_iou


def rectangle_support(boxes,target,robot,other):
    """Evaluation-only floor/ceil raster enclosure, never a grounded label."""
    boxes=np.asarray(boxes,float).reshape(-1,4);h,w=target.shape
    x0=np.clip(np.floor(boxes[:,0]).astype(int),0,w);y0=np.clip(np.floor(boxes[:,1]).astype(int),0,h)
    x1=np.clip(np.ceil(boxes[:,2]).astype(int),0,w);y1=np.clip(np.ceil(boxes[:,3]).astype(int),0,h)
    area=np.maximum(0,x1-x0)*np.maximum(0,y1-y0)
    def counts(mask):
        integral=np.pad(np.asarray(mask,np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
        value=integral[y1,x1]-integral[y1,x0]-integral[y0,x1]+integral[y0,x0]
        return np.where(area>0,value,0)
    tp,r,o=counts(target),counts(robot),counts(other);actor=tp+r+o;visible=int(target.sum())
    return [dict(roi_pixels=int(a),target_pixels=int(t),robot_pixels=int(rv),other_object_pixels=int(ov),
        target_fraction=float(t/a) if a else None,nonbackground_target_dominance=float(t/fg) if fg else None,
        target_coverage=float(t/visible) if visible else None) for a,t,rv,ov,fg in zip(area,tp,r,o,actor)]


def run(root,expected,baseline,challenger,references,candidate):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino
    root.mkdir(parents=True,exist_ok=True)
    entries=read(expected)['entries'];lookup={str(e['episode_id']):e for e in entries};calls=[];unique={};inputs=[]
    for condition,source in [('B0',baseline),(candidate,challenger)]:
        assert read(source/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
        for e in entries:
            eid=str(e['episode_id']);folder=source/'perception'/eid;rr=read(folder/'receipt.json')
            assert rr['raw_sha256']==e['raw_sha256'] and all(sha(folder/p)==h for p,h in rr['files'].items())
            rows=read(folder/'detector_timing.json');inputs.append(dict(path=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json')))
            for index,r in enumerate(rows):
                assert r['prompt'] in ('red and white block','cardboard box')
                key=hashlib.sha256((r['input_rgb_sha256']+'|'+r['prompt']).encode()).hexdigest()
                assert r['condition']==condition
                call=dict(r,condition=condition,episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),
                    original_call_index=index,original_trace=str(folder/'detector_timing.json'),trace_sha256=sha(folder/'detector_timing.json'),
                    unique_diagnostic_key=key)
                calls.append(call);unique.setdefault(key,call)
    paths=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json');identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
    assert all(sha(Path(paths['dino'])/p)==h for p,h in identity['dino_files'].items())
    protocol=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),candidate=candidate,
        detector_files=identity['dino_files'],actual_original_calls=len(calls),unique_RGB_prompt_inputs=len(unique),
        original_raw900_tensors_were_not_saved=True,recomputed_raw_queries=True,
        comparison_level='exact original postprocessed boxes/scores; original below-threshold raw logits unavailable for byte comparison',
        GT_labels_after_all_recomputed_forwards=True,new_thresholds_or_prompts=False,not_primary_adoption_data=True)
    frozen=root/'protocol_frozen.json'
    if frozen.exists():assert read(frozen)==protocol
    else:write(frozen,protocol)
    torch.set_num_threads(1);torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_Final_DINO_recomputed_raw_diagnostic',seed=42,
            root=root/'experiments',job_type='evaluation',config=protocol) as logger:
        detector=AuditedDino('cuda',model_id=paths['dino'],output=root/'unused_output');receipts={}
        try:
            for key,r in unique.items():
                folder=root/'unique'/key
                if (folder/'receipt.json').exists():
                    rr=read(folder/'receipt.json');assert rr['protocol_sha256']==sha(frozen)
                    assert all(sha(folder/p)==h for p,h in rr['files'].items());receipts[key]=rr;continue
                if folder.exists():folder.rename(folder.with_name(key+'.partial_'+str(time.time_ns())))
                folder.mkdir(parents=True);e=lookup[r['episode']];assert sha(e['raw_path'])==e['raw_sha256']
                with h5py.File(e['raw_path']) as raw:
                    t=r['decision_timestep'];rgb=raw[r['camera']+'/rgb'][t]
                    if hashlib.sha256(rgb.tobytes()).hexdigest()!=r['input_rgb_sha256']:
                        candidates=[i for i in range(t+1) if hashlib.sha256(raw[r['camera']+'/rgb'][i].tobytes()).hexdigest()==r['input_rgb_sha256']]
                        assert candidates,'original detector image cannot be located in observed prefix'
                        rgb=raw[r['camera']+'/rgb'][candidates[0]]
                    else:candidates=[t]
                detector.output=folder;detector.events=[];detector.context=dict(episode_id=r['episode'],camera=r['camera'],decision_timestep=t)
                boxes,scores=detector(rgb,r['prompt'])
                assert boxes==r['postprocessed_boxes'] and scores==r['postprocessed_scores'],'recomputed detector postprocessing differs; do not use as an original raw proxy'
                event=detector.events[0];write(folder/'event.json',event)
                with np.load(event['raw_path']) as z:
                    assert z['pred_boxes'].shape[1]==z['logits'].shape[1]==900
                    tokens=detector.proc.tokenizer.convert_ids_to_tokens(z['input_ids'][0].tolist())
                    shapes={k:list(z[k].shape) for k in z.files}
                rr=dict(unique_key=key,protocol_sha256=sha(frozen),RGB_sha256=r['input_rgb_sha256'],prompt=r['prompt'],tokens=tokens,
                    shapes=shapes,original_postprocessed_equal=True,recomputed_not_original_raw=True,
                    representative_input_timestep_candidates=candidates,raw_npz=str(Path(event['raw_path'])),
                    files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
                write(folder/'receipt.json',rr);receipts[key]=rr
                write(root/'pipeline_state.json',dict(status='RECOMPUTING_RAW_QUERIES',unique_completed=len(receipts),unique_expected=len(unique),pid=__import__('os').getpid()))
                if len(receipts)%20==0:print('Final diagnostic DINO',len(receipts),'/',len(unique),flush=True)
        finally:detector.close()
        assert len(receipts)==len(unique)
        recomputed_post={}
        for key,receipt in receipts.items():
            with np.load(receipt['raw_npz']) as z:recomputed_post[key]=(z['selected_boxes'].tolist(),z['selected_scores'].tolist())
        for call in calls:
            boxes,scores=recomputed_post[call['unique_diagnostic_key']]
            assert boxes==call['postprocessed_boxes'] and scores==call['postprocessed_scores'],'same-input original calls differ; diagnostic cannot claim equivalence'
        write(root/'all_recomputed_forwards_complete.json',dict(actual_extra_diagnostic_forwards=len(unique),original_calls=len(calls),
            all_original_postprocessed_outputs_exact_equal=True,GT_opened_for_labels=False))
        # Only this second pass opens evaluation actor labels.
        query_fields=['condition','episode','parent_id','stratum','camera','object_id','prompt','decision_timestep','input_timestep',
            'original_call_index','query_index','raw_query_rank','max_token_index','max_token_score','pixel_box','gt_box_iou',
            'roi_actor_support','raw_npz','raw_sha256','recomputed_not_original_raw','graph_eligible','reference_kind']
        totals=[];t0_rows=[];no_call=[];query_count=0
        with (root/'dino_recomputed_raw_queries_audit.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=query_fields);writer.writeheader()
            for e in entries:
                eid=str(e['episode_id']);ref=references/'references'/eid;rr=read(ref/'receipt.json')
                assert rr['raw_sha256']==e['raw_sha256'] and all(sha(ref/p)==h for p,h in rr['files'].items())
                with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                for condition,source in [('B0',baseline),(candidate,challenger)]:
                    ec=[r for r in calls if r['episode']==eid and r['condition']==condition]
                    with np.load(source/'perception'/eid/(condition+'.npz')) as z:pred=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
                    original_prompts=read(source/'perception'/eid/'original_initialization_prompts.json')
                    for camera in CAMERAS:
                        for oid,prompt in [(0,'red and white block'),(1,'cardboard box')]:
                            relevant=[r for r in ec if r['camera']==camera and r['prompt']==prompt]
                            requested={r['decision_timestep'] for r in relevant}
                            no_call.extend(dict(condition=condition,episode=eid,camera=camera,object_id=oid,timestep=t,
                                status='NO_ACTUAL_DETECTOR_FORWARD_NOT_RAW_PROPOSAL_MISSING') for t in range(e['actions']+1) if t not in requested)
                    for r in ec:
                        oid=0 if r['prompt']=='red and white block' else 1;ci=CAMERAS.index(r['camera']);t=r['decision_timestep']
                        with h5py.File(e['raw_path']) as raw:
                            matches=[i for i in range(t+1) if hashlib.sha256(raw[r['camera']+'/rgb'][i].tobytes()).hexdigest()==r['input_rgb_sha256']]
                        assert matches
                        masks=[np.isin(seg[i,ci],rr['object_ids'][('peg','box')[oid]]) for i in matches]
                        supported=all(exact[i] for i in matches) and all(np.array_equal(masks[0],m) for m in masks)
                        input_t=matches[0];target=masks[0];gtbox=bbox(target) if supported else None
                        receipt=receipts[r['unique_diagnostic_key']];path=Path(receipt['raw_npz'])
                        with np.load(path) as z:
                            logits=z['logits'][0].astype(float);boxes=pixel_boxes(z['pred_boxes'][0]);scores=(1/(1+np.exp(-np.clip(logits,-700,700)))).max(-1)
                            max_token=logits.argmax(-1)
                        order=np.argsort(-scores);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
                        ious=np.array([box_iou(b,gtbox) if gtbox else 0 for b in boxes])
                        support=rectangle_support(boxes,target,np.isin(seg[input_t,ci],rr['robot_ids']),
                            np.isin(seg[input_t,ci],rr['object_ids'][('box','peg')[oid]])) if supported else [None]*len(boxes)
                        meta=dict(condition=condition,episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),camera=r['camera'],
                            object_id=oid,prompt=r['prompt'],decision_timestep=t,input_timestep=input_t,original_call_index=r['original_call_index'],
                            raw_npz=str(path),raw_sha256=sha(path),recomputed_not_original_raw=True,
                            graph_eligible=not(ci==1 and oid==1),reference_kind='EXACT_GT' if supported else 'AMBIGUOUS_OR_UNSUPPORTED_INPUT_REFERENCE')
                        for i in range(len(boxes)):
                            writer.writerow(dict(**meta,query_index=i,raw_query_rank=int(rank[i]),max_token_index=int(max_token[i]),
                                max_token_score=float(scores[i]),pixel_box=boxes[i].tolist(),gt_box_iou=float(ious[i]) if gtbox else None,
                                roi_actor_support=support[i]));query_count+=1
                        totals.append(dict(**meta,visible_pixels=int(target.sum()) if supported else None,raw_localized_queries=int((ious>=.5).sum()) if supported else None,
                            raw_query_count=len(boxes),postprocessed_count=len(r['postprocessed_boxes']),scores_are_not_calibrated=True))
                    for ci,camera in enumerate(CAMERAS):
                        pair={}
                        for oid,prompt in [(0,'red and white block'),(1,'cardboard box')]:
                            t0=[r for r in ec if r['camera']==camera and r['prompt']==prompt and r['decision_timestep']==0]
                            assert len(t0)==1,'t0 initialization audit requires exactly one actual call per phrase/camera'
                            r=t0[0];post=list(zip(r['postprocessed_boxes'],r['postprocessed_scores']))
                            area=[(b,s) for b,s in post if (b[2]-b[0])*(b[3]-b[1])<.25*128*128]
                            pair[oid]=dict(call=r,post=post,area=area,top5=area[:5])
                        chosen=assign(pair[0]['top5'],pair[1]['top5'])
                        for oid in (0,1):
                            p=pair[oid];r=p['call'];target=np.isin(seg[0,ci],rr['object_ids'][('peg','box')[oid]]);gtbox=bbox(target)
                            prompt=next((v for v in original_prompts if v['camera']==camera and v['object_id']==oid and v['timestep']==0),None)
                            actual=chosen[oid][0] if chosen[oid] else None
                            assert (prompt is None)==(actual is None)
                            if prompt is not None:np.testing.assert_array_equal(prompt['box'],actual)
                            with np.load(receipts[r['unique_diagnostic_key']]['raw_npz']) as z:rawboxes=pixel_boxes(z['pred_boxes'][0])
                            localized=lambda seq:any(gtbox is not None and box_iou(b,gtbox)>=.5 for b in seq)
                            good=lambda seq:localized([b for b,s in seq])
                            if not exact[0] or target.sum()<8:drop='NOT_EXACT_GT_VISIBLE'
                            elif not localized(rawboxes):drop='RECOMPUTED_RAW_NO_LOCALIZED_QUERY'
                            elif not good(p['post']):drop='ORIGINAL_POSTPROCESS_THRESHOLD'
                            elif not good(p['area']):drop='ORIGINAL_AREA_FILTER'
                            elif not good(p['top5']):drop='ORIGINAL_TOP5'
                            elif actual is None:drop='ORIGINAL_JOINT_NULL'
                            elif box_iou(actual,gtbox)<.5:drop='ORIGINAL_JOINT_SELECTS_DIFFERENT_LOCALIZATION'
                            else:drop='LOCALIZED_PROMPT_ADOPTED_SEMANTIC_CAUSE_UNRESOLVED'
                            m=pixel_metrics(pred[0,ci,oid],target,np.isin(seg[0,ci],rr['robot_ids']),np.isin(seg[0,ci],rr['object_ids'][('box','peg')[oid]])) if exact[0] else {}
                            if m.get('correct_mask_availability')==1:operational='NONE_T0_CORRECT_MASK'
                            elif drop=='LOCALIZED_PROMPT_ADOPTED_SEMANTIC_CAUSE_UNRESOLVED':operational='LOCALIZED_PROMPT_BUT_INCORRECT_SAM_MASK_NOT_PROVEN_MODEL_ONLY'
                            else:operational=drop
                            t0_rows.append(dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),condition=condition,camera=camera,object_id=oid,
                                graph_eligible=not(ci==1 and oid==1),reference_kind='EXACT_GT' if exact[0] else 'GT_UNSUPPORTED',
                                localization_first_drop=drop,operational_first_failure=operational,chosen_box=actual,
                                chosen_rectangle_GT_support=rectangle_support([actual],target,np.isin(seg[0,ci],rr['robot_ids']),
                                    np.isin(seg[0,ci],rr['object_ids'][('box','peg')[oid]]))[0] if actual is not None and exact[0] else None,
                                actual_SAM_metrics=m,semantic_identity_not_proven_by_box_IoU=True,recomputed_raw_proxy=True))
        csv_write(root/'dino_recomputed_call_accounting.csv',totals);csv_write(root/'dino_not_called_frames.csv',no_call)
        write(root/'dino_independent_t0_assignment_cases.json',t0_rows)
        count=Counter((r['condition'],r['camera'],r['object_id'],r['operational_first_failure']) for r in t0_rows)
        csv_write(root/'dino_independent_t0_restriction_funnel.csv',[dict(condition=c,camera=cam,object_id=o,reason=d,count=n) for (c,cam,o,d),n in sorted(count.items())])
        write(root/'scope.json',dict(status='FINAL_RECOMPUTED_DINO_RAW_DIAGNOSTIC_COMPLETE',run_id=logger.run_id,**protocol,
            original_input_receipts=inputs,actual_calls=len(calls),unique_extra_forwards=len(unique),query_rows=query_count,
            missing_forward_frames_not_counted_as_missing_raw_proposal=True,original_postprocessed_outputs_all_exact=True,
            no_original_below_threshold_raw_byte_equivalence_claim=True,GT_labels_only_after_all_recomputed_forwards=True,
            no_candidate_or_registered_metric_changed=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ('pipeline_state.json',)}))
        write(root/'pipeline_state.json',dict(status='FINAL_RECOMPUTED_DINO_RAW_DIAGNOSTIC_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','expected','baseline','challenger','references'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
