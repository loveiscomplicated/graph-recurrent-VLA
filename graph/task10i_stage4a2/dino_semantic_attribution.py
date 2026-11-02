"""Historical same-forward first-drop and actor-support audit, evaluation only.

Box overlap is reported separately from actor support and actual SAM semantics.
Changing a null/exclusion constant here is a labelled oracle diagnostic, never
a practical improvement. No new model forward or validation access occurs.
"""
import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from .dino_proposal_audit import pixel_boxes
from .evaluation_metrics import pixel_metrics
from .dinov2_matching_audit import csv_write
from .prompt_audit import AUDITS, bbox
from visual.runtime.initialization import assign, box_iou
from graph.task10i_stage4a.run_pipeline import read, write, sha, CAMERAS


def counterfactual_assign(peg, box, missing=.3, exclusion=.3):
    best=(None,None);score=2*missing
    for a,b in product([*peg,None],[*box,None]):
        if a is not None and b is not None and box_iou(a[0],b[0])>=exclusion:continue
        value=(a[1] if a else missing)+(b[1] if b else missing)
        if value>score:best,score=(a,b),value
    return best


def actor_rectangles(boxes, seg, reference):
    """Vectorized rectangle sums; raster rule matches the saved crop audit."""
    h,w=seg.shape
    low=np.clip(np.floor(boxes[:,:2]).astype(int),0,[w,h])
    high=np.clip(np.ceil(boxes[:,2:]).astype(int),0,[w,h])
    counts=[];visible=[];bounds=[]
    for name in ('peg','box','robot'):
        mask=np.isin(seg,reference['robot_ids'] if name=='robot' else reference['object_ids'][name])
        integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        a,b=low[:,0],low[:,1];c,d=high[:,0],high[:,1]
        count=integral[d,c]-integral[b,c]-integral[d,a]+integral[b,a]
        count=np.where((c>a)&(d>b),count,0)
        counts.append(count);visible.append(int(mask.sum()));bounds.append(bbox(mask))
    counts=np.stack(counts,axis=1);nonbackground=counts.sum(axis=1)
    winner=counts.argmax(axis=1);best=counts[np.arange(len(boxes)),winner]
    dominance=np.divide(best,nonbackground,out=np.zeros(len(boxes),float),where=nonbackground>0)
    labels=[];details=[]
    for i,actor in enumerate(winner):
        name=('peg','box','robot')[actor]
        label=None;reason='insufficient_actor_pixels'
        coverage=best[i]/visible[actor] if visible[actor] else None
        rect=np.r_[low[i],high[i]].tolist()
        iou=box_iou(rect,bounds[actor]) if bounds[actor] else None
        if nonbackground[i]==0:label='background';reason='no_actor_support'
        elif nonbackground[i]>=6:
            if dominance[i]<.8:reason='mixed_nonbackground_actors'
            elif actor==2:label='robot';reason='dominant_robot'
            elif visible[actor]>=8 and coverage>=.5 and iou>=.3:label=name;reason='localized_dominant_actor'
            else:reason='insufficient_actor_localization'
        labels.append(label)
        details.append(dict(actor_label=label,actor_counts=counts[i].tolist(),nonbackground_dominance=float(dominance[i]),
                            actor_coverage=coverage,rectangle_actor_bbox_iou=iou,reason=reason))
    return labels,details


def first_drop(raw, threshold, area, top, chosen_good, chosen_null):
    if not np.any(raw):return 'NO_RAW_SUPPORTED_LOCALIZATION'
    if not np.any(raw&threshold):return 'SCORE_THRESHOLD'
    if not np.any(raw&threshold&area):return 'AREA_FILTER'
    if not np.any(raw&threshold&area&top):return 'TOP5'
    if chosen_good:return 'ADOPTED_SUPPORTED_LOCALIZATION'
    return 'JOINT_NULL_SELECTION' if chosen_null else 'JOINT_SELECTS_DIFFERENT_CANDIDATE'


def run(root):
    from utils.experiment_logger import ExperimentLogger
    from transformers import AutoTokenizer
    import inspect
    import visual.runtime.initialization as initialization
    import visual.runtime.grounding as grounding
    root.mkdir(parents=True,exist_ok=False)
    original=Path('/workspace/task10i_stage4a_initialization_bottlenecks_20261101_v1/run/t0_reconstruction_rows.json')
    rows=read(original)
    runtime=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    tokenizer=AutoTokenizer.from_pretrained(runtime['dino'],local_files_only=True)
    config=dict(source_sha256=sha(Path(__file__)),input_sha256=sha(original),new_model_forwards=0,
        partition='historical_Development24',box_localization_threshold=.5,
        actor_label='>=6 nonbackground actor pixels, >=.8 actor dominance; Peg/Box additionally visible>=8, coverage>=.5, rounded rectangle actor bbox IoU>=.3',
        actor_label_is_not_model_phrase_identity=True,GT_posthoc_only=True,thresholds_selected=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_DINO_semantic_first_drop',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        by_episode={};raw_cache={};groups=defaultdict(dict);inputs=[];query_rows=[]
        for r in rows:
            eid=str(r['episode']);ci=CAMERAS.index(r['camera']);oid=r['object_id'];name=('peg','box')[oid]
            if eid not in by_episode:
                cr=read(AUDITS/'corrected/perception'/eid/'receipt.json');ref=Path(cr['corrected_reference'])
                assert sha(ref)==cr['corrected_reference_sha256'];rr=read(ref)
                assert sha(ref.parent/'segmentation.npz')==rr['segmentation_sha256']
                with np.load(ref.parent/'segmentation.npz') as z:seg=z['actor_segmentation'][0].copy();exact=bool(z['exact_frame'][0])
                masks=AUDITS/'baseline/perception'/eid/'masks.npz';assert sha(masks)==cr['masks_sha256']
                with np.load(masks) as z:pred=np.unpackbits(z['mask_bits'][0]).reshape(2,2,128,128).astype(bool)
                by_episode[eid]=(rr,seg,exact,pred)
                inputs.append(dict(reference=str(ref),sha256=sha(ref),masks=str(masks),masks_sha256=sha(masks)))
            rr,seg,exact,pred=by_episode[eid]
            assert exact and r['reference_kind']=='EXACT_GT'
            target=np.isin(seg[ci],rr['object_ids'][name]);assert int(target.sum())==r['raw_gt_pixels']
            path=Path(r['cached_forward_source'])
            if str(path) not in raw_cache:
                with np.load(path) as z:
                    logits=z['logits'][0].astype(np.float64);boxes=pixel_boxes(z['pred_boxes'][0])
                    scores=(1/(1+np.exp(-np.clip(logits,-700,700)))).max(axis=1)
                    raw_cache[str(path)]=dict(boxes=boxes,scores=scores,post_boxes=z['selected_boxes'].tolist(),
                        post_scores=z['selected_scores'].tolist(),tokens=tokenizer.convert_ids_to_tokens(z['input_ids'][0].tolist()),
                        token_argmax=logits.argmax(axis=1).tolist(),raw_sha256=sha(path),shapes={k:list(z[k].shape) for k in z.files})
            data=raw_cache[str(path)];boxes=data['boxes'];scores=data['scores']
            labels,details=actor_rectangles(boxes,seg[ci],rr)
            localized=np.array([r['gt_box'] is not None and box_iou(b,r['gt_box'])>=.5 for b in boxes])
            supported=localized&np.array([a==name for a in labels])
            area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)<.25*128*128
            # Map actual HF-postprocessed output to original query IDs, preserving
            # its sort order rather than inventing a differently rounded ranking.
            post_ids=[]
            for b,s in zip(data['post_boxes'],data['post_scores']):
                ids=np.flatnonzero(np.max(np.abs(boxes-np.asarray(b)),axis=1)<1e-4)
                ids=[i for i in ids if abs(scores[i]-s)<1e-6];assert ids
                post_ids.append(int(ids[0]))
            threshold=np.zeros(len(boxes),bool);threshold[post_ids]=True
            assert np.array_equal(threshold,scores>.1),'installed threshold contract differs'
            eligible_ids=[i for i in post_ids if area[i]][:5]
            top=np.zeros(len(boxes),bool);top[eligible_ids]=True
            eligible=[(data['post_boxes'][j],data['post_scores'][j]) for j,i in enumerate(post_ids) if i in eligible_ids]
            chosen_ids=[] if r['chosen_box'] is None else np.flatnonzero(np.max(np.abs(boxes-np.asarray(r['chosen_box'])),axis=1)<1e-4).tolist()
            chosen_good=any(supported[i] for i in chosen_ids)
            m=pixel_metrics(pred[ci,oid],target,np.isin(seg[ci],rr['robot_ids']),
                            np.isin(seg[ci],rr['object_ids'][('box','peg')[oid]]))
            record=dict(episode=eid,parent_id=r['parent_id'],camera=r['camera'],object_id=oid,
                graph_eligible=not(ci==1 and oid==1),visible_pixels=int(target.sum()),
                reference_kind='EXACT_GT',cached_forward_source=str(path),raw_sha256=data['raw_sha256'],
                actual_initialization_timestep=r['actual_initialization_timestep'],
                chosen_box=r['chosen_box'],chosen_score=r['chosen_score'],
                chosen_query_ids=chosen_ids,chosen_actor_support=[details[i] for i in chosen_ids],
                chosen_GT_box_iou=r['chosen_gt_iou'],chosen_supported_localization=bool(chosen_good),
                raw_localization_count=int(localized.sum()),raw_supported_localization_count=int(supported.sum()),
                raw_localized_but_not_actor_supported_count=int((localized&~supported).sum()),
                threshold_supported_count=int((supported&threshold).sum()),
                area_supported_count=int((supported&threshold&area).sum()),
                top5_supported_count=int((supported&top).sum()),
                supported_top5_all_scores_at_most_null=bool(np.any(supported&top) and np.max(scores[supported&top])<=.3),
                localization_only_first_drop=first_drop(localized,threshold,area,top,any(localized[i] for i in chosen_ids),r['chosen_box'] is None),
                actor_supported_first_drop=first_drop(supported,threshold,area,top,chosen_good,r['chosen_box'] is None),
                actual_SAM_mask_metrics=m,semantic_identity_from_box_alone_confirmed=False)
            if not target.any():
                record['localization_only_first_drop']=record['actor_supported_first_drop']='NOT_GT_VISIBLE'
            if chosen_good:
                record['downstream_first_failure']='SAM_MASK_NOT_CORRECT' if m['correct_mask_availability']!=1 else 'NONE_T0_CORRECT_MASK'
            else:record['downstream_first_failure']='INITIAL_SELECTION_OR_UNRESOLVED_ACTOR_SUPPORT'
            groups[eid,r['camera']][oid]=(record,eligible,supported,boxes)
            order=np.argsort(-scores);rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
            for i in range(len(boxes)):
                query_rows.append(dict(episode=eid,parent_id=r['parent_id'],camera=r['camera'],object_id=oid,
                    query_index=i,query_rank=int(rank[i]),max_token=data['tokens'][data['token_argmax'][i]],
                    score=float(scores[i]),localized=bool(localized[i]),actor_supported_localized=bool(supported[i]),
                    actor_label=labels[i],actor_counts=details[i]['actor_counts'],actor_label_reason=details[i]['reason'],
                    threshold_survives=bool(threshold[i]),area_eligible=bool(area[i]),top5_eligible=bool(top[i]),
                    chosen_query=i in chosen_ids))
        records=[];counterfactuals=[]
        for key,pair in groups.items():
            assert set(pair)=={0,1}
            original_pair=assign(pair[0][1],pair[1][1])
            for oid in (0,1):
                actual=original_pair[oid][0] if original_pair[oid] else None
                assert actual==pair[oid][0]['chosen_box']
            for intervention,missing,exclusion in [('B0',.3,.3),('ORACLE_NULL_ZERO',0.,.3),
                    ('ORACLE_NO_EXCLUSION',.3,float('inf')),('ORACLE_NULL_ZERO_NO_EXCLUSION',0.,float('inf'))]:
                selected=counterfactual_assign(pair[0][1],pair[1][1],missing,exclusion)
                for oid in (0,1):
                    r,eligible,supported,boxes=pair[oid];choice=selected[oid]
                    ids=[] if choice is None else np.flatnonzero(np.max(np.abs(boxes-np.asarray(choice[0])),axis=1)<1e-4).tolist()
                    counterfactuals.append(dict(episode=key[0],camera=key[1],object_id=oid,parent_id=r['parent_id'],
                        intervention=intervention,chosen_box=choice[0] if choice else None,
                        chosen_score=choice[1] if choice else None,
                        actor_supported_localization=any(supported[i] for i in ids),
                        null_selected=choice is None,GT_diagnostic_only=True))
            for oid in (0,1):
                r,eligible,supported,boxes=pair[oid];peer=pair[1-oid][0]['chosen_box']
                good_indices=np.flatnonzero(supported)
                r['top5_supported_candidates_overlap_selected_peer']=[
                    dict(box=b,score=s,peer_bbox_iou=box_iou(b,peer) if peer else None)
                    for b,s in eligible if any(np.allclose(b,boxes[i],atol=1e-4,rtol=0) for i in good_indices)]
                records.append(r)
        counts=Counter((r['camera'],r['object_id'],r['actor_supported_first_drop']) for r in records if r['visible_pixels']>=8)
        write(root/'dino_assignment_error_cases.json',records)
        csv_write(root/'dino_candidate_restriction_funnel.csv',[dict(camera=c,object_id=o,first_drop=d,count=n,
            graph_eligible=not(c=='hand_camera' and o==1)) for (c,o,d),n in sorted(counts.items())])
        csv_write(root/'dino_same_output_assignment_counterfactual.csv',counterfactuals)
        csv_write(root/'dino_raw_query_actor_support.csv',query_rows)
        write(root/'raw_query_token_identity.json',{p:{k:v for k,v in data.items() if k in
            ('tokens','raw_sha256','shapes')} for p,data in raw_cache.items()})
        write(root/'source_line_identity.json',dict(files={str(Path(m.__file__)):sha(Path(m.__file__)) for m in (initialization,grounding)},
            candidates_line=inspect.getsourcelines(initialization.candidates)[1],assign_line=inspect.getsourcelines(initialization.assign)[1],
            grounded_forward_line=inspect.getsourcelines(grounding.GDino.__call__)[1],
            baseline_already_uses_HF_grounded_postprocess=True,baseline_discards_postprocessed_labels=True,
            null_is_joint_alternative_not_independent_t0_threshold=True))
        write(root/'scope.json',dict(status='HISTORICAL_SAME_OUTPUT_SEMANTIC_SUPPORT_ATTRIBUTION_COMPLETE',
            run_id=logger.run_id,**config,episodes=len(by_episode),parents=len({r['parent_id'] for r in records}),
            rows=len(records),query_rows=len(query_rows),unique_raw_artifacts=len(raw_cache),inputs=inputs,
            original_assignments_reproduced=True,box_actor_support_is_not_model_semantic_proof=True,
            mixed_rectangles_retained=True,SAM_semantics_measured_separately=True,
            counterfactuals_are_NOT_practical_benefits=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
