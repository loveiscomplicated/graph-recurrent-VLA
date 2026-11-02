"""Post-forward reference/candidate audit using immutable saved DINOv2 vectors.

No model forward or practical decision is modified. Actor labels for rectangular
crops separate background area from semantic identity. Same-time diagnostics are
kept separate from strictly past-reference comparisons; empty references remain
missing. Baseline had no DINOv2 bank, so B0-t0 references are counterfactual only.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .crop_feature_audit import MODES
from .dinov2_matching_audit import csv_write
from .prompt_audit import inputs,AUDITS
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS
from visual.runtime.initialization import box_iou


def rectangle_identity(mask,seg,rr):
    """Evaluation label only; no GT-based acceptance or crop is produced."""
    counts={name:int((mask&np.isin(seg,rr['robot_ids'] if name=='robot' else rr['object_ids'][name])).sum())
        for name in ('peg','box','robot')}
    nonbackground=sum(counts.values());size=int(mask.sum())
    if nonbackground<6:
        return ('background' if nonbackground==0 else None),dict(counts=counts,reason='insufficient_actor_pixels')
    actor=max(counts,key=counts.get);dominance=counts[actor]/nonbackground
    if dominance<.8:return None,dict(counts=counts,reason='mixed_nonbackground_actors',dominance=dominance)
    if actor=='robot':return actor,dict(counts=counts,dominance=dominance,reason='dominant_robot_pixels')
    target=np.isin(seg,rr['object_ids'][actor]);visible=int(target.sum())
    def bounds(m):
        y,x=np.nonzero(m);return [float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)]
    coverage=counts[actor]/visible if visible else None
    iou=box_iou(bounds(mask),bounds(target)) if visible else None
    good=visible>=8 and coverage>=.5 and iou>=.3
    return actor if good else None,dict(counts=counts,dominance=dominance,actor_visible_pixels=visible,
        actor_coverage=coverage,actor_bbox_iou=iou,reason='localized_dominant_actor' if good else 'insufficient_actor_localization')


def curves(rows,keys,positive_key='intended_positive'):
    groups=defaultdict(list);result=[]
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    for group,rr in groups.items():
        score=np.asarray([r['similarity'] for r in rr]);positive=np.asarray([r[positive_key] for r in rr],bool)
        p=int(positive.sum());n=len(rr)-p
        for threshold in np.unique(np.r_[np.linspace(-1,1,201),.6]):
            accept=score>=threshold;tp=int((accept&positive).sum());fp=int((accept&~positive).sum())
            result.append(dict(zip(keys,group),threshold=float(threshold),positives=p,negatives=n,
                tp=tp,fp=fp,tpr=tp/p if p else None,fpr=fp/n if n else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/p if p else None,
                false_rejection_rate=1-tp/p if p else None))
    return result


def run(root,features,causal):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source_sha256=sha(Path(__file__)),stored_embeddings_sha256=sha(features/'embeddings.npz'),
        provenance_sha256=sha(features/'dinov2_reference_provenance.json'),
        label_protocol='rectangles: >=6 actor pixels, >=80% dominance among nonbackground; Peg/Box also GT-visible>=8, actor coverage>=.5 and bbox IoU>=.3; robot/background negatives separately',
        positive_definition='candidate is the intended physical Peg/Box actor in the same parent; actual reference contamination is retained, not relabelled as successful robot identity matching',
        threshold_selected=False,practical_forward_changed=False,baseline_DINOv2_active=False,
        original_diagnostic_preserved=True,independent_test=False)
    write(root/'registered_protocol.json',config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_reference_pair_label_audit',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        metadata=read(features/'dinov2_reference_provenance.json')
        with np.load(features/'embeddings.npz') as z:vectors=z['embeddings'].copy()
        assert len(metadata)==len(vectors) and vectors.shape[1]==384
        events=read('/workspace/task10i_stage4a2_controls_20261101_v1/prompt/dino_events.json')
        raw={}
        for event in events:
            if event['prompt'] in ('red and white block','cardboard box'):
                raw[str(event['episode_id']),event['camera'],'peg' if event['prompt']=='red and white block' else 'box']=event
        chosen=inputs();by_episode={str(e['episode_id']):(e,receipt) for e,receipt in chosen.values()}
        reference_sources=[];labels=[];cached={}
        def gt(eid):
            if eid not in cached:
                e,r=by_episode[eid];path=Path(r['corrected_reference']);rr=read(path)
                assert sha(path)==r['corrected_reference_sha256']
                with np.load(path.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                cached[eid]=(rr,seg,exact)
            return cached[eid]
        for m in metadata:
            label=m['actual_actor'];detail=dict(reason='original_mask_semantic_purity_rule',target_purity=m['target_purity'])
            if m['branch']=='ACTUAL_DINO_BOX_CROP':
                rr,seg,exact=gt(m['episode']);assert m['timestep']==0 and exact[0]
                event=raw[m['episode'],m['camera'],m['requested']];path=Path(event['raw_path'])
                with np.load(path) as z:box=z['selected_boxes'][m['post_rank']]
                low=np.maximum(np.floor(box[:2]).astype(int),0);high=np.minimum(np.ceil(box[2:]).astype(int),128)
                mask=np.zeros((128,128),bool);mask[low[1]:high[1],low[0]:high[0]]=True
                assert int(mask.sum())==m['mask_pixels']
                label,detail=rectangle_identity(mask,seg[0,CAMERAS.index(m['camera'])],rr)
                reference_sources.append(dict(path=str(path),sha256=sha(path)))
            labels.append(dict(**m,original_actor_label=m['actual_actor'],actor_label_v2=label,label_detail=detail))
        references=[]
        availability=[]
        for parent,(e,receipt) in chosen.items():
            path=AUDITS/'baseline/perception'/str(e['episode_id'])/'masks.npz'
            assert sha(path)==receipt['masks_sha256']
            with np.load(path) as z:predicted=np.unpackbits(z['mask_bits'][0]).reshape(2,2,128,128).astype(bool)
            for camera,oid in (('base_camera',0),('hand_camera',0),('base_camera',1)):
                for mode in MODES:
                    exists=any(m['parent_id']==parent and m['camera']==camera and m['requested']==('peg','box')[oid]
                        and m['timestep']==0 and m['mode']==mode and m['branch']=='PREDICTION_CROP_DIAGNOSTIC' for m in labels)
                    pixels=int(predicted[CAMERAS.index(camera),oid].sum())
                    availability.append(dict(parent_id=parent,camera=camera,object_id=oid,mode=mode,
                        origin='HYPOTHETICAL_B0_T0',reference_available=exists,predicted_pixels=pixels,
                        unavailable_reason=None if exists else 'empty_prediction' if pixels==0 else 'feature_extraction_support_rejection'))
        for m in labels:
            if m['timestep']!=0 or m['requested'] not in ('peg','box'):continue
            if m['camera']=='hand_camera' and m['requested']=='box':continue
            if m['branch'] not in ('ORACLE_DIAGNOSTIC','PREDICTION_CROP_DIAGNOSTIC'):continue
            references.append(dict(parent_id=m['parent_id'],episode=m['episode'],camera=m['camera'],
                timestep=0,target=m['requested'],mode=m['mode'],vector=vectors[m['index']],
                origin='ORACLE_IDEAL_T0' if m['branch']=='ORACLE_DIAGNOSTIC' else 'HYPOTHETICAL_B0_T0',
                target_purity=m['target_purity'],actual_actor=m['actor_label_v2'],stored_embedding_index=m['index']))
        for path in sorted((causal/'episodes').glob('*/BC/reference_provenance.json')):
            reference_sources.append(dict(path=str(path),sha256=sha(path)))
            for r in read(path):
                if r['episode'] not in by_episode:continue
                rr,seg,exact=gt(r['episode']);t=r['timestep'];assert exact[t]
                target=('peg','box')[r['object_id']];mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                purity=float(np.isin(seg[t,CAMERAS.index(r['camera'])],rr['object_ids'][target])[mask].mean())
                vector=np.asarray(r['embedding'],np.float32)
                assert vector.shape==(384,) and abs(np.linalg.norm(vector)-1)<1e-4
                references.append(dict(parent_id=by_episode[r['episode']][0]['parent_id'],episode=r['episode'],
                    camera=r['camera'],timestep=t,target=target,mode='full_patch448',vector=vector,
                    origin='ACTUAL_BC_FROZEN_REFERENCE',target_purity=purity,
                    actual_actor=target if purity>=.8 else None,stored_embedding_index=None))
        pairs=[];skipped=Counter()
        for reference in references:
            for m in labels:
                if m['parent_id']!=reference['parent_id'] or m['mode']!=reference['mode']:continue
                if m['branch']=='ORACLE_DIAGNOSTIC':continue
                if m['actor_label_v2'] is None:skipped['unknown_candidate_actor']+=1;continue
                if m['timestep']<reference['timestep']:skipped['future_reference_prohibited']+=1;continue
                time_relation='PAST_REFERENCE' if reference['timestep']<m['timestep'] else 'SIMULTANEOUS_DIAGNOSTIC'
                if time_relation=='SIMULTANEOUS_DIAGNOSTIC' and reference['camera']==m['camera']:
                    skipped['same_image_reference_not_ReID']+=1;continue
                pairs.append(dict(parent_id=m['parent_id'],reference_origin=reference['origin'],mode=m['mode'],
                    reference_camera=reference['camera'],reference_timestep=reference['timestep'],
                    target=reference['target'],reference_purity=reference['target_purity'],
                    contaminated_reference=reference['target_purity']<.8,candidate_index=m['index'],
                    candidate_branch=m['branch'],candidate_camera=m['camera'],candidate_timestep=m['timestep'],
                    candidate_actor=m['actor_label_v2'],time_relation=time_relation,
                    cross_camera=reference['camera']!=m['camera'],
                    intended_positive=m['actor_label_v2']==reference['target'],
                    actual_reference_actor=reference['actual_actor'],
                    similarity=float(reference['vector']@vectors[m['index']]),
                    physical_unit='same parent/physical actor; no cross-parent instance negatives'))
        metrics=curves(pairs,('reference_origin','mode','time_relation','cross_camera'))
        csv_write(root/'dinov2_positive_negative_pairs.csv',pairs)
        csv_write(root/'dinov2_reidentification_metrics.csv',metrics)
        write(root/'crop_actor_label_audit.json',labels)
        provenance=[{k:(v.tolist() if k=='vector' else v) for k,v in r.items()} for r in references]
        write(root/'dinov2_reference_provenance.json',provenance)
        write(root/'reference_availability.json',availability)
        write(root/'dinov2_reference_contamination_audit.json',dict(
            reference_counts=dict(Counter(r['origin'] for r in references)),
            contaminated_counts=dict(Counter(r['origin'] for r in references if r['target_purity']<.8)),
            unavailable_hypothetical_references=dict(Counter(r['unavailable_reason'] for r in availability if not r['reference_available'])),
            empty_B0_references='not fabricated: absent nonempty crop embeddings remain missing',
            baseline_bank_used=False,actual_bank_origin='ACTUAL_BC_FROZEN_REFERENCE only',skipped_pairs=dict(skipped)))
        rectangle_counts=Counter(r['actor_label_v2'] or 'unknown' for r in labels if r['branch']=='ACTUAL_DINO_BOX_CROP' and r['mode']=='full_patch448')
        write(root/'scope.json',dict(status='POST_FORWARD_REFERENCE_PAIR_AUDIT_COMPLETE',run_id=logger.run_id,
            embeddings_recomputed=0,model_forwards=0,pairs=len(pairs),rectangle_labels=dict(rectangle_counts),
            input_sources=reference_sources,criteria_or_matching_threshold_changed=False,
            same_time_pairs_separate=True,no_future_reference_pairs=True,no_cross_parent_instance_negatives=True,
            source=config,independent_test=False,whole_goal_complete=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','features','causal'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.features,a.causal)
