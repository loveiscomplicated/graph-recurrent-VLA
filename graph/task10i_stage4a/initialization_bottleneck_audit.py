"""Exact t=0 original assignment reconstruction and first localization restriction.

Cached DINO outputs are joined by exact input RGB hash/prompt, including memo
hits in other branches. No future image or new model forward is used.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import h5py
import numpy as np
from .run_pipeline import CAMERAS,STAGE3,read,write
from .detector_audit import bbox,pixel_boxes
from visual.runtime.initialization import PROMPTS,MAX_BOX_FRAC,MISSING_SCORE,candidates,assign,box_iou


def run(output,audits):
    from utils.experiment_logger import ExperimentLogger
    output.mkdir(parents=True,exist_ok=False)
    events={(r['input_rgb_sha256'],r['prompt']):r for r in read(audits/'baseline/dino_events.json')}
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    initial={(r['episode'],r['camera'],r['object_id']):r for r in read(audits/'extended/actual_initializations.json')}
    cache={};rows=[]
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_t0_initialization_restriction_audit',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(model_forwards=0,partition='development',timestep=0,no_candidate_retuning=True)) as log:
        for p in sorted((audits/'corrected/perception').glob('*/receipt.json')):
            cr=read(p);eid=str(cr['episode_id']);e=entries[eid];rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
            with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:
                assert z['exact_frame'][0];seg=z['actor_segmentation'][0].copy()
            with h5py.File(e['raw_path'],'r') as f:
                for ci,camera in enumerate(CAMERAS):
                    rgb=f[camera+'/rgb'][0];key_rgb=hashlib.sha256(rgb.tobytes()).hexdigest();pools=[];raw=[]
                    def detector(image,prompt):
                        key=hashlib.sha256(image.tobytes()).hexdigest(),prompt
                        if key not in cache:
                            event=events[key]
                            with np.load(event['raw_path'],allow_pickle=False) as z:
                                cache[key]=(z['selected_boxes'].tolist(),z['selected_scores'].tolist(),pixel_boxes(z['pred_boxes'][0]))
                        boxes,scores,_=cache[key]
                        return boxes,scores
                    for name in ('peg','box'):
                        detector(rgb,PROMPTS[name]);pools.append(candidates(detector,rgb,PROMPTS[name]));raw.append(cache[key_rgb,PROMPTS[name]])
                    selected=assign(*pools)
                    for oid,name in enumerate(('peg','box')):
                        target=np.isin(seg[ci],rr['object_ids'][name]);gtbox=bbox(target);n=int(target.sum());boxes,scores,queries=raw[oid]
                        localized=lambda box:gtbox is not None and box_iou(box,gtbox)>=.5
                        raw_local=int(sum(localized(box) for box in queries));post_local=sum(localized(box) for box in boxes)
                        area=[(box,score) for box,score in zip(boxes,scores) if (box[2]-box[0])*(box[3]-box[1])<MAX_BOX_FRAC*128*128]
                        area_local=sum(localized(box) for box,score in area);top_local=sum(localized(box) for box,score in pools[oid])
                        above_missing=sum(localized(box) and score>MISSING_SCORE for box,score in pools[oid])
                        chosen=selected[oid];hit=chosen is not None and localized(chosen[0]);actual=initial.get((eid,camera,oid))
                        assert bool(chosen is not None)==bool(actual and actual['decision_timestep']==0)
                        if chosen is not None:np.testing.assert_array_equal(chosen[0],actual['selected_box'])
                        if n<8:reason='TINY_OR_ABSENT_GT_SUPPORT'
                        elif hit:reason='LOCALIZED_INITIALIZATION'
                        elif not raw_local:reason='NO_RAW_LOCALIZED_QUERY'
                        elif not post_local:reason='POST_THRESHOLD_REMOVES_ALL_LOCALIZED'
                        elif not area_local:reason='AREA_FILTER_REMOVES_ALL_LOCALIZED'
                        elif not top_local:reason='TOP5_REMOVES_ALL_LOCALIZED'
                        elif not above_missing:reason='LOCALIZED_SCORES_DO_NOT_BEAT_MISSING_ALTERNATIVE'
                        else:reason='JOINT_RANK_OR_EXCLUSION_SELECTS_OTHER_OR_MISSING'
                        rows.append(dict(episode=eid,parent_id=e['parent_id'],camera=camera,object_id=oid,timestep=0,
                            raw_gt_pixels=n,gt_box=gtbox,raw_localized=raw_local,post_threshold_localized=post_local,
                            area_eligible_localized=area_local,top5_localized=top_local,localized_score_beats_missing=above_missing,
                            chosen_box=chosen[0] if chosen else None,chosen_score=chosen[1] if chosen else None,
                            chosen_gt_iou=box_iou(chosen[0],gtbox) if chosen and gtbox else None,
                            actual_initialization_timestep=actual['decision_timestep'] if actual else None,first_localization_restriction=reason,
                            cached_forward_source=events[key_rgb,PROMPTS[name]]['raw_path'],reference_kind='EXACT_GT'))
        counts=Counter((r['camera'],r['object_id'],r['first_localization_restriction']) for r in rows)
        write(output/'t0_reconstruction_rows.json',rows)
        write(output/'restriction_counts.json',[dict(camera=c,object_id=o,restriction=k,logical_episodes=n) for (c,o,k),n in sorted(counts.items())])
        write(output/'scope.json',dict(status='T0_ASSIGNMENT_RECONSTRUCTION_PASS',run_id=log.run_id,rows=len(rows),
            all_development_t0_inputs_including_memo_hits=True,no_model_forwards=True,original_assignment_bytes_equal=True,
            area_fraction_limit=MAX_BOX_FRAC,missing_alternative_score=MISSING_SCORE,
            localization_not_semantic_identity=True,restriction_is_first_pool_stage_with_no_localized_candidate=True,
            no_candidate_or_numeric_selection_changes=True))
        log.link_output(output/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.audits)
