"""Reconstruct historical t0 proposal funnel from unchanged raw forwards.

All query logits remain in their source NPZ, referenced by SHA. CSV query rows
retain original query indices; localization is never labelled semantic proof.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha
from visual.runtime.initialization import assign,box_iou


def pixel_boxes(boxes):
    boxes=np.asarray(boxes)
    center,extent=boxes[:,:2],boxes[:,2:]
    return np.concatenate((center-extent/2,center+extent/2),axis=1)*np.array([128,128,128,128])


def run(root):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    source=Path('/workspace/task10i_stage4a_initialization_bottlenecks_20261101_v1/run')
    rows=read(source/'t0_reconstruction_rows.json')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_raw_query_funnel',
                          root=root/'experiments',job_type='evaluation',seed=42,
                          config=dict(partition='historical_development',new_forwards=0,
                                      selection_changed=False)) as logger:
        records=[]
        raw_receipts={}
        query_count=0
        with (root/'dino_raw_queries_audit.csv').open('w') as f:
            keys=['episode','parent_id','camera','object_id','query_index','query_rank',
                  'max_token_index','max_token_score','box','gt_box_iou','area_eligible',
                  'threshold_survives','top5_eligible','beats_missing','chosen_box_match']
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            grouped={}
            for r in rows:
                p=Path(r['cached_forward_source'])
                with np.load(p) as z:
                    logits=z['logits'][0].astype(np.float64)
                    boxes=pixel_boxes(z['pred_boxes'][0])
                    probs=1/(1+np.exp(-np.clip(logits,-700,700)))
                    scores=probs.max(axis=-1)
                    token_indices=probs.argmax(axis=-1)
                    input_ids=z['input_ids'].tolist()
                    post_boxes=z['selected_boxes'].tolist()
                    post_scores=z['selected_scores'].tolist()
                    shapes={k:list(z[k].shape) for k in ('logits','pred_boxes','input_ids','pixel_values_shape')}
                raw_receipts[str(p)]=dict(sha256=sha(p),shapes=shapes,input_ids=input_ids,
                                        score_is_calibrated_probability=False)
                order=np.argsort(-scores)
                rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
                eligible=[(b,s) for b,s in zip(post_boxes,post_scores)
                          if (b[2]-b[0])*(b[3]-b[1])<.25*128*128][:5]
                grouped.setdefault((r['episode'],r['camera']),{})[r['object_id']]=(r,eligible)
                for i,box in enumerate(boxes):
                    top=any(np.allclose(box,b,atol=1e-4,rtol=0) for b,s in eligible)
                    chosen=r['chosen_box'] is not None and np.allclose(box,r['chosen_box'],atol=1e-4,rtol=0)
                    writer.writerow(dict(episode=r['episode'],parent_id=r['parent_id'],camera=r['camera'],
                        object_id=r['object_id'],query_index=i,query_rank=int(rank[i]),
                        max_token_index=int(token_indices[i]),max_token_score=float(scores[i]),
                        box=np.asarray(box).tolist(),gt_box_iou=box_iou(box,r['gt_box']) if r['gt_box'] else None,
                        area_eligible=(box[2]-box[0])*(box[3]-box[1])<.25*128*128,
                        threshold_survives=bool(scores[i]>.1),top5_eligible=top,
                        beats_missing=bool(scores[i]>.3),chosen_box_match=bool(chosen)))
                    query_count+=1
                reason=r['first_localization_restriction']
                line={'AREA_FILTER_REMOVES_ALL_LOCALIZED':32,
                      'TOP5_REMOVES_ALL_LOCALIZED':32,
                      'LOCALIZED_SCORES_DO_NOT_BEAT_MISSING_ALTERNATIVE':38,
                      'JOINT_RANK_OR_EXCLUSION_SELECTS_OTHER_OR_MISSING':40,
                      'POST_THRESHOLD_REMOVES_ALL_LOCALIZED':70}.get(reason)
                records.append(dict(**r,source_file='visual/runtime/grounding.py' if reason.startswith('POST_')
                    else 'visual/runtime/initialization.py',source_line=line,
                    semantic_identity_confirmed=False,scope='t0 historical Development localization funnel'))
            for key,pair in grouped.items():
                chosen=assign(pair[0][1],pair[1][1])
                for oid in (0,1):
                    expected=pair[oid][0]['chosen_box']
                    actual=chosen[oid][0] if chosen[oid] else None
                    assert (expected is None)==(actual is None),key
                    if expected is not None:
                        np.testing.assert_array_equal(actual,expected)
        write(root/'dino_candidate_restriction_funnel.json',records)
        write(root/'raw_query_artifact_identity.json',raw_receipts)
        write(root/'scope.json',dict(status='HISTORICAL_RAW_QUERY_AUDIT_COMPLETE',
              run_id=logger.run_id,object_camera_episode_rows=len(rows),query_rows=query_count,
              raw_artifacts=len(raw_receipts),raw_tensors_duplicated=False,
              original_assignments_reproduced=True,new_model_forwards=0,
              semantic_identity_analysis_pending=True,source_sha256=sha(Path(__file__))))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
