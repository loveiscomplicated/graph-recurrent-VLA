"""Actual first-loss attribution; uninitialized and tracking failures stay distinct."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from .run_pipeline import read,write


def run(output,audits):
    baseline=audits/'baseline';corrected=audits/'corrected'
    init={(r['episode'],r['camera'],r['object_id']):r for r in read(audits/'extended/actual_initializations.json')}
    raw=defaultdict(list)
    for r in read(audits/'extended/detector_raw_query_metrics.json'):
        raw[r['episode'],r['camera'],r['object_id'],r['source_timestep']].append(r)
    rows=[];funnel=defaultdict(Counter)
    for receipt in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(receipt);eid=str(cr['episode_id']);grouped=defaultdict(list)
        records={(r['camera'],r['timestep']):r for r in read(baseline/'perception'/eid/'vision_records.json')}
        points={(r['object_id'],r['timestep']):r for r in read(receipt.parent/'representative_points.json')}
        for r in read(receipt.parent/'stage_rows.json'):
            grouped[r['camera'],r['object_id'],r['timestep']].append(r)
            m=r['metrics']
            if m:
                for scope in ('all_supported','visible_8_plus' if m['input_gt_support']>=8 else 'tiny_1_7' if m['input_gt_support'] else 'absent'):
                    f=funnel[r['camera'],r['object_id'],r['stage'],scope]
                    f.update({k:m[k] for k in ('input_gt_support','output_support','true_positive','false_positive','false_negative','robot_pixels','wrong_object_pixels','background_pixels')});f['frames']+=1
        for (camera,oid,t),rr in grouped.items():
            by={r['stage']:r for r in rr};m=by['sam2']['metrics'];found=any(oid+1 in g for g in records[camera,t]['tracking_groups'])
            first=None;detail=None
            if m is None:status='INSUFFICIENT_REFERENCE'
            elif m['input_gt_support']==0:status='PHYSICALLY_NOT_VISIBLE'
            elif m['input_gt_support']<8:status='TINY_VISIBLE_SUPPORT'
            elif not m['true_positive']:
                first='sam2'
                final_m=by['final_graph_sources']['metrics']
                status='GT_VISIBLE_BUT_DETECTOR_MISSING' if not found else 'DETECTION_PRESENT_BUT_SEGMENTATION_MISSING' if not m['output_support'] else 'WRONG_OBJECT_ACCEPTED' if final_m['output_support'] else 'WRONG_MASK_REJECTED_DOWNSTREAM'
                queries=[q for q in raw[eid,camera,oid,t] if q['decision_timestep']<=t]
                if not found:
                    detail='DETECTOR_NOT_CALLED_ON_THIS_SOURCE_FRAME' if not queries else 'RAW_LOCALIZED_QUERY_BELOW_POSTPROCESSING' if any(q['raw_localized_queries']>0 and q['post_threshold_localized_count']==0 for q in queries) else 'POSTPROCESSING_LOCALIZED_BOX_NOT_INITIALIZED' if any(q['post_threshold_localized_count']>0 for q in queries) else 'NO_RAW_QUERY_LOCALIZED_AT_IOU_0_5'
            else:
                first=next((r['stage'] for r in rr if r['metrics']['true_positive']==0),None)
                status='CORRECT_GRAPH_SOURCE_SUPPORT' if first is None else 'SEGMENTATION_PRESENT_BUT_FILTERED' if first in ('depth_valid','robot_confirmed_removed','robot_uncertain_removed','overlap_removed','gate_accepted') else 'VALID_MASK_BUT_GRAPH_MISSING'
            if camera=='hand_camera' and oid==1:graph_scope='INTENTIONALLY_EXCLUDED_WRIST_BOX'
            else:graph_scope='ACTIVE_CAMERA_OBJECT_SOURCE'
            final=by['final_graph_sources']['metrics'];g=points[oid,t]
            rows.append(dict(episode=eid,parent_id=cr['parent_id'],timestep=t,camera=camera,object_id=oid,
                status=status,first_zero_correct_stage=first,detector_detail=detail,accepted_initialization_present=found,
                initialization=init.get((eid,camera,oid)),graph_scope=graph_scope,
                wrong_source_accepted=bool(final and final['false_positive']>0),
                final_source_metrics=final,global_graph_points=g['points'],global_graph_correct=g['correct'],
                camera_source_loss_is_not_necessarily_global_object_loss=True))
    output.mkdir(parents=True,exist_ok=False)
    write(output/'frame_attribution.json',rows)
    write(output/'status_counts.json',[dict(camera=c,object_id=oid,status=status,count=n) for (c,oid,status),n in sorted(Counter((r['camera'],r['object_id'],r['status']) for r in rows).items())])
    write(output/'retention_funnel.json',[dict(camera=k[0],object_id=k[1],stage=k[2],scope=k[3],**v) for k,v in funnel.items()])
    write(output/'scope.json',dict(status='ACTUAL_FIRST_LOSS_ATTRIBUTION_FINISHED',all_current_source_frames_retained=True,
        raw_query_score_is_not_probability=True,localization_iou_does_not_prove_semantic_identity=True,
        detector_not_called_not_counted_as_raw_proposal_absence=True,wrist_box_exclusion_not_counted_as_graph_error=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.audits)
