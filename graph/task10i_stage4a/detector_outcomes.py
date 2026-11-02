"""Separate actual-call proposal localization, accepted initialization and mask identity.

No new forwards, candidate selection or unsupported-frame relabeling. Proposal
FPs use an explicit one-GT-box IoU match and include duplicate proposals. This
is distinct from the selected initialization's false-positive rate.
"""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import numpy as np
from .run_pipeline import CAMERAS,read,write
from .instrumentation import support


def run(output,audits):
    from utils.experiment_logger import ExperimentLogger
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_detector_outcome_accounting',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(model_forwards=0,partition='development',proposal_match_iou=.5,candidate_retuning=False)) as log:
        raw={}
        for r in read(audits/'extended/detector_raw_query_metrics.json'):
            key=r['episode'],r['camera'],r['object_id'],r['source_timestep']
            if key in raw:
                assert all(raw[key][k]==r[k] for k in ('selected_count','post_threshold_localized_count','gt_visible_pixels'))
            else:raw[key]=r
        counts=defaultdict(Counter)
        for r in raw.values():
            n=r['gt_visible_pixels'];stratum='unsupported' if n is None else 'absent' if n==0 else 'tiny_1_7' if n<8 else 'visible_8_plus'
            c=counts[r['camera'],r['object_id'],stratum];c['called_source_frames']+=1
            if stratum=='unsupported':continue
            tp=int(n>=8 and r['post_threshold_localized_count']>0)
            c['localized_detection_frames']+=tp;c['selected_proposals']+=r['selected_count']
            if stratum!='tiny_1_7':c['unmatched_or_duplicate_proposals']+=r['selected_count']-tp
            c['no_proposal_frames']+=int(r['selected_count']==0)
        initial=[]
        for r in read(audits/'extended/actual_initializations.json'):
            n=r['gt_visible_pixels'];initial.append(dict(**r,
                visibility_stratum='absent' if n==0 else 'tiny_1_7' if n<8 else 'visible_8_plus',
                localized_at_iou_0_5=bool(n>=8 and r['gt_iou'] is not None and r['gt_iou']>=.5),
                localization_failure_does_not_prove_semantic_confusion=True))
        identity=defaultdict(Counter)
        for p in sorted((audits/'corrected/perception').glob('*/receipt.json')):
            cr=read(p);eid=str(cr['episode_id']);rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
            with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:segs=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            with np.load(audits/'baseline/perception'/eid/'masks.npz',allow_pickle=False) as z:masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            for t in range(len(exact)):
                for ci,camera in enumerate(CAMERAS):
                    for oid in (0,1):
                        c=identity[camera,oid];c['all_frames']+=1
                        if not exact[t]:c['unsupported']+=1;continue
                        m=support(masks[t,ci,oid],segs[t,ci],oid,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']])
                        if not m['output_support']:c['empty_mask']+=1;continue
                        c['nonempty_supported_masks']+=1
                        fields={'correct':'true_positive','other_object':'wrong_object_pixels','robot':'robot_pixels','background':'background_pixels'}
                        label=next((name for name,field in fields.items() if m[field]>.5*m['output_support']),'mixed_no_majority')
                        c[label+'_majority_masks']+=1
        write(output/'actual_call_proposal_counts.json',[dict(camera=k[0],object_id=k[1],support_stratum=k[2],**v) for k,v in sorted(counts.items())])
        write(output/'actual_initialization_outcomes.json',initial)
        write(output/'mask_identity_counts.json',[dict(camera=k[0],object_id=k[1],**v) for k,v in sorted(identity.items())])
        write(output/'scope.json',dict(status='DETECTOR_AND_IDENTITY_OUTCOMES_FINISHED',run_id=log.run_id,
            unique_called_source_frames=len(raw),model_forwards=0,not_full_frame_detector_recall=True,
            proposal_FP='one localized match per prompted GT object; remaining returned boxes unmatched/duplicate; tiny stratum excluded from FP rate',
            initialization_FP='assigned box with absent target or IoU<0.5 on visible>=8 support; tiny cases separate',
            identity_confusion='other-object majority (>50%) of nonempty exact-reference masks; robot/background separate; fixed semantic slots are not tracker ID switches',
            no_candidate_changes=True))
        log.link_output(output/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.audits)
