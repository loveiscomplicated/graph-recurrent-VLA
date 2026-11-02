"""Supplemental weak-reference metrics, NEVER promoted into primary EXACT_GT."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
from .run_pipeline import CAMERAS,read,write


def eligible(comparisons):
    return set(comparisons)==set(CAMERAS) and all(
        all(fields[k]['exact'] for k in ('depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'))
        and fields['rgb']['max_abs']<=1 and fields['rgb']['different_values']<=32
        for fields in comparisons.values())


def run(root,final):
    from utils.experiment_logger import ExperimentLogger
    from .instrumentation import support
    root.mkdir(parents=True,exist_ok=False)
    definition=dict(reference_kind='APPROXIMATE_REFERENCE_RGB_ULP_ONLY',not_exact_GT=True,
        criterion='both-camera depth and all3 calibration fields byte-equal; RGB max_abs<=1 and <=32 differing channel values/camera',
        not_mixed_with_primary_exact_metrics=True,no_primary_reclassification=True,no_candidate_retuning=True)
    write(root/'definition.json',definition)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_separate_approximate_reference_supplement',seed=42,
        root=root/'experiments',job_type='evaluation',config=definition) as log:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_FINAL_FROZEN_EVALUATION',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        rows=[];points=[];counts=[]
        for partition in ('internal_test','validation'):
            total=Counter()
            for p in sorted((final/partition/'corrected/perception').glob('*/receipt.json')):
                cr=read(p);eid=str(cr['episode_id']);rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
                with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:segs=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                for t in range(len(exact)):
                    total['all_frames']+=1
                    if exact[t]:total['EXACT_GT']+=1;continue
                    if not eligible(rr['rows'][t]['comparisons']):total['INSUFFICIENT_REFERENCE']+=1;continue
                    total['APPROXIMATE_REFERENCE_RGB_ULP_ONLY']+=1
                    for condition in ('baseline','candidate'):
                        folder=final/partition/condition
                        with np.load(folder/'perception'/eid/'masks.npz',allow_pickle=False) as z:mask=np.unpackbits(z['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
                        for ci,camera in enumerate(CAMERAS):
                            for oid in (0,1):rows.append(dict(partition=partition,condition=condition,episode=eid,parent_id=cr['parent_id'],timestep=t,camera=camera,object_id=oid,
                                reference_kind=definition['reference_kind'],metrics=support(mask[ci,oid],segs[t,ci],oid,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']])))
                        trace=read(folder/'traces'/eid/f't{t:04d}/trace.json')
                        labels={0:Counter(),1:Counter()}
                        for r in trace['representative_sources']:
                            _,_,camera,pixel=r['representative'];actor=int(segs[t,CAMERAS.index(camera)].ravel()[pixel]);oid=r['object_id']
                            label='correct' if actor in rr['object_ids'][('peg','box')[oid]] else 'robot' if actor in rr['robot_ids'] else 'other_object' if actor in rr['object_ids'][('box','peg')[oid]] else 'background'
                            labels[oid][label]+=1
                        for oid in (0,1):points.append(dict(partition=partition,condition=condition,episode=eid,parent_id=cr['parent_id'],timestep=t,object_id=oid,reference_kind=definition['reference_kind'],**labels[oid]))
            counts.append(dict(partition=partition,**total))
        write(root/'pixel_rows.json',rows);write(root/'point_rows.json',points)
        write(root/'reference_accounting.json',dict(status='SEPARATE_APPROXIMATE_REFERENCE_SUPPLEMENT_FINISHED',counts=counts,
            exact_reference_flags_unchanged=True,primary_decisions_unchanged=True,GT_labels_not_student_inputs=True))
        log.link_output(root/'reference_accounting.json',kind='metrics')
        write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='SEPARATE_APPROXIMATE_REFERENCE_SUPPLEMENT_FINISHED',run_id=log.run_id))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('final',type=Path)
    a=p.parse_args();run(a.root,a.final)
