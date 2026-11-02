"""ORACLE_DIAGNOSTIC: partial Peg anchor versus complete current GT box.

Failure time16 and actor boxes are evaluator interventions, never practical
triggers. The practical t6 prompt is independently replayed byte-for-byte.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from .lazy_feature_control import wait_for
from .prompt_audit import bbox
from .evaluation_metrics import pixel_metrics
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


def run(root,development,references,dependency):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.tracking import SAM2Tracker
    from graph.task10i_stage4a.reliable_reanchor import AnchorState
    root.mkdir(parents=True,exist_ok=False)
    eid='10i_stage3_10i_s1_0250_full_task';camera='hand_camera';ci=1;oid=0;anchor_t=6;failure_t=16
    entry=next(e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values() if str(e['episode_id'])==eid)
    folder=development/'perception'/eid
    receipt=read(folder/'receipt.json');assert all(sha(folder/p)==h for p,h in receipt['files'].items())
    events=read(folder/'C_KEEP/events.json');event=next(e for e in events if e['accepted'] and e['camera']==camera and e['object_id']==oid and e['timestep']==anchor_t)
    practical_box=event['chosen']['box']
    ref=references/'references'/eid;rr=read(ref/'receipt.json');assert all(sha(ref/p)==h for p,h in rr['files'].items())
    with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
    with np.load(folder/'C_KEEP.npz') as z:practical=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
    with np.load(folder/'B0.npz') as z:baseline=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
    assert exact[anchor_t] and exact[failure_t]
    target=lambda t:np.isin(seg[t,ci],rr['object_ids']['peg'])
    boxes={t:bbox(target(t)) for t in (anchor_t,failure_t)}
    config=dict(episode=eid,source_sha256=sha(Path(__file__)),anchor_t=anchor_t,oracle_failure_t=failure_t,
        practical_box=practical_box,oracle_boxes=boxes,GT_oracle_only=True,practical_metrics_modified=False,
        inference_receipt_sha256=sha(folder/'receipt.json'),reference_receipt_sha256=sha(ref/'receipt.json'))
    write(root/'protocol.json',config)
    wait_for(root,dependency/'development/pipeline_state.json','DEVELOPMENT_SCOPED_CORRECTION_INFERENCE_COMPLETE',
             'graph.task10i_stage4a2.main_box_pipeline',dependency)
    torch.set_num_threads(1);torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    paths=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json');assert sha(paths['sam'])==identity['sam_sha256']
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_ORACLE_partial_Peg_prompt',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        tracker=SAM2Tracker('cuda',checkpoint=paths['sam']);rows=[];result={}
        for condition in ('PRACTICAL_T6_REPRODUCTION','ORACLE_COMPLETE_BOX_T6','ORACLE_FRESH_COMPLETE_BOX_T16'):
            state=None;values=[]
            try:
                with h5py.File(entry['raw_path']) as raw:
                    for t in range(anchor_t,entry['actions']+1):
                        rgb=raw[camera+'/rgb'][t]
                        if t==anchor_t:
                            state=AnchorState(tracker,rgb,boxes[t] if condition=='ORACLE_COMPLETE_BOX_T6' else practical_box,t)
                        elif t==failure_t and condition=='ORACLE_FRESH_COMPLETE_BOX_T16':
                            state.close();state=AnchorState(tracker,rgb,boxes[t],t)
                        else:state.append(rgb,t)
                        mask=state.current()
                        if condition=='PRACTICAL_T6_REPRODUCTION':assert np.array_equal(mask,practical[t,ci,oid]),('practical replay mismatch',t)
                        values.append(np.packbits(mask.ravel()))
                        row=dict(condition=condition,timestep=t,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                            intervention_is_oracle=condition!='PRACTICAL_T6_REPRODUCTION')
                        if exact[t]:row.update(pixel_metrics(mask,target(t),np.isin(seg[t,ci],rr['robot_ids']),np.isin(seg[t,ci],rr['object_ids']['box'])))
                        rows.append(row)
                np.savez_compressed(root/(condition+'.npz'),mask_bits=np.stack(values),start_timestep=anchor_t)
                result[condition]=dict(frames=len(values),source='actual SAM2 CUDA',practical_bytes_equal=condition=='PRACTICAL_T6_REPRODUCTION')
            finally:
                if state is not None:state.close()
                torch.cuda.empty_cache()
        for condition,masks in [('B0',baseline),('C_KEEP_SAVED',practical)]:
            for t in range(anchor_t,entry['actions']+1):
                row=dict(condition=condition,timestep=t,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',intervention_is_oracle=False)
                if exact[t]:row.update(pixel_metrics(masks[t,ci,oid],target(t),np.isin(seg[t,ci],rr['robot_ids']),np.isin(seg[t,ci],rr['object_ids']['box'])))
                rows.append(row)
        csv_write(root/'partial_prompt_tracking_metrics.csv',rows)
        write(root/'scope.json',dict(status='ORACLE_PARTIAL_PEG_PROMPT_CONTROL_COMPLETE',run_id=logger.run_id,**config,
            conditions=result,oracle_not_practical_benefit=True,whole_goal_complete=False,
            files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
        write(root/'pipeline_state.json',dict(status='ORACLE_PARTIAL_PEG_PROMPT_CONTROL_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','development','references','dependency'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.development,a.references,a.dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
