"""Causal prefixes with explicitly ORACLE_DIAGNOSTIC initialization/correction.

Practical baseline calls take RGB only; GT-derived boxes enter only named oracle
conditions. All predictions are saved before the separate pixel evaluator runs.
"""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .prompt_audit import bbox, AUDITS
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read, write, sha, STAGE3, CAMERAS

WINDOWS=[dict(episode='10i_stage3_10i_s1_0924_full_task',target_camera='base_camera',target_oid=0,
              correction_t=0,reason='historical both-camera Peg initial failure'),
         dict(episode='10i_stage3_10i_s1_0121_full_task',target_camera='base_camera',target_oid=1,
              correction_t=95,reason='historical Main Box robot drift'),
         dict(episode='2000003',target_camera='base_camera',target_oid=0,
              correction_t=50,reason='healthy tracking unnecessary correction negative control')]
CONDITIONS=('B0','ORACLE_CORRECT_INIT','ORACLE_WRONG_INIT',
            'ORACLE_CORRECTION_EXISTING_MEMORY','ORACLE_CORRECTION_FRESH_MEMORY')


def memory_summary(adapter):
    result=[]
    for group,state in adapter.states.items():
        result.append(dict(group=list(group),obj_ids=list(state['obj_ids']),num_frames=state['num_frames'],
            object_state=[dict(index=int(i),conditioning_frames=sorted(v['cond_frame_outputs']),
                              nonconditioning_frames=sorted(v['non_cond_frame_outputs']))
                          for i,v in state['output_dict_per_obj'].items()],
            tracked={str(i):sorted(v) for i,v in state['frames_tracked_per_obj'].items()},
            cached_feature_frames=sorted(state['cached_features'])))
    return result


def correct_existing(adapter,camera,oid,t,box):
    """Official installed API; no assertion of per-object-memory reset."""
    group=next((g for g in adapter.states if oid+1 in g),None)
    if group is None:
        return None,dict(status='NO_INITIALIZED_OBJECT_SLOT')
    state=adapter.states[group]
    before=memory_summary(adapter)
    adapter.tracker.predictor.add_new_points_or_box(
        state,frame_idx=t,obj_id=oid+1,box=np.asarray(box,np.float32),clear_old_points=True)
    iterator=adapter.tracker.track(state,start_frame=t)
    try:frame,values=next(iterator)
    finally:iterator.close()
    assert frame==t
    return {o-1:m.cpu().numpy() for o,m in values.items()},dict(status='CORRECTED',before=before,
        after=memory_summary(adapter),clear_old_points=True,full_memory_reset=False)


def evaluate(root,windows):
    from graph.task10i_stage4a.instrumentation import support
    rows=[]
    for window in windows:
        eid=window['episode'];folder=root/'episodes'/eid
        cr=read(AUDITS/'corrected/perception'/eid/'receipt.json')
        refpath=Path(cr['corrected_reference']);rr=read(refpath)
        assert sha(refpath)==cr['corrected_reference_sha256']
        with np.load(refpath.parent/'segmentation.npz') as z:
            exact=z['exact_frame'].copy();seg=z['actor_segmentation'].copy()
        for condition in CONDITIONS:
            with np.load(folder/(condition+'.npz')) as z:
                masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            for t in range(len(masks)):
                for ci,camera in enumerate(CAMERAS):
                    for oid in (0,1):
                        row=dict(condition=condition,episode=eid,parent_id=cr['parent_id'],timestep=t,
                            camera=camera,object_id=oid,correction_t=window['correction_t'],
                            reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                            condition_type='BASELINE' if condition=='B0' else 'ORACLE_DIAGNOSTIC')
                        if exact[t]:
                            row.update(support(masks[t,ci,oid],seg[t,ci],oid,rr['robot_ids'],
                                       [rr['object_ids']['peg'],rr['object_ids']['box']]))
                        rows.append(row)
    csv_write(root/'sam2_causal_recovery_metrics.csv',rows)
    return rows


def run(root):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.task10i_stage4a.development_perception import Memo
    from graph.task10i_stage4a.reliable_reanchor import AnchorState
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    windows=[dict(w,last_t=min(entries[w['episode']]['actions'],w['correction_t']+20)) for w in WINDOWS]
    write(root/'registered_windows.json',dict(windows=windows,conditions=CONDITIONS,
        source_sha256=sha(Path(__file__)),oracle_timing_not_practical_trigger=True,
        purpose='causal intervention at historically known failure and healthy control; not population rate'))
    rt=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    events=[];timing=[]
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_SAM2_memory_controls',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(condition_type='ORACLE_DIAGNOSTIC',windows=windows)) as logger:
        dino=GDino(device='cuda',model_id=rt['dino']);dino.context={};memo=Memo(dino)
        tracker=SAM2Tracker(device='cuda',checkpoint=rt['sam'])
        for window in windows:
            eid=window['episode'];e=entries[eid]
            cr=read(AUDITS/'corrected/perception'/eid/'receipt.json')
            refpath=Path(cr['corrected_reference']);rr=read(refpath)
            assert sha(e['raw_path'])==e['raw_sha256']
            with np.load(refpath.parent/'segmentation.npz') as z:
                segmentation=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            assert exact[0] and exact[window['correction_t']]
            initial_boxes={camera:{name:bbox(np.isin(segmentation[0,ci],rr['object_ids'][name]))
                                  for name in ('peg','box')} for ci,camera in enumerate(CAMERAS)}
            folder=root/'episodes'/eid;folder.mkdir(parents=True)
            for condition in CONDITIONS:
                # Execute conditions sequentially: the initial parallel adapter
                # schedule exhausted 24GB on long prefixes. No state pruning or
                # changed arithmetic is needed to bound concurrent memory.
                adapters={}
                for camera in CAMERAS:
                    if condition in ('ORACLE_CORRECT_INIT','ORACLE_WRONG_INIT'):
                        def oracle_detect(rgb,prompt,camera=camera,condition=condition):
                            name='peg' if prompt=='red and white block' else 'box'
                            if condition=='ORACLE_WRONG_INIT':name='box' if name=='peg' else 'peg'
                            box=initial_boxes[camera][name]
                            return ([box],[1.]) if box is not None else ([],[])
                        detector=oracle_detect
                    else:detector=memo
                    adapter=OnlineVisionAdapter(detector,tracker);adapter.reset(eid)
                    adapters[camera]=adapter
                bits=[];fresh=None
                try:
                    with h5py.File(e['raw_path']) as raw,np.load(AUDITS/'baseline/perception'/eid/'masks.npz') as old:
                        for t in range(window['last_t']+1):
                            tick=time.perf_counter();masks=[]
                            for camera in CAMERAS:
                                dino.context=dict(episode_id=eid,camera=camera,decision_timestep=t)
                                mask,_=adapters[camera].process(raw[camera+'/rgb'][t],eid,t)
                                masks.append(mask)
                            masks=np.stack(masks)
                            if condition=='B0':
                                assert np.array_equal(np.packbits(masks.ravel()),old['mask_bits'][t]),('BASELINE_MASK_MISMATCH',eid,t)
                            camera=window['target_camera'];ci=CAMERAS.index(camera);oid=window['target_oid']
                            if condition in ('ORACLE_CORRECTION_EXISTING_MEMORY','ORACLE_CORRECTION_FRESH_MEMORY'):
                                if t==window['correction_t']:
                                    box=bbox(np.isin(segmentation[t,ci],rr['object_ids'][('peg','box')[oid]]))
                                    assert box is not None
                                    if condition=='ORACLE_CORRECTION_EXISTING_MEMORY':
                                        corrected,detail=correct_existing(adapters[camera],camera,oid,t,box)
                                        if corrected is not None:
                                            for slot,mask in corrected.items():masks[ci,slot]=mask
                                    else:
                                        fresh=AnchorState(tracker,raw[camera+'/rgb'][t],box,t)
                                        detail=dict(status='FRESH_OBJECT_MEMORY',local_frame=0)
                                    events.append(dict(**window,timestep=t,oracle_box=box,condition=condition,
                                        memory=detail,condition_type='ORACLE_DIAGNOSTIC',practical_trigger=False))
                                elif fresh is not None:fresh.append(raw[camera+'/rgb'][t],t)
                                if fresh is not None:masks[ci,oid]=fresh.current()
                            bits.append(np.packbits(masks.ravel()))
                            timing.append(dict(episode=eid,timestep=t,condition=condition,
                                elapsed_ms=1000*(time.perf_counter()-tick),observer_present=True))
                            if t%20==0:print('sam2 controls',eid,condition,t,flush=True)
                    assert len(bits)==window['last_t']+1
                    np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(bits))
                    write(root/'sam2_memory_state_trace.json',events)
                finally:
                    if fresh is not None:fresh.close()
                    for adapter in adapters.values():adapter.close()
                    adapters.clear();fresh=None;memo.cache.clear();torch.cuda.empty_cache()
            write(folder/'receipt.json',dict(episode=eid,frames=window['last_t']+1,
                raw_sha256=e['raw_sha256'],baseline_masks_bytes_equal=True,
                sequential_conditions=True,files={p.name:sha(p) for p in folder.glob('*.npz')}))
        # Evaluation happens only after all condition outputs have been saved.
        rows=evaluate(root,windows)
        csv_write(root/'diagnostic_timing.csv',timing)
        write(root/'scope.json',dict(status='SAM2_ORACLE_MEMORY_CONTROLS_COMPLETE',run_id=logger.run_id,
            episodes=len(windows),object_camera_condition_frame_rows=len(rows),
            baseline_masks_bytes_equal=True,oracle_intervention_not_practical=True,
            full_population_estimate=False,GT_evaluation_after_prediction=True,
            source_sha256=sha(Path(__file__))))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
