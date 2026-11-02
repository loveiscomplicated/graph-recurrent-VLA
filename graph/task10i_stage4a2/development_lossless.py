"""Development-only P: same128 RGB, original pipeline, lossless SAM container.

Reads RGB/proprioception raw records and verified B0 predictions. No GT sidecar
or evaluator is imported. Rejected/failed episodes restart from causal t0.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def run(root,baseline,partition,engineering=False):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino
    from graph.task10i_stage4a.development_perception import Memo
    from visual.runtime.tracking import SAM2Tracker
    from .preprocessing_audit import LosslessAdapter
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        files={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils')
            for p in (source/folder).rglob('*.py')}
        digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        expected=read(baseline/'expected_counts.json')
        assert {e['parent_id'] for e in expected['entries']}==set(read(partition)['development'])
        assert read(baseline/'pipeline_state.json')['status']=='NEW_DEVELOPMENT_B0_A_INFERENCE_COMPLETE'
        entries=expected['entries']
        if engineering:entries=[dict(entries[0],actions=2,engineering_only=True)]
        expected=dict(parents=len({e['parent_id'] for e in entries}),episodes=len(entries),
            frames=sum(e['actions']+1 for e in entries),entries=entries)
        rt=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
        identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        assert sha(rt['sam'])==identity['sam_sha256']
        assert all(sha(Path(rt['dino'])/p)==h for p,h in identity['dino_files'].items())
        dinov2=Path('/workspace/task10i_stage4a2_models/dinov2-small')
        v2=read(dinov2/'download_identity.json');assert v2['model']=='facebook/dinov2-small'
        assert all(sha(dinov2/p)==h for p,h in v2['files'].items())
        import sam2
        installed=Path(sam2.__file__).parent
        spec=dict(source_digest=digest,files=files,partition_sha256=sha(partition),
            baseline_source_config_sha256=sha(baseline/'source_config_frozen.json'),
            hypothesis='P: SAM2 temporary frame encoding JPEG->lossless PNG only; original detector/selection/memory; DINO receives unchanged raw RGB',
            model_identity=identity,dinov2_identity=v2,dinov2_enabled=False,
            packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','tokenizers','numpy')},
            sam_installed_sources={str(p):sha(p) for p in installed.rglob('*.py')},
            sam_config_sha256=sha(installed/'configs/sam2.1/sam2.1_hiera_s.yaml'),
            adoption_criteria_sha256=sha(Path(__file__).with_name('adoption_criteria.json')),
            GT_forward_input=False,Internal_or_Final_access=False,engineering=engineering)
        if (root/'source_config_frozen.json').exists():assert read(root/'source_config_frozen.json')==spec
        else:write(root/'source_config_frozen.json',spec)
        write(root/'expected_counts.json',expected)
        torch.set_num_threads(1);torch.manual_seed(42)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_development_P_lossless',seed=42,
                root=root/'experiments',job_type='evaluation',config=spec) as logger:
            detector=AuditedDino(device='cuda',model_id=rt['dino'],output=root/'dino_raw');memo=Memo(detector)
            tracker=SAM2Tracker(device='cuda',checkpoint=rt['sam'])
            original=tracker.predictor._run_single_frame_inference
            sam_calls=[];context={}
            def observe(*args,**kwargs):
                output=original(*args,**kwargs)
                sam_calls.append(dict(**context,local_frame_idx=kwargs.get('frame_idx'),
                    object_score_logits=output[0]['object_score_logits'].detach().float().cpu().reshape(-1).tolist(),
                    output_keys=list(output[0]),mask_shape=list(output[0]['pred_masks'].shape),
                    is_init_cond_frame=kwargs.get('is_init_cond_frame'),run_mem_encoder=kwargs.get('run_mem_encoder')))
                return output
            tracker.predictor._run_single_frame_inference=observe
            progress=[]
            try:
                for e in entries:
                    eid=str(e['episode_id']);folder=root/'perception'/eid
                    if (folder/'receipt.json').exists():
                        r=read(folder/'receipt.json')
                        assert r['source_digest']==digest and r['raw_sha256']==e['raw_sha256']
                        assert all(sha(folder/p)==h for p,h in r['files'].items())
                        progress.append(r);continue
                    if folder.exists():folder.rename(folder.with_name(eid+'.partial_'+str(time.time_ns())))
                    folder.mkdir(parents=True)
                    detector.output=folder/'dino_raw';detector.output.mkdir()
                    base=baseline/'perception'/eid;br=read(base/'receipt.json')
                    assert br['raw_sha256']==e['raw_sha256']==sha(e['raw_path'])
                    assert all(sha(base/p)==h for p,h in br['files'].items())
                    with np.load(base/'B0.npz') as z:b0=z['mask_bits'][:e['actions']+1].copy()
                    np.savez_compressed(folder/'B0.npz',mask_bits=b0)
                    write(root/'pipeline_state.json',dict(status='RUNNING',episode=eid,pid=os.getpid(),
                        completed_episodes=len(progress),run_id=logger.run_id))
                    adapters={c:LosslessAdapter(memo,tracker) for c in CAMERAS}
                    for adapter in adapters.values():adapter.reset(eid)
                    values=[];records=[];times=[];sam_start=len(sam_calls);dino_start=len(detector.events)
                    try:
                        with h5py.File(e['raw_path']) as raw:
                            for t in range(e['actions']+1):
                                masks=[];tick=time.perf_counter()
                                for c in CAMERAS:
                                    context=dict(episode=eid,camera=c,timestep=t)
                                    detector.context=dict(episode_id=eid,parent_id=e['parent_id'],camera=c,decision_timestep=t,condition='P')
                                    mask,record=adapters[c].process(raw[c+'/rgb'][t],eid,t)
                                    masks.append(mask);records.append(dict(camera=c,**record))
                                values.append(np.packbits(np.stack(masks).ravel()))
                                times.append(dict(timestep=t,diagnostic_inference_ms=1000*(time.perf_counter()-tick),
                                    observer_free_latency=False,baseline_replayed_from_verified_prediction_cache=True))
                                if t%20==0:
                                    partial=folder/'P.partial.npz';tmp=folder/'P.partial.npz.tmp'
                                    with tmp.open('wb') as stream:np.savez_compressed(stream,mask_bits=np.stack(values))
                                    tmp.replace(partial)
                                    write(folder/'partial_state.json',dict(last_completed_timestep=t,frames=len(values),
                                        completed_episode=False,restart_from_t0_required=True))
                                    write(folder/'sam_calls.partial.json',sam_calls[sam_start:])
                        assert len(values)==len(b0)==e['actions']+1
                        np.savez_compressed(folder/'P.npz',mask_bits=np.stack(values))
                        write(folder/'vision_records.json',records);write(folder/'sam_calls.json',sam_calls[sam_start:])
                        write(folder/'dino_events.json',detector.events[dino_start:]);write(folder/'timing.json',times)
                        r=dict(episode=eid,parent_id=e['parent_id'],frames=len(values),raw_sha256=e['raw_sha256'],
                            source_digest=digest,GT_forward_input=False,causal_reset=True,
                            B0_receipt_sha256=sha(base/'receipt.json'),actual_sam_frame_inferences=len(sam_calls)-sam_start,
                            actual_dino_forwards=len(detector.events)-dino_start,actual_dinov2_forwards=0,
                            files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
                        write(folder/'receipt.json',r);progress.append(r)
                        write(root/'progress.json',progress)
                        print('lossless Development',len(progress),'/',len(entries),eid,len(values),flush=True)
                    finally:
                        for adapter in adapters.values():adapter.close()
                        adapters.clear();memo.cache.clear();sam_calls.clear();detector.events.clear();torch.cuda.empty_cache()
                assert len(progress)==expected['episodes'] and sum(r['frames'] for r in progress)==expected['frames']
                write(root/'pipeline_state.json',dict(status='DEVELOPMENT_P_INFERENCE_COMPLETE',run_id=logger.run_id,
                    episodes=len(progress),frames=sum(r['frames'] for r in progress),GT_evaluation_pending=True,whole_goal_complete=False))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),
                    pid=os.getpid(),partial_preserved=True,restart_incomplete_episode_from_t0=True))
                raise
            finally:
                tracker.predictor._run_single_frame_inference=original;detector.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition'):p.add_argument(name,type=Path)
    p.add_argument('--engineering',action='store_true');a=p.parse_args();run(a.root,a.baseline,a.partition,a.engineering)
