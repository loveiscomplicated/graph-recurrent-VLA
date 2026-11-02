"""Full new Development36 C_KEEP/BC_KEEP causal corrections of frozen B0.

No GT/evaluator imports. Each condition/episode gets isolated fresh memory and
reference bank. The negative suppression control remains a separate frozen run.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .dinov2_runtime import CurrentFeatures
from .tri_model_integration import CausalCorrection
from .model_trace import SAMTrace
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS

CONDITIONS=('C_KEEP','BC_KEEP')


def run(root,baseline,partition,engineering=False,conditions=CONDITIONS,geometry_cache=False,
        engineering_episode=None,engineering_last=None):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino
    from graph.task10i_stage4a.development_perception import Memo
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.task10i_stage4a.perframe_cache import install
    assert conditions and set(conditions)<={'C_KEEP','BC_KEEP','BC_MAIN_BOX','BC_WB_HANDOFF'} and len(set(conditions))==len(conditions)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        files={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils') for p in (source/folder).rglob('*.py')}
        digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        expected=read(baseline/'expected_counts.json')
        assert {e['parent_id'] for e in expected['entries']}==set(read(partition)['development'])
        entries=expected['entries']
        if engineering:
            selected=next(e for e in entries if str(e['episode_id'])==engineering_episode) if engineering_episode else entries[0]
            last=engineering_last if engineering_last is not None else 2
            assert 0<=last<=selected['actions']
            entries=[dict(selected,actions=last,engineering_only=True)]
        else:assert engineering_episode is None and engineering_last is None
        expected=dict(parents=len({e['parent_id'] for e in entries}),episodes=len(entries),
            frames=sum(e['actions']+1 for e in entries),entries=entries)
        rt=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(rt/'runtime_paths.json')
        identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        assert sha(paths['sam'])==identity['sam_sha256']
        assert all(sha(Path(paths['dino'])/p)==h for p,h in identity['dino_files'].items())
        spec=dict(source_digest=digest,files=files,partition_sha256=sha(partition),
            baseline_config_sha256=sha(baseline/'source_config_frozen.json'),conditions=list(conditions),geometry_cache=geometry_cache,
            model_identity=identity,uncertain_output_policy='preserve_current',
            feature='same-camera448 masked patch + frozen first supported reference; .6 cosine/.05 negative margin',
            GT_forward_input=False,Internal_or_Final_access=False,engineering=engineering)
        if (root/'source_config_frozen.json').exists():assert read(root/'source_config_frozen.json')==spec
        else:write(root/'source_config_frozen.json',spec)
        write(root/'expected_counts.json',expected)
        torch.set_num_threads(1);torch.manual_seed(42)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_Development_C_BC_KEEP',seed=42,
                root=root/'experiments',job_type='evaluation',config=spec) as logger:
            detector=AuditedDino(device='cuda',model_id=paths['dino'],output=root/'dino_raw');memo=Memo(detector)
            tracker=SAM2Tracker(device='cuda',checkpoint=paths['sam'])
            features=CurrentFeatures('/workspace/task10i_stage4a2_models/dinov2-small')
            write(root/'dinov2_model_identity.json',features.identity)
            trace=SAMTrace(tracker,root/'prompt_logits');progress=[]
            try:
                for e in entries:
                    eid=str(e['episode_id']);folder=root/'perception'/eid
                    if (folder/'receipt.json').exists():
                        r=read(folder/'receipt.json');assert r['source_digest']==digest and r['raw_sha256']==e['raw_sha256']
                        assert all(sha(folder/p)==h for p,h in r['files'].items());progress.append(r);continue
                    if folder.exists():folder.rename(folder.with_name(eid+'.partial_'+str(time.time_ns())))
                    folder.mkdir(parents=True);detector.output=folder/'dino_raw';detector.output.mkdir()
                    source_folder=baseline/'perception'/eid;br=read(source_folder/'receipt.json')
                    assert br['raw_sha256']==e['raw_sha256']==sha(e['raw_path'])
                    assert all(sha(source_folder/p)==h for p,h in br['files'].items())
                    with np.load(source_folder/'B0.npz') as z:bits=z['mask_bits'][:e['actions']+1].copy()
                    baseline_records=read(source_folder/'vision_records.json')
                    np.savez_compressed(folder/'B0.npz',mask_bits=bits)
                    for condition in conditions:
                        write(root/'pipeline_state.json',dict(status='RUNNING',episode=eid,condition=condition,
                            completed_episodes=len(progress),pid=os.getpid(),run_id=logger.run_id))
                        out=folder/condition;out.mkdir();trace.logits=out/'prompt_logits';trace.clear()
                        features.calls.clear();features.reset(eid)
                        geometry=CurrentGeometry(rt/'inputs',schema)
                        if geometry_cache:install(geometry)
                        policy=CausalCorrection(memo,tracker,geometry,features=features if condition.startswith('BC') else None,
                            uncertain_output_policy='preserve_current',lazy_matching=condition in ('BC_MAIN_BOX','BC_WB_HANDOFF'),
                            correction_slots=[('base_camera',1)] if condition=='BC_MAIN_BOX' else
                                [('hand_camera',0),('base_camera',1)] if condition=='BC_WB_HANDOFF' else None,
                            late_baseline_handoff=condition=='BC_WB_HANDOFF')
                        policy.reset(eid);found={};values=[];times=[]
                        by_t={}
                        for record in baseline_records:by_t.setdefault(record['timestep'],[]).append(record)
                        try:
                            with h5py.File(e['raw_path']) as raw:
                                for t in range(e['actions']+1):
                                    for r in by_t.get(t,[]):
                                        if r['condition']=='B0':
                                            for oid in r['new_object_ids']:found[r['camera'],oid-1]=True
                                    cameras,data,surfaces=geometry.from_h5(raw,t,eid)
                                    masks=dict(zip(CAMERAS,np.unpackbits(bits[t]).reshape(2,2,128,128).astype(bool)))
                                    detector.context=dict(episode_id=eid,parent_id=e['parent_id'],condition=condition,decision_timestep=t)
                                    tick=time.perf_counter();current=policy.process(cameras,data,surfaces,masks,found)
                                    times.append(dict(timestep=t,diagnostic_correction_ms=1000*(time.perf_counter()-tick),fair_latency=False))
                                    values.append(np.packbits(np.stack([current[c] for c in CAMERAS]).ravel()))
                                    if t%20==0:
                                        tmp=out/'mask_prefix.npz.tmp'
                                        with tmp.open('wb') as stream:np.savez_compressed(stream,mask_bits=np.stack(values))
                                        tmp.replace(out/'mask_prefix.npz')
                                        write(out/'partial_state.json',dict(frames=len(values),complete=False,restart_from_t0=True))
                            np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(values))
                            for name,data in [('events.json',policy.events),('candidate_masks.json',policy.candidate_masks),
                                ('reference_provenance.json',policy.reference_provenance),('feature_calls.json',features.calls),
                                ('sam_calls.json',trace.calls),('sam_modules.json',trace.modules),('timing.json',times),
                                ('baseline_handoff_events.json',policy.handoffs)]:write(out/name,data)
                            write(out/'state_machine.json',{str(k):v.events for k,v in policy.slots.items()})
                            write(out/'receipt.json',dict(condition=condition,frames=len(values),
                                accepted=sum(r['accepted'] for r in policy.events),
                                actual_dinov2_forwards=sum(r['model_forward'] for r in features.calls),
                                actual_reid_comparisons=sum(p['similarity'] is not None and p['reference_available'] for ev in policy.events for p in ev['proposals'])+
                                    sum(h['similarity'] is not None for h in policy.handoffs),
                                actual_sam_frame_inferences=len(trace.calls),GT_forward_input=False))
                        finally:policy.close();torch.cuda.empty_cache()
                    write(folder/'dino_events.json',detector.events)
                    r=dict(episode=eid,parent_id=e['parent_id'],frames=len(bits),raw_sha256=e['raw_sha256'],source_digest=digest,
                        B0_receipt_sha256=sha(source_folder/'receipt.json'),causal_reset=True,GT_forward_input=False,
                        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
                    write(folder/'receipt.json',r);progress.append(r);write(root/'progress.json',progress)
                    print('Development C/BC KEEP',len(progress),'/',len(entries),eid,len(bits),flush=True)
                    memo.cache.clear();detector.events.clear();trace.clear();features.calls.clear();torch.cuda.empty_cache()
                assert len(progress)==expected['episodes'] and sum(r['frames'] for r in progress)==expected['frames']
                write(root/'pipeline_state.json',dict(status='DEVELOPMENT_C_BC_KEEP_INFERENCE_COMPLETE' if tuple(conditions)==CONDITIONS else 'DEVELOPMENT_SCOPED_CORRECTION_INFERENCE_COMPLETE',run_id=logger.run_id,
                    episodes=len(progress),frames=expected['frames'],GT_evaluation_pending=True,whole_goal_complete=False))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),
                    partial_preserved=True,restart_incomplete_episode_from_t0=True));raise
            finally:trace.close();detector.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','baseline','partition'):p.add_argument(name,type=Path)
    p.add_argument('--engineering',action='store_true');p.add_argument('--geometry-cache',action='store_true')
    p.add_argument('--engineering-episode');p.add_argument('--engineering-last',type=int)
    p.add_argument('--conditions',nargs='+',default=list(CONDITIONS));a=p.parse_args()
    run(a.root,a.baseline,a.partition,a.engineering,a.conditions,a.geometry_cache,a.engineering_episode,a.engineering_last)
