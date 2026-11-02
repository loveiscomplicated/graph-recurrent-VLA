"""New Development36 B0/A mask forwards; no GT imports or evaluator calls.

A changes only Wrist Peg text to the historically supported prompt. B0 keeps
its original detector, initializer, camera states and propagation. Other model
and geometry changes must be independently evaluated in later ablations.
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
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


class WristPrompt:
    def __init__(self,detector):self.detector=detector
    def __call__(self,rgb,prompt):
        return self.detector(rgb,'red and white peg' if prompt=='red and white block' else prompt)


def run(root,partition_manifest,engineering=False):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino
    from graph.task10i_stage4a.development_perception import Memo
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=Path(__file__).resolve().parents[2]
        hashes={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils')
                for p in (source/folder).rglob('*.py')}
        digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        spec=dict(source_digest=digest,files=hashes,partition_manifest_sha256=sha(partition_manifest),
             partition='engineering' if engineering else 'new_development36',
             B0='original OnlineVisionAdapter both cameras',
             A='same SAM2 and original consumer; Wrist red and white block -> red and white peg only',
             dinov2_enabled=False,internal_or_final_access=False,GT_forward_input=False)
        if (root/'source_config_frozen.json').exists():assert read(root/'source_config_frozen.json')==spec
        else:write(root/'source_config_frozen.json',spec)
        parents=set(read(partition_manifest)['development'])
        entries=sorted((e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()
                        if e['parent_id'] in parents),key=lambda e:str(e['episode_id']))
        if engineering:entries=[dict(entries[0],actions=min(2,entries[0]['actions']),engineering_only=True)]
        write(root/'expected_counts.json',dict(parents=len({e['parent_id'] for e in entries}),
              episodes=len(entries),frames=sum(e['actions']+1 for e in entries),entries=entries))
        torch.set_num_threads(1);torch.manual_seed(42)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        rt=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_new_development_B0_A',
             seed=42,job_type='evaluation',root=root/'experiments',config=spec) as logger:
            dino=AuditedDino(device='cuda',model_id=rt['dino'],output=root/'dino_raw');memo=Memo(dino)
            if (root/'dino_events.json').exists():dino.events=read(root/'dino_events.json')
            tracker=SAM2Tracker(device='cuda',checkpoint=rt['sam'])
            progress=[]
            try:
                for e in entries:
                    eid=str(e['episode_id']);folder=root/'perception'/eid
                    if (folder/'receipt.json').exists():
                        receipt=read(folder/'receipt.json')
                        assert receipt['source_digest']==digest and receipt['raw_sha256']==e['raw_sha256']
                        assert receipt['frames']==e['actions']+1
                        assert all(sha(folder/p)==h for p,h in receipt['files'].items())
                        progress.append(receipt);continue
                    if folder.exists():
                        # Never append a suffix to a missing causal tracker memory.
                        folder.rename(folder.with_name(eid+'.partial_preserved_'+str(time.time_ns())))
                    folder.mkdir(parents=True)
                    assert sha(e['raw_path'])==e['raw_sha256']
                    adapters={c:OnlineVisionAdapter(memo,tracker) for c in CAMERAS}
                    changed=OnlineVisionAdapter(WristPrompt(memo),tracker)
                    for adapter in [*adapters.values(),changed]:adapter.reset(eid)
                    bits={'B0':[],'A':[]};records=[];times=[]
                    write(root/'pipeline_state.json',dict(status='RUNNING',stage='NEW_DEVELOPMENT_MASK_FORWARD',
                        episode=eid,completed_episodes=len(progress),pid=os.getpid(),run_id=logger.run_id))
                    try:
                        with h5py.File(e['raw_path']) as f:
                            for t in range(e['actions']+1):
                                masks=[];tick=time.perf_counter()
                                for c in CAMERAS:
                                    dino.context=dict(episode_id=eid,parent_id=e['parent_id'],camera=c,decision_timestep=t,condition='B0')
                                    m,record=adapters[c].process(f[c+'/rgb'][t],eid,t)
                                    masks.append(m);records.append(dict(condition='B0',camera=c,**record))
                                base=np.stack(masks);b0_ms=1000*(time.perf_counter()-tick)
                                tick=time.perf_counter()
                                dino.context=dict(episode_id=eid,parent_id=e['parent_id'],camera='hand_camera',decision_timestep=t,condition='A')
                                wrist,record=changed.process(f['hand_camera/rgb'][t],eid,t)
                                candidate=base.copy();candidate[1]=wrist
                                bits['B0'].append(np.packbits(base.ravel()));bits['A'].append(np.packbits(candidate.ravel()))
                                records.append(dict(condition='A',camera='hand_camera',**record))
                                times.append(dict(timestep=t,b0_ms=b0_ms,a_wrist_only_ms=1000*(time.perf_counter()-tick),
                                    fair_end_to_end_latency=False,shared_dino_memo=True))
                        for condition,values in bits.items():
                            assert len(values)==e['actions']+1
                            np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(values))
                        write(folder/'vision_records.json',records);write(folder/'timing.json',times)
                        receipt=dict(episode=eid,parent_id=e['parent_id'],frames=e['actions']+1,
                            raw_sha256=e['raw_sha256'],source_digest=digest,causal_reset=True,
                            GT_forward_input=False,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
                        write(folder/'receipt.json',receipt);progress.append(receipt)
                        write(root/'progress.json',progress);write(root/'dino_events.json',dino.events)
                        print('new development',len(progress),'/',len(entries),eid,e['actions']+1,flush=True)
                    finally:
                        for adapter in [*adapters.values(),changed]:adapter.close()
                        memo.cache.clear();torch.cuda.empty_cache()
                assert len(progress)==len(entries) and sum(r['frames'] for r in progress)==sum(e['actions']+1 for e in entries)
                write(root/'pipeline_state.json',dict(status='NEW_DEVELOPMENT_B0_A_INFERENCE_COMPLETE',
                    run_id=logger.run_id,episodes=len(progress),frames=sum(r['frames'] for r in progress),
                    GT_evaluation_pending=True,whole_goal_complete=False))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),
                    completed_episodes=len(progress),resume='verify receipts, restart incomplete episode from t0'))
                raise
            finally:dino.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('partition_manifest',type=Path)
    p.add_argument('--engineering',action='store_true');a=p.parse_args();run(a.root,a.partition_manifest,a.engineering)
