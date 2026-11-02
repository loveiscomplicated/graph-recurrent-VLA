"""Serialize live frozen-condition compatibility checks after Development GPU.

Historical B0/P/C/BC masks and identical Graph tensors are the expected fixture;
this engineering job cannot open Internal/Final or choose a challenger.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .frozen_inference import array_digest
from .integration_ablation import WINDOWS
from .prompt_audit import AUDITS
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3


def run(root,dependency):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
            dependency=str(dependency),Internal_and_Final_access=False,
            windows=WINDOWS,latency='engineering only; concurrent CPU evaluation may contend')
        freeze=root/'source_config_frozen.json'
        if freeze.exists():assert read(freeze)==identity
        else:write(freeze,identity)
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        selected=[dict(entries[w['episode']],actions=w['last'],engineering_only=True) for w in WINDOWS]
        expected=dict(entries=selected,parents=len({e['parent_id'] for e in selected}),episodes=len(selected),
                      frames=sum(e['actions']+1 for e in selected))
        write(root/'expected_counts.json',expected)
        deadline=time.monotonic()+24*3600
        while read(dependency/'pipeline_state.json')['status']!='DEVELOPMENT_C_BC_KEEP_INFERENCE_COMPLETE':
            state=read(dependency/'pipeline_state.json')
            if state['status']=='FAILED':raise RuntimeError('Development model stage failed')
            pid=state.get('pid');assert pid and Path('/proc',str(pid)).exists(),'missing live inference dependency'
            write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_GPU_DEPENDENCY',pid=os.getpid(),dependency_pid=pid))
            if time.monotonic()>deadline:raise TimeoutError('GPU dependency deadline')
            time.sleep(10)
        conditions=[('B0',[]),('P',[]),('C_KEEP',['--geometry-cache']),
                    ('BC_KEEP',['--geometry-cache','--lazy-matching'])]
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_live_wrapper_compatibility',seed=42,
                root=root/'experiments',job_type='evaluation',config=identity) as logger:
            subprocess.run([sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts',
                'graph.task10i_stage4a2.test_registered_evaluation','graph.task10i_stage4a2.test_dino_attribution',
                'graph.task10i_stage4a.test_contracts','graph.task10i_stage4a.test_perframe_cache','-v'],check=True)
            for condition,options in conditions:
                child=root/condition;receipt=root/'stages'/(condition+'.json')
                if receipt.exists():
                    r=read(receipt);assert r['identity']==identity
                    assert all(sha(root/p)==h for p,h in r['files'].items());continue
                command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.frozen_inference',str(child),
                    str(root/'expected_counts.json'),'--condition',condition,'--engineering',*options]
                for attempt in range(2):
                    write(root/'pipeline_state.json',dict(status='RUNNING',condition=condition,attempt=attempt,
                        pid=os.getpid(),command=command))
                    log=root/(condition+f'_attempt{attempt}.log')
                    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode==0:break
                    if attempt or 'out of memory' not in log.read_text().lower():raise RuntimeError('preserved live engineering failure: '+str(log))
                state=read(child/'pipeline_state.json');assert state['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
                checks=[]
                for e in selected:
                    eid=str(e['episode_id']);folder=child/'perception'/eid
                    if condition=='B0':
                        old=AUDITS/'baseline/perception'/eid/'masks.npz';key='mask_bits'
                    elif condition=='P':
                        old=Path('/workspace/task10i_stage4a2_component_controls_20261101_v1/run/preprocessing/episodes')/eid/'LOSSLESS_PNG.npz';key='mask_bits'
                    else:
                        old=Path('/workspace/task10i_stage4a2_keep_control_20261101_v1/run/historical/episodes')/eid/(('BC' if condition=='BC_KEEP' else 'C')+'.npz');key='mask_bits'
                    with np.load(old) as z:target=z[key][:e['actions']+1].copy()
                    with np.load(folder/(condition+'.npz')) as z:actual=z['mask_bits'].copy()
                    assert np.array_equal(actual,target),('live mask mismatch',condition,eid,np.flatnonzero(np.any(actual!=target,axis=1)).tolist())
                    graph_count=0
                    if condition!='P':
                        label={'B0':'B0','C_KEEP':'C','BC_KEEP':'BC'}[condition]
                        base=Path('/workspace/task10i_stage4a2_keep_control_20261101_v1/run/historical/graphs')/eid/label
                        for frame in read(folder/'graph_fingerprints.json'):
                            with np.load(base/f't{frame["timestep"]:04d}.npz') as z:
                                for k,digest in frame['tensors'].items():assert array_digest(z[k])==digest,('live Graph mismatch',condition,eid,frame['timestep'],k)
                            graph_count+=1
                    checks.append(dict(episode=eid,mask_frames=len(actual),mask_bytes_equal=True,
                        Graph_five_tensors_equal_frames=graph_count,fixture=str(old),fixture_sha256=sha(old)))
                write(root/('equivalence_'+condition+'.json'),dict(status='PASS',condition=condition,checks=checks,
                    cache_or_lazy_changes_output=False,GT_forward_input=False,latency_is_engineering_only=True))
                write(receipt,dict(status='PASS',identity=identity,
                    files={str(p.relative_to(root)):sha(p) for p in child.rglob('*') if p.is_file()}))
            write(root/'scope.json',dict(status='LIVE_WRAPPER_COMPATIBILITY_COMPLETE',run_id=logger.run_id,
                conditions=[c for c,_ in conditions],frames_per_condition=expected['frames'],Internal_and_Final_access=False,
                optimized_BC_semantics_equal_historical=True,GT_forward_input=False,fair_latency_executed=False,
                files={str(p.relative_to(root)):sha(p) for p in root.glob('equivalence_*.json')},whole_goal_complete=False))
            write(root/'pipeline_state.json',dict(status='LIVE_WRAPPER_COMPATIBILITY_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dependency',type=Path);a=p.parse_args()
    try:run(a.root,a.dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
