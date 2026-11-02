"""Finite, hash-checked tmux coordinator for registered statistical gates only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read, write, sha


def run(root, jobs_path, criteria):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True, exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        package=Path(__file__).resolve().parent
        identity=dict(jobs_sha256=sha(jobs_path), criteria_sha256=sha(criteria),
                      source={str(p):sha(p) for p in package.glob('*.py')})
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==identity
        else:write(frozen,identity)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_registered_gates_pipeline',
                seed=1011202643,root=root/'experiments',job_type='evaluation',config=identity) as logger:
            subprocess.run([sys.executable,'-B','-m','unittest',
                            'graph.task10i_stage4a2.test_registered_evaluation','-v'],check=True)
            for job in read(jobs_path):
                child=root/job['name']
                receipt=root/'stages'/(job['name']+'.json')
                if receipt.exists():
                    r=read(receipt)
                    assert r['identity']==identity
                    assert all(sha(root/p)==h for p,h in r['files'].items())
                    continue
                dependency=job.get('dependency')
                if dependency:
                    deadline=time.monotonic()+24*3600
                    while True:
                        state=read(Path(dependency)/'pipeline_state.json')
                        if state['status']==job['dependency_status']:break
                        if state['status']=='FAILED':raise RuntimeError('dependency failed, inspect preserved output')
                        live=[]
                        for p in Path('/proc').glob('[0-9]*/cmdline'):
                            try:
                                args=p.read_bytes().split(b'\0')
                                module=job['dependency_module'].encode()
                                if module in args and str(dependency).encode() in args:live.append(p.parent.name)
                            except OSError:pass
                        if not live:raise RuntimeError('dependency coordinator missing without terminal completion')
                        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_DEPENDENCY',
                            name=job['name'],pids=live,pid=os.getpid(),whole_goal_complete=False))
                        if time.monotonic()>deadline:raise TimeoutError('finite dependency deadline')
                        time.sleep(10)
                if child.exists():child.rename(child.with_name(child.name+'.partial_'+str(time.time_ns())))
                command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.registered_evaluation',
                         str(child),job['paired'],job['expected'],str(criteria),
                         '--candidate',job['candidate'],'--scope','development']
                write(root/'pipeline_state.json',dict(status='RUNNING',name=job['name'],command=command,pid=os.getpid()))
                log=root/(job['name']+'.log')
                with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                assert read(child/'statistical_gate_receipt.json')['candidate']==job['candidate']
                write(receipt,dict(status='PASS',identity=identity,command=command,
                    files={str(p.relative_to(root)):sha(p) for p in child.rglob('*') if p.is_file()}))
            write(root/'pipeline_state.json',dict(status='DEVELOPMENT_REGISTERED_GATES_COMPLETE',
                run_id=logger.run_id,candidate_selected=False,Internal_and_Final_opened=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','jobs','criteria'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.jobs,a.criteria)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
