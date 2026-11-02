"""Serialize GPU experiments; run contracts, live engineering and C/BC research."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,dependency):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'source_identity.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    deadline=time.monotonic()+24*3600
    while True:
        state=read(dependency/'pipeline_state.json')
        if state['status']=='NEW_DEVELOPMENT_INFERENCE_COMPLETE':break
        if state['status']=='FAILED':raise RuntimeError('new Development inference exhausted its retry')
        live=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command=path.read_bytes()
                if b'graph.task10i_stage4a2.mask_forward_runner' in command and str(dependency).encode() in command:
                    live.append(path.parent.name)
            except OSError:pass
        if not live:raise RuntimeError('GPU dependency coordinator missing without terminal receipt')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_GPU_DEPENDENCY',pids=live,pid=os.getpid()))
        if time.monotonic()>deadline:raise TimeoutError('GPU serialization deadline')
        time.sleep(10)
    commands=[('contracts',[sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts','-v']),
              ('engineering',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.integration_ablation',str(root/'engineering'),'--engineering']),
              ('historical',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.integration_ablation',str(root/'historical')])]
    for name,command in commands:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),command=command))
        with (root/f'{name}.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=name,returncode=result.returncode,partial_preserved=True))
            raise RuntimeError(f'{name} failed; preserve output and inspect log')
        if name!='contracts':
            scope=read(root/name/'scope.json')
            assert scope['status']==('TRI_MODEL_ENGINEERING_COMPLETE' if name=='engineering' else 'HISTORICAL_C_BC_ABLATION_COMPLETE')
    write(root/'pipeline_state.json',dict(status='TRI_MODEL_ENGINEERING_AND_HISTORICAL_COMPLETE',
        independent_test_executed=False,candidate_selected=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dependency',type=Path)
    a=p.parse_args();run(a.root,a.dependency)
