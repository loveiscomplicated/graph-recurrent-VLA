"""Serialize same-input preprocessing and matched-feature crop controls after C/BC."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,causal):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'source_identity.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    deadline=time.monotonic()+24*3600
    while True:
        statefile=causal/'pipeline_state.json'
        if statefile.exists():
            state=read(statefile)
            if state['status'] in ('TRI_MODEL_ENGINEERING_AND_HISTORICAL_COMPLETE','FAILED'):break
        live=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                value=p.read_bytes()
                if b'graph.task10i_stage4a2.tri_model_runner' in value and str(causal).encode() in value:live.append(p.parent.name)
            except OSError:pass
        if not live:raise RuntimeError('C/BC runner missing without terminal receipt')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_C_BC_RUNNER',pids=live,pid=os.getpid()))
        if time.monotonic()>deadline:raise TimeoutError('component control dependency deadline')
        time.sleep(10)
    # These diagnostic tracks remain valid even if a practical candidate fails.
    # Available actual proposals are read only; missing proposals are not invented.
    commands=[('preprocessing',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.preprocessing_audit',str(root/'preprocessing')]),
        ('matched_crop',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.crop_feature_audit',str(root/'matched_crop'),str(causal/'historical')])]
    for name,command in commands:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),command=command,
            causal_dependency_status=state['status']))
        with (root/f'{name}.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=name,returncode=result.returncode,partial_preserved=True))
            raise RuntimeError(f'{name} failed; preserve output and inspect')
    write(root/'pipeline_state.json',dict(status='COMPONENT_PREPROCESSING_AND_MATCHED_CROP_COMPLETE',
        independent_test=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('causal',type=Path)
    a=p.parse_args();run(a.root,a.causal)
