"""Independent orchestrator: wait for actual baseline proof, then diagnostics.

Never run an enabled DINOv2 diagnostic in the original baseline process. No
automatic independent-test opening or unsupported candidate promotion.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read, write, sha


def run(root, prerequisite, model_folder):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'source_identity.json', {str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    deadline=time.monotonic()+1800
    while True:
        state=read(prerequisite/'pipeline_state.json')
        if state['status']=='FAILED':
            raise RuntimeError('PREREQUISITE_FAILED: '+str(state))
        if state['status']=='AUDIT_PHASE_COMPLETE':
            break
        if time.monotonic()>deadline:
            raise TimeoutError('baseline prerequisite did not finish within 1800s')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_BASELINE_PROOF',pid=os.getpid()))
        time.sleep(10)
    assert read(prerequisite/'baseline_forward_equivalence.json')['status']=='PASS'
    commands=[
        [sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts','-v'],
        [sys.executable,'-B','-u','-m','graph.task10i_stage4a2.dinov2_matching_audit',
         str(root/'matching'),str(model_folder)],
    ]
    for i,cmd in enumerate(commands):
        write(root/'pipeline_state.json',dict(status='RUNNING',stage=i,pid=os.getpid(),command=cmd))
        with (root/f'stage{i}.log').open('w') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=i,exit_code=result.returncode,
                partial_outputs_preserved=True,whole_goal_complete=False))
            raise RuntimeError(f'Diagnostic stage {i} failed: see stage{i}.log')
    write(root/'pipeline_state.json',dict(status='HISTORICAL_MATCHING_DIAGNOSTIC_COMPLETE',
        whole_goal_complete=False, independent_final_executed=False,
        next_stage='DINO_PROPOSAL_AND_SAM2_CAUSAL_CANDIDATE_RESEARCH'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('prerequisite',type=Path)
    p.add_argument('model_folder',type=Path)
    a=p.parse_args()
    run(a.root,a.prerequisite,a.model_folder)
