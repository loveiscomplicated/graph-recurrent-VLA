"""Development-only candidate execution after completed root-cause audits."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .run_pipeline import read, write, sha


def run(root, audits, robot_memo=False, reuse=None, smoke_evidence=None):
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    write(root / 'source_freeze.json', dict(files={str(p.relative_to(source)):sha(p)
          for p in source.rglob('*.py') if '__pycache__' not in p.parts}))
    write(root / 'pipeline_state.json', dict(status='RUNNING', stage='WAITING_ROOT_CAUSE_AUDITS', pid=os.getpid()))
    while read(audits / 'coordinator_state.json')['status'] != 'AUDITS_FINISHED':
        assert read(audits / 'coordinator_state.json')['status'] != 'FAILED'
        time.sleep(10)
    assert read(audits / 'corrected/reevaluation_summary.json')['exact_frames'] == 4162
    extra=(['--robot-memo'] if robot_memo else [])+(['--reuse-completed',str(reuse)] if reuse else [])
    stages=[]
    if smoke_evidence:
        assert read(smoke_evidence)['status']=='FEASIBILITY_SMOKE_FINISHED'
        write(root/'reused_smoke_evidence.json',dict(path=str(smoke_evidence),sha256=sha(smoke_evidence),policy_source_unchanged=True))
    else:stages.append(('V2_ENGINEERING_SMOKE', ['-m','graph.task10i_stage4a.reanchor_smoke',str(root/'smoke'),str(audits/'baseline'),str(audits/'corrected'),'--refine']))
    stages.append(('FULL_DEVELOPMENT_V2', ['-m','graph.task10i_stage4a.paired_candidate',str(root/'candidate'),str(audits/'baseline'),str(audits/'corrected'),*extra]))
    try:
        for stage,args in stages:
            write(root/'pipeline_state.json',dict(status='RUNNING',stage=stage,pid=os.getpid()))
            with (root/(stage.lower()+'.log')).open('x') as output:
                subprocess.run([sys.executable,'-B','-u',*args],stdout=output,stderr=subprocess.STDOUT,check=True)
        write(root/'pipeline_state.json',dict(status='DEVELOPMENT_CANDIDATE_FINISHED',whole_goal_complete=False))
    except BaseException as exc:
        write(root/'pipeline_state.json',dict(status='FAILED',stage=stage,error=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('audits',type=Path)
    p.add_argument('--robot-memo',action='store_true');p.add_argument('--reuse-completed',type=Path);p.add_argument('--smoke-evidence',type=Path)
    a=p.parse_args();run(a.root,a.audits,a.robot_memo,a.reuse_completed,a.smoke_evidence)
