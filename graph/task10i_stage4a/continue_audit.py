"""Resume evidence into a new frozen source/output without repeating DINO."""
import argparse
import os
from pathlib import Path
from .run_pipeline import audit,read,write,sha

def main(root,previous):
    import fcntl
    root.mkdir(parents=True,exist_ok=False)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from utils.experiment_logger import ExperimentLogger
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_reference_and_stagewise_audit',seed=42,root=root/'experiments',job_type='evaluation',config=dict(previous_root=str(previous),production_adoption=False)) as log:
            write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),previous_root=str(previous)))
            stage='SOURCE_AND_DATA_AUDIT'
            try:
                audit(root)
                old=read(previous/'source_freeze.json')['files'];current=read(root/'source_freeze.json')['files']
                changed=[k for k,v in old.items() if not k.startswith('graph/task10i_stage4a/') and current.get(k)!=v]
                assert not changed,changed
                baseline=read(previous/'baseline_forward_equivalence.json');assert baseline['status']=='PASS'
                write(root/'baseline_forward_equivalence.json',dict(**baseline,reused_evidence=str(previous/'baseline_forward_equivalence.json'),reused_evidence_sha256=sha(previous/'baseline_forward_equivalence.json'),dependency_source_equality=True))
                from .replay_reference import run as replay
                from .diagnose_cached import run as diagnose
                for stage,fn in [('GT_REFERENCE_VALIDATION',replay),('STAGEWISE_FAILURE_INSTRUMENTATION',diagnose)]:
                    write(root/'pipeline_state.json',dict(stage=stage,status='RUNNING',run_id=log.run_id,pid=os.getpid()))
                    fn(root)
                write(root/'pipeline_state.json',dict(stage='FAILURE_CASE_COLLECTION_AND_ROOT_CAUSE_QUANTIFICATION',status='PENDING_IMPLEMENTATION',completed=False,run_id=log.run_id))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(stage=stage,status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);a=p.parse_args();main(a.root,a.previous)
