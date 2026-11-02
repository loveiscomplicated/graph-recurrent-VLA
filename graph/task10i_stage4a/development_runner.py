"""Autonomous broad development audit with isolated stage processes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from .run_pipeline import read,write,sha

def main(root,previous):
    root.mkdir(parents=True,exist_ok=False)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from utils.experiment_logger import ExperimentLogger
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_all_development_baseline_audit',seed=42,root=root/'experiments',job_type='evaluation',config=dict(previous_root=str(previous),partition='preregistered_development24',validation_accessed=False)) as log:
            stage='SOURCE_EQUIVALENCE'
            try:
                source=Path(__file__).resolve().parents[2];old=read(previous/'source_freeze.json')['files']
                changed=[k for k,v in old.items() if not k.startswith('graph/task10i_stage4a/') and sha(source/k)!=v];assert not changed,changed
                assert read(previous/'baseline_forward_equivalence.json')['status']=='PASS'
                write(root/'protocol.json',read(previous/'protocol.json'))
                write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),previous_root=str(previous),source_root=str(source)))
                write(root/'source_freeze.json',dict(files={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py') if '__pycache__' not in p.parts},unchanged_baseline_dependencies=True))
                engineering=root/'engineering';engineering.mkdir();write(engineering/'protocol.json',read(root/'protocol.json'))
                stages=[('CONTRACT_TESTS',[sys.executable,'-B','-m','unittest','graph.task10i_stage4a.test_contracts','graph.integration_baseline.tests.test_online_vision','graph.integration_baseline.tests.test_causality','graph.plan_a_geometry.tests','graph.plan_a_bc.tests','-v']),
                    ('ENGINEERING_REFERENCE_SMOKE',[sys.executable,'-B','-u','-m','graph.task10i_stage4a.development_reference',str(engineering),str(previous),'--smoke']),
                    ('ENGINEERING_PERCEPTION_SMOKE',[sys.executable,'-B','-u','-m','graph.task10i_stage4a.development_perception',str(engineering),str(previous),'--smoke']),
                    ('ALL_DEVELOPMENT_GT_REFERENCE',[sys.executable,'-B','-u','-m','graph.task10i_stage4a.development_reference',str(root),str(previous)]),
                    ('ALL_DEVELOPMENT_CAUSAL_BASELINE',[sys.executable,'-B','-u','-m','graph.task10i_stage4a.development_perception',str(root),str(previous)]),
                    ('DEVELOPMENT_ROOT_CAUSE_TABLES',[sys.executable,'-B','-u','-m','graph.task10i_stage4a.aggregate_development',str(root)])]
                for stage,command in stages:
                    write(root/'pipeline_state.json',dict(stage=stage,status='RUNNING',run_id=log.run_id,pid=os.getpid()))
                    with (root/(stage.lower()+'.log')).open('x') as out:subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,check=True)
                    print(stage,'FINISHED',flush=True)
                for name in ('development_reference_summary.json','development_perception_summary.json','development_parent_macro.json','failure_case_inventory.json'):log.link_output(root/name,kind='metrics')
                write(root/'pipeline_state.json',dict(stage='ALL_DEVELOPMENT_BASELINE_AUDIT_FINISHED',status='SUBSTAGE_FINISHED',completed_goal=False,run_id=log.run_id))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(stage=stage,status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);a=p.parse_args();main(a.root,a.previous)
