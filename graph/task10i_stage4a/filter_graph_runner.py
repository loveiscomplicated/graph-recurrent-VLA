import argparse
import os
import time
from pathlib import Path
from .run_pipeline import read,write,sha

def run(root,baseline,corrected):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_robot_gate_graph_source_audit',seed=42,root=root/'experiments',job_type='evaluation',config=dict(baseline_root=str(baseline),corrected_root=str(corrected),model_forwards=0)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(Path(__file__).with_name('filter_graph_audit.py'))))
        try:
            write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_BASELINE_CORRECTED_REFERENCE',run_id=log.run_id,pid=os.getpid()))
            while not (corrected/'reevaluation_summary.json').exists():
                for r in (baseline,corrected):
                    if read(r/'pipeline_state.json').get('status')=='FAILED':raise RuntimeError('prerequisite failed '+str(r))
                time.sleep(10)
            write(root/'pipeline_state.json',dict(status='RUNNING',stage='INDEPENDENT_FILTER_GATE_GRAPH_SOURCE_AUDIT',run_id=log.run_id,pid=os.getpid()))
            from .filter_graph_audit import run as audit
            audit(root,baseline,corrected);log.link_output(root/'filter_graph_scope.json',kind='metrics')
            write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='INDEPENDENT_FILTER_GATE_GRAPH_SOURCE_AUDIT_FINISHED',completed_goal=False,run_id=log.run_id))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path);a=p.parse_args();run(a.root,a.baseline,a.corrected)
