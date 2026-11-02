"""Queue evidence analysis after the full baseline and corrected-reference gate."""
import argparse
import os
import time
from pathlib import Path
from .run_pipeline import read,write,sha

def run(root,baseline,corrected):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_detector_temporal_failure_audit',seed=42,root=root/'experiments',job_type='evaluation',config=dict(baseline_root=str(baseline),corrected_root=str(corrected),model_forward_calls=0)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(__file__)))
        stage='WAITING_BASELINE_CORRECTED_REFERENCE'
        try:
            write(root/'pipeline_state.json',dict(stage=stage,status='RUNNING',run_id=log.run_id,pid=os.getpid()))
            while not (corrected/'reevaluation_summary.json').exists():
                for r in (baseline,corrected):
                    if read(r/'pipeline_state.json').get('status')=='FAILED':raise RuntimeError('prerequisite failed '+str(r))
                time.sleep(10)
            from .detector_audit import run as detector
            from .temporal_audit import run as temporal
            for stage,fn in [('RAW_DETECTOR_AND_ACTUAL_INITIALIZATION',detector),('TEMPORAL_AND_PHASE_AUDIT',temporal)]:
                write(root/'pipeline_state.json',dict(stage=stage,status='RUNNING',run_id=log.run_id,pid=os.getpid()));fn(root,baseline,corrected)
            for name in ('detector_audit_scope.json','temporal_scope.json'):log.link_output(root/name,kind='metrics')
            write(root/'pipeline_state.json',dict(stage='EXTENDED_DEVELOPMENT_AUDIT_FINISHED',status='SUBSTAGE_FINISHED',completed_goal=False,run_id=log.run_id))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(stage=stage,status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path);a=p.parse_args();run(a.root,a.baseline,a.corrected)
