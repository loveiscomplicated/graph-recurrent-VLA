"""Finite autonomous preregistered challenger freeze and independent evaluation.

The challenger is selected for confirmation/falsification from Development
mechanistic evidence. A failed Development gate is retained, never converted
into an adoption claim or used to relax the registered criteria.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,development,engineering,prefix,forward,collector,partition,protocol,design,profile):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[2]
    candidate='BC_WB_HANDOFF'
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=dict(candidate=candidate,source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
            development=str(development),engineering=str(engineering),prefix=str(prefix),forward=str(forward),
            collector=str(collector),partition_sha256=sha(partition),protocol_sha256=sha(protocol),design_sha256=sha(design),
            selection_rationale='Development-only causal investigation: preserve Main Peg and Wrist Box, correct Wrist Peg/Main Box, hand off to a later current-evidence-confirmed original initialization instead of retaining an incomplete early anchor; independently test one challenger even if Development gates fail; preserve all failures and the original criteria',
            test_driven_retuning=False,adoption_authorized=False)
        frozen=root/'coordinator_config_frozen.json'
        if frozen.exists():assert read(frozen)==config
        else:write(frozen,config)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_registered_independent_coordinator',seed=42,
                root=root/'experiments',job_type='evaluation',config=config) as logger:
            wait_for(root,development/'pipeline_state.json','DEVELOPMENT_WRIST_BOX_HANDOFF_EVALUATION_COMPLETE',
                'graph.task10i_stage4a2.main_box_pipeline',development)
            wait_for(root,engineering/'pipeline_state.json','FINAL_SERIALIZATION_MODEL_GRAPH_GT_ENGINEERING_COMPLETE',
                'graph.task10i_stage4a2.final_path_engineering',engineering)
            assert read(prefix/'pipeline_state.json')['status']=='ACTUAL_SHARED_PREFIX_INVARIANCE_PASS'
            # Evaluation postprocessors were executed by the miniature path;
            # no independent RGB-D has been opened at this barrier.
            engsource=engineering.parent/'source/graph/task10i_stage4a2'
            for name in ('evaluation_inputs.py','paired_evaluation.py','aggregate_evaluation.py',
                         'registered_evaluation.py','matching_outcomes.py','latency_report.py'):
                assert sha(Path(__file__).with_name(name))==sha(engsource/name)
            freeze=root/'freeze';python=[sys.executable,'-B','-u','-m']
            commands=[]
            if not (freeze/'candidate_selection_receipt.json').exists():
                assert not freeze.exists(),'partial freeze retained; requires explicit diagnosis, never silently reuse'
                commands.append(('freeze',python+['graph.task10i_stage4a2.freeze_candidate',str(freeze),str(development),
                    str(forward),str(source),str(collector),str(profile),str(design),str(protocol),str(partition),
                    '--candidate',candidate,'--rationale',config['selection_rationale']]))
            commands.append(('independent',python+['graph.task10i_stage4a2.independent_pipeline',str(root/'evaluation'),
                str(freeze),str(partition),str(protocol),str(design),str(collector)]))
            for name,command in commands:
                write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),run_id=logger.run_id,
                    command=command,whole_goal_complete=False))
                with (root/(name+'.log')).open('a') as stream:
                    subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            write(root/'pipeline_state.json',dict(status='FROZEN_INDEPENDENT_RESULTS_READY_FOR_FINAL_REPORT',
                run_id=logger.run_id,candidate=candidate,whole_goal_complete=False,production_changed=False,
                Stage4B_started=False,training_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','development','engineering','prefix','forward','collector','partition','protocol','design','profile'):
        p.add_argument(n,type=Path)
    a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            whole_goal_complete=False,source_or_threshold_retuned=False))
        raise
