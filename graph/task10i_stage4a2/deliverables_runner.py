"""Wait for verified independent reports, then collect reviewable artifacts."""
import argparse
import os
from pathlib import Path
from .lazy_feature_control import wait_for
from .deliverables import run as collect
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,independent,reporting):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        independent=str(independent),reporting=str(reporting),no_models_or_tuning=True)
    write(root/'source_config_frozen.json',config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_named_deliverables_pipeline',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        wait_for(root,reporting/'pipeline_state.json','INDEPENDENT_DECISION_GALLERY_READBACK_COMPLETE',
            'graph.task10i_stage4a2.reporting_runner',reporting)
        write(root/'pipeline_state.json',dict(status='RUNNING',stage='collect_named_artifacts',pid=os.getpid(),run_id=logger.run_id))
        collect(root/'named',independent,reporting)
        write(root/'pipeline_state.json',dict(status='NAMED_DELIVERABLES_READY_FOR_FULL_MARKDOWN_AND_LOCAL_READBACK',
            run_id=logger.run_id,whole_goal_complete=False,Stage4B_started=False,training_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','independent','reporting'):p.add_argument(n,type=Path)
    a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            whole_goal_complete=False))
        raise
