"""Autonomous post-evaluation decision/gallery; never schedules model tuning."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,independent):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        independent=str(independent),posthoc_only=True,tuning=False)
    frozen=root/'source_config_frozen.json'
    if frozen.exists():assert read(frozen)==config
    else:write(frozen,config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_post_evaluation_reporting_pipeline',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        wait_for(root,independent/'pipeline_state.json','FROZEN_INDEPENDENT_RESULTS_READY_FOR_FINAL_REPORT',
            'graph.task10i_stage4a2.frozen_pipeline_runner',independent)
        candidate=read(independent/'freeze/candidate_config_frozen.json')['candidate']
        commands=[('decision',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.final_decision',
            str(root/'decision'),str(independent/'evaluation'),str(independent/'freeze')]),
            ('gallery',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.reliability_gallery',
                str(root/'gallery'),str(independent/'evaluation'),'--candidate',candidate])]
        for name,command in commands:
            receipt=root/'stages'/(name+'.json')
            if receipt.exists():
                prior=read(receipt);assert prior['command']==command
                assert all(sha(root/p)==h for p,h in prior['files'].items());continue
            write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),run_id=logger.run_id))
            with (root/(name+'.log')).open('a') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            write(receipt,dict(status='PASS',command=command,
                files={str(p.relative_to(root)):sha(p) for p in (root/name).rglob('*') if p.is_file()}))
        write(root/'pipeline_state.json',dict(status='INDEPENDENT_DECISION_GALLERY_READBACK_COMPLETE',
            run_id=logger.run_id,full_Markdown_report_and_local_readback_pending=True,whole_goal_complete=False,
            Stage4B_started=False,training_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','independent'):p.add_argument(n,type=Path)
    a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            whole_goal_complete=False,no_tuning=True))
        raise
