"""Final descriptive slices after all independent/supplemental evidence."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,independent,supplemental,mini):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze=independent/'freeze';candidate=read(freeze/'candidate_config_frozen.json')['candidate']
        criteria=freeze/'adoption_criteria_frozen.json'
        config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
            independent=str(independent),supplemental=str(supplemental),mini=str(mini),candidate=candidate,
            criteria_sha256=sha(criteria),candidate_config_sha256=sha(freeze/'candidate_config_frozen.json'),
            posthoc_only=True,models_run=False,adoption_criteria_changed=False)
        frozen=root/'protocol_frozen.json'
        if frozen.exists():assert read(frozen)==config
        else:write(frozen,config)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_final_descriptive_strata_pipeline',seed=42,
                root=root/'experiments',job_type='analysis',config=config) as logger:
            wait_for(root,supplemental/'pipeline_state.json','POST_TEST_SUPPLEMENTAL_ROOT_CAUSE_DIAGNOSTICS_COMPLETE',
                'graph.task10i_stage4a2.supplemental_pipeline',supplemental)
            for name,base in [('mini',mini),('internal',independent/'evaluation/internal'),('final',independent/'evaluation/final')]:
                output=root/name;receipt=root/'stages'/(name+'.json')
                command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.stratified_evaluation',str(output),
                    str(base/'paired'),str(base/'expected_counts.json'),str(criteria),'--candidate',candidate]
                if receipt.exists():
                    r=read(receipt);assert r['command']==command and r['protocol_sha256']==sha(frozen)
                    assert all(sha(root/p)==h for p,h in r['files'].items());continue
                assert not output.exists(),'retain partial descriptive output; diagnose before retry'
                write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),run_id=logger.run_id))
                with (root/(name+'.log')).open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                write(receipt,dict(status='PASS',command=command,protocol_sha256=sha(frozen),
                    files={str(p.relative_to(root)):sha(p) for p in output.rglob('*') if p.is_file()}))
            write(root/'pipeline_state.json',dict(status='ALL_DESCRIPTIVE_STRATA_COMPLETE',run_id=logger.run_id,
                source_or_threshold_retuned=False,adoption_criteria_changed=False,models_run=False,
                full_report_and_completion_readback_pending=True,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','independent','supplemental','mini'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            no_candidate_tuning=True,whole_goal_complete=False))
        raise
