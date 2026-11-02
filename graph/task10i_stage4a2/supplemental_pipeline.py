"""Post-test diagnostic completion, serialized after frozen live latency runs.

Adds root-cause evidence only. Cannot tune the challenger, update any registered
metric, or change the already frozen source/config/adoption rules.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from .lazy_feature_control import wait_for
from .final_capture import freeze_barrier
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,independent,mini):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze=independent/'freeze';frozen_identity=freeze_barrier(freeze,None,None,scope='internal')
        candidate=read(freeze/'candidate_config_frozen.json')['candidate'];criteria=freeze/'adoption_criteria_frozen.json'
        config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},independent=str(independent),mini=str(mini),
            frozen_challenger=frozen_identity,candidate=candidate,changes_registered_criteria=False,
            model_jobs_wait_for_all_independent_live_runs=True,
            raw_query_limit='recomputed frozen detector diagnostics, original postprocessed outputs checked exactly; not original below-threshold tensor equivalence',
            Oracle='GT t0 boxes only, future RGB prefix only thereafter; no practical benefit claim',
            extra_metrics='posthoc geometry residual, phase/occlusion/grasp strata, trigger miss/recovery and actual Re-ID event delays')
        frozen=root/'protocol_frozen.json'
        if frozen.exists():assert read(frozen)==config
        else:write(frozen,config)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_post_test_supplemental_diagnostics',seed=42,
                root=root/'experiments',job_type='evaluation',config=config) as logger:
            wait_for(root,independent/'pipeline_state.json','FROZEN_INDEPENDENT_RESULTS_READY_FOR_FINAL_REPORT',
                'graph.task10i_stage4a2.frozen_pipeline_runner',independent)
            assert freeze_barrier(freeze,None,None,scope='internal')==frozen_identity
            evaluation=independent/'evaluation';final=evaluation/'final';internal=evaluation/'internal'
            original_metrics={name:sha(evaluation/name/'statistical_gate_receipt.json') for name in
                ('internal/registered','final/registered_normal','final/registered_stress')}
            write(root/'independent_metrics_before_diagnostics.json',original_metrics)
            prefix=[sys.executable,'-B','-u','-m'];jobs=[]
            crops=Path('/workspace/task10i_stage4a2_component_controls_20261101_v1/run/matched_crop')
            # Read-only view for new Internal posthoc outputs. Never add files
            # to the committed independent-evaluation directories.
            view=root/'internal_analysis_view';view.mkdir(exist_ok=True)
            for name,target in [('candidate',internal/'candidate'),('baseline',internal/'baseline'),
                                ('paired',internal/'paired'),('matching',root/'internal_matching')]:
                link=view/name
                if link.is_symlink():assert link.resolve()==target.resolve()
                else:link.symlink_to(target,target_is_directory=True)
            ie=internal/'expected_counts.json';ir=internal/'references'
            jobs.extend([
                ('internal_matching','graph.task10i_stage4a2.matching_outcomes',[root/'internal_matching',ie,internal/'baseline',internal/'candidate',ir,'--candidate',candidate],False),
                ('internal_temporal','graph.task10i_stage4a2.temporal_mask_diagnostics',[root/'internal_temporal',internal/'masks',ir,ie,criteria,'--candidate',candidate],False),
                ('internal_geometry','graph.task10i_stage4a2.geometry_drift_audit',[root/'internal_geometry',internal/'paired',ir,ie,criteria,'--candidate',candidate],False),
                ('internal_recovery','graph.task10i_stage4a2.recovery_event_audit',[root/'internal_recovery',view,ie,'--candidate',candidate],False),
                ('internal_phase','graph.task10i_stage4a2.matching_phase_audit',[root/'internal_phase',crops,view,ie,ir,'--candidate',candidate,'--evaluation-scope','internal'],False)])
            def add(scope,base):
                expected=base/'expected_counts.json';references=base/'references'
                jobs.extend([
                    (scope+'_raw','graph.task10i_stage4a2.final_dino_diagnostic',[root/(scope+'_raw'),expected,base/'baseline',base/'candidate',references,'--candidate',candidate],True),
                    (scope+'_oracle','graph.task10i_stage4a2.oracle_initialization',[root/(scope+'_oracle'),expected,references,base/'baseline',base/'candidate',criteria,'--candidate',candidate],True),
                    (scope+'_geometry','graph.task10i_stage4a2.geometry_drift_audit',[root/(scope+'_geometry'),base/'paired',references,expected,criteria,'--candidate',candidate],False),
                    (scope+'_recovery','graph.task10i_stage4a2.recovery_event_audit',[root/(scope+'_recovery'),base,expected,'--candidate',candidate],False),
                    (scope+'_phase','graph.task10i_stage4a2.matching_phase_audit',[root/(scope+'_phase'),crops,base,expected,references,'--candidate',candidate],False)])
            add('mini',mini);add('final',final)
            for name,module,args,model in jobs:
                command=prefix+[module,*map(str,args)];receipt=root/'stages'/(name+'.json');output=Path(args[0])
                if receipt.exists():
                    r=read(receipt);assert r['command']==command and r['protocol_sha256']==sha(frozen)
                    assert all(sha(root/p)==h for p,h in r['files'].items());continue
                assert freeze_barrier(freeze,None,None,scope='internal')==frozen_identity
                if not model and output.exists():output.rename(output.with_name(output.name+'.uncommitted_'+str(time.time_ns())))
                for attempt in range(2 if model else 1):
                    log=root/(name+f'_attempt{attempt}.log')
                    write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),run_id=logger.run_id,
                        command=command,attempt=attempt,no_candidate_tuning=True))
                    with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                    if result.returncode==0:break
                    if not model or attempt or 'out of memory' not in log.read_text().lower():raise RuntimeError('supplemental evidence failed; preserve source: '+str(log))
                assert read(output/'scope.json')['whole_goal_complete'] is False
                assert original_metrics=={key:sha(evaluation/key/'statistical_gate_receipt.json') for key in original_metrics}
                write(receipt,dict(status='PASS',command=command,protocol_sha256=sha(frozen),
                    files={str(p.relative_to(root)):sha(p) for p in output.rglob('*') if p.is_file()}))
            assert freeze_barrier(freeze,None,None,scope='internal')==frozen_identity
            write(root/'pipeline_state.json',dict(status='POST_TEST_SUPPLEMENTAL_ROOT_CAUSE_DIAGNOSTICS_COMPLETE',run_id=logger.run_id,
                original_independent_metrics_unchanged=True,original_metrics=original_metrics,
                frozen_candidate_source_config_unchanged=True,Final_driven_retuning=False,
                Stage4B_started=False,training_started=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','independent','mini'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            no_candidate_tuning=True,whole_goal_complete=False))
        raise
