"""Autonomous freeze -> internal test -> final freeze -> full validation sequence."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .run_pipeline import read,write,sha
from .selection import contract,evaluate,freeze


def run(top,reanchor=None,depth=None,both=None):
    from utils.experiment_logger import ExperimentLogger
    root=top/'run';root.mkdir(parents=True,exist_ok=False)
    source=top/'source';audits=Path('/workspace/task10i_stage4a_completion_20261101_v1/run')
    protocol=read(Path('/workspace/task10i_stage4a_20261031_v5/run/protocol.json'))
    definitions=Path('/workspace/task10i_stage4a_completion_20261101_v1/selection_contract.json')
    assert read(definitions)['numeric_gates']==protocol['selection']
    shutil.copy2(definitions,root/'selection_contract.json')
    reanchor=reanchor or Path('/workspace/task10i_stage4a_candidate_development_20261101_v1/run/candidate')
    depth=depth or Path('/workspace/task10i_stage4a_depth_candidate_20261101_v1/run')
    candidates={
      'reanchor':dict(path=reanchor,source=reanchor.parents[1]/'source',
          session='task10i-stage4a-candidate-'+reanchor.parents[1].name.split('_')[-1] if reanchor.parents[1].name.split('_')[-1]!='v1' else 'task10i-stage4a-candidate'),
      'depth_component':dict(path=depth,source=depth.parent/'source',
          session='task10i-stage4a-depth-'+depth.parent.name.split('_')[-1] if depth.parent.name.split('_')[-1]!='v1' else 'task10i-stage4a-depth')}
    if both:candidates['depth_component_both']=dict(path=both,source=both.parent/'source',session='task10i-stage4a-depth-both-'+both.parent.name.split('_')[-1] if both.parent.name.split('_')[-1]!='v1' else 'task10i-stage4a-depth-both')
    stage='WAITING_DEVELOPMENT_CANDIDATES'
    def execute(stage,args):
        write(root/'pipeline_state.json',dict(status='RUNNING',stage=stage,run_id=log.run_id))
        with (root/(stage.lower()+'.log')).open('x') as output:
            subprocess.run([sys.executable,'-B','-u',*args],cwd=source,stdout=output,stderr=subprocess.STDOUT,check=True)
        print(stage,'FINISHED',flush=True)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_final_frozen_evaluation',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(numeric_selection=protocol['selection'],
         source_freeze_before_test=True,validation_once=True,production_adoption=False)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,source_root=str(source)))
        try:
            waited=0
            while not all((v['path']/'candidate_summary.json').exists() for v in candidates.values()):
                write(root/'pipeline_state.json',dict(status='RUNNING',stage=stage,run_id=log.run_id,wait_seconds=waited))
                for v in candidates.values():
                    p=v['path']/'pipeline_state.json'
                    if p.exists():assert read(p)['status']!='FAILED',str(p)
                    if waited>60 and not (v['path']/'candidate_summary.json').exists():
                        assert subprocess.run(['tmux','has-session','-t',v['session']],capture_output=True).returncode==0,'candidate coordinator exited: '+v['session']
                time.sleep(10);waited+=10
            for mode,v in candidates.items():
                stage='DEVELOPMENT_PAIR_'+mode.upper()
                execute(stage,['-m','graph.task10i_stage4a.paired_summary',str(root/('comparison_'+mode)),str(audits/'corrected'),str(v['path'])])
            stage='ORACLE_CAUSAL_ABLATIONS'
            execute(stage,['-m','graph.task10i_stage4a.oracle_windows',str(root/'oracles'),str(audits/'baseline'),str(audits/'corrected')])
            decisions={}
            for mode,v in candidates.items():
                stage='OBSERVER_FREE_LATENCY_'+mode.upper()
                execute(stage,['-m','graph.task10i_stage4a.latency_benchmark',str(root/('latency_'+mode)),'--mode',mode])
                decisions[mode]=evaluate(root/('comparison_'+mode),root/('latency_'+mode),protocol)
            write(root/'development_decisions.json',decisions)
            chosen=min(decisions,key=lambda mode:(decisions[mode]['measurable_safety_failures'],
                -(decisions[mode]['coverage_gain'] if decisions[mode]['coverage_gain'] is not None else -1),mode))
            info=candidates[chosen]
            # Preserve the selected Development inference wrapper byte-for-byte.
            # Only new evaluator/orchestrator modules are added around it.
            for name in ('paired_candidate.py','reliable_reanchor.py','depth_component.py','depth_box_component.py','perframe_cache.py'):
                p=info['source']/'graph/task10i_stage4a'/name
                if p.exists():shutil.copy2(p,source/'graph/task10i_stage4a'/name)
            spec=read(info['path']/'candidate_spec.json')
            first=root/'development_candidate_freeze.json'
            freeze(source,first,'DEVELOPMENT_CANDIDATE_FROZEN',spec['candidate'],chosen,decisions[chosen])
            log.link_output(first,kind='metrics')
            stage='INTERNAL_TEST'
            execute(stage,['-m','graph.task10i_stage4a.partition_runner',str(root/'internal_test'),str(first),'--partition','internal_test'])
            internal=evaluate(root/'internal_test/comparison',root/('latency_'+chosen),protocol)
            write(root/'internal_test_decision.json',internal)
            final=root/'final_candidate_freeze.json'
            freeze(source,final,'FINAL_CANDIDATE_FROZEN',spec['candidate'],chosen,internal,previous=first)
            assert read(first)['inference_files']==read(final)['inference_files']
            log.link_output(final,kind='metrics')
            stage='INDEPENDENT_VALIDATION'
            execute(stage,['-m','graph.task10i_stage4a.partition_runner',str(root/'validation'),str(final),'--partition','independent_validation'])
            validation=evaluate(root/'validation/comparison',root/('latency_'+chosen),protocol)
            write(root/'validation_decision.json',validation)
            full_go=all(x['decision']=='GO_RESEARCH_ONLY' for x in (decisions[chosen],internal,validation))
            write(root/'handoff.json',dict(status='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING',
                overall_decision='GO_RESEARCH_ONLY' if full_go else 'NO_GO',selected_challenger=spec['candidate'],mode=chosen,
                selected_development_root=str(info['path']),
                development=decisions[chosen],internal_test=internal,validation=validation,
                final_freeze=str(final),final_freeze_sha256=sha(final),
                baseline_root=str(audits/'baseline'),corrected_development_root=str(audits/'corrected'),
                final_source_root=str(source),production_adoption=False,policy_training=False,
                required_remaining='failure galleries, final report, protection/requirements handoff audit',whole_goal_complete=False))
            for n in ('development_decisions.json','internal_test_decision.json','validation_decision.json','handoff.json'):
                log.link_output(root/n,kind='metrics')
            write(root/'pipeline_state.json',dict(status='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING',run_id=log.run_id,whole_goal_complete=False))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=stage,error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('top',type=Path)
    p.add_argument('--development-reanchor',type=Path);p.add_argument('--development-depth',type=Path);p.add_argument('--development-both',type=Path)
    a=p.parse_args();run(a.top,a.development_reanchor,a.development_depth,a.development_both)
