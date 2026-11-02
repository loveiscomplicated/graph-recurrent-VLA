"""Resume finished Internal inference after a phase-metadata-only evaluator fix.

All old outputs remain intact. Inference source/config hashes must match the
original Development freeze. Only paired aggregation and subsequent stages run.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from .run_pipeline import read,write,sha
from .selection import evaluate,freeze


def run(top,previous):
    from utils.experiment_logger import ExperimentLogger
    source=top/'source';root=top/'run';root.mkdir(parents=True,exist_ok=False)
    first=previous/'development_candidate_freeze.json';original=read(first)
    for name,digest in original['inference_files'].items():assert sha(source/name)==digest,name
    old=previous/'internal_test'
    assert read(old/'candidate/candidate_summary.json')['status']=='PAIRED_CANDIDATE_FINISHED'
    assert read(old/'candidate/candidate_summary.json')['frames']==5903
    assert not (previous/'validation').exists(),'do not silently rerun an opened validation'
    fixed_artifacts=['selection_contract.json','development_decisions.json','development_candidate_freeze.json']
    fixed_artifacts += [name+'_'+mode for name in ('comparison','latency') for mode in ('reanchor','depth_component','depth_component_both')]
    fixed_artifacts += ['oracles']
    for name in fixed_artifacts:
        p=previous/name
        if p.is_dir():(root/name).symlink_to(p,target_is_directory=True)
        else:shutil.copy2(p,root/name)
    partition=root/'internal_test';partition.mkdir()
    for name in ('reference','repaired','baseline','corrected','candidate'):(partition/name).symlink_to(old/name,target_is_directory=True)
    protocol=read(old/'baseline/protocol.json');mode=original['mode'];first=root/'development_candidate_freeze.json'
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_phase_metadata_recovery_and_final_validation',seed=42,
        root=root/'experiments',job_type='evaluation',config=dict(previous=str(previous),
        inference_changed=False,numeric_selection_changed=False,completed_internal_inference_reused=True)) as log:
        stage='PHASE_METADATA_RECOVERY'
        def execute(name,args):
            write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,run_id=log.run_id))
            with (root/(name.lower()+'.log')).open('x') as out:
                subprocess.run([sys.executable,'-B','-u',*args],cwd=source,stdout=out,stderr=subprocess.STDOUT,check=True)
            print(name,'FINISHED',flush=True)
        try:
            write(root/'run_identity.json',dict(run_id=log.run_id,source_root=str(source)))
            write(root/'recovery_receipt.json',dict(previous_failed_stage='Internal paired aggregation: partial legacy trace has no expert phase labels',
                old_source_phase_sha256=sha(previous.parent/'source/graph/task10i_stage4a/temporal_audit.py'),
                corrected_source_phase_sha256=sha(source/'graph/task10i_stage4a/temporal_audit.py'),
                phase_correction='preserve full raw action length as EXPERT_PHASE_UNAVAILABLE when suffix trace has no phase labels; complete HDF5 segment remains authoritative',
                original_inference_files_unchanged=True,original_freeze_sha256=sha(first),
                baseline_summary_sha256=sha(old/'baseline/development_perception_summary.json'),
                corrected_summary_sha256=sha(old/'corrected/reevaluation_summary.json'),
                candidate_summary_sha256=sha(old/'candidate/candidate_summary.json'),
                internal_model_forwards_repeated=0,old_failure_outputs_preserved=True,validation_previously_opened=False))
            stage='INTERNAL_PAIRED_PARENT_MACRO'
            execute(stage,['-m','graph.task10i_stage4a.paired_summary',str(partition/'comparison'),str(partition/'corrected'),str(partition/'candidate')])
            stage='INTERNAL_PAIRED_TEMPORAL'
            execute(stage,['-m','graph.task10i_stage4a.temporal_pair',str(partition/'temporal'),str(partition/'baseline'),str(partition/'corrected'),str(partition/'candidate')])
            r=read(old/'corrected/reevaluation_summary.json')
            write(partition/'partition_summary.json',dict(status='FROZEN_PARTITION_EVALUATION_FINISHED',partition='internal_test',run_id=log.run_id,
                frames=r['frames'],exact_frames=r['exact_frames'],unsupported_frames=r['frames']-r['exact_frames'],
                complete_registered_partition=True,no_candidate_changes_after_freeze=True,independent_validation_accessed=False,
                freeze_sha256=sha(first),reused_complete_inference_from=str(old),phase_metadata_only_reaggregation=True))
            internal=evaluate(partition/'comparison',root/('latency_'+mode),protocol)
            write(root/'internal_test_decision.json',internal)
            final=root/'final_candidate_freeze.json'
            freeze(source,final,'FINAL_CANDIDATE_FROZEN',original['candidate'],mode,internal,previous=first)
            assert read(first)['inference_files']==read(final)['inference_files']
            stage='INDEPENDENT_VALIDATION'
            execute(stage,['-m','graph.task10i_stage4a.partition_runner',str(root/'validation'),str(final),'--partition','independent_validation'])
            validation=evaluate(root/'validation/comparison',root/('latency_'+mode),protocol)
            write(root/'validation_decision.json',validation)
            development=read(root/'development_decisions.json')[mode]
            full_go=all(r['decision']=='GO_RESEARCH_ONLY' for r in (development,internal,validation))
            selected=read(root/('comparison_'+mode)/'summary.json')['candidate']
            write(root/'handoff.json',dict(status='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING',
                overall_decision='GO_RESEARCH_ONLY' if full_go else 'NO_GO',selected_challenger=original['candidate'],mode=mode,
                selected_development_root=selected,development=development,internal_test=internal,validation=validation,
                final_freeze=str(final),final_freeze_sha256=sha(final),
                baseline_root='/workspace/task10i_stage4a_completion_20261101_v1/run/baseline',
                corrected_development_root='/workspace/task10i_stage4a_completion_20261101_v1/run/corrected',
                final_source_root=str(source),production_adoption=False,policy_training=False,
                required_remaining='failure galleries, final report, protection/requirements handoff audit',whole_goal_complete=False))
            for name in ('recovery_receipt.json','internal_test_decision.json','validation_decision.json','handoff.json','final_candidate_freeze.json'):
                log.link_output(root/name,kind='metrics')
            write(root/'pipeline_state.json',dict(status='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING',run_id=log.run_id,whole_goal_complete=False))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=stage,error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('top',type=Path);p.add_argument('previous',type=Path)
    a=p.parse_args();run(a.top,a.previous)
