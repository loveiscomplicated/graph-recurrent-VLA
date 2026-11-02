"""Autonomous evidence gallery, matched statistics and protected-data handoff."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .run_pipeline import read,write,sha


def run(top,final,reuse_attribution=None):
    from utils.experiment_logger import ExperimentLogger
    root=top/'run';root.mkdir(parents=True,exist_ok=False)
    audits=Path('/workspace/task10i_stage4a_completion_20261101_v1/run')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_readonly_report_and_handoff_audit',seed=42,
        root=root/'experiments',job_type='evaluation',config=dict(final_root=str(final),model_forwards=0,read_only_source_data=True)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(__file__)))
        try:
            write(root/'pipeline_state.json',dict(status='RUNNING',stage='DEVELOPMENT_ACTUAL_ATTRIBUTION',run_id=log.run_id))
            if reuse_attribution is None:
                from .attribution import run as attribution
                attribution(root/'attribution',audits)
            else:
                assert read(reuse_attribution/'scope.json')['status']=='ACTUAL_FIRST_LOSS_ATTRIBUTION_FINISHED'
                (root/'attribution').symlink_to(reuse_attribution,target_is_directory=True)
                write(root/'reused_attribution.json',dict(path=str(reuse_attribution),scope_sha256=sha(reuse_attribution/'scope.json'),
                    unchanged_development_inputs=True,phase_metadata_not_used_by_attribution=True))
            write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_FROZEN_VALIDATION',run_id=log.run_id))
            while not (final/'handoff.json').exists():
                state=final/'pipeline_state.json'
                if state.exists():assert read(state)['status']!='FAILED',read(state)
                time.sleep(10)
            handoff=read(final/'handoff.json')
            candidates={mode:Path(read(final/('comparison_'+mode)/'summary.json')['candidate']) for mode in ('reanchor','depth_component','depth_component_both')}
            selected=candidates[handoff['mode']]
            from .report_analysis import run as statistics
            jobs=[(mode,final/('comparison_'+mode),audits/'corrected',path) for mode,path in candidates.items()]
            jobs.extend((name,final/name/'comparison',final/name/'corrected',final/name/'candidate') for name in ('internal_test','validation'))
            for name,comparison,base,candidate in jobs:
                write(root/'pipeline_state.json',dict(status='RUNNING',stage='MATCHED_REPORT_'+name,run_id=log.run_id))
                statistics(root/('statistics_'+name),comparison,dict(baseline=base,candidate=candidate))
            from .failure_gallery import run as gallery
            gallery(root/'gallery',audits,selected)
            for name,base,corrected,candidate in [('development',audits/'baseline',audits/'corrected',selected),
                 ('internal_test',final/'internal_test/baseline',final/'internal_test/corrected',final/'internal_test/candidate'),
                 ('validation',final/'validation/baseline',final/'validation/corrected',final/'validation/candidate')]:
                write(root/'pipeline_state.json',dict(status='RUNNING',stage='CORRECT_SOURCE_COVERAGE_'+name,run_id=log.run_id))
                with (root/('valid_coverage_'+name+'.log')).open('x') as out:
                    subprocess.run([sys.executable,'-B','-u','-m','graph.task10i_stage4a.valid_coverage',str(root/('valid_coverage_'+name)),str(base),str(corrected),str(candidate)],stdout=out,stderr=subprocess.STDOUT,check=True)
            from .handoff_audit import run as protection
            protection(root/'handoff_audit',final,audits,selected)
            for n in ('attribution/scope.json','gallery/case_manifest.json','handoff_audit/protection_checks.json','handoff_audit/causal_graph_checks.json'):
                log.link_output(root/n,kind='metrics')
            write(root/'report_evidence_ready.json',dict(status='REPORT_EVIDENCE_READY',run_id=log.run_id,
                scientific_handoff=str(final/'handoff.json'),scientific_handoff_sha256=sha(final/'handoff.json'),
                no_new_candidate_or_threshold_selection=True,production_adoption=False,whole_goal_complete=False,
                remaining='final report publication to repository docs and human-readable evidence review'))
            write(root/'pipeline_state.json',dict(status='REPORT_EVIDENCE_READY',run_id=log.run_id,whole_goal_complete=False))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('top',type=Path);p.add_argument('final',type=Path)
    p.add_argument('--reuse-attribution',type=Path)
    a=p.parse_args();run(a.top,a.final,a.reuse_attribution)
