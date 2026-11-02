"""Read-only temporal comparison for all completed Development challengers."""
import argparse
from pathlib import Path
from .run_pipeline import read,write,sha


def run(output,final,audits):
    from utils.experiment_logger import ExperimentLogger
    from .temporal_pair import run as temporal
    output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_Development_temporal_supplement',seed=42,
        root=output/'experiments',job_type='evaluation',config=dict(model_forwards=0,partition='development',post_selection_diagnostic=True,no_candidate_retuning=True)) as log:
        rows=[]
        for mode in ('reanchor','depth_component','depth_component_both'):
            candidate=Path(read(final/('comparison_'+mode)/'summary.json')['candidate'])
            assert (candidate/'candidate_summary.json').exists()
            temporal(output/mode,audits/'baseline',audits/'corrected',candidate)
            rows.append(dict(mode=mode,candidate_root=str(candidate),candidate_spec_sha256=sha(candidate/'candidate_spec.json')))
            print('Development temporal',mode,'FINISHED',flush=True)
        write(output/'scope.json',dict(status='DEVELOPMENT_TEMPORAL_COMPARISONS_FINISHED',run_id=log.run_id,
            model_forwards=0,rows=rows,post_selection_diagnostic=True,no_selection_rule_changes=True,
            no_candidate_source_or_config_changes=True,not_independent_test_evidence=True))
        log.link_output(output/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('final',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.output,a.final,a.audits)
