"""Register numeric adoption rules and Development-only headroom before tests."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,criteria,partition,expected,summary):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    specification=read(criteria);assert specification['status']=='FROZEN'
    split=read(partition);manifest=read(expected)
    parents={e['parent_id'] for e in manifest['entries']}
    assert parents==set(split['development']) and not parents&set(split['internal_holdout'])
    assert read(summary/'scope.json')['status']=='NEW_DEVELOPMENT_PAIRED_SUMMARY_COMPLETE'
    selected={('base_camera',0),('hand_camera',0),('base_camera',1)}
    values={}
    for row in read(summary/'paired_parent_metrics.json'):
        if row['scope']=='pixel' and row['metric']=='correct_mask_availability' and (row['camera'],row['object_id']) in selected:
            for p in row['parent_values']:
                values.setdefault(p['parent_id'],[]).append([p['baseline'],p['candidate']])
    assert set(values)==parents
    paired=np.stack([np.mean(v,axis=0) for _,v in sorted(values.items())])
    delta=paired[:,1]-paired[:,0];sd=float(np.std(delta,ddof=1))
    review=dict(development_parents=len(values),baseline_composite_availability=float(paired[:,0].mean()),
        baseline_error_headroom=float(1-paired[:,0].mean()),A_composite_availability=float(paired[:,1].mean()),
        A_delta=float(delta.mean()),A_paired_parent_delta_SD=sd,
        normal32_approx_80pct_power_min_effect=2.8*sd/np.sqrt(32),
        stress12_approx_80pct_power_min_effect=2.8*sd/np.sqrt(12),
        normal32_zero_event_one_sided_95pct_upper=1-.05**(1/32),
        stress12_zero_event_one_sided_95pct_upper=1-.05**(1/12),
        interpretation='A variance is a Development planning diagnostic; C/BC or Final variances may differ. Minimum32+12 does not ensure rare-event safety or effect power.',
        slots_per_parent={p:len(v) for p,v in values.items()})
    shutil.copy2(criteria,root/'adoption_criteria_frozen.json')
    write(root/'development_headroom_and_power_review.json',review)
    identity=dict(criteria_sha256=sha(criteria),partition_sha256=sha(partition),expected_sha256=sha(expected),
        development_summary_scope_sha256=sha(summary/'scope.json'),
        development_parent_metrics_sha256=sha(summary/'paired_parent_metrics.json'),
        source_sha256=sha(Path(__file__)),internal_results_opened=False,final_results_opened=False,
        observed_data='historical+Development B0/A; historical C/BC audit ongoing',
        candidate_source_frozen=False,candidate_selected=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_adoption_preregistration',
            seed=42,root=root/'experiments',job_type='analysis',config=identity) as logger:
        write(root/'registration_receipt.json',dict(status='ADOPTION_CRITERIA_REGISTERED',run_id=logger.run_id,
            **identity,files={p.name:sha(p) for p in root.iterdir() if p.is_file()},
            no_test_execution_started=True,whole_goal_complete=False))
        logger.link_output(root/'registration_receipt.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','criteria','partition','expected','summary'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.criteria,a.partition,a.expected,a.summary)
