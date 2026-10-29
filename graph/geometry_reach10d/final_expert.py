"""Post-selection held-out Expert Reach ceiling; never a student input/cache."""
from pathlib import Path
import csv
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,now
from graph.geometry_reach10d.collect import run as collect
from graph.geometry_reach10d.profile import Resources


def run(root):
    assert (root/'model_selection_frozen.json').exists()
    if (root/'final_expert_reference.json').exists():return
    protocol=read(root/'collection_protocol.json');folder=root/'final_expert_reference';folder.mkdir(exist_ok=True)
    for name in ['normalizer.json','inputs','renderer_gpu_gate.json']:
        if not (folder/name).exists():(folder/name).symlink_to(root/name,target_is_directory=name=='inputs')
    specs=[dict(seed=s['seed'],variant=s['variant'],offset_mm=s['offset_mm'],split='final_expert_reference',
        parent_id=s['seed'],episode_id=1600000+i) for i,s in enumerate(protocol['final_cases'])]
    with ExperimentLogger(team='graph',model='Task10D_heldout_expert_reach_ceiling',seed=42,
            config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],after_model_selection=True,
                used_for_student_training=False),root=root/'evaluation_runs',job_type='evaluate') as log,Resources(root,'final_expert_reference'):
        result=collect(folder,specs)
        summary=dict(status='COMPLETE',created_at=now(),attempts=len(result),
            successes=sum(r['status']=='ACCEPTED' for r in result),episodes=result,
            used_for_student_training=False,used_for_checkpoint_selection=False,
            no_exclusion_of_student_cases_based_on_expert_outcome=True)
        write(root/'final_expert_reference.json',summary,immutable=True)
        with (root/'expert_final_reference_metrics.csv').open('w',newline='') as f:
            rows=[dict(seed=r['spec']['seed'],variant=r['spec']['variant'],status=r['status'],actions=r.get('actions',0),
                final_position_error_m=r.get('final_position_error_m'),final_orientation_error_rad=r.get('final_orientation_error_rad')) for r in result]
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        log.link_output(root/'final_expert_reference.json',kind='metrics')
