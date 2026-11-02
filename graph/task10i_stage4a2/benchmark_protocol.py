"""Commit fresh final seed queues without opening final images or model outputs."""
import argparse
import csv
from pathlib import Path
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3


def integer_values(value):
    if isinstance(value,dict):
        for x in value.values():yield from integer_values(x)
    elif isinstance(value,list):
        for x in value:yield from integer_values(x)
    elif isinstance(value,int) and not isinstance(value,bool) and 0<=value<2**31:
        yield value


def register(root):
    root.mkdir(parents=True,exist_ok=False)
    pool=Path('/workspace/task10i_stage1_20261031/verified_pool_v2')
    inventory=read(pool/'existing_parent_inventory.json')
    excluded=set(inventory['conservative_integers'])
    ranges=inventory['literal_range_exclusions']
    paths=set([pool/'existing_parent_inventory.json',pool/'candidate_seed_manifest.json',
        STAGE3/'stage4_recache_input_manifest.json',STAGE3/'stage4_raw_dataset_receipt.json'])
    paths.update(Path('/workspace/task10i_stage1_20261031').glob('*/candidate_seed_manifest.json'))
    paths.update(Path('/workspace/task10g_f1_abc_20261031/full').glob('*.json'))
    identities=[]
    for p in sorted(paths):
        value=read(p);excluded.update(integer_values(value))
        identities.append(dict(path=str(p),sha256=sha(p)))
    # Actual historical Final rollout rows, not only planned seed manifests.
    final_csv=Path('/workspace/task10g_f1_abc_20261031/full/final_rollouts.csv')
    with final_csv.open() as stream:
        for row in csv.DictReader(stream):
            for key,value in row.items():
                if 'seed' in key.lower() and value:
                    excluded.add(int(value))
    identities.append(dict(path=str(final_csv),sha256=sha(final_csv)))
    generator=np.random.default_rng(1011202642)
    queues={};seen=set()
    for stratum in ('normal_conditional_random','occlusion_reappearance_negative_stress'):
        seeds=[]
        while len(seeds)<1024:
            seed=int(generator.integers(1_200_000_000,1_900_000_000))
            if seed in excluded or seed in seen or any(a<=seed<=b for a,b in ranges):continue
            seeds.append(seed);seen.add(seed)
        queues[stratum]=[dict(attempt=i,parent_id=f'10i4a2_final_{stratum}_{i:04d}',seed=s)
                         for i,s in enumerate(seeds)]
    assert len(seen)==2048 and not seen&excluded
    write(root/'seed_collision_audit.json',dict(status='PASS',source_files=identities,
          conservative_excluded_integers=len(excluded),seed_queue_count=len(seen),collisions=0,
          stage1_all_registered_attempt_seeds_excluded=True,task10g_final_actual_rows_checked=True,
          stage2_stage3_parents_inherit_stage1_or_original_inventory=True,
          note='Integer exclusion is deliberately broader than actual parent seeds'))
    write(root/'new_final_benchmark_protocol.json',dict(version=1,queues=queues,
        seed_generator='NumPy PCG64(1011202642)',normal_minimum_parents=32,stress_minimum_parents=12,
        max_attempts_per_stratum=1024,seed_collision_audit_sha256=sha(root/'seed_collision_audit.json'),
        normal_sampling='first32 registered native random parents passing unchanged Stage1 visibility+physical+replay gates',
        stress_sampling='first12 disjoint registered parents passing same initial gates; all registered stress interventions retained',
        mask_success_selection=False,expert_success_selection=False,student_training_data=False,
        final_pixels_opened=False,capture_after_candidate_freeze=True,
        exact_GT='synchronous same-render actor segmentation with RGB/depth/calibration repeat equality; sidecar only',
        evaluation='parent paired bootstrap; normal and stress reported separately; no final-driven retuning',
        stress_intervention_geometry_frozen=False,statistical_power_review_pending=True,
        full_collection_protocol_ready=False,source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);register(p.parse_args().root)
