"""Prospective decisions, never selected from Final performance."""
import argparse
from pathlib import Path
from graph.plan_a_bc.common import read, write, sha

F1_SHA = '781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6'


def create(root, f1, miniature=False, miniature_root=None,reuse_verified_data=None):
    root = root.resolve(); f1 = f1.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'config.json').exists():
        raise FileExistsError('configuration already exists')
    old = read(f1/'config.json')
    manifest = read(f1/'final_dataset_manifest.json')
    reach=Path(read(f1/'run_identity.json')['reach'])
    strict=read(reach/'collection_protocol.json')['threshold']
    val = sorted({s['seed'] for s in manifest['episode_specs'] if s['split']=='validation'})[:6]
    final = list(range(98200,98212))
    historical = set(range(91000,91012)) | set(range(61000,61006))
    assert not set(final) & (historical | {s['seed'] for s in manifest['episode_specs']})
    cfg = dict(task='10-G', miniature=miniature, root=str(root), f1_root=str(f1), seed=42,
        f1_checkpoint=str(f1/'gnn_pretrained.pt'), f1_checkpoint_sha256=F1_SHA,
        dataset_manifest_sha256=sha(f1/'final_dataset_manifest.json'),
        model=old['model'], geometry=old['geometry'], training={**old['training'],
            'updates':4 if miniature else 11487, 'validation_interval':2 if miniature else 1000,
            'checkpoint_steps':[2,4] if miniature else [3000,6000,9000,11487],
            'scheduler_clock':'per optimizer update cosine T_max=common update budget',
            'sampling':'seed42 epoch permutation, episode batch2, contiguous seq32, hidden carry/detach; all-masked chunks scan without optimizer update'},
        evaluation={**old['evaluation'], 'horizon':2 if miniature else 250,
            'smoke_horizon':2, 'dev_seeds':val[:1] if miniature else val,
            'final_seeds':[98190] if miniature else final,
            'strict_reach_thresholds':strict, 'strict_reach_protocol_sha256':sha(reach/'collection_protocol.json'),
            'offset_mm':[0,0,0], 'smoke_seed':98180,
            'selection':['insertion descending','lift descending','grasp descending',
                         'alignment descending','mean final TCP-peg distance ascending',
                         'validation BC loss ascending','step ascending'],
            'sampling_identity':'task10g:seed{seed}, identical across conditions/checkpoints',
            'paired_snapshot_claim':'seed reset plus exact simulator/controller/RNG snapshot hashes checked; rendering/vision isolated per episode'},
        calibration=dict(batches=8, ratio=.25, clamp=[.0001,10.],fallback=.01,
            unstable_rule='nonfinite/zero norms or fewer than4 finite positive batches; fallback fixed before calibration'),
        geometry_pairs_per_update=16, benchmark_steps=20 if miniature else 30,
        minimum_free_vram_bytes=3*1024**3,
        concurrency_throughput_floor=.85,
        miniature_root=str(miniature_root.resolve()) if miniature_root else None,
        reuse_verified_data=str(reuse_verified_data.resolve()) if reuse_verified_data else None,
        no_publish=True, no_callback=True, no_extra_experiments=True)
    write(root/'config.json',cfg,immutable=True)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--f1-root',type=Path,required=True);p.add_argument('--miniature',action='store_true')
    p.add_argument('--miniature-root',type=Path);p.add_argument('--reuse-verified-data',type=Path)
    a=p.parse_args();create(a.root,a.f1_root,a.miniature,a.miniature_root,a.reuse_verified_data)
