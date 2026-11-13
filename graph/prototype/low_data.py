"""Small, parameter-matched Flat/Surface experiment using the existing v2 BC."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from .common import config, seed_all, versions, write_json
from .data import PHASES, load_dataset, sha256, split_indices
from .models import parameter_count
from .normalization import FeatureNormalizer
from .v2.analyze import offline
from .v2.diagnostics import qvel_dataset
from .v2.models import CommonPolicy
from .v2.state import StateNormalizer

REPO = Path(__file__).resolve().parents[2]
MODELS = ('flat', 'surface')
DEFAULT_CONFIG = Path(__file__).with_name('v2') / 'config.toml'


def subset(data, train_episodes=10, subset_seed=0):
    """Keep complete reset groups from train; preserve every held-out row."""
    manifest = copy.deepcopy(data['manifest'])
    splits = manifest['splits']
    all_ids = sum((list(splits[k]) for k in ('train', 'validation', 'test')), [])
    episodes = {e['episode_id']: e for e in manifest['episodes']}
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(episodes):
        raise ValueError('source episodes/splits must be complete and disjoint')
    if any(not splits[k] for k in splits) or train_episodes < 1:
        raise ValueError('nonempty splits and a positive train episode count required')
    groups = [{episodes[e]['episode_seed'] for e in splits[k]}
              for k in ('train', 'validation', 'test')]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('reset group crosses a source split')
    if train_episodes > len(splits['train']):
        raise ValueError('requested more training episodes than the source split')
    order = np.random.default_rng(subset_seed).permutation(sorted(groups[0])).tolist()
    selected = []
    for group in order:
        selected.extend(e for e in splits['train'] if episodes[e]['episode_seed'] == group)
        if len(selected) >= train_episodes:
            break
    splits['train'] = sorted(selected)
    retained = set(sum((list(splits[k]) for k in ('train', 'validation', 'test')), []))
    rows = torch.tensor([int(e) in retained for e in data['episode_id']], dtype=torch.bool)
    if set(data['episode_id'].tolist()) != set(episodes):
        raise ValueError('source rows and episode inventory differ')
    result = {k: v[rows].clone() for k, v in data.items() if torch.is_tensor(v)}
    result['geometry'] = {k: v[rows].clone() for k, v in data['geometry'].items()}
    manifest['episodes'] = [e for e in manifest['episodes'] if e['episode_id'] in retained]
    for key in ('snapshot_checks', 'replay_checks'):
        if key in manifest:
            manifest[key] = [e for e in manifest[key] if e['episode_id'] in retained]
    manifest['samples'] = int(rows.sum())
    manifest['phase_counts'] = {name: int((result['phase'] == i).sum())
                               for i, name in enumerate(PHASES)}
    manifest['low_data'] = dict(subset_seed=subset_seed, requested_train_episodes=train_episodes,
                                actual_train_episodes=len(selected),
                                selection='shuffle train reset groups; retain whole episodes/groups')
    result['manifest'] = manifest
    return result


def paths(root, model, seed):
    name = f'{model}_seed{seed}'
    return root / 'training' / name, root / 'evaluation' / f'{name}.json'


def prepare(args):
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('output root is not empty; use a new root or an explicit --phase')
    if args.epochs < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError('positive epochs and unique nonnegative training seeds required')
    cfg = config(args.config)
    cfg.update(flat_width=args.flat_width, epochs=args.epochs)
    cfg.setdefault('head_width', 128)
    seed_all(args.subset_seed, cfg['threads'])
    counts = {name: parameter_count(CommonPolicy(name, cfg, 'goal', 'regression')) for name in MODELS}
    gap = abs(counts['flat'] - counts['surface']) / counts['surface']
    if gap > .01:
        raise ValueError(f'parameter mismatch {gap:.1%}; choose a Flat width within 1% of Surface')
    source_path = args.dataset.resolve()
    data, _ = load_dataset(source_path)
    for key in ('peg_points', 'hole_points', 'sampling_seed', 'control_mode'):
        if cfg[key] != data['manifest']['config'][key]:
            raise ValueError(f'source/config mismatch: {key}')
    trajectory = (args.trajectory or Path(data['manifest']['source'])).resolve()
    if sha256(trajectory) != data['manifest']['source_sha256']:
        raise ValueError('trajectory hash differs from the source dataset')
    reduced = subset(data, args.train_episodes, args.subset_seed)
    reduced['manifest']['source'] = str(trajectory)
    source_hash = sha256(source_path)
    reduced['manifest']['low_data'].update(parent_dataset=str(source_path), parent_sha256=source_hash)
    # Validate qpos/qvel ordering before any training, without a simulator.
    qvel_dataset(reduced)
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / 'dataset.pt'
    torch.save(reduced, dataset)
    write_json(root / 'dataset.json', reduced['manifest'])
    # v2 uses flat TOML configs; JSON primitive syntax is compatible with TOML.
    cfg_path = root / 'config.toml'
    cfg_path.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in cfg.items()))
    if config(cfg_path) != cfg:
        raise ValueError('configuration roundtrip failed')
    plan = dict(schema_version=1, dataset=str(dataset), dataset_sha256=sha256(dataset),
                config=cfg, config_sha256=sha256(cfg_path), source_dataset=str(source_path),
                source_dataset_sha256=source_hash, trajectory=str(trajectory),
                trajectory_sha256=data['manifest']['source_sha256'],
                subset=reduced['manifest']['low_data'], splits=reduced['manifest']['splits'],
                samples={s: len(split_indices(reduced, s)) for s in ('train', 'validation', 'test')},
                seeds=args.seeds, parameters=counts, parameter_relative_gap=gap,
                primary_metric='test sample-weighted raw 7D action MSE',
                checkpoint_selection='minimum validation action MSE',
                state_mode='goal', gripper_mode='regression', device='cpu', versions=versions())
    write_json(root / 'plan.json', plan)
    print(f"Prepared {plan['subset']['actual_train_episodes']} train episodes; "
          f"samples {plan['samples']}; parameters {counts}", flush=True)
    return plan


def load_plan(root):
    plan = json.loads((root / 'plan.json').read_text())
    if sha256(plan['dataset']) != plan['dataset_sha256'] or sha256(root / 'config.toml') != plan['config_sha256']:
        raise ValueError('prepared dataset/config changed; use a fresh root')
    if sha256(plan['trajectory']) != plan['trajectory_sha256']:
        raise ValueError('source trajectory changed')
    return plan


def completed_run(directory, plan, model, seed):
    """Never silently restart or overwrite an interrupted training run."""
    record = directory / 'training.json'
    checkpoint = directory / 'best.pt'
    if not record.exists() or not checkpoint.exists():
        return False
    training = json.loads(record.read_text())
    expected = dict(plan['config'], seed=seed)
    return (training['config'] == expected and training['model'] == model
            and training['dataset_sha256'] == plan['dataset_sha256']
            and training['splits'] == plan['splits']
            and training['state_mode'] == 'goal' and training['gripper_mode'] == 'regression'
            and training['smoke_max_batches'] is None
            and [r['epoch'] for r in training['history']] == list(range(1, expected['epochs'] + 1))
            and training['optimizer_steps'] == math.ceil(plan['samples']['train'] / expected['batch_size']) * expected['epochs'])


def train(root, plan):
    for seed in plan['seeds']:
        for model in MODELS:
            directory, _ = paths(root, model, seed)
            if completed_run(directory, plan, model, seed):
                print(f'Skip completed {model} seed {seed}', flush=True)
                continue
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f'incomplete/incompatible training: {directory}; use a new root')
            log = root / 'logs' / f'{model}_seed{seed}.log'
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-m', 'graph.prototype.v2.train', '--model', model,
                       '--state', 'goal', '--gripper', 'regression', '--dataset', plan['dataset'],
                       '--config', str(root / 'config.toml'), '--epochs', str(plan['config']['epochs']),
                       '--seed', str(seed), '--output', str(directory)]
            print(f'Train {model} seed {seed}; log: {log}', flush=True)
            with log.open('a') as stream:
                subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if not completed_run(directory, plan, model, seed):
                raise ValueError(f'incomplete training: {directory}')


def audit_checkpoints(root, plan, data, geometry, qvel):
    """Check all runs and selected-train normalizers before looking at test loss."""
    ids = split_indices(data, 'train')
    expected = dict(normalizer=FeatureNormalizer.fit(geometry.select(ids)).as_dict(),
                    state_normalizer=StateNormalizer.fit(geometry.select(ids), qvel[ids], 'goal').as_dict())
    for seed in plan['seeds']:
        for model in MODELS:
            directory, _ = paths(root, model, seed)
            if not completed_run(directory, plan, model, seed):
                raise ValueError(f'missing/incompatible completed run: {directory}')
            ck = torch.load(directory / 'best.pt', weights_only=True, map_location='cpu')
            if (ck['config'] != dict(plan['config'], seed=seed) or ck['model_name'] != model
                    or ck['dataset_sha256'] != plan['dataset_sha256'] or ck['splits'] != plan['splits']
                    or ck['parameters'] != plan['parameters'][model]
                    or ck['state_mode'] != 'goal' or ck['gripper_mode'] != 'regression'):
                raise ValueError(f'checkpoint identity mismatch: {directory}')
            for field, values in expected.items():
                for key, value in values.items():
                    if not torch.equal(value, ck[field][key]):
                        raise ValueError(f'normalizer does not match selected train data: {directory}/{field}/{key}')


def evaluate(root, plan):
    seed_all(plan['config']['seed'], plan['config']['threads'])
    data, geometry = load_dataset(plan['dataset'])
    qvel = qvel_dataset(data)
    audit_checkpoints(root, plan, data, geometry, qvel)
    for seed in plan['seeds']:
        for model in MODELS:
            directory, output = paths(root, model, seed)
            checkpoint = directory / 'best.pt'
            digest = sha256(checkpoint)
            if output.exists():
                old = json.loads(output.read_text())
                if (old.get('checkpoint_sha256') != digest or old.get('dataset_sha256') != plan['dataset_sha256']
                        or old.get('seed') != seed or old.get('model') != model or old.get('splits') != plan['splits']):
                    raise ValueError(f'evaluation identity mismatch: {output}')
                print(f'Skip completed evaluation {model} seed {seed}', flush=True)
                continue
            print(f'Evaluate {model} seed {seed}', flush=True)
            offline(checkpoint, data, geometry, qvel, plan['dataset'], per_episode=True, output=output)


def summarize(rows, seeds):
    """Pair model results by initialization seed, without selecting a winner run."""
    indexed = {(r['model'], r['seed']): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {(m, s) for m in MODELS for s in seeds}:
        raise ValueError('comparison needs exactly one result per planned model/seed')
    aggregate = {}
    for model in MODELS:
        values = [indexed[model, s] for s in seeds]
        aggregate[model] = {}
        for metric in ('action_mse', 'translation_rmse', 'rotation_rmse', 'gripper_mae', 'latency_ms'):
            samples = np.array([r[metric] for r in values])
            aggregate[model][metric] = dict(mean=float(samples.mean()),
                                           std=float(samples.std(ddof=1)) if len(samples) > 1 else None)
    pairs = []
    for seed in seeds:
        flat, surface = (indexed[m, seed]['action_mse'] for m in MODELS)
        pairs.append(dict(seed=seed, flat_mse=flat, surface_mse=surface,
                          reduction_percent=100 * (flat - surface) / flat if flat else None,
                          surface_lower=surface < flat))
    flat = aggregate['flat']['action_mse']['mean']
    surface = aggregate['surface']['action_mse']['mean']
    return dict(aggregate=aggregate, pairs=pairs,
                surface_lower_seeds=sum(p['surface_lower'] for p in pairs), total_seeds=len(seeds),
                mean_mse_reduction_percent=100 * (flat - surface) / flat if flat else None)


def report(root, plan):
    rows = []
    for seed in plan['seeds']:
        for model in MODELS:
            directory, output = paths(root, model, seed)
            result = json.loads(output.read_text())
            training = json.loads((directory / 'training.json').read_text())
            if (result['dataset_sha256'] != plan['dataset_sha256'] or result['splits'] != plan['splits']
                    or result['seed'] != seed or result['model'] != model
                    or result['checkpoint_sha256'] != sha256(directory / 'best.pt')):
                raise ValueError(f'unpaired/stale result: {output}')
            rows.append(dict(model=model, seed=seed, parameters=result['parameters'],
                             selected_epoch=result['checkpoint_epoch'], **result['offline'],
                             episode_mean_action_mse=float(np.mean([r['action_mse'] for r in result['episode_offline']])),
                             latency_ms=result['latency']['total_ms'], training_seconds=training['training_seconds']))
    comparison = dict(plan=plan, rows=rows, **summarize(rows, plan['seeds']))
    lines = ['# Graph low-data comparison', '',
             f"Train episodes: {len(plan['splits']['train'])}; validation/test: "
             f"{len(plan['splits']['validation'])}/{len(plan['splits']['test'])}. "
             f"Subset seed {plan['subset']['subset_seed']}; training seeds {plan['seeds']}; "
             f"{plan['config']['epochs']} complete epochs. CPU only.", '',
             'Primary: sample-weighted raw normalized 7D action MSE; checkpoint chosen only by validation MSE. '
             'All runs use identical selected train rows, train-only normalizers, batches per seed, and goal31 bypass.', '',
             '| Model | Seed | Params | Epoch | Action MSE | Translation RMSE | Rotation RMSE | Gripper MAE | Episode mean MSE | CPU ms |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['parameters']} | {r['selected_epoch']} | "
                     f"{r['action_mse']:.8f} | {r['translation_rmse']:.6f} | {r['rotation_rmse']:.6f} | "
                     f"{r['gripper_mae']:.6f} | {r['episode_mean_action_mse']:.8f} | {r['latency_ms']:.3f} |")
    lines += ['', '| Model | Mean action MSE | Sample std across training seeds |', '|---|---:|---:|']
    for model in MODELS:
        metric = comparison['aggregate'][model]['action_mse']
        std = f"{metric['std']:.8f}" if metric['std'] is not None else 'N/A (one seed)'
        lines.append(f"| {model} | {metric['mean']:.8f} | {std} |")
    reduction = comparison['mean_mse_reduction_percent']
    lines += ['', f"Surface has lower MSE in {comparison['surface_lower_seeds']}/{comparison['total_seeds']} training seeds.",
              f"Reduction relative to Flat mean MSE: {reduction:.2f}%." if reduction is not None else 'Relative reduction: N/A (Flat MSE is zero).',
              '', 'Positive reduction means lower Surface error. No seed, phase or checkpoint is selected by test results. '
              'These are small offline results, not a statistical significance claim, closed-loop success or a data-efficiency curve. '
              'Translation/rotation RMSE are normalized commands, not realized motion. CPU batch-1 mean latency includes '
              'normalization/graph, shared state and network, excluding simulator/GT extraction. '
              'Training and evaluation run sequentially. No simulator, CUDA or MPS is executed.', '']
    result_path = root / 'comparison' / 'results.json'
    markdown = result_path.with_suffix('.md')
    with ExperimentLogger(team='graph', config=plan, seed=plan['subset']['subset_seed'],
                          model='flat_vs_surface', job_type='compare') as experiment:
        experiment.record(dataset_sha256=plan['dataset_sha256'])
        write_json(result_path, comparison)
        markdown.write_text('\n'.join(lines))
        experiment.log(comparison['aggregate'], prefix='comparison')
        experiment.log(dict(surface_lower_seeds=comparison['surface_lower_seeds'],
                            mean_mse_reduction_percent=reduction), prefix='comparison')
        experiment.link_output(root / 'plan.json')
        experiment.link_output(result_path)
        experiment.link_output(markdown)
    print('\n'.join(lines[-10:]), flush=True)
    print(f'Report: {markdown}', flush=True)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('all', 'prepare', 'train', 'evaluate', 'report'), default='all')
    parser.add_argument('--root', type=Path, default=Path('outputs/graph/low_data'))
    parser.add_argument('--dataset', type=Path, default=Path('outputs/graph/dataset.pt'))
    parser.add_argument('--trajectory', type=Path, help='relocated source H5; original hash must match')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--train-episodes', type=int, default=10)
    parser.add_argument('--subset-seed', type=int, default=0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--flat-width', type=int, default=192)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.mplconfig'))
    plan = prepare(args) if args.phase in ('all', 'prepare') else load_plan(root)
    if args.phase in ('all', 'train'):
        train(root, plan)
    if args.phase in ('all', 'evaluate'):
        evaluate(root, plan)
    if args.phase in ('all', 'report'):
        report(root, plan)


if __name__ == '__main__':
    main()
