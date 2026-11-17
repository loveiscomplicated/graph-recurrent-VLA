"""Audit the frozen frontend and paired-policy success and failure differences."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from math import comb

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_manifest(folder, summary):
    manifest = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
    assert manifest['status'] == 'finished'
    assert manifest['git_commit'].startswith('b2e8e4c') and not manifest['git_dirty']
    return manifest


def verify_runs(root):
    prior = OUTPUTS / 'recurrent_full_insertion/source_runs/train'
    checkpoints = {
        'baseline': [sha(prior / f'mamba_s{i}/last.pt') for i in range(5)],
        'finetuned': [sha(root / f'train/mamba_s{i}/last.pt') for i in range(5)],
    }
    demos = []
    for folder in sorted((root / 'demos').glob('shard_*')):
        summary = read(folder / 'summary.json')
        run_manifest(folder, summary)
        assert summary['config']['streaming_parity']
        assert summary['sha256'] == sha(folder / 'demos.npz')
        demos.append(dict(run_id=summary['run_id'], sha256=summary['sha256']))
    assert len(demos) == 8
    training = []
    for i in range(5):
        folder = root / f'train/mamba_s{i}'
        stats = read(folder / 'summary.json')
        run_manifest(folder, stats)
        assert stats['checkpoint_sha256'] == checkpoints['finetuned'][i]
        assert stats['config']['prior_sha256'] == checkpoints['baseline'][i]
        assert stats['config']['alpha'] == [1.0] and not stats['config']['filter']
        assert stats['config']['sources'] == demos
        history = stats['history']
        best = min(history, key=lambda row: row['validation']['balanced'])
        assert best['step'] == stats['best_step'] and best['validation']['balanced'] == stats['best_loss']
        training.append(dict(seed=i, start=history[0]['validation'], best=best['validation'],
                             best_step=best['step'], iterations=stats['final_step'], stop=stats['stop'],
                             counts=stats['config']['counts']))
    return checkpoints, training


def load_rollouts(root, arm, models):
    original = OUTPUTS / 'gtfree_frontend_retrain/source_runs/finetuned'
    tracker, gnn = (sha(original / f'{name}/best.pt') for name in ('tracker', 'gnn'))
    all_rows = []
    configs = []
    for a in range(0, 32, 4):
        shard = root / arm / f'shard_{a}_{a + 4}'
        summary = read(shard / 'summary.json')
        run_manifest(shard, summary)
        cfg = summary['config']
        assert cfg['tracker_sha256'] == tracker and cfg['gnn_sha256'] == gnn
        assert cfg['no_filter'] and cfg['fixed_horizon'] == 250
        assert cfg['recurrent_sha256'] == models[arm]
        if arm == 'finetuned':
            assert cfg['old_recurrent_sha256'] == models['baseline']
        with (shard / 'rollouts.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        assert [row['seed'] for row in rows] == cfg['seeds']
        for key in ('fixed', 'moving'):
            assert sum(row[key]['success'] for row in rows) == summary['success'][key]
            for row in rows:
                trial = row[key]
                assert trial['success'] == (trial['end_reason'] == 'success')
                assert 1 <= trial['steps'] == len(trial['trace']) <= 250
                assert all(step['filtered'] == step['object21'] for step in trial['trace'])
        all_rows.extend(rows)
        configs.append({name: value for name, value in cfg.items()
                        if name not in ('recurrent_sha256', 'old_recurrent_sha256')})
    assert len(all_rows) == len({row['seed'] for row in all_rows}) == 32
    return all_rows, configs


def load_classification():
    file = OUTPUTS / 'recurrent_full_insertion/failure_classification/classify.py'
    spec = importlib.util.spec_from_file_location('prior_failure_classification', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def compare(old, new):
    classify = load_classification()
    summary = {}
    for key in ('fixed', 'moving'):
        gained, lost = [], []
        stats = {'baseline': [], 'finetuned': []}
        both, old_count, new_count, moving_seen = 0, 0, 0, Counter()
        for a, b in zip(old, new, strict=True):
            seed = a['seed']
            assert seed == b['seed']
            x, y = a[key], b[key]
            if key == 'moving':
                assert x['motion_plan'] == y['motion_plan']
            for name in ('object21_gt', 'object21'):
                np.testing.assert_allclose(x['trace'][0][name], y['trace'][0][name], atol=2e-5, rtol=0)
            old_count += x['success']
            new_count += y['success']
            both += x['success'] and y['success']
            if y['success'] and not x['success']:
                gained.append(seed)
            if x['success'] and not y['success']:
                lost.append(seed)
            for arm, trial in (('baseline', x), ('finetuned', y)):
                if not trial['success']:
                    stats[arm].append(classify(seed, trial))
                moving_seen[arm] += trial.get('box_max_shift_mm', 0) > 1
        discordant = len(gained) + len(lost)
        p_value = min(1., 2 * sum(comb(discordant, i) for i in range(min(len(gained), len(lost)) + 1))
                      / (2 ** discordant))
        summary[key] = dict(n=len(old), baseline_success=old_count, finetuned_success=new_count,
                            success_both=both, gained=gained, lost=lost, exact_paired_p=p_value,
                            measured_motion=dict(moving_seen),
                            failed_after_entry={arm:sum(row['entry_frames'] > 0 for row in rows)
                                                for arm, rows in stats.items()},
                            failure_groups={arm:dict(Counter(row['group'] for row in rows)) for arm, rows in stats.items()},
                            failures=stats)
    return summary


def object21_errors(rows, condition):
    trace = [step for row in rows for step in row[condition]['trace']]
    target = np.asarray([step['object21_gt'] for step in trace])
    prediction = np.asarray([step['object21'] for step in trace])
    tip = np.asarray([step['actual_relative_mm'] for step in trace])
    stored = np.asarray([step['raw_lateral_mm'] for step in trace])
    delta = (prediction[:, 7:10] - target[:, 7:10]) * 1000
    inverse = -target[:, 14:17]
    cross = 2 * np.cross(inverse, delta)
    rotated = delta + target[:, 13:14] * cross + np.cross(inverse, cross)
    np.testing.assert_allclose(np.linalg.norm(rotated[:, 1:], axis=1), stored, rtol=0, atol=1e-3)
    depth = tip[:, 0] + target[:, 17] * 1000
    in_aperture = (np.abs(tip[:, 1:]) <= target[:, 20:21] * 1000).all(axis=1)
    stages = dict(all=np.ones(len(trace), dtype=bool), pre=depth <= 0,
                  shallow=in_aperture & (depth > 0) & (depth <= 20),
                  deep=in_aperture & (depth > 20))
    return {key: dict(frames=int(mask.sum()), mean_mm=float(stored[mask].mean()),
                      p90_mm=float(np.percentile(stored[mask], 90)))
            for key, mask in stages.items()}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('root', type=Path)
    root = cli.parse_args().root
    models, training = verify_runs(root)
    old, cfg_old = load_rollouts(root, 'baseline', models)
    new, cfg_new = load_rollouts(root, 'finetuned', models)
    assert cfg_old == cfg_new
    object21 = {condition: {arm: object21_errors(rows, condition)
                            for arm, rows in (('baseline', old), ('finetuned', new))}
                for condition in ('fixed', 'moving')}
    result = dict(evaluation=compare(old, new), training=training, config=cfg_old,
                  object21_lateral=object21)
    (root / 'verified.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, condition in result['evaluation'].items():
        print(name, {k:v for k,v in condition.items() if k != 'failures'})
    print('validation start/best and updates', [(x['start']['balanced'], x['best']['balanced'], x['best_step']) for x in training])


if __name__ == '__main__':
    main()
