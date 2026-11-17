"""Validate and summarize the paired per-frame Object21 ablation results."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ('detected', 'remove_front', 'visible_gt_front')
METRICS = ('hole_center_mm', 'axis_deg', 'tip_lateral_mm')


def paired(rows, left, right, metric):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row['seed']].append(row[right]['errors'][metric] - row[left]['errors'][metric])
    if not by_seed:
        return None
    values = np.concatenate([np.asarray(v) for v in by_seed.values()])
    seed_means = np.asarray([np.mean(v) for v in by_seed.values()])
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(seed_means, size=(5000, len(seed_means)), replace=True).mean(axis=1)
    return dict(frames=len(values), seeds=len(by_seed), median_delta=float(np.median(values)),
                mean_seed_delta=float(seed_means.mean()), improved=int((values < -1e-5).sum()),
                worsened=int((values > 1e-5).sum()),
                mean_seed_delta_ci95=np.quantile(bootstrap, [.025, .975]).tolist())


def analyze(root):
    result = json.loads((root / 'summary.json').read_text())
    rows = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
    initial = [r['seed'] for r in rows if 'initial' in r['stages']]
    assert initial == result['config']['seeds']
    assert len({(r['seed'], r['step']) for r in rows}) == len(rows)
    assert len(result['expert_terminals']) == len(result['config']['expert_seeds'])
    for row in rows:
        for vector in [row['gt21'], *(row[c]['output21'] for c in CONDITIONS)]:
            assert len(vector) == 21 and np.isfinite(vector).all()
        if 'initial' in row['stages'] and not row['counts']['rim_flagged']:
            np.testing.assert_allclose(row['detected']['output21'], row['remove_front']['output21'],
                                       atol=1e-5, rtol=0)
        if 'initial' in row['stages'] and not row['counts']['visible_gt_front']:
            np.testing.assert_allclose(row['remove_front']['output21'], row['visible_gt_front']['output21'],
                                       atol=1e-5, rtol=0)
    groups = {}
    for stage, summary in result['metrics'].items():
        group = [r for r in rows if stage in r['stages']]
        assert len(group) == summary['frames']
        groups[stage] = group
    completed = {int(seed) for seed, status in result['expert_terminals'].items()
                 if status == 'insert_entry'}
    approach = defaultdict(list)
    for row in groups['held_align']:
        if row['seed'] in completed:
            approach[row['seed']].append(row)
    groups['last_5_held_align'] = [row for frames in approach.values() for row in frames[-5:]]

    report = {}
    for stage, group in groups.items():
        report[stage] = dict(
            frames=len(group),
            seeds=len({r['seed'] for r in group}),
            visible_gt_hist=dict(Counter(r['counts']['visible_gt_front'] for r in group)),
            flagged_hist=dict(Counter(r['counts']['rim_flagged'] for r in group)),
            flagged_near_front=sum(r['counts']['flagged_within_5mm_front'] for r in group),
            flagged_total=sum(r['counts']['rim_flagged'] for r in group),
            untagged_near_front=sum(r['counts']['untagged_within_5mm_front'] for r in group),
            all={condition: {metric: distribution([r[condition]['errors'][metric] for r in group])
                             for metric in METRICS} for condition in CONDITIONS},
            comparisons={})
        subgroups = {'all': group,
                     'gt_visible': [r for r in group if r['counts']['visible_gt_front']],
                     'gt_visible_detector_missing': [r for r in group
                                                     if r['counts']['visible_gt_front']
                                                     and not r['counts']['rim_flagged']]}
        for name, subgroup in subgroups.items():
            report[stage]['comparisons'][name] = {
                f'{left}_to_{right}': {m: paired(subgroup, left, right, m) for m in METRICS}
                for left, right in (('detected', 'remove_front'), ('detected', 'visible_gt_front'),
                                    ('remove_front', 'visible_gt_front'))}
    (root / 'paired.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def distribution(values):
    if not values:
        return None
    return {label: float(value) for label, value in zip(
        ('median', 'p90'), np.quantile(values, [.5, .9]), strict=True)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
