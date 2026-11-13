"""Summarize fixed test errors and an explicitly post-hoc visibility diagnostic."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation.gnn.vertex_metrics import statistics
from utils.experiment_logger import ExperimentLogger


def frame_mean(frame, metric):
    """Average errors, not predictions, over the three model seeds."""
    rows = [frame['models'][str(seed)] for seed in (0, 1, 2)]
    if not all(row['available'] for row in rows):
        return None
    return float(np.mean([row['errors'][metric] for row in rows]))


def summarize_groups(episodes, passed):
    """Retain every test case and describe initial-visibility groups separately."""
    result = {}
    for mode in ('expert', 'random_after_lift'):
        result[mode] = {}
        for label, predicate in (('all', lambda seed: True),
                                 ('initial_visibility_pass', lambda seed: passed[seed]),
                                 ('initial_visibility_fail', lambda seed: not passed[seed])):
            group = [episode for episode in episodes if episode['mode'] == mode and predicate(episode['seed'])]
            result[mode][label] = {'episodes': len(group)}
            for period in ('all', 'post_lift'):
                frames = [frame for episode in group for frame in episode['frames']
                          if period == 'all' or frame['after_lift'] > 0]
                metrics = {}
                for name in ('peg_position_error_mm', 'hole_position_error_mm'):
                    values = [frame_mean(frame, name) for frame in frames]
                    valid = [value for value in values if value is not None]
                    metrics[name] = dict(statistics(np.asarray(valid)), missing=len(values)-len(valid))
                result[mode][label][period] = dict(frames=len(frames), metrics=metrics)
    return result


def reference_summary(paths):
    """Read the original development arrays with exact frame and GT alignment."""
    references = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            references.append({name: archive[name].copy() for name in
                               ('sample_ids', 'target', 'peg_position_error_mm', 'hole_position_error_mm')})
    for item in references[1:]:
        if not np.array_equal(item['sample_ids'], references[0]['sample_ids']) or not np.array_equal(item['target'], references[0]['target']):
            raise ValueError('Original reference frame IDs or GT differ by model seed')
    return {name: statistics(np.mean(np.stack([item[name] for item in references]), axis=0))
            for name in ('peg_position_error_mm', 'hole_position_error_mm')}


def draw_figure(episodes, passed, reference, output):
    """Plot paired episode means, separately marking the diagnostic visibility gate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    seeds = sorted(passed)
    by_key = {(episode['seed'], episode['mode']): episode for episode in episodes}
    for axis, name, title in zip(axes, ('peg_position_error_mm', 'hole_position_error_mm'), ('Peg', 'Hole')):
        for seed in seeds:
            means = []
            for mode in ('expert', 'random_after_lift'):
                frames = [frame for frame in by_key[(seed, mode)]['frames'] if frame['after_lift'] > 0]
                values = [frame_mean(frame, name) for frame in frames]
                values = [value for value in values if value is not None]
                means.append(float(np.mean(values)) if values else np.nan)
            color = '#2d9a66' if passed[seed] else '#d35b4d'
            axis.plot([0, 1], means, color=color, marker='o', alpha=.65)
        axis.axhline(reference[name]['mean'], linestyle='--', color='#345b9c',
                     label='Development mean (all frames)')
        axis.plot([], [], color='#2d9a66', marker='o', label='Initial gate passed')
        axis.plot([], [], color='#d35b4d', marker='o', label='Initial gate failed')
        axis.set_xticks([0, 1], ['Expert', 'Random after lift'])
        axis.set_ylabel('Episode mean 3D position error (mm)')
        axis.set_title(f'{title} — independent test after lift')
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    path = output / 'independent_test.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def report(config):
    """Aggregate the fixed test without selecting or discarding test episodes."""
    source = json.loads(Path(config['results']).read_text())
    gate = json.loads(Path(config['visibility_audit']).read_text())
    passed = {row['seed']: row['requested_pass'] for row in gate['rows']}
    episode_seeds = {episode['seed'] for episode in source['episodes']}
    if set(passed) != episode_seeds:
        raise ValueError('Initial visibility audit does not cover the fixed test seeds')
    reference = reference_summary(config['references'])
    groups = summarize_groups(source['episodes'], passed)
    with ExperimentLogger(team='graph', model='independent_test_report', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        result = dict(config=config, reference_development_all_frames=reference, groups=groups,
                      initial_gate_pass=sum(passed.values()), initial_gate_fail=len(passed)-sum(passed.values()),
                      caveat='Visibility grouping is a post-hoc diagnostic, not a replacement for the fixed test.')
        path = output / 'report.json'
        path.write_text(json.dumps(result, indent=2))
        figure = draw_figure(source['episodes'], passed, reference, output)
        logger.record(initial_gate_pass=result['initial_gate_pass'], initial_gate_fail=result['initial_gate_fail'])
        logger.link_output(path)
        logger.link_output(figure)
        print(f'Independent test report: {output}', flush=True)
