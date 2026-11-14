"""Summarize held-out per-state camera visibility against the task default."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_logger import ExperimentLogger


def sample_key(row):
    """Identify a predeclared sampled physical state."""
    return row['seed'], row['action_seed'], row['frame']


def summarize(candidate, baseline, seeds):
    """Calculate seed-weighted, paired visibility and robot occlusion metrics."""
    rows = {sample_key(row): row for row in candidate['frames']}
    base = {sample_key(row): row for row in baseline['frames']}
    if rows.keys() != base.keys():
        raise ValueError(f'Sampled states differ for {candidate["name"]}')
    per_seed = []
    for seed in seeds:
        values = [row for row in rows.values() if row['seed'] == seed]
        references = [base[sample_key(row)] for row in values]
        if not values:
            raise ValueError(f'No visibility samples for seed {seed}')
        visible = float(np.mean([row['visible_fraction'] for row in values]))
        base_visible = float(np.mean([row['visible_fraction'] for row in references]))
        per_seed.append(dict(seed=seed, mean_visibility=visible, baseline_visibility=base_visible,
                             difference=visible-base_visible,
                             robot_blocked=float(np.mean([row['robot_blocked_fraction'] for row in values]))))
    differences = [row['difference'] for row in per_seed]
    values = [row['mean_visibility'] for row in per_seed]
    phases = {}
    predicates = {'initial': lambda row: row['frame'] == 0,
                  'approach_and_grasp': lambda row: row['frame'] > 0 and
                      (row['relative_step'] is None or row['relative_step'] < 0),
                  'post_lift_random': lambda row: row['relative_step'] is not None and row['relative_step'] >= 0}
    for name, predicate in predicates.items():
        phase_rows = [row for row in rows.values() if predicate(row)]
        phases[name] = dict(samples=len(phase_rows),
                            mean_visibility=float(np.mean([row['visible_fraction'] for row in phase_rows])) if phase_rows else None,
                            mean_robot_blocked=float(np.mean([row['robot_blocked_fraction'] for row in phase_rows])) if phase_rows else None)
    return dict(mean_visibility=float(np.mean(values)), p10_seed_visibility=float(np.percentile(values, 10)),
                median_seed_visibility=float(np.median(values)), min_seed_visibility=float(np.min(values)),
                mean_robot_blocked=float(np.mean([row['robot_blocked'] for row in per_seed])),
                improved_seeds=sum(delta > 1e-9 for delta in differences),
                unchanged_seeds=sum(abs(delta) <= 1e-9 for delta in differences),
                worse_seeds=sum(delta < -1e-9 for delta in differences),
                mean_difference=float(np.mean(differences)), per_seed=per_seed, phases=phases)


def draw_comparison(summaries, chosen, output):
    """Show paired seed scores and visibility by action phase."""
    result = summaries[chosen]
    per_seed = result['per_seed']
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for item in per_seed:
        axes[0].plot([0, 1], [item['baseline_visibility'], item['mean_visibility']],
                     marker='o', color='#5388ab', alpha=.4)
    axes[0].set_xticks([0, 1], ['Default', 'Selected'])
    axes[0].set_ylabel('Mean aperture visibility for each seed')
    axes[0].set_title('Held-out initial states (each line is one seed)')
    axes[0].set_ylim(0, 1.03)
    phase_names = ['initial', 'approach_and_grasp', 'post_lift_random']
    positions = np.arange(len(phase_names))
    for offset, name, color in ((-.18, 'default', '#647a8a'), (.18, chosen, '#2d9a66')):
        values = [summaries[name]['phases'][phase]['mean_visibility'] for phase in phase_names]
        axes[1].bar(positions + offset, values, width=.36, label=name, color=color)
    axes[1].set_xticks(positions, ['Initial', 'Approach / grasp', 'After lift'])
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel('Mean sampled aperture visibility')
    axes[1].set_title('Visibility across trajectory phases')
    axes[1].legend()
    for axis in axes:
        axis.grid(axis='y', alpha=.2)
    path = output / 'distribution_comparison.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def compare(config):
    """Validate held-out membership and report paired camera statistics."""
    selection = json.loads(Path(config['selection_results']).read_text())
    heldout = json.loads(Path(config['heldout_results']).read_text())
    chosen = selection['selected']
    if chosen != config['expected_candidate']:
        raise ValueError('Selected candidate differs from expected candidate')
    selection_seeds = set(selection['config']['selection_environment_seeds'])
    heldout_seeds = heldout['config']['selection_environment_seeds']
    if selection_seeds.intersection(heldout_seeds):
        raise ValueError('Camera selection and held-out environment seeds overlap')
    if selection['config']['source_results'] != heldout['config']['source_results']:
        raise ValueError('Selection and held-out scans use different trajectory sources')
    candidates = {entry['name']: entry for entry in heldout['ranked_candidates']}
    if 'default' not in candidates or chosen not in candidates:
        raise ValueError('Held-out scan must include the default and selected camera')
    chosen_source = next(item for item in selection['ranked_candidates'] if item['name'] == chosen)
    if chosen_source['eye'] != candidates[chosen]['eye']:
        raise ValueError('Selected camera eye differs between selection and held-out scans')
    if selection['config']['camera_target'] != heldout['config']['camera_target']:
        raise ValueError('Camera look-at target differs between scans')
    summaries = {name: summarize(candidate, candidates['default'], heldout_seeds)
                 for name, candidate in candidates.items()}
    with ExperimentLogger(team='graph', model='camera_distribution_comparison', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        result = {'config': config, 'selected': chosen, 'selection_seeds': sorted(selection_seeds),
                  'heldout_seeds': heldout_seeds, 'summaries': summaries}
        path = output / 'distribution_results.json'
        path.write_text(json.dumps(result, indent=2))
        figure = draw_comparison(summaries, chosen, output)
        logger.record(selected=chosen, heldout_seeds=len(heldout_seeds))
        logger.link_output(path)
        logger.link_output(figure)
        print(f'Distribution comparison: {output}', flush=True)
