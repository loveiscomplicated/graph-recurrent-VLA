"""Compare three camera settings on the same held-out action trajectories."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation.compare_camera_diagnostics import episode_key, frame_metrics, validate_pair
from utils.experiment_logger import ExperimentLogger


def load_episodes(paths, seeds):
    """Load one complete episode per environment and action seed."""
    episodes = {}
    checkpoints = None
    settings = None
    for path in paths:
        data = json.loads(Path(path).read_text())
        identity = (data['gnn_sha256'], data['temporal_sha256'])
        if checkpoints is None:
            checkpoints = identity
        elif checkpoints != identity:
            raise ValueError('Checkpoint hashes differ within a camera condition')
        comparable = {key: value for key, value in data['config'].items()
                      if key not in ('camera_view', 'environment_seeds', 'output_root')}
        if settings is None:
            settings = comparable
        elif comparable != settings:
            raise ValueError('Non-camera evaluation settings differ within a condition')
        for episode in data['episodes']:
            if episode['environment_seed'] not in seeds:
                continue
            key = episode_key(episode)
            if key in episodes:
                raise ValueError(f'Duplicate episode {key} in camera condition')
            episodes[key] = episode
    return episodes, checkpoints, settings


def plot_comparison(groups, output):
    """Plot per-trajectory paired position errors and detection gaps."""
    names = list(groups)
    colors = ['#2672b8', '#e08b2d', '#2e985e', '#b4549c']
    keys = sorted(key for key in groups[names[0]] if key[1] is not None)
    figure, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
    for axis, part in zip(axes[:2], ('peg', 'hole')):
        for key in keys:
            values = [groups[name][key]['summary'][f'original_{part}_mean_mm'] for name in names]
            axis.plot(range(len(names)), values, color='gray', alpha=.35, marker='o')
        for index, name in enumerate(names):
            values = [groups[name][key]['summary'][f'original_{part}_mean_mm'] for key in keys]
            axis.scatter([index], [np.mean(values)], s=100, color=colors[index], zorder=3)
        axis.set_title(f'{part.title()} position error by rollout')
        axis.set_ylabel('Mean position error (mm)')
    for key in keys:
        values = [sum(frame['detections'] == 0 for frame in groups[name][key]['frames']
                      if frame['relative_step'] is not None and frame['relative_step'] > 0)
                  for name in names]
        axes[2].plot(range(len(names)), values, color='gray', alpha=.35, marker='o')
    axes[2].set_title('Frames with zero detections')
    axes[2].set_ylabel('Frames / 35')
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=12)
        axis.grid(alpha=.2)
    figure.suptitle('Held-out initial states: paired random actions; colored dots show means')
    path = output / 'heldout_camera_comparison.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_visibility(visibility, output):
    """Show sampled front-opening visibility and robot occlusion per camera."""
    names = list(visibility)
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(9, 5), layout='constrained')
    axis.bar(positions - .18, [visibility[name]['mean_visibility'] for name in names],
             width=.36, label='Opening samples visible', color='#2e985e')
    axis.bar(positions + .18, [visibility[name]['mean_robot_blocked'] for name in names],
             width=.36, label='Opening samples robot-blocked', color='#cf5148')
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel('Fraction of sampled opening points')
    axis.set_title('Held-out states: sampled depth and segmentation')
    axis.legend()
    path = output / 'heldout_visibility.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def compare(config):
    """Verify checkpoint and per-frame parity before aggregation."""
    groups = {}
    checkpoints = None
    settings = None
    for name, paths in config['camera_results'].items():
        groups[name], identity, comparable = load_episodes(paths, config['environment_seeds'])
        if checkpoints is None:
            checkpoints = identity
            settings = comparable
        elif identity != checkpoints:
            raise ValueError('Checkpoint hashes differ across cameras')
        elif comparable != settings:
            raise ValueError('Non-camera evaluation settings differ across cameras')
    names = list(groups)
    if len(names) < 2:
        raise ValueError('At least two camera conditions are required')
    keys = groups[names[0]].keys()
    if any(group.keys() != keys for group in groups.values()):
        raise ValueError('Camera conditions do not contain the same episodes')
    required = {(seed, action) for seed in config['environment_seeds']
                for action in [None, *config['action_seeds']]}
    if set(keys) != required:
        raise ValueError('Camera conditions must include every configured seed and action')
    failed = [str(key) for key in keys if any(not group[key]['switched'] for group in groups.values())]
    if failed:
        raise ValueError(f'Cannot compare requested action windows; grasp/lift failed for {failed}')
    maximum_difference = max(validate_pair(groups[names[0]][key], groups[name][key])
                             for name in names[1:] for key in keys)
    with ExperimentLogger(team='graph', model='camera_holdout_comparison', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        metrics = {}
        per_episode = {}
        for name, episodes in groups.items():
            metrics[name] = {kind: frame_metrics([episode for key, episode in episodes.items()
                                                  if (key[1] is None) == (kind == 'expert')])
                             for kind in ('expert', 'random')}
            per_episode[name] = [{'environment_seed': key[0], 'action_seed': key[1],
                                  **episode['summary']} for key, episode in episodes.items()]
        visibility = None
        if config.get('visibility_results'):
            scan = json.loads(Path(config['visibility_results']).read_text())
            if set(scan['config']['selection_environment_seeds']) != set(config['environment_seeds']):
                raise ValueError('Visibility and model evaluation initial states differ')
            by_name = {candidate['name']: candidate for candidate in scan['ranked_candidates']}
            visibility = {name: {key: by_name[source][key] for key in
                                 ('mean_visibility', 'worst_seed_mean_visibility',
                                  'mean_robot_blocked', 'min_projected_bbox_width_px')}
                          for name, source in config['visibility_name_map'].items()}
        result = {'config': config, 'paired_episodes': len(keys), 'maximum_gt_action_difference': maximum_difference,
                  'gnn_sha256': checkpoints[0], 'temporal_sha256': checkpoints[1],
                  'metrics': metrics, 'per_episode': per_episode, 'visibility': visibility}
        path = output / 'holdout_results.json'
        path.write_text(json.dumps(result, indent=2))
        figure = plot_comparison(groups, output)
        logger.record(paired_episodes=len(keys), maximum_gt_action_difference=maximum_difference)
        logger.link_output(path)
        logger.link_output(figure)
        if visibility is not None:
            logger.link_output(plot_visibility(visibility, output))
        print(f'Holdout comparison: {output}', flush=True)
