"""Compare two random-action diagnostic runs on verified paired trajectories."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_logger import ExperimentLogger


def episode_key(episode):
    """Identify the environment and expert or random-action trajectory."""
    return episode['environment_seed'], episode['action_seed']


def validate_pair(original, selected):
    """Require identical actions and simulator object trajectories at every frame."""
    if episode_key(original) != episode_key(selected):
        raise ValueError('Environment and action seeds must match')
    if original['switch_frame'] != selected['switch_frame'] or len(original['frames']) != len(selected['frames']):
        raise ValueError(f'Trajectory length or switch frame mismatch: {episode_key(original)}')
    maximum_difference = 0.0
    for old, new in zip(original['frames'], selected['frames']):
        if old['frame'] != new['frame'] or old['relative_step'] != new['relative_step']:
            raise ValueError(f'Frame mismatch: {episode_key(original)}')
        if old['grasp'] != new['grasp']:
            raise ValueError(f'Grasp mismatch: {episode_key(original)} frame {old["frame"]}')
        for field in ('peg_gt', 'hole_gt', 'action'):
            if (field in old) != (field in new):
                raise ValueError(f'{field} availability mismatch: {episode_key(original)}')
            if field not in old:
                continue
            delta = float(np.max(np.abs(np.asarray(old[field]) - np.asarray(new[field]))))
            maximum_difference = max(maximum_difference, delta)
            if delta > 1e-6:
                raise ValueError(f'{field} mismatch ({delta}): {episode_key(original)} frame {old["frame"]}')
    return maximum_difference


def frame_metrics(episodes):
    """Compute frame-weighted errors and node-weighted diagnostic RMSE."""
    frames = [frame for episode in episodes for frame in episode['frames']
              if frame['relative_step'] is not None and frame['relative_step'] > 0]
    result = {'frames': len(frames), 'zero_detections': sum(frame['detections'] == 0 for frame in frames),
              'grasp_frames': sum(frame['grasp'] for frame in frames)}
    for variant in ('original', 'correct_observed', 'correct_predicted', 'correct_all',
                    'assigned_gt_only', 'assigned_gt_as_observed', 'complete_gt'):
        for part in ('peg', 'hole'):
            values = [frame['errors'][variant][part] for frame in frames
                      if frame['errors'][variant][part] is not None]
            result[f'{variant}_{part}_mean_mm'] = float(np.mean(values)) if values else None
    detection = [distance for frame in frames for distance in frame['detection_nearest_mm']]
    result['detection_nearest_rmse_mm'] = float(np.sqrt(np.mean(np.square(detection)))) if detection else None
    result['detection_count'] = len(detection)
    for observed, name in ((True, 'observed'), (False, 'predicted')):
        distances = [distance for frame in frames
                     for distance, flag in zip(frame['track_fixed_mm'], frame['track_observed'])
                     if distance is not None and flag == observed]
        result[f'{name}_track_rmse_mm'] = float(np.sqrt(np.mean(np.square(distances)))) if distances else None
        result[f'{name}_track_count'] = len(distances)
    result['unassigned_nodes'] = sum(sum(identity is None for identity in frame['track_reference_ids']) for frame in frames)
    result['track_nodes'] = sum(frame['tracks'] for frame in frames)
    return result


def plot_pairs(pairs, output):
    """Show per-trajectory mean errors and missing detections for both cameras."""
    random_pairs = [(old, new) for old, new in pairs if old['action_seed'] is not None]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for axis, part in zip(axes[0], ('peg', 'hole')):
        for old, new in random_pairs:
            before = old['summary'][f'original_{part}_mean_mm']
            after = new['summary'][f'original_{part}_mean_mm']
            axis.plot([0, 1], [before, after], marker='o', alpha=.6)
        axis.set_xticks([0, 1], ['Default', 'Selected'])
        axis.set_ylabel('Mean 3D error (mm)')
        axis.set_title(f'{part.title()} error per paired rollout')
        axis.grid(alpha=.2)
    for old, new in random_pairs:
        axis = axes[1, 0]
        zero = [sum(frame['detections'] == 0 for frame in episode['frames']
                    if frame['relative_step'] is not None and frame['relative_step'] > 0)
                for episode in (old, new)]
        axis.plot([0, 1], zero, marker='o', alpha=.6)
        axis = axes[1, 1]
        values = [episode['summary']['predicted_fixed_rmse_mm'] for episode in (old, new)]
        axis.plot([0, 1], [np.nan if value is None else value for value in values], marker='o', alpha=.6)
    axes[1, 0].set_title('Frames with no detected corners')
    axes[1, 0].set_ylabel('Frames out of 35')
    axes[1, 1].set_title('Predicted tracks: birth-reference RMSE')
    axes[1, 1].set_ylabel('Node-weighted RMSE (mm)')
    for axis in axes[1]:
        axis.set_xticks([0, 1], ['Default', 'Selected'])
        axis.grid(alpha=.2)
    fig.suptitle('Same initial states and actions: camera comparison (each line is one rollout)')
    path = output / 'camera_comparison.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compare(config):
    """Check experiment identity, pair trajectories, then write metrics and a plot."""
    original = json.loads(Path(config['default_results']).read_text())
    selected = json.loads(Path(config['selected_results']).read_text())
    for key in ('gnn_sha256', 'temporal_sha256'):
        if original[key] != selected[key]:
            raise ValueError(f'{key} differs between runs')
    old_settings = {key: value for key, value in original['config'].items() if key not in ('camera_view', 'output_root')}
    new_settings = {key: value for key, value in selected['config'].items() if key not in ('camera_view', 'output_root')}
    if old_settings != new_settings:
        raise ValueError('Effective settings differ beyond camera position and output root')
    old_episodes = {episode_key(episode): episode for episode in original['episodes']}
    new_episodes = {episode_key(episode): episode for episode in selected['episodes']}
    if old_episodes.keys() != new_episodes.keys():
        raise ValueError('Episode seed sets differ')
    pairs = [(episode, new_episodes[key]) for key, episode in old_episodes.items()]
    maximum_difference = max(validate_pair(old, new) for old, new in pairs)
    with ExperimentLogger(team='graph', model='camera_diagnostics_comparison', config=config,
                          seed=old_settings['model_seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        summary = {'config': config, 'maximum_paired_gt_action_difference': maximum_difference,
                   'paired_episodes': len(pairs), 'metrics': {}}
        for name, index in (('default', 0), ('selected', 1)):
            for group in ('random', 'expert'):
                episodes = [pair[index] for pair in pairs if (pair[index]['action_seed'] is None) == (group == 'expert')]
                summary['metrics'][f'{name}_{group}'] = frame_metrics(episodes)
        result = output / 'comparison.json'
        result.write_text(json.dumps(summary, indent=2))
        figure = plot_pairs(pairs, output)
        logger.record(paired_episodes=len(pairs), maximum_paired_difference=maximum_difference)
        logger.link_output(result)
        logger.link_output(figure)
        print(f'Camera comparison: {output}', flush=True)
