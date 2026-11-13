"""Compare frozen GNN, unchanged rule memory and learned gate on identical episodes."""

from pathlib import Path

import numpy as np
import torch

from data_collection.storage import write_json
from data_collection.episode_dataset import file_sha256
from evaluation.gnn.temporal_memory import (
    error_summary, lifecycle_metrics, replay_memory, scoring_groups,
)
from graph.temporal_gate import LearnedTemporalGate, correct_object21, rollout
from graph.utils.hole_memory import MemorySettings
from training.gnn.temporal_gate import (
    infer_observations, prepare, resolve_gate_config,
)
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def replay_gate(gate, data, device):
    """Replay sorted episodes with explicit reset and no GT passed to the model."""
    positions, alphas = [], []
    gate.eval()
    with torch.no_grad():
        for episode in np.unique(data['episode']):
            mask = data['episode'] == episode
            inputs = tuple(torch.as_tensor(data[name][mask], device=device, dtype=torch.float32)
                           for name in ('prediction', 'tcp_world', 'features'))
            world, alpha = rollout(gate, *inputs)
            positions.append(world.cpu().numpy())
            alphas.append(alpha.cpu().numpy())
    world = np.concatenate(positions)
    relative = torch.from_numpy(world - data['tcp_world'])
    prediction = torch.from_numpy(data['prediction'])
    return {'world': world, 'alpha': np.concatenate(alphas),
            'object21': correct_object21(prediction, relative).numpy(),
            'hole_only_object21': correct_object21(prediction, relative, False).numpy()}


def update_audit(data, world, error, threshold):
    """Score harmful updates against a hold counterfactual using current GT only."""
    target_world = data['target'][:, 10:13] + data['tcp_world']
    same_episode = np.r_[False, data['episode'][1:] == data['episode'][:-1]]
    held_world = np.vstack((world[0], world[:-1]))
    hold_error = np.linalg.norm(held_world - target_world, axis=1) * 1000
    worsening = same_episode & (error > hold_error + 1e-5)
    return {'worsening_updates': int(worsening.sum()),
            'updates_crossing_bad_error_threshold': int(np.sum(
                worsening & (hold_error <= threshold) & (error > threshold))),
            'worsening_error_increase': error_summary((error - hold_error)[worsening])}


def recovery_events(data, error, threshold, available_mask=None):
    """Reobservation: first observed BOX/HOLE after a run with none (diagnostic only)."""
    events = []
    for episode in np.unique(data['episode']):
        indices = np.flatnonzero(data['episode'] == episode)
        available = (data['quality'][indices, 0] > 0 if available_mask is None
                     else available_mask[indices])
        for start in np.flatnonzero(available[1:] & ~available[:-1]) + 1:
            stop = start + 1
            while stop < len(indices) and available[stop]:
                stop += 1
            segment = indices[start:stop]
            recovered = segment[error[segment] <= threshold]
            events.append({'episode': str(episode), 'return_frame': int(data['frame'][segment[0]]),
                           'before_error_mm': float(error[indices[start - 1]]),
                           'return_error_mm': float(error[segment[0]]),
                           'recovery_latency_frames': int(data['frame'][recovered[0]] - data['frame'][segment[0]])
                           if len(recovered) else None,
                           'end_error_mm': float(error[segment[-1]])})
    return events


def save_plots(output, data, errors, alpha, maximum):
    """Save per-episode alpha and all three position-error traces on shared time axes."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    for number, episode in enumerate(np.unique(data['episode'])[:maximum]):
        mask = data['episode'] == episode
        frames = data['frame'][mask]
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        for name, error in errors.items():
            axes[0].plot(frames, error[mask], label=name)
        axes[0].set_ylabel('Hole error (mm)')
        axes[0].legend()
        axes[0].set_title(str(episode))
        axes[1].plot(frames, alpha[mask], color='tab:purple')
        axes[1].set_ylabel('Effective alpha')
        axes[1].set_ylim(-0.03, 1.03)
        axes[1].set_xlabel('Frame')
        absent = data['quality'][mask, 0] == 0
        axes[1].fill_between(frames, 0, 1, where=absent, alpha=0.15, label='No BOX/HOLE')
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output / f'episode_trace_{number:03d}.png', dpi=140)
        plt.close(figure)


def evaluate(config):
    """Evaluate once on original GNN validation; never fit or select gate weights."""
    effective = resolve_gate_config(config)
    gate_path = Path(effective['gate_checkpoint']).expanduser().resolve()
    payload = torch.load(gate_path, map_location='cpu', weights_only=True)
    if payload.get('kind') != 'learned_temporal_gate':
        raise ValueError('gate_checkpoint must be a learned_temporal_gate best.pt')
    device = set_device(effective['device'])
    frozen, loaders, identity = prepare(effective, device, ('test',))
    trained = payload['config']
    for name in ('source_checkpoint_sha256', 'split_manifest_sha256', 'gate', 'feature_names'):
        if effective[name] != trained[name]:
            raise ValueError(f'Evaluation differs from trained gate provenance: {name}')
    if identity['dataset_sha256'] != payload['dataset_sha256']:
        raise ValueError('Gate training and evaluation dataset identities differ')
    gate = LearnedTemporalGate(**effective['gate']).to(device)
    gate.load_state_dict(payload['gate'], strict=True)
    rule_settings = MemorySettings(**effective['rule_memory'])
    with ExperimentLogger(team='graph', model='learned_temporal_gate', config=effective,
                          seed=effective['seed'], job_type='evaluation') as experiment:
        output = Path(effective['output']['root']).expanduser().resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(gate_path, checkpoint=True)
        experiment.link_output(effective['checkpoint'], checkpoint=True)
        data = infer_observations(frozen, loaders['test'], device)
        learned = replay_gate(gate, data, device)
        rule = replay_memory(data['prediction'], data['tcp_world'], data['quality'],
                             data['episode'], data['frame'], rule_settings)
        # All predictions are fixed before any GT/occlusion diagnostic is consumed.
        settings = effective['evaluation']
        groups = scoring_groups(data, settings['visibility_labels'])
        groups['no_box_or_hole_vertices'] = data['quality'][:, 0] == 0
        target_world = data['target'][:, 10:13] + data['tcp_world']
        drift = max(float(np.linalg.norm(target_world[data['episode'] == episode]
                    - target_world[data['episode'] == episode][0], axis=1).max())
                    for episode in np.unique(data['episode']))
        if drift > settings['stationary_tolerance_m']:
            raise ValueError(f'Stationary-hole GT drift exceeds tolerance: {drift} m')
        predictions = {'gnn': data['prediction'], 'rule': rule['object21'],
                       'learned': learned['object21']}
        errors = {name: np.linalg.norm(value[:, 10:13] - data['target'][:, 10:13], axis=1) * 1000
                  for name, value in predictions.items()}
        summary = {'split': 'gate_test_original_gnn_validation', 'frames': len(data['frame']),
                   'episodes': len(np.unique(data['episode'])), 'stationary_gt_drift_m': drift,
                   'occlusion_labels_available': settings['visibility_labels'] is not None,
                   'methods': {}}
        events = {}
        first = np.r_[True, data['episode'][1:] != data['episode'][:-1]]
        for name, prediction in predictions.items():
            error = errors[name]
            world = prediction[:, 10:13] + data['tcp_world']
            method = {'groups': {group: error_summary(error[mask]) for group, mask in groups.items()},
                      'episode_balanced_mean_mm': float(np.mean([
                          error[data['episode'] == ep].mean() for ep in np.unique(data['episode'])])),
                      'endpoint_vector_error': error_summary(np.linalg.norm(
                          prediction[:, 7:10] - data['target'][:, 7:10], axis=1) * 1000),
                      'first_frame_error': error_summary(error[first]),
                      'first_frame_failures': int(np.sum(first & (error > settings['bad_error_mm']))),
                      'update_audit': update_audit(data, world, error, settings['bad_error_mm'])}
            events[name] = recovery_events(data, error, settings['bad_error_mm'])
            if 'strong_occlusion_proxy' in groups:
                events[name + '_occlusion'] = recovery_events(
                    data, error, settings['bad_error_mm'], ~groups['strong_occlusion_proxy'])
            latencies = [event['recovery_latency_frames'] for event in events[name]
                         if event['recovery_latency_frames'] is not None]
            method['recovery'] = {'events': len(events[name]), 'unrecovered': len(events[name]) - len(latencies),
                                  'mean_latency_frames': float(np.mean(latencies)) if latencies else None}
            if name == 'rule':
                method['lifecycle'], _ = lifecycle_metrics(data, rule, errors['gnn'], error, settings['bad_error_mm'])
            if name == 'learned':
                method['initialization'] = {'uninitialized_episodes': 0,
                    'bad_initializations': int(np.sum(first & (error > settings['bad_error_mm'])))}
                initially_bad = []
                for episode in np.unique(data['episode']):
                    indices = np.flatnonzero(data['episode'] == episode)
                    if error[indices[0]] > settings['bad_error_mm']:
                        recovered = indices[error[indices] <= settings['bad_error_mm']]
                        initially_bad.append(int(data['frame'][recovered[0]] - data['frame'][indices[0]])
                                             if len(recovered) else None)
                method['initialization']['unrecovered_episodes'] = initially_bad.count(None)
                method['initialization']['recovery_latencies_frames'] = initially_bad
            if name == 'learned':
                method['alpha_by_group'] = {}
                for group, mask in groups.items():
                    values = learned['alpha'][mask]
                    method['alpha_by_group'][group] = {
                        'frames': int(mask.sum()),
                        'mean': float(values.mean()) if len(values) else None,
                        'median': float(np.median(values)) if len(values) else None,
                        'p90': float(np.percentile(values, 90)) if len(values) else None,
                    }
            summary['methods'][name] = method
        hole_only = learned['hole_only_object21']
        consistent_tip = learned['object21'][:, 7:10] + learned['object21'][:, 10:13]
        original_tip = data['prediction'][:, 7:10] + data['prediction'][:, 10:13]
        summary['assembly_ablation'] = {
            'hole_only_endpoint_vector_error': error_summary(np.linalg.norm(
                hole_only[:, 7:10] - data['target'][:, 7:10], axis=1) * 1000),
            'consistent_endpoint_vector_error': summary['methods']['learned']['endpoint_vector_error'],
            'consistent_tip_preservation_error': error_summary(np.linalg.norm(consistent_tip - original_tip, axis=1) * 1000),
            'hole_only_tip_shift': error_summary(np.linalg.norm(
                hole_only[:, 7:10] + hole_only[:, 10:13] - original_tip, axis=1) * 1000)}
        np.savez_compressed(output / 'predictions.npz', **data,
                            learned_world=learned['world'], learned_alpha=learned['alpha'],
                            learned_object21=learned['object21'], hole_only_object21=hole_only,
                            rule_object21=rule['object21'])
        save_plots(output, data, errors, learned['alpha'], settings['plot_episodes'])
        write_json(output / 'summary.json', summary)
        write_json(output / 'recovery_events.json', events)
        write_json(output / 'effective_config.json', effective)
        write_json(output / 'provenance.json', {
            'gate_checkpoint_sha256': file_sha256(gate_path),
            'source_checkpoint_sha256': effective['source_checkpoint_sha256'],
            'split_manifest_sha256': effective['split_manifest_sha256'],
            'visibility_sha256': file_sha256(Path(settings['visibility_labels']))
            if settings['visibility_labels'] else None})
        experiment.record(device=str(device), dataset_sha256=identity['dataset_sha256'],
                          memory_uses_gt=False, teacher_forcing=False)
        for name, method in summary['methods'].items():
            experiment.log(method['groups']['all'], prefix=name)
        for path in list(output.iterdir()):
            experiment.link_output(path, kind='metrics')
        print(f'Temporal gate comparison: {output}', flush=True)
        return output
