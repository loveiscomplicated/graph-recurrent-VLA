"""Fine-tune a selected GNN and a temporal insertion model together.

The original causal observations supply robot motion and visibility features.
Only their Object21 prefix is rebuilt from the trainable GNN on every step.
"""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset, InsertionWindowDataset
from data_collection.temporal.node_tokens import collate_node_windows
from evaluation.gnn.temporal.insertion import check_development_provenance
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.pose_geometry import geometric_endpoint, unit_quaternion
from graph.temporal.node_features import capture_pre_pool_features, ordered_node_features
from graph.temporal.node_token_mamba import NodeTokenMamba
from training.gnn.data import collate_graph_samples
from training.gnn.temporal.config import resolve_insertion_config, resolve_joint_job_config
from training.gnn.temporal.frozen_source import infer_insertion_data, prepare_insertion_sequences
from training.gnn.temporal.insertion import (
    build_insertion_model, configure_training_memory, update_early_stopping,
)
from training.gnn.temporal.joint_continuation import (
    capture_random_state, load_joint_continuation, restore_random_state,
)
from training.gnn.temporal.node_features import infer_node_data
from training.gnn.temporal.pose_loss import insertion_loss
from training.gnn.trainer import atomic_torch_save, seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


class GraphWindows:
    """Align cached graph frames with the existing episode-safe temporal windows."""

    def __init__(self, episodes, loader, sequence_length):
        self.windows = InsertionWindowDataset(episodes, sequence_length)
        self.episodes = episodes
        subset = loader.dataset
        if not hasattr(subset, 'indices') or not hasattr(subset.dataset, 'samples'):
            raise ValueError('Joint fine-tuning expects the selected-M0 cached episode loader')
        self.source = subset.dataset
        self.indices = {self.source.samples[index].sample_id: index for index in subset.indices}
        ids = [str(value) for value in episodes.data['sample_ids']]
        if len(self.indices) != len(ids) or set(self.indices) != set(ids):
            raise ValueError('Cached graph and temporal observation frame IDs must match exactly')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start, stop = self.windows.windows[index]
        observations, target = self.windows[index]
        rows = self.episodes.indices[episode][start:stop]
        sample_ids = self.episodes.data['sample_ids'][rows]
        samples = [self.source[self.indices[str(sample_id)]] for sample_id in sample_ids]
        return observations, target, samples


def collate_windows(windows):
    """Pad temporal fields while preserving chronological graph order."""
    observations, targets, graphs = zip(*windows)
    lengths = [len(sequence) for sequence in observations]
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    batch = collate_graph_samples([sample for sequence in graphs for sample in sequence])
    return pad_sequence(observations, batch_first=True), pad_sequence(targets, batch_first=True), mask, batch


def replace_gnn_predictions(observations, mask, predictions, tcp_world):
    """Replace Object21 with differentiable world-frame predictions, keeping causal context."""
    if predictions.shape != (int(mask.sum()), 21) or tcp_world.shape != (len(predictions), 3):
        raise ValueError('Graph outputs must have one Object21 and TCP position per valid frame')
    world = torch.cat((predictions[:, :3] + tcp_world, unit_quaternion(predictions[:, 3:7]),
                       predictions[:, 7:10], predictions[:, 10:13] + tcp_world,
                       unit_quaternion(predictions[:, 13:17]), predictions[:, 17:21]), dim=-1)
    lengths = mask.sum(dim=1).tolist()
    padded = pad_sequence(world.split(lengths), batch_first=True)
    return torch.cat((padded, observations[..., 21:]), dim=-1)


def build_joint_temporal(effective, node_width=None):
    """Restore the pretrained input architecture without changing its parameters."""
    if node_width is None:
        return build_insertion_model(effective)
    return NodeTokenMamba(node_width=node_width, mode=effective['mode'],
                          with_robot=effective['with_robot'], **effective['model'])


def forward_joint_predictions(gnn, temporal, graph, observations, mask):
    """Recompute Object21 and optional node tokens with an intact autograd graph."""
    if not isinstance(temporal, NodeTokenMamba):
        output = gnn(graph)['object21']
        inputs = replace_gnn_predictions(observations, mask, output, graph.tcp_xyz_world)
        return temporal.forward_sequence(inputs, mask)

    captured = []

    # Retain the original tensor so temporal supervision reaches message passing.
    with capture_pre_pool_features(gnn, captured.append):
        output = gnn(graph)['object21']
    if len(captured) != 1:
        raise ValueError('Expected exactly one pre-pooling node representation')
    inputs = replace_gnn_predictions(observations, mask, output, graph.tcp_xyz_world)
    ordered = ordered_node_features(graph, captured[0])
    nodes = pad_sequence(ordered, batch_first=True)
    counts = torch.tensor([len(frame) for frame in ordered], device=nodes.device)
    node_mask = torch.arange(nodes.shape[1], device=nodes.device)[None] < counts[:, None]
    lengths = mask.sum(1).tolist()
    nodes = pad_sequence(nodes.split(lengths), batch_first=True)
    node_mask = pad_sequence(node_mask.split(lengths), batch_first=True)
    return temporal.forward_nodes(inputs, nodes, node_mask, mask)


def run_epoch(gnn, temporal, windows, device, settings, optimizer=None, generator=None):
    """Measure frame-weighted objective and physical error statistics for one split."""
    training = optimizer is not None
    gnn.train(training)
    temporal.train(training)
    loader = DataLoader(windows, batch_size=settings['episode_batch_size'], shuffle=training,
                        generator=generator, collate_fn=collate_windows)
    totals, count, errors = {}, 0, {}
    with torch.set_grad_enabled(training):
        for observations, target, mask, batch in loader:
            observations, target, mask = (value.to(device) for value in (observations, target, mask))
            graph = batch.graph.to(device)
            prediction = forward_joint_predictions(gnn, temporal, graph, observations, mask)
            loss, terms = insertion_loss(prediction, target, mask, temporal.mode, settings['loss'])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                           for p in gnn.parameters()):
                    raise RuntimeError('No finite nonzero gradient reached the GNN')
                torch.nn.utils.clip_grad_norm_(list(gnn.parameters()) + list(temporal.parameters()),
                                               settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
            frames = int(mask.sum())
            for name, value in {'loss': loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * frames
            count += frames
            if not training:
                valid = prediction[mask].detach().cpu()
                actual = target[mask].detach().cpu()
                measures = per_frame_errors({'object21': valid, 'endpoint_geom': geometric_endpoint(valid),
                                             'endpoint_residual': torch.zeros_like(valid[:, :3])}, actual)
                for name, value in measures.items():
                    if not name.startswith('endpoint_residual'):
                        errors.setdefault(name, []).append(value)
    if not count:
        raise ValueError('Joint fine-tuning requires nonempty windows')
    result = {name: value / count for name, value in totals.items()}
    result['frames'] = count
    if errors:
        result['physical_errors'] = {name: statistics(np.concatenate(values)) for name, values in errors.items()}
    return result


def train(config):
    """Train or continue paired models, selecting only on development validation."""
    effective = resolve_insertion_config(config)
    if not effective.get('selected_m0'):
        raise ValueError('Joint fine-tuning currently requires selected_m0')
    checkpoint_path = Path(config.get('temporal_checkpoint', '')).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError('temporal_checkpoint must point to a pretrained temporal best.pt')
    settings = effective['training']
    rates = config.get('joint_finetuning', {})
    for key in ('gnn_lr', 'temporal_lr'):
        value = rates.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'joint_finetuning.{key} must be finite and positive')
    device = set_device(effective['device'])
    configure_training_memory(device, settings)
    seed_everything(effective['seed'], device)
    print(f'Loading paired GNN and temporal checkpoints for seed {effective["seed"]}', flush=True)
    gnn, loaders, identity = prepare_insertion_sequences(effective, device, ('train', 'validation'))
    previous = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if previous.get('kind') not in ('temporal_insertion', 'node_token_mamba'):
        raise ValueError('temporal_checkpoint must contain insertion or node-token best weights')
    node_width = None
    if previous['kind'] == 'node_token_mamba':
        node_width = previous['config']['node_width']
        effective['node_width'] = node_width
        effective['node_token_contract'] = previous['config']['node_token_contract']
    check_development_provenance(previous, effective, identity)
    for key in ('seed', 'mode', 'model', 'with_robot', 'feature_names', 'observation_contract'):
        if previous['config'][key] != effective[key]:
            raise ValueError(f'Temporal checkpoint differs in {key}')
    temporal = build_joint_temporal(effective, node_width).to(device)
    temporal.load_state_dict(previous['model'], strict=True)
    effective['normalization'] = previous['config']['normalization']
    repository = Path(__file__).resolve().parents[3]
    for name in ('graph/gated_mpnn.py', 'graph/temporal/node_token_mamba.py',
                 'training/gnn/temporal/joint_finetuning.py',
                 'training/gnn/temporal/joint_continuation.py'):
        effective['source_sha256'][name] = file_sha256(repository / name)
    effective['joint_finetuning'] = dict(
        rates, temporal_checkpoint=str(checkpoint_path),
        temporal_checkpoint_sha256=file_sha256(checkpoint_path),
        source_sha256=file_sha256(Path(__file__)),
        optimizer='new Adam; both pretrained models are updated',
    )
    continuation = None
    best_payload = None
    history = []
    early_state = None
    starting_epoch = 0
    continuation_path = config.get('continuation_checkpoint')
    if continuation_path is not None:
        continuation, best_payload, history, early_state, metadata = load_joint_continuation(
            continuation_path, effective, identity)
        starting_epoch = continuation['epoch']
        gnn.load_state_dict(continuation['gnn'], strict=True)
        temporal.load_state_dict(continuation['temporal'], strict=True)
        effective['continuation'] = metadata
        effective['joint_finetuning']['optimizer'] = metadata['optimizer']
        print(f'Continuing paired weights and Adam from joint epoch {starting_epoch}', flush=True)
    datasets = {}
    for split, loader in loaders.items():
        print(f'Preparing {split} observations', flush=True)
        data = infer_insertion_data(gnn, loader, device, identity, effective, split)
        episodes = InsertionEpisodeDataset(data)
        datasets[split] = GraphWindows(episodes, loader, settings['sequence_length'])
    gnn.requires_grad_(True)
    optimizer = torch.optim.Adam([{'params': gnn.parameters(), 'lr': rates['gnn_lr']},
                                  {'params': temporal.parameters(), 'lr': rates['temporal_lr']}])
    generator = torch.Generator().manual_seed(effective['seed'])
    if continuation is not None:
        optimizer.load_state_dict(continuation['optimizer'])
    run_settings = dict(settings, loss=effective['loss'])
    early_settings = settings.get('early_stopping')
    if early_settings and early_state is None:
        early_state = {'reference_loss': float('inf'), 'stale_epochs': 0}
    model_name = 'joint_node_token_mamba' if node_width is not None else 'joint_' + temporal.architecture
    with ExperimentLogger(team='graph', model=model_name,
                          config=effective, seed=effective['seed']) as experiment:
        output = Path(effective['output']['root']) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'effective_config.json', effective)
        experiment.link_output(effective['checkpoint'], checkpoint=True)
        experiment.link_output(checkpoint_path, checkpoint=True)
        if continuation is not None:
            experiment.link_output(continuation_path, checkpoint=True)
        experiment.record(dataset_sha256=identity['dataset_sha256'],
                          gnn_parameter_count=sum(p.numel() for p in gnn.parameters()),
                          temporal_parameter_count=sum(p.numel() for p in temporal.parameters()),
                          test_frames_read=False, backend=temporal.backend,
                          torch_num_threads=torch.get_num_threads())
        best = float('inf') if best_payload is None else best_payload['validation_loss']
        if best_payload is not None:
            atomic_torch_save(output / 'best.pt', dict(best_payload, config=deepcopy(effective)))

        def save_epoch(epoch, validation, improved):
            """Keep paired best weights and a self-contained resumable last state."""
            payload = {
                'kind': 'joint_gnn_temporal', 'epoch': epoch,
                'gnn': gnn.state_dict(), 'temporal': temporal.state_dict(),
                'optimizer': optimizer.state_dict(), 'config': deepcopy(effective),
                'dataset_sha256': identity['dataset_sha256'],
                'validation_loss': validation['loss'], 'history': history,
                'early_stopping_state': deepcopy(early_state),
                'random_state': capture_random_state(generator, device),
            }
            if improved:
                atomic_torch_save(output / 'best.pt', payload)
                experiment.link_output(output / 'best.pt')
            payload['best_checkpoint'] = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            atomic_torch_save(output / 'last.pt', payload)
            write_json(output / 'history.json', history)

        validation = run_epoch(gnn, temporal, datasets['validation'], device, run_settings)
        if continuation is not None:
            if not math.isclose(validation['loss'], continuation['validation_loss'], rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError('Restored joint validation loss differs from the continuation checkpoint')
            write_json(output / 'resume_baseline.json', {'epoch': starting_epoch, 'validation': validation})
            experiment.log({'resume_baseline': validation}, step=starting_epoch)
            if not history:
                history.append({'epoch': starting_epoch, 'train': None, 'validation': validation})
            restore_random_state(continuation.get('random_state'), generator, device, effective['seed'])
            save_epoch(starting_epoch, validation, improved=False)
        else:
            history.append({'epoch': 0, 'train': None, 'validation': validation})
            best = validation['loss']
            if early_settings:
                early_state = update_early_stopping(early_state, best, early_settings)
            experiment.log(history[-1], step=0)
            save_epoch(0, validation, improved=True)
        print(f'Joint {temporal.architecture} epoch {starting_epoch}/{settings["epochs"]}: '
              f'validation={validation["loss"]:.6f}', flush=True)
        early_stopped = bool(early_settings and starting_epoch >= early_settings['min_epochs']
                             and early_state['stale_epochs'] >= early_settings['patience'])
        epoch = starting_epoch
        final_epoch_limit = starting_epoch if early_stopped else settings['epochs']
        for epoch in range(starting_epoch + 1, final_epoch_limit + 1):
            trained = run_epoch(gnn, temporal, datasets['train'], device, run_settings, optimizer, generator)
            validation = run_epoch(gnn, temporal, datasets['validation'], device, run_settings)
            if early_settings:
                early_state = update_early_stopping(early_state, validation['loss'], early_settings)
            row = {'epoch': epoch, 'train': trained, 'validation': validation}
            history.append(row)
            experiment.log(row, step=epoch)
            improved = validation['loss'] < best
            if improved:
                best = validation['loss']
            save_epoch(epoch, validation, improved)
            print(f'Joint {temporal.architecture} epoch {epoch}/{settings["epochs"]}: '
                  f'train={trained["loss"]:.6f}, validation={validation["loss"]:.6f}', flush=True)
            early_stopped = bool(early_settings and epoch >= early_settings['min_epochs']
                                 and early_state['stale_epochs'] >= early_settings['patience'])
            if early_stopped:
                print(f'Early stopping at joint epoch {epoch}; '
                      f'{early_state["stale_epochs"]} epochs without improvement', flush=True)
                break
        completion = {'completed_epoch': epoch, 'starting_epoch': starting_epoch,
                      'early_stopped': early_stopped, 'best_validation_loss': best,
                      'stop_reason': 'early_stopping' if early_stopped else 'epoch_limit'}
        write_json(output / 'training_summary.json', completion)
        experiment.record(**completion)
        for name in ('effective_config.json', 'history.json', 'last.pt', 'best.pt', 'training_summary.json'):
            experiment.link_output(output / name)
        if continuation is not None:
            experiment.link_output(output / 'resume_baseline.json')
        return output


def evaluate(config):
    """Evaluate paired joint weights on complete held-out episodes and selected frames."""
    import json

    effective = resolve_insertion_config(config)
    checkpoint_path = Path(config.get('joint_checkpoint', '')).expanduser().resolve()
    selection_path = Path(config.get('evaluation_selection', '')).expanduser().resolve()
    if not checkpoint_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError('joint_checkpoint and evaluation_selection must be existing files')
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if payload.get('kind') != 'joint_gnn_temporal':
        raise ValueError('joint_checkpoint must contain paired GNN and temporal weights')
    device = set_device(effective['device'])
    configure_training_memory(device, effective['training'])
    gnn, loaders, identity = prepare_insertion_sequences(effective, device, ('test',))
    saved = payload['config']
    for name in ('source_checkpoint_sha256', 'split_manifest_sha256', 'mode', 'model',
                 'with_robot', 'feature_names', 'observation_contract'):
        if saved[name] != effective[name]:
            raise ValueError(f'Joint checkpoint evaluation differs in {name}')
    if payload['dataset_sha256'] != identity['dataset_sha256']:
        raise ValueError('Joint checkpoint and evaluation dataset identities differ')
    gnn.load_state_dict(payload['gnn'], strict=True)
    gnn.eval().requires_grad_(False)
    temporal = build_joint_temporal(effective, saved.get('node_width')).to(device)
    temporal.load_state_dict(payload['temporal'], strict=True)
    temporal.eval()
    if isinstance(temporal, NodeTokenMamba):
        data = infer_node_data(gnn, loaders['test'], device, identity, effective, 'test')
    else:
        data = infer_insertion_data(gnn, loaders['test'], device, identity, effective, 'test')
    episodes = InsertionEpisodeDataset(data)
    predictions = []
    with torch.no_grad():
        for index in range(len(episodes)):
            observations, _ = episodes[index]
            observations = observations[None].to(device)
            mask = torch.ones(observations.shape[:2], dtype=torch.bool, device=device)
            if isinstance(temporal, NodeTokenMamba):
                rows = episodes.indices[index]
                nodes = [data['node_features'][row] for row in rows]
                batch = collate_node_windows([(observations[0].cpu(), episodes[index][1], nodes)])
                node_values, node_mask = (value.to(device) for value in batch[2:4])
                output = temporal.forward_nodes(observations, node_values, node_mask, mask)
            else:
                output = temporal.forward_sequence(observations, mask)
            predictions.append(output[0].cpu())
    predictions = torch.cat(predictions)
    target = torch.cat([episodes[index][1] for index in range(len(episodes))])
    ordered_ids = [str(episodes.data['sample_ids'][row]) for rows in episodes.indices for row in rows]
    records = [json.loads(line) for line in selection_path.read_text().splitlines()]
    lookup = {sample_id: index for index, sample_id in enumerate(ordered_ids)}
    if (len(lookup) != len(ordered_ids) or len({row['sample_id'] for row in records}) != len(records)
            or any(row['sample_id'] not in lookup for row in records)):
        raise ValueError('Evaluation selection must contain unique held-out frame IDs')
    indices = [lookup[row['sample_id']] for row in records]
    selected = predictions[indices]
    actual = target[indices]
    errors = per_frame_errors({'object21': selected, 'endpoint_geom': geometric_endpoint(selected),
                               'endpoint_residual': torch.zeros_like(selected[:, :3])}, actual)
    errors = {name: value for name, value in errors.items() if not name.startswith('endpoint_residual')}
    groups = {'all': np.ones(len(records), dtype=bool),
              'physical_both_pass': np.array([row['both_pass'] for row in records], dtype=bool)}
    result = {'checkpoint_sha256': file_sha256(checkpoint_path),
              'selection_sha256': file_sha256(selection_path),
              'counts': {name: int(mask.sum()) for name, mask in groups.items()},
              'metrics': {name: {metric: statistics(values[mask]) for metric, values in errors.items()}
                          for name, mask in groups.items()}}
    effective['joint_checkpoint'] = str(checkpoint_path)
    effective['evaluation_selection'] = str(selection_path)
    model_name = 'joint_node_token_mamba' if saved.get('node_width') is not None else 'joint_' + temporal.architecture
    with ExperimentLogger(team='graph', model=model_name, job_type='evaluate',
                          config=effective, seed=effective['seed']) as experiment:
        output = Path(effective['output']['root']) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'effective_config.json', effective)
        write_json(output / 'original_validation_21d.json', result)
        np.savez_compressed(output / 'selected_errors.npz', sample_ids=np.array([r['sample_id'] for r in records]),
                            prediction=selected.numpy(), target=actual.numpy(), **errors)
        experiment.link_output(checkpoint_path, checkpoint=True)
        for name in ('effective_config.json', 'original_validation_21d.json', 'selected_errors.npz'):
            experiment.link_output(output / name)
        return output


def train_from_base(config):
    """Resolve a seed job before invoking the existing joint training workflow."""
    return train(resolve_joint_job_config(config))


def train_and_evaluate(config):
    """Train a paired model and evaluate its selected best exactly once at the end."""
    if not config.get('comparison_output'):
        raise ValueError('Joint training and evaluation requires comparison_output')
    output = train(config)
    evaluation_config = deepcopy(config)
    evaluation_config['trainer'] = 'training.gnn.temporal.joint_finetuning:evaluate'
    evaluation_config['joint_checkpoint'] = str(output / 'best.pt')
    evaluation_config['output']['root'] = config['comparison_output']
    return evaluate(evaluation_config)


def continue_and_evaluate(config):
    """Continue joint optimization and evaluate its selected best after completion."""
    if not config.get('continuation_checkpoint') or not config.get('comparison_output'):
        raise ValueError('Joint continuation needs continuation_checkpoint and comparison_output')
    return train_and_evaluate(config)
