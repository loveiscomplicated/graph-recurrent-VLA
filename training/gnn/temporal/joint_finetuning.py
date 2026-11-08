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
from evaluation.gnn.temporal.insertion import check_development_provenance
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.pose_geometry import geometric_endpoint, unit_quaternion
from training.gnn.data import collate_graph_samples
from training.gnn.temporal.config import resolve_insertion_config, resolve_joint_job_config
from training.gnn.temporal.frozen_source import infer_insertion_data, prepare_insertion_sequences
from training.gnn.temporal.insertion import build_insertion_model, configure_training_memory
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
            gnn_output = gnn(graph)['object21']
            inputs = replace_gnn_predictions(observations, mask, gnn_output, graph.tcp_xyz_world)
            prediction = temporal.forward_sequence(inputs, mask)
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
    """Run a fresh joint fine-tuning job from matched pretrained checkpoints."""
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
    if settings.get('early_stopping'):
        raise ValueError('Disable early stopping for the requested fixed-length joint run')
    device = set_device(effective['device'])
    configure_training_memory(device, settings)
    seed_everything(effective['seed'], device)
    gnn, loaders, identity = prepare_insertion_sequences(effective, device, ('train', 'validation'))
    previous = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if previous.get('kind') != 'temporal_insertion':
        raise ValueError('temporal_checkpoint must be temporal_insertion best weights')
    check_development_provenance(previous, effective, identity)
    for key in ('seed', 'mode', 'model', 'with_robot', 'feature_names', 'observation_contract'):
        if previous['config'][key] != effective[key]:
            raise ValueError(f'Temporal checkpoint differs in {key}')
    temporal = build_insertion_model(effective).to(device)
    temporal.load_state_dict(previous['model'], strict=True)
    effective['normalization'] = previous['config']['normalization']
    effective['joint_finetuning'] = dict(
        rates, temporal_checkpoint=str(checkpoint_path),
        temporal_checkpoint_sha256=file_sha256(checkpoint_path),
        source_sha256=file_sha256(Path(__file__)),
        optimizer='new Adam; both pretrained models are updated',
    )
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
    run_settings = dict(settings, loss=effective['loss'])
    with ExperimentLogger(team='graph', model='joint_' + temporal.architecture,
                          config=effective, seed=effective['seed']) as experiment:
        output = Path(effective['output']['root']) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'effective_config.json', effective)
        experiment.link_output(effective['checkpoint'], checkpoint=True)
        experiment.link_output(checkpoint_path, checkpoint=True)
        history = []
        best = float('inf')
        for epoch in range(settings['epochs'] + 1):
            trained = None if epoch == 0 else run_epoch(gnn, temporal, datasets['train'], device,
                                                        run_settings, optimizer, generator)
            validation = run_epoch(gnn, temporal, datasets['validation'], device, run_settings)
            row = {'epoch': epoch, 'train': trained, 'validation': validation}
            history.append(row)
            write_json(output / 'history.json', history)
            experiment.log(row, step=epoch)
            payload = {'kind': 'joint_gnn_temporal', 'epoch': epoch, 'gnn': gnn.state_dict(),
                       'temporal': temporal.state_dict(), 'optimizer': optimizer.state_dict(),
                       'config': deepcopy(effective), 'dataset_sha256': identity['dataset_sha256'],
                       'validation_loss': validation['loss']}
            atomic_torch_save(output / 'last.pt', payload)
            if validation['loss'] < best:
                best = validation['loss']
                atomic_torch_save(output / 'best.pt', payload)
            print(f'Joint {temporal.architecture} epoch {epoch}/{settings["epochs"]}: '
                  f'validation={validation["loss"]:.6f}', flush=True)
        for name in ('effective_config.json', 'history.json', 'last.pt', 'best.pt'):
            experiment.link_output(output / name)
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
    temporal = build_insertion_model(effective).to(device)
    temporal.load_state_dict(payload['temporal'], strict=True)
    temporal.eval()
    data = infer_insertion_data(gnn, loaders['test'], device, identity, effective, 'test')
    episodes = InsertionEpisodeDataset(data)
    predictions = []
    with torch.no_grad():
        for index in range(len(episodes)):
            observations, _ = episodes[index]
            observations = observations[None].to(device)
            mask = torch.ones(observations.shape[:2], dtype=torch.bool, device=device)
            predictions.append(temporal.forward_sequence(observations, mask)[0].cpu())
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
    with ExperimentLogger(team='graph', model='joint_' + temporal.architecture, job_type='evaluate',
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
