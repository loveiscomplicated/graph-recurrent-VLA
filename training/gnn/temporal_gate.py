"""Frozen vertex inference and independent temporal-gate training orchestration."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal_gate_data import selected_loaders
from evaluation.gnn.temporal_memory import sample_order
from graph.temporal_gate import LearnedTemporalGate, QUALITY_NAMES, observation_features, rollout
from training.gnn.trainer import build_model, resolve_config, seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def resolve_gate_config(config):
    """Resolve workflow defaults without modifying the frozen checkpoint config."""
    effective = deepcopy(config)
    effective.setdefault('device', 'cpu')
    effective.setdefault('seed', 0)
    if type(effective['seed']) is not int or not 0 <= effective['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    defaults = {
        'gate': {'hidden': 32, 'position_scale_m': 0.1},
        'training': {'epochs': 20, 'lr': 0.001, 'loss': 'huber', 'huber_delta_m': 0.01,
                     'gradient_clip_norm': 1.0},
        'evaluation': {'batch_size': 64, 'visibility_labels': None, 'bad_error_mm': 10.0,
                       'plot_episodes': 3, 'stationary_tolerance_m': 0.0001},
        'output': {'root': 'outputs/learned_temporal_gate'},
    }
    for section, values in defaults.items():
        for name, value in values.items():
            effective.setdefault(section, {}).setdefault(name, value)
    for name in ('checkpoint', 'split_manifest'):
        path = Path(effective[name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}: {path}')
        effective[name] = str(path)
    cache = Path(effective['data']['cache_dir']).expanduser().resolve()
    effective['data'] = {'cache_dir': str(cache)}
    for section, name in (('training', 'epochs'), ('evaluation', 'batch_size'),
                          ('evaluation', 'plot_episodes')):
        value = effective[section][name]
        if type(value) is not int or value < 1:
            raise ValueError(f'{section}.{name} must be an integer >= 1')
    for section, names in (('training', ('lr', 'huber_delta_m', 'gradient_clip_norm')),
                           ('evaluation', ('bad_error_mm', 'stationary_tolerance_m'))):
        for name in names:
            value = effective[section][name]
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{section}.{name} must be finite and positive')
    if effective['training']['loss'] not in ('huber', 'l1'):
        raise ValueError('training.loss must be huber or l1')
    LearnedTemporalGate(**effective['gate'])
    return effective


def load_frozen_source(effective, device):
    """Load exact best.pt architecture/state head and disable all parameter gradients."""
    checkpoint = torch.load(effective['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'best_weights':
        raise ValueError('checkpoint must be the original best_weights best.pt')
    source = resolve_config(checkpoint['config'])
    if source['model']['name'] != 'vertex_edge_gated_mpnn':
        raise ValueError('Expected the existing vertex_edge_gated_mpnn checkpoint')
    if source['data']['factory'] != 'training.gnn.cached_data:build_data':
        raise ValueError('Expected a published cache-only GNN checkpoint')
    if source['data']['empty_inputs'] != 'keep':
        raise ValueError('Temporal gate requires empty_inputs=keep')
    source['data']['cache_dir'] = effective['data']['cache_dir']
    source['training']['batch_size'] = effective['evaluation']['batch_size']
    model = build_model(source['model'], device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, source, checkpoint['dataset_sha256']


def infer_observations(model, loader, device):
    """Capture frozen outputs/observations; keep labels separate from rollout inputs."""
    arrays = {name: [] for name in ('prediction', 'target', 'tcp_world', 'features', 'quality')}
    identifiers = []
    with torch.no_grad():
        for batch in loader:
            graph = batch.graph
            features = observation_features(graph)
            box_count, hole_count = features[:, 0], features[:, 8]
            known = (box_count * features[:, 3] + hole_count * features[:, 11])
            confidence_sum = (box_count * features[:, 3] * features[:, 5]
                              + hole_count * features[:, 11] * features[:, 13])
            count = box_count + hole_count
            quality = torch.stack((count, confidence_sum / known.clamp_min(1),
                                   known / count.clamp_min(1)), dim=1)
            values = (model(graph.to(device))['object21'].cpu(), batch.targets['object21'],
                      graph.tcp_xyz_world, features, quality)
            identifiers.extend(batch.sample_ids)
            for name, value in zip(arrays, values):
                arrays[name].append(value.numpy())
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('Selected frames must be nonempty and unique')
    order = sorted(range(len(identifiers)), key=lambda i: sample_order(identifiers[i]))
    data = {name: np.concatenate(values)[order] for name, values in arrays.items()}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite frozen predictions, observations or targets')
    data['sample_ids'] = np.asarray(identifiers)[order]
    data['episode'] = np.asarray([sample_order(value)[0] for value in data['sample_ids']])
    data['frame'] = np.asarray([sample_order(value)[1] for value in data['sample_ids']])
    for episode in np.unique(data['episode']):
        frames = data['frame'][data['episode'] == episode]
        if np.any(np.diff(frames) != 1):
            raise ValueError(f'Temporal episodes require contiguous cached frames: {episode}')
    return data


def episode_tensors(data, episode, device):
    """Transfer one complete episode; labels remain outside the gate input tuple."""
    mask = data['episode'] == episode
    inputs = tuple(torch.as_tensor(data[name][mask], device=device, dtype=torch.float32)
                   for name in ('prediction', 'tcp_world', 'features'))
    target_world = torch.as_tensor(data['target'][mask, 10:13] + data['tcp_world'][mask],
                                   device=device, dtype=torch.float32)
    return inputs, target_world


def run_gate_epoch(gate, data, device, settings, optimizer=None, generator=None):
    """Full-episode autoregressive BPTT; shuffle episodes only, never their frames."""
    gate.train(optimizer is not None)
    episodes = np.unique(data['episode'])
    if optimizer is not None:
        episodes = generator.permutation(episodes)
    total_loss, total_frames = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for episode in episodes:
            inputs, target = episode_tensors(data, episode, device)
            memory, _ = rollout(gate, *inputs)
            if settings['loss'] == 'huber':
                loss = F.huber_loss(memory, target, delta=settings['huber_delta_m'])
            else:
                loss = F.l1_loss(memory, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite temporal loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), settings['gradient_clip_norm'],
                                               error_if_nonfinite=True)
                optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_frames += len(target)
    return total_loss / total_frames


def prepare(effective, device, requested):
    """Verify source/cache identity before any selected frozen inference."""
    frozen, source, expected_identity = load_frozen_source(effective, device)
    loaders, identity, manifest = selected_loaders(
        source, torch.Generator().manual_seed(effective['seed']), effective['split_manifest'], requested)
    if identity['dataset_sha256'] != expected_identity:
        raise ValueError('Frozen checkpoint and graph cache dataset identities differ')
    effective['source_training_config'] = source
    effective['source_checkpoint_sha256'] = file_sha256(Path(effective['checkpoint']))
    effective['split_manifest_sha256'] = file_sha256(Path(effective['split_manifest']))
    effective['gate_split_manifest'] = manifest
    effective['feature_names'] = list(QUALITY_NAMES)
    repository = Path(__file__).resolve().parents[2]
    source_files = ('graph/temporal_gate.py', 'graph/gated_mpnn.py', 'graph/utils/hole_memory.py',
                    'training/gnn/temporal_gate.py', 'evaluation/gnn/temporal_gate.py',
                    'data_collection/temporal_gate_data.py')
    effective['source_sha256'] = {name: file_sha256(repository / name) for name in source_files}
    return frozen, loaders, identity


def train(config):
    """Train only the new gate; select best on held-out GNN-train parents."""
    effective = resolve_gate_config(config)
    device = set_device(effective['device'])
    seed_everything(effective['seed'], device)
    frozen, loaders, identity = prepare(effective, device, ('train', 'validation'))
    with ExperimentLogger(team='graph', model='learned_temporal_gate', config=effective,
                          seed=effective['seed']) as experiment:
        output = Path(effective['output']['root']).expanduser().resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'effective_config.json', effective)
        experiment.link_output(effective['checkpoint'], checkpoint=True)
        data = {split: infer_observations(frozen, loader, device) for split, loader in loaders.items()}
        gate = LearnedTemporalGate(**effective['gate']).to(device)
        optimizer = torch.optim.Adam(gate.parameters(), lr=effective['training']['lr'])
        generator = np.random.default_rng(effective['seed'])
        best, history = float('inf'), []
        for epoch in range(1, effective['training']['epochs'] + 1):
            training_loss = run_gate_epoch(gate, data['train'], device, effective['training'], optimizer, generator)
            validation_loss = run_gate_epoch(gate, data['validation'], device, effective['training'])
            history.append(dict(epoch=epoch, train_loss=training_loss, validation_loss=validation_loss))
            experiment.log(history[-1], step=epoch)
            if validation_loss < best:
                best = validation_loss
                payload = {'kind': 'learned_temporal_gate', 'gate': gate.state_dict(),
                           'config': effective, 'dataset_sha256': identity['dataset_sha256'],
                           'epoch': epoch, 'validation_loss': best}
                torch.save(payload, output / 'best.pt')
                experiment.link_output(output / 'best.pt')
            print(f'Gate epoch {epoch}: train={training_loss:.8f}, val={validation_loss:.8f}', flush=True)
        write_json(output / 'history.json', history)
        for name in ('history.json', 'effective_config.json'):
            experiment.link_output(output / name)
        experiment.record(parameter_count=sum(p.numel() for p in gate.parameters()),
                          frozen_parameter_count=sum(p.numel() for p in frozen.parameters()),
                          dataset_sha256=identity['dataset_sha256'], device=str(device),
                          teacher_forcing=False, gnn_validation_used_for_training=False,
                          source_checkpoint_sha256=effective['source_checkpoint_sha256'])
        return output
