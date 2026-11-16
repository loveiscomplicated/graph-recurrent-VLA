"""Run detector-only tracking and the historical Expert validation baseline."""

import numpy as np
import torch

from data_collection.expert_corner_curriculum import checked_path
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn import rollout_episode, run_epoch
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger, write


def load_baseline(settings, device):
    """Load the paired leaderboard weights strictly for evaluation."""
    gnn_path = checked_path(settings['gnn_checkpoint'], settings['gnn_sha256'])
    saved = torch.load(gnn_path, map_location='cpu', weights_only=False)
    config = saved['config']
    if config['temporal']['sha256'] != settings['temporal_sha256']:
        raise ValueError('Historical GNN and Transformer checkpoints must be a matched pair')
    config = dict(config, temporal=dict(config['temporal'], checkpoint=settings['temporal_checkpoint']))
    temporal, _ = load_temporal(config, device)
    gnn = VertexEdgeGatedMPNN(**config['gnn']).to(device).eval().requires_grad_(False)
    gnn.load_state_dict(saved['model'], strict=True)
    return temporal, gnn


def rollout_tensors(temporal, episodes, gate_m, device):
    """Pack actual detector-only tracked graphs for a frozen GNN pass."""
    frames, counts = [], []
    for episode in episodes:
        tracked, details = rollout_episode(temporal, episode, gate_m, device)
        frames.extend(tracked)
        counts.append(dict(episode=episode['episode'], split=episode['split'], **details))
        if len(counts) % 10 == 0 or len(counts) == len(episodes):
            print(f'Detector-only rollout {len(counts)}/{len(episodes)} Expert episodes', flush=True)
    maximum = max(1, max(len(frame['xyz']) for frame in frames))
    positions = np.zeros((len(frames), maximum, 3), dtype=np.float32)
    kinds = np.full((len(frames), maximum), -1, dtype=np.int64)
    valid = np.zeros((len(frames), maximum), dtype=bool)
    observed = np.zeros_like(valid)
    for index, frame in enumerate(frames):
        length = len(frame['xyz'])
        positions[index, :length] = frame['xyz']
        kinds[index, :length] = frame['kinds']
        valid[index, :length] = True
        observed[index, :length] = frame['observed']
    arrays = dict(xyz=positions, kinds=kinds, valid=valid, observed=observed,
                  tcp=np.concatenate([episode['robot'][:, :3] for episode in episodes]),
                  target=np.concatenate([episode['target'] for episode in episodes]),
                  split=np.concatenate([np.full(len(episode['robot']), episode['split'] == 'validation')
                                        for episode in episodes]))
    return {name: torch.as_tensor(value, device=device) for name, value in arrays.items()}, counts


def evaluate_baseline(config, validation, device, output):
    """Score the original pair on the same Expert validation frames as scratch training."""
    output.mkdir()
    settings = config['baseline']
    temporal, gnn = load_baseline(settings, device)
    with ExperimentLogger(team='graph', model='fixed_hole_expert_baseline', config=config,
                          seed=config['seed'], job_type='evaluate') as logger:
        logger.link_output(settings['temporal_checkpoint'], checkpoint=True)
        logger.link_output(settings['gnn_checkpoint'], checkpoint=True)
        arrays, counts = rollout_tensors(temporal, validation, config['association_gate_m'], device)
        metrics = run_epoch(gnn, arrays, torch.arange(len(arrays['target']), device=device),
                            config['gnn'], None, 0, logger, output)
        result = dict(temporal_sha256=settings['temporal_sha256'], gnn_sha256=settings['gnn_sha256'],
                      metrics=metrics, tracking=counts)
        write(output / 'metrics.json', result)
        logger.link_output(output / 'metrics.json', kind='metrics')
        return result
