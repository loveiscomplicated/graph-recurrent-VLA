"""Validate and restore paired joint checkpoints, optimizer and random state."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.temporal.insertion import check_development_provenance


def capture_random_state(generator, device):
    """Save all RNGs used by the trainer in a weights-only-loadable structure."""
    numpy_state = np.random.get_state()
    return {
        'generator': generator.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_random_state(state, generator, device, seed):
    """Restore after data preparation and baseline validation have consumed RNGs."""
    if state is None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        generator.manual_seed(seed)
        return
    generator.set_state(state['generator'].cpu())
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise ValueError('CUDA joint continuation requires a saved CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))


def validate_joint_checkpoint(payload, effective, identity):
    """Reject changes to paired models, data, objective or optimizer protocol."""
    if payload.get('kind') != 'joint_gnn_temporal':
        raise ValueError('continuation_checkpoint must contain paired joint weights')
    required = {'gnn', 'temporal', 'optimizer', 'epoch', 'validation_loss'}
    if not required <= payload.keys():
        raise ValueError(f'Joint continuation is missing fields: {sorted(required - payload.keys())}')
    check_development_provenance(payload, effective, identity)
    saved = payload['config']
    for name in ('seed', 'mode', 'model', 'with_robot', 'feature_names', 'observation_contract',
                 'loss', 'normalization', 'node_width', 'node_token_contract', 'split_manifest_sha256'):
        if saved.get(name) != effective.get(name):
            raise ValueError(f'Joint continuation differs in {name}')
    for name in ('sequence_length', 'episode_batch_size', 'context_policy', 'gradient_clip_norm'):
        if saved['training'][name] != effective['training'][name]:
            raise ValueError(f'Joint continuation differs in training.{name}')
    for name in ('gnn_lr', 'temporal_lr', 'temporal_checkpoint_sha256'):
        if saved['joint_finetuning'][name] != effective['joint_finetuning'][name]:
            raise ValueError(f'Joint continuation differs in joint_finetuning.{name}')
    previous_stopping = saved['training'].get('early_stopping')
    if previous_stopping is not None and previous_stopping != effective['training'].get('early_stopping'):
        raise ValueError('Joint continuation must retain its existing early-stopping settings')


def load_joint_continuation(path, effective, identity):
    """Load current and best paired states, retaining only history up to restart."""
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    validate_joint_checkpoint(payload, effective, identity)
    start = payload['epoch']
    if type(start) is not int or start < 0 or start >= effective['training']['epochs']:
        raise ValueError('training.epochs must exceed the cumulative continuation epoch')
    if path.name == 'best.pt':
        best = payload
    elif 'best_checkpoint' in payload:
        best = payload['best_checkpoint']
    else:
        best_path = path.with_name('best.pt')
        if not best_path.is_file():
            raise FileNotFoundError('Legacy joint last.pt continuation requires the adjacent best.pt')
        best = torch.load(best_path, map_location='cpu', weights_only=True)
    validate_joint_checkpoint(best, effective, identity)
    if best['epoch'] > start or best['validation_loss'] > payload['validation_loss'] + 1e-6:
        raise ValueError('Continuation best checkpoint is inconsistent with its current state')
    history = payload.get('history')
    history_path = path.with_name('history.json')
    if history is None and history_path.is_file():
        history = json.loads(history_path.read_text())
    history = [row for row in (history or []) if row['epoch'] <= start]
    if history:
        epochs = [row['epoch'] for row in history]
        if epochs != sorted(set(epochs)) or epochs[-1] != start:
            raise ValueError('Continuation history must end at its checkpoint epoch without duplicates')
    state = deepcopy(payload.get('early_stopping_state'))
    if effective['training'].get('early_stopping') and state is None:
        state = {'reference_loss': best['validation_loss'], 'stale_epochs': start - best['epoch']}
    metadata = {
        'checkpoint': str(path), 'checkpoint_sha256': file_sha256(path),
        'starting_epoch': start, 'optimizer': 'restored paired Adam state',
        'random_state': ('restored' if payload.get('random_state') is not None else
                         'reseeded: legacy joint checkpoint did not save RNG state'),
        'history_rows_retained': len(history),
    }
    return payload, best, history, state, metadata
