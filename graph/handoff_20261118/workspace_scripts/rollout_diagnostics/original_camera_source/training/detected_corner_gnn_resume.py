"""Continue detector-only GNN training with optimizer state and early stopping."""

from copy import deepcopy
import os
from pathlib import Path
import random
import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn_fresh import checkpoint_state, quiet_epoch, save_atomic
from utils.experiment_logger import ExperimentLogger, write


def validate_source(config, checkpoint, history, summary):
    """Reject a mismatched or incomplete source before starting a continuation."""
    current = deepcopy(config)
    previous = deepcopy(checkpoint['config'])
    for value in (current, previous):
        for key in ('trainer', 'output', 'resume'):
            value.pop(key, None)
        value['training'].pop('epochs')
    if current != previous:
        raise ValueError('Continuation settings differ from the source training run')
    if checkpoint['epoch'] != 40 or len(history) != 40 or summary['epochs'] != 40:
        raise ValueError('Source must be the completed 40-epoch last checkpoint')
    if summary['best_validation_loss'] != min(item['validation']['loss'] for item in history):
        raise ValueError('Source best loss does not match its training history')
    for name in ('optimizer', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng'):
        if name not in checkpoint:
            raise ValueError(f'Source checkpoint lacks {name}')


def train(config):
    """Restore the 40th epoch and retain the best model across both runs."""
    resume = config['resume']
    source_path = Path(resume['checkpoint'])
    if file_sha256(source_path) != resume['checkpoint_sha256']:
        raise ValueError('Resume checkpoint checksum mismatch')
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    history = json.loads((source_path.parent / 'history.json').read_text())
    summary = json.loads((source_path.parent / 'completed.json').read_text())
    validate_source(config, source, history, summary)
    best_path = source_path.parent / 'best.pt'
    if file_sha256(best_path) != resume['best_sha256']:
        raise ValueError('Source best checkpoint checksum mismatch')
    prior_best = torch.load(best_path, map_location='cpu', weights_only=False)
    if prior_best['epoch'] != summary['best_epoch']:
        raise ValueError('Source best checkpoint does not match its summary')
    cache_path = source_path.parent / 'rollout_inputs.pt'
    if file_sha256(cache_path) != source['rollout_cache_sha256']:
        raise ValueError('Rollout cache checksum mismatch')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if cache['identity'] != source['identity']:
        raise ValueError('Rollout cache identity mismatch')
    if file_sha256(config['temporal']['checkpoint']) != source['temporal_sha256']:
        raise ValueError('Frozen temporal checkpoint checksum mismatch')
    patience = resume['patience']
    if patience < 1 or config['training']['epochs'] <= source['epoch']:
        raise ValueError('Patience must be positive and max epochs must exceed 40')
    torch.set_num_threads(config['training']['cpu_threads'])
    device = torch.device(config['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This matched training recipe requires CUDA')
    arrays = {key: torch.as_tensor(value, device=device) for key, value in cache['arrays'].items()
              if key != 'sample_ids'}
    train_indices = torch.nonzero(~arrays['split'], as_tuple=True)[0]
    validation_indices = torch.nonzero(arrays['split'], as_tuple=True)[0]
    model = VertexEdgeGatedMPNN(**config['gnn']).to(device)
    model.load_state_dict(source['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])
    optimizer.load_state_dict(source['optimizer'])
    shuffle = np.random.default_rng()
    shuffle.bit_generator.state = source['numpy_rng']
    best = summary['best_validation_loss']
    best_epoch = summary['best_epoch']
    stale_epochs = 0
    with ExperimentLogger(team='graph', model='detected_corner_gnn_resume', config=config,
                          seed=config['seed']) as logger:
        output = Path(config['output']['root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        os.link(cache_path, output / 'rollout_inputs.pt')
        prior_best['config'] = config
        prior_best['run_id'] = logger.run_id
        save_atomic(output / 'best.pt', prior_best)
        logger.link_output(source_path, checkpoint=True)
        logger.link_output(best_path, checkpoint=True)
        logger.link_output(output / 'rollout_inputs.pt')
        logger.record(source_run_id=source['run_id'], source_checkpoint_sha256=resume['checkpoint_sha256'],
                      temporal_frozen=True, gt_ids_used_in_tracker=False,
                      optimizer_restored=True, patience=patience)
        # Restore after initialization and logging so the next epoch continues the saved RNG stream.
        torch.set_rng_state(source['torch_rng'])
        torch.cuda.set_rng_state_all(source['cuda_rng'])
        random.setstate(source['python_rng'])
        print(f'Resumed epoch 40; best_epoch={best_epoch} best_loss={best:.6f} patience={patience}', flush=True)
        for epoch in range(source['epoch'] + 1, config['training']['epochs'] + 1):
            order = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
            train_metrics = quiet_epoch(model, arrays, train_indices[order], config, optimizer, epoch, logger, output)
            validation = quiet_epoch(model, arrays, validation_indices, config, None, epoch, logger, output)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation))
            state = checkpoint_state(model, optimizer, config, source, logger.run_id, epoch, shuffle)
            if validation['loss'] < best:
                best, best_epoch, stale_epochs = validation['loss'], epoch, 0
                save_atomic(output / 'best.pt', state)
            else:
                stale_epochs += 1
            state['early_stopping'] = dict(best_loss=best, best_epoch=best_epoch, stale_epochs=stale_epochs,
                                           patience=patience)
            save_atomic(output / 'last.pt', state)
            write(output / 'history.json', history)
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={validation['loss']:.6f} "
                  f"best_epoch={best_epoch} patience={stale_epochs}/{patience}", flush=True)
            if stale_epochs >= patience:
                break
        result = dict(epochs=epoch, best_epoch=best_epoch, best_validation_loss=best,
                      validation=history[best_epoch - 1]['validation'], optimizer_saved_each_epoch=True,
                      source_run_id=source['run_id'], early_stopped=stale_epochs >= patience,
                      patience=patience, resumed_from_epoch=source['epoch'])
        write(output / 'completed.json', result)
        for name in ('best.pt', 'last.pt', 'history.json', 'completed.json', 'progress.json'):
            if (output / name).exists():
                logger.link_output(output / name)
        logger.record(**result)
        print(f'Completed: {output}; {result}', flush=True)
        return str(output)
