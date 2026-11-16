"""Train a fresh Expert GNN on rollouts from the validation-selected Transformer."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import yaml

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.expert_corner_curriculum import checked_path, load_expert
from evaluation.expert_corner_curriculum import rollout_tensors
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.detected_corner_gnn import run_epoch
from training.detected_corner_gnn_fresh import save_atomic
from training.fixed_id_corners import update_early_stopping
from utils.experiment_logger import ExperimentLogger, write


def load_sources(config):
    """Load the prior recipe and only the newly selected Transformer weights."""
    source = config['source']
    recipe_path = checked_path(source['recipe'], source['recipe_sha256'])
    recipe = yaml.safe_load(recipe_path.read_text())
    tracker_path = checked_path(source['tracker_checkpoint'], source['tracker_sha256'])
    tracker = torch.load(tracker_path, map_location='cpu', weights_only=False)
    best = min(tracker['history'], key=lambda row: row['validation']['loss'])
    if (tracker['format'] != 'fixed_id_corner_predictor_v1'
            or tracker['epoch'] != source['tracker_epoch'] or best['epoch'] != tracker['epoch']
            or tracker['dataset_sha256'] != source['tracker_dataset_sha256']
            or tracker['config']['model'] != recipe['tracker']['model']
            or tracker['config']['model']['architecture'] != 'transformer'):
        raise ValueError('Expected the validation-selected Transformer from the original Expert split')
    return recipe, tracker


def cache_rollouts(config, tracker, recipe, device, output, logger):
    """Regenerate detector-only inputs and save them with the ordered frame IDs."""
    episodes = load_expert(recipe['data'])
    sample_ids = [sample for episode in episodes for sample in episode['sample_ids'].tolist()]
    validation_ids = [sample for episode in episodes if episode['split'] == 'validation'
                      for sample in episode['sample_ids'].tolist()]
    validation_hash = canonical_hash(validation_ids)
    if (len(validation_ids) != config['validation']['frames']
            or validation_hash != config['validation']['frame_ids_sha256']):
        raise ValueError('Ordered validation frame IDs differ from the original Expert GNN split')

    temporal = FixedIdCornerPredictor(**tracker['config']['model']).to(device).eval().requires_grad_(False)
    temporal.load_state_dict(tracker['model'], strict=True)
    arrays, counts = rollout_tensors(temporal, episodes, recipe['association_gate_m'], device)
    del temporal
    frame_counts = dict(train_frames=len(sample_ids) - len(validation_ids), validation_frames=len(validation_ids))
    if len(arrays['split']) != len(sample_ids) or int(arrays['split'].sum().item()) != len(validation_ids):
        raise ValueError('Rollout tensors and frame IDs are misaligned')
    identity = dict(temporal_sha256=config['source']['tracker_sha256'], temporal_epoch=tracker['epoch'],
                    dataset_sha256=tracker['dataset_sha256'], split_sha256=recipe['data']['split_manifest_sha256'],
                    frame_ids_sha256=canonical_hash(sample_ids), validation_frame_ids_sha256=validation_hash,
                    association_gate_m=recipe['association_gate_m'], **frame_counts)
    path = output / 'rollout_inputs.pt'
    save_atomic(path, dict(arrays={name: value.cpu() for name, value in arrays.items()},
                           sample_ids=sample_ids, identity=identity))
    write(output / 'rollout.json', counts)
    logger.link_output(path, kind='rollouts')
    logger.link_output(output / 'rollout.json')
    identity['rollout_cache_sha256'] = file_sha256(path)
    logger.record(**identity, temporal_frozen=True, gt_ids_used_in_tracker=False)
    print(f"Saved {frame_counts['train_frames']} train and {frame_counts['validation_frames']} validation rollout frames", flush=True)
    return arrays, identity


def fit_gnn(config, settings, arrays, identity, output, logger):
    """Initialize GNN and AdamW, then stop when validation loss plateaus."""
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shuffle = np.random.default_rng(seed)
    device = arrays['xyz'].device
    model = VertexEdgeGatedMPNN(**settings['model']).to(device)
    initial_hash = sha256(b''.join(value.detach().cpu().numpy().tobytes()
                                   for value in model.state_dict().values())).hexdigest()
    training = settings['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    train_indices = torch.nonzero(~arrays['split'], as_tuple=True)[0]
    validation_indices = torch.nonzero(arrays['split'], as_tuple=True)[0]
    logger.record(gnn_fresh_initialization=True, gnn_initial_sha256=initial_hash)

    history, best_loss, best_epoch = [], float('inf'), 0
    stop_reason = 'maximum_epochs'
    patience = config['early_stopping_patience']
    for epoch in range(1, config['maximum_epochs'] + 1):
        order = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
        train_metrics = run_epoch(model, arrays, train_indices[order], settings, optimizer, epoch, logger, output)
        validation = run_epoch(model, arrays, validation_indices, settings, None, epoch, logger, output)
        best_loss, best_epoch, stale, improved, should_stop = update_early_stopping(
            validation['loss'], epoch, best_loss, best_epoch, patience,
        )
        history.append(dict(epoch=epoch, train=train_metrics, validation=validation, best_epoch=best_epoch,
                            epochs_without_improvement=stale))
        state = dict(format='fixed_hole_expert_gnn_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                     config=config, epoch=epoch, history=history, run_id=logger.run_id,
                     dataset_sha256=identity['dataset_sha256'], temporal_sha256=identity['temporal_sha256'],
                     initial_model_sha256=initial_hash, identity=identity, shuffle_state=shuffle.bit_generator.state,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        save_atomic(output / 'last.pt', state)
        if improved:
            save_atomic(output / 'best.pt', state)
        write(output / 'history.json', history)
        print(f"Epoch {epoch}/{config['maximum_epochs']} train={train_metrics['loss']:.6f} "
              f"validation={validation['loss']:.6f} peg_rotation={validation['peg_rotation_deg']:.3f} deg "
              f"hole_rotation={validation['hole_rotation_deg']:.3f} deg endpoint={validation['endpoint_mm']:.3f} mm "
              f"best={best_epoch} no_improvement={stale}/{patience}", flush=True)
        if should_stop:
            stop_reason = 'early_stopping'
            break
    return dict(epochs=len(history), best_epoch=best_epoch, best_validation_loss=best_loss, stop_reason=stop_reason,
                best_validation=history[best_epoch - 1]['validation'], final_validation=history[-1]['validation'])


def train(config):
    """Build one new rollout cache and fit only the downstream GNN."""
    for name in ('maximum_epochs', 'early_stopping_patience'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    recipe, tracker = load_sources(config)
    device = torch.device(recipe['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The fixed-Hole Expert GNN run requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(recipe['tracker']['training']['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = True
    gnn = dict(recipe['gnn'], training=dict(recipe['gnn']['training'], epochs=config['maximum_epochs']))
    effective = dict(config, seed=recipe['seed'], data=recipe['data'], tracker=recipe['tracker'],
                     gnn=gnn, device=str(device), association_gate_m=recipe['association_gate_m'], tf32=True)
    with ExperimentLogger(team='graph', model='fixed_hole_expert_gnn_early_stopping', config=effective,
                          seed=effective['seed'], job_type='training') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True)
        print(f'GNN early stopping output: {output}', flush=True)
        write(output / 'effective_config.yaml', effective)
        logger.link_output(config['source']['tracker_checkpoint'], checkpoint=True)
        arrays, identity = cache_rollouts(config, tracker, recipe, device, output, logger)
        summary = fit_gnn(effective, gnn, arrays, identity, output, logger)
        write(output / 'completed.json', summary)
        for name in ('best.pt', 'last.pt', 'history.json', 'completed.json', 'effective_config.yaml'):
            logger.link_output(output / name)
        logger.record(**summary)
        print(f"Completed GNN at epoch {summary['epochs']}, best={summary['best_epoch']}, reason={summary['stop_reason']}", flush=True)
        return str(output)
