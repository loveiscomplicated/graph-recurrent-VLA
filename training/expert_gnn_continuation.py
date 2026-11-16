"""Continue a fixed-Hole Expert GNN with its original frozen Transformer."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash
from data_collection.expert_corner_curriculum import checked_path, load_expert
from evaluation.expert_corner_curriculum import rollout_tensors
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.detected_corner_gnn import run_epoch
from training.detected_corner_gnn_fresh import save_atomic
from utils.experiment_logger import ExperimentLogger, write


COMPARISON_METRICS = ('loss', 'peg_rotation_deg', 'hole_rotation_deg', 'endpoint_mm')


def read_source(config):
    """Verify the original paired checkpoints and the completed GNN history."""
    source = config['source']
    gnn_path = checked_path(source['gnn_checkpoint'], source['gnn_sha256'])
    tracker_path = checked_path(source['tracker_checkpoint'], source['tracker_sha256'])
    rollout_path = checked_path(source['rollout_receipt'], source['rollout_sha256'])
    gnn = torch.load(gnn_path, map_location='cpu', weights_only=False)
    tracker = torch.load(tracker_path, map_location='cpu', weights_only=False)
    start = source['gnn_epoch']
    epochs = [row['epoch'] for row in gnn['history']]
    if (gnn['format'] != 'fixed_hole_expert_gnn_v1' or gnn['epoch'] != start
            or epochs != list(range(1, start + 1)) or config['total_epochs'] <= start
            or not gnn['optimizer']['state']):
        raise ValueError('Expected a completed GNN checkpoint with optimizer state and a larger epoch budget')
    if (tracker['format'] != 'fixed_id_corner_predictor_v1' or tracker['epoch'] != source['tracker_epoch']
            or gnn['temporal_sha256'] != source['tracker_sha256']
            or gnn['dataset_sha256'] != tracker['dataset_sha256']
            or gnn['config']['tracker']['model'] != tracker['config']['model']):
        raise ValueError('The GNN must continue with the exact tracker and dataset recorded in the checkpoint')
    best = min(gnn['history'], key=lambda row: row['validation']['loss'])
    if best['epoch'] != start:
        raise ValueError('The configured continuation source must also be the validation-selected checkpoint')
    return gnn, tracker, json.loads(rollout_path.read_text())


def restore_optimizer(checkpoint, settings, device):
    """Restore model weights, optimizer moments and the next shuffle position."""
    model = VertexEdgeGatedMPNN(**settings['model']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    training = settings['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    shuffle = np.random.default_rng()
    shuffle.bit_generator.state = checkpoint['shuffle_state']
    if 'torch_rng' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return model, optimizer, shuffle


def validation_record(row):
    """Keep physical units and epoch numbers alongside the validation loss."""
    return {'epoch': row['epoch'], **{name: row['validation'][name] for name in COMPARISON_METRICS}}


def train(config):
    """Resume GNN updates and compare physical errors on the same validation frames."""
    source = config['source']
    original, tracker, expected_rollout = read_source(config)
    original_config = original['config']
    device = torch.device(original_config['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The original fixed-Hole Expert run requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(original_config['tracker']['training']['cpu_threads'])
    # The original GNN followed tracker training in the same process.
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = original_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    settings = original_config['gnn']
    start = original['epoch']
    history = list(original['history'])
    best = history[-1]
    effective = dict(config, inherited_data=original_config['data'], inherited_gnn=settings,
                     inherited_association_gate_m=original_config['association_gate_m'], tf32=True)
    with ExperimentLogger(team='graph', model='fixed_hole_expert_gnn_continuation', config=effective,
                          seed=seed, job_type='training') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        print(f'GNN continuation output: {output}', flush=True)
        write(output / 'effective_config.yaml', effective)
        logger.link_output(source['tracker_checkpoint'], checkpoint=True)
        logger.link_output(source['gnn_checkpoint'], checkpoint=True)
        logger.record(resumed_epoch=start, dataset_sha256=original['dataset_sha256'],
                      temporal_sha256=source['tracker_sha256'], source_gnn_sha256=source['gnn_sha256'])
        episodes = load_expert(original_config['data'])
        settings_tracker = tracker['config']['model']
        temporal = FixedIdCornerPredictor(**settings_tracker).to(device).eval().requires_grad_(False)
        temporal.load_state_dict(tracker['model'], strict=True)
        arrays, rollout = rollout_tensors(temporal, episodes, original_config['association_gate_m'], device)
        del temporal
        if rollout != expected_rollout:
            raise ValueError('Frozen tracker rollouts changed from the source run')
        write(output / 'rollout.json', rollout)

        model, optimizer, shuffle = restore_optimizer(original, settings, device)
        train_indices = torch.nonzero(~arrays['split'], as_tuple=True)[0]
        validation_indices = torch.nonzero(arrays['split'], as_tuple=True)[0]
        sample_ids = [sample for episode in episodes if episode['split'] == 'validation'
                      for sample in episode['sample_ids'].tolist()]
        source_metrics = history[-1]['validation']
        replay = run_epoch(model, arrays, validation_indices, settings, None, start, logger, output)
        if len(sample_ids) != source_metrics['frames'] or replay['frames'] != source_metrics['frames']:
            raise ValueError('Validation frame count differs from the prior run')
        np.testing.assert_allclose([replay[name] for name in source_metrics], list(source_metrics.values()),
                                   rtol=1e-4, atol=1e-5, err_msg='Epoch-20 validation replay differs')
        validation_hash = canonical_hash(sample_ids)
        write(output / 'validation_replay.json', dict(epoch=start, source=source_metrics, replay=replay,
                                                     frame_ids_sha256=validation_hash))
        logger.record(validation_frames=len(sample_ids), validation_frame_ids_sha256=validation_hash)
        print(f'Verified epoch {start} on the same {len(sample_ids)} validation frames', flush=True)
        shutil.copyfile(source['gnn_checkpoint'], output / 'best.pt')
        write(output / 'history.json', history)

        for epoch in range(start + 1, config['total_epochs'] + 1):
            order = shuffle.permutation(len(train_indices))
            indices = train_indices[torch.as_tensor(order, device=device)]
            training = run_epoch(model, arrays, indices, settings, optimizer, epoch, logger, output)
            validation = run_epoch(model, arrays, validation_indices, settings, None, epoch, logger, output)
            row = dict(epoch=epoch, train=training, validation=validation,
                       batch_order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest())
            history.append(row)
            state = dict(original, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         epoch=epoch, run_id=logger.run_id, history=history,
                         shuffle_state=shuffle.bit_generator.state, torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all(), continuation=effective)
            save_atomic(output / 'last.pt', state)
            if validation['loss'] < best['validation']['loss']:
                best = row
                save_atomic(output / 'best.pt', state)
            write(output / 'history.json', history)
            print(f'Epoch {epoch}/{config["total_epochs"]} train={training["loss"]:.6f} '
                  f'validation={validation["loss"]:.6f} peg_rot_deg={validation["peg_rotation_deg"]:.3f} '
                  f'hole_rot_deg={validation["hole_rotation_deg"]:.3f} endpoint_mm={validation["endpoint_mm"]:.3f}',
                  flush=True)

        comparison = dict(status='COMPLETE', validation_frames=len(sample_ids),
                          validation_frame_ids_sha256=validation_hash,
                          starting_checkpoint=validation_record(history[start - 1]),
                          final_checkpoint=validation_record(history[-1]),
                          best_by_validation_loss=validation_record(best),
                          validation_curve=[validation_record(row) for row in history[start - 1:]])
        write(output / 'comparison.json', comparison)
        for filename in ('last.pt', 'best.pt', 'history.json', 'validation_replay.json', 'comparison.json'):
            logger.link_output(output / filename)
        print(f'GNN continuation complete: {output}', flush=True)
        return str(output)
