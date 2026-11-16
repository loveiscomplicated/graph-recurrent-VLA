"""Train the first fixed-Hole Expert stage from newly initialized weights."""

import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.expert_corner_curriculum import load_expert, write_sequences
from evaluation.expert_corner_curriculum import evaluate_baseline, rollout_tensors
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.detected_corner_gnn import run_epoch
from training.detected_corner_gnn_fresh import save_atomic
from training.fixed_id_corners import train as fit_tracker
from utils.experiment_logger import ExperimentLogger, write


def fit_frame_gnn(config, episodes, checkpoint, dataset_hash, device, output):
    """Freeze this run's tracker, then fit a new GNN on its online rollouts."""
    checkpoint_sha = file_sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    tracker_settings = saved['config']
    if (saved['format'] != 'fixed_id_corner_predictor_v1' or saved['dataset_sha256'] != dataset_hash
            or tracker_settings['model'] != config['tracker']['model']
            or tracker_settings.get('resume', {}).get('checkpoint') is not None):
        raise ValueError('GNN input tracker must come from this scratch Expert run')
    temporal = FixedIdCornerPredictor(**tracker_settings['model']).to(device).eval().requires_grad_(False)
    temporal.load_state_dict(saved['model'], strict=True)
    arrays, rollout = rollout_tensors(temporal, episodes, config['association_gate_m'], device)
    del temporal

    output.mkdir()
    seed = config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shuffle = np.random.default_rng(seed)
    settings = config['gnn']
    gnn = VertexEdgeGatedMPNN(**settings['model']).to(device)
    initial = hashlib.sha256()
    for value in gnn.state_dict().values():
        initial.update(value.detach().cpu().numpy().tobytes())
    training = settings['training']
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=training['learning_rate'],
                                  weight_decay=training['weight_decay'])
    train_indices = torch.nonzero(~arrays['split'], as_tuple=True)[0]
    validation_indices = torch.nonzero(arrays['split'], as_tuple=True)[0]
    history, best_loss, best_epoch = [], float('inf'), 0
    with ExperimentLogger(team='graph', model='fixed_hole_expert_gnn_scratch', config=config, seed=seed) as logger:
        logger.link_output(checkpoint, checkpoint=True)
        logger.record(dataset_sha256=dataset_hash, temporal_sha256=checkpoint_sha,
                      gnn_initial_sha256=initial.hexdigest(), gt_ids_used_in_tracker=False,
                      train_frames=len(train_indices), validation_frames=len(validation_indices))
        write(output / 'rollout.json', rollout)
        for epoch in range(1, training['epochs'] + 1):
            permutation = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
            train_metrics = run_epoch(gnn, arrays, train_indices[permutation], settings, optimizer, epoch, logger, output)
            validation = run_epoch(gnn, arrays, validation_indices, settings, None, epoch, logger, output)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation))
            if validation['loss'] < best_loss:
                best_loss, best_epoch = validation['loss'], epoch
            state = dict(format='fixed_hole_expert_gnn_v1', model=gnn.state_dict(), optimizer=optimizer.state_dict(),
                         run_id=logger.run_id, epoch=epoch, config=config, dataset_sha256=dataset_hash,
                         temporal_sha256=checkpoint_sha, initial_model_sha256=initial.hexdigest(),
                         shuffle_state=shuffle.bit_generator.state, history=history)
            save_atomic(output / 'last.pt', state)
            if best_epoch == epoch:
                save_atomic(output / 'best.pt', state)
            write(output / 'history.json', history)
            print(f'GNN epoch {epoch}/{training["epochs"]} train={train_metrics["loss"]:.6f} '
                  f'validation={validation["loss"]:.6f} best_epoch={best_epoch}', flush=True)
        for name in ('last.pt', 'best.pt', 'history.json', 'rollout.json'):
            logger.link_output(output / name)
        return dict(best_epoch=best_epoch, validation=history[best_epoch - 1]['validation'],
                    checkpoint=str(output / 'best.pt'), temporal_sha256=checkpoint_sha)


def train(config):
    """Run the frozen Expert split through baseline, scratch tracker and scratch GNN."""
    device = torch.device(config['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The fixed-Hole Expert recipe requires CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(config['tracker']['training']['cpu_threads'])
    with ExperimentLogger(team='graph', model='fixed_hole_expert_curriculum', config=config,
                          seed=config['seed'], job_type='training') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        print(f'Expert curriculum output: {output}', flush=True)
        write(output / 'effective_config.yaml', config)
        episodes = load_expert(config['data'])
        lineage = {key: config['data'][key] for key in
                   ('cache_manifest_sha256', 'split_manifest_sha256', 'robot_archive_sha256')}
        sequence_root = output / 'tracker_sequences'
        dataset_hash = write_sequences(episodes, sequence_root, config['offline_match_tolerance_m'], lineage)
        logger.record(dataset_sha256=dataset_hash, **lineage, train_episodes=80, validation_episodes=10,
                      policy_mode='expert', initialization='scratch')
        write(output / 'stage.json', {'phase': 'historical_baseline'})
        validation = [episode for episode in episodes if episode['split'] == 'validation']
        baseline = evaluate_baseline(config, validation, device, output / 'baseline')

        write(output / 'stage.json', {'phase': 'training_transformer_from_scratch'})
        tracker_config = dict(config['tracker'], seed=config['seed'], resume={'checkpoint': None},
                              data=dict(root=str(sequence_root), expected_dataset_sha256=dataset_hash),
                              output=dict(root=str(output / 'tracker')))
        tracker_output = Path(fit_tracker(tracker_config))
        write(output / 'stage.json', {'phase': 'training_gnn_from_scratch'})
        gnn = fit_frame_gnn(config, episodes, tracker_output / 'best.pt', dataset_hash, device, output / 'gnn')
        if baseline['metrics']['frames'] != gnn['validation']['frames']:
            raise ValueError('Baseline and scratch validation frames differ')
        report = dict(status='COMPLETE', baseline=baseline['metrics'], scratch=gnn,
                      scratch_tracker_checkpoint=str(tracker_output / 'best.pt'), **lineage)
        write(output / 'comparison.json', report)
        write(output / 'stage.json', {'phase': 'complete'})
        logger.link_output(output / 'comparison.json', kind='metrics')
        return str(output)
