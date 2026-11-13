"""Check pretrained node-token joint fine-tuning on real CUDA graph windows."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.node_tokens import collate_node_windows
from training.gnn.temporal.node_features import infer_node_data
from evaluation.gnn.temporal.insertion import check_development_provenance
from training.gnn.temporal.config import resolve_insertion_config
from training.gnn.temporal.frozen_source import prepare_insertion_sequences
from training.gnn.temporal.insertion import configure_training_memory
from training.gnn.temporal.joint_continuation import load_joint_continuation, restore_random_state
from training.gnn.temporal.joint_finetuning import (
    build_joint_temporal, collate_windows, forward_joint_predictions, run_epoch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--continuation', action='store_true')
    parser.add_argument('--config', help='Override the seed-specific joint training recipe')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    prefix = 'continue_joint_node_token_mamba' if arguments.continuation else 'train_joint_node_token_mamba'
    config = Path(arguments.config) if arguments.config else Path(
        f'graph/configs/temporal/insertion/{prefix}_{arguments.seed}.yaml')
    effective = resolve_insertion_config(yaml.safe_load(config.read_text()))
    if effective['seed'] != arguments.seed:
        raise ValueError('Smoke --seed must match the selected config seed')
    device = torch.device('cuda:0')
    configure_training_memory(device, effective['training'])
    gnn, loaders, identity = prepare_insertion_sequences(effective, device, ('train',))
    previous = torch.load(effective['temporal_checkpoint'], map_location='cpu', weights_only=True)
    check_development_provenance(previous, effective, identity)
    print(f'Loaded source pair for seed {arguments.seed}: temporal epoch {previous["epoch"]}; '
          f'GNN SHA-256 {effective["source_checkpoint_sha256"]}', flush=True)
    temporal = build_joint_temporal(effective, previous['config']['node_width']).to(device)
    continuation = None
    metadata = None
    if arguments.continuation:
        effective['normalization'] = previous['config']['normalization']
        effective['node_width'] = previous['config']['node_width']
        effective['node_token_contract'] = previous['config']['node_token_contract']
        effective['joint_finetuning']['temporal_checkpoint_sha256'] = file_sha256(effective['temporal_checkpoint'])
        continuation, _, _, _, metadata = load_joint_continuation(
            effective['continuation_checkpoint'], effective, identity)
        gnn.load_state_dict(continuation['gnn'], strict=True)
        temporal.load_state_dict(continuation['temporal'], strict=True)
    else:
        temporal.load_state_dict(previous['model'], strict=True)
    loader = loaders['train']
    source = loader.dataset.dataset
    indices = loader.dataset.indices
    first_episode = str(Path(source.samples[indices[0]].sample_id).parent)
    positions = [index for index, row in enumerate(indices)
                 if str(Path(source.samples[row].sample_id).parent) == first_episode][:4]
    if len(positions) != 4:
        raise ValueError('Smoke requires four contiguous frames from one training episode')
    mini_loader = DataLoader(Subset(loader.dataset, positions), batch_size=4,
                             collate_fn=loader.collate_fn, shuffle=False)
    data = infer_node_data(gnn, mini_loader, device, identity, effective, 'train')
    episodes = InsertionEpisodeDataset(data)
    observations, target = episodes[0]
    samples = [source[indices[position]] for position in positions]
    batch = collate_windows([(observations, target, samples)])
    obs, _, mask, graphs = batch
    obs, mask = obs.to(device), mask.to(device)
    temporal.eval()
    with torch.no_grad():
        actual = forward_joint_predictions(gnn, temporal, graphs.graph.to(device), obs, mask)
        token_batch = collate_node_windows([(observations, target, data['node_features'])])
        nodes, node_mask = (value.to(device) for value in token_batch[2:4])
        expected = temporal.forward_nodes(obs, nodes, node_mask, mask)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    gnn.requires_grad_(True)
    rates = effective['joint_finetuning']
    optimizer = torch.optim.Adam([{'params': gnn.parameters(), 'lr': rates['gnn_lr']},
                                  {'params': temporal.parameters(), 'lr': rates['temporal_lr']}])
    if continuation is not None:
        optimizer.load_state_dict(continuation['optimizer'])
        restore_random_state(continuation.get('random_state'), torch.Generator(), device, effective['seed'])
    before_steps = {parameter: float(state['step']) for parameter, state in optimizer.state.items()
                    if 'step' in state}
    # Repeated real frames exercise the exact requested batch and window shapes.
    item = (observations.repeat(16, 1), target.repeat(16, 1), samples * 16)
    windows = [item] * 4
    before_gnn = {name: value.detach().clone() for name, value in gnn.named_parameters()}
    before_temporal = {name: value.detach().clone() for name, value in temporal.named_parameters()}
    settings = dict(effective['training'], loss=effective['loss'])
    trained = run_epoch(gnn, temporal, windows, device, settings, optimizer)
    validation = run_epoch(gnn, temporal, windows, device, settings)
    for model, before in ((gnn, before_gnn), (temporal, before_temporal)):
        if not any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters()):
            raise RuntimeError('Joint optimizer failed to update one of the models')
    if continuation is not None:
        for parameter, before_step in before_steps.items():
            if parameter.grad is not None and float(optimizer.state[parameter]['step']) != before_step + 1:
                raise RuntimeError('Restored Adam step counter did not advance by exactly one')
        print(f'Restored joint epoch {continuation["epoch"]}; optimizer states={len(before_steps)}; '
              f'RNG: {metadata["random_state"]}', flush=True)
    torch.cuda.synchronize()
    print(f'Joint CUDA smoke seed {arguments.seed}: batch=4 frames=64; '
          f'train={trained["loss"]:.6f}, validation={validation["loss"]:.6f}; both models updated', flush=True)


if __name__ == '__main__':
    main()
