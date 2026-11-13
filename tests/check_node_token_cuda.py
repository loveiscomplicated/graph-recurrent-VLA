"""Real-cache CUDA smoke for the node-token training path; no optimizer step."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.node_tokens import collate_node_windows
from training.gnn.temporal.node_features import infer_node_data
from graph.temporal.node_token_mamba import NodeTokenMamba
from training.gnn.temporal.config import resolve_insertion_config
from training.gnn.temporal.frozen_source import prepare_insertion_sequences
from training.gnn.temporal.pose_loss import insertion_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='graph/configs/temporal/insertion/train_node_token_mamba_0.yaml')
    config_path = Path(parser.parse_args().config)
    effective = resolve_insertion_config(yaml.safe_load(config_path.read_text()))
    device = torch.device('cuda:0')
    frozen, loaders, identity = prepare_insertion_sequences(effective, device, ('train',))
    loader = loaders['train']
    source = loader.dataset.dataset
    source_indices = loader.dataset.indices
    first_episode = str(Path(source.samples[source_indices[0]].sample_id).parent)
    indices = [i for i, source_index in enumerate(source_indices)
               if str(Path(source.samples[source_index].sample_id).parent) == first_episode][:4]
    if len(indices) != 4:
        raise ValueError('CUDA smoke needs four contiguous source frames')
    sample_loader = DataLoader(Subset(loader.dataset, indices), batch_size=4,
                               shuffle=False, collate_fn=loader.collate_fn)
    data = infer_node_data(frozen, sample_loader, device, identity, effective, 'train')
    dataset = InsertionEpisodeDataset(data)
    if len(dataset) != 1:
        raise ValueError('CUDA smoke frames must belong to one episode')
    observations, target = dataset[0]
    batch = collate_node_windows([(observations, target, data['node_features'])])
    observations, target, nodes, node_mask, frame_mask = (value.to(device) for value in batch)
    torch.cuda.set_per_process_memory_fraction(effective['training']['cuda_memory_fraction'], device)
    model = NodeTokenMamba(node_width=nodes.shape[-1], mode=effective['mode'],
                           with_robot=effective['with_robot'], **effective['model']).to(device)
    prediction = model.forward_nodes(observations, nodes, node_mask, frame_mask)
    loss, _ = insertion_loss(prediction, target, frame_mask, model.mode, effective['loss'])
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(model.node_projection[0].weight.grad).all():
        raise ValueError('CUDA smoke loss and node gradients must be finite')
    if not model.node_projection[0].weight.grad.abs().sum().item():
        raise ValueError('CUDA smoke node projection received no gradient')
    # Exercise the actual four-window, 64-frame training shape using real features.
    model.zero_grad(set_to_none=True)
    long_observations = observations.repeat(4, 16, 1)
    long_target = target.repeat(4, 16, 1)
    long_nodes = nodes.repeat(4, 16, 1, 1)
    long_node_mask = node_mask.repeat(4, 16, 1)
    long_frame_mask = frame_mask.repeat(4, 16)
    output = model.forward_nodes(long_observations, long_nodes, long_node_mask, long_frame_mask)
    long_loss, _ = insertion_loss(output, long_target, long_frame_mask, model.mode, effective['loss'])
    long_loss.backward()
    torch.optim.Adam(model.parameters(), lr=effective['training']['lr']).step()
    torch.cuda.synchronize()
    print(f'CUDA real-data smoke passed: batch=4, frames=64, '
          f'nodes={int(long_node_mask.sum())}, loss={long_loss.item():.6f}', flush=True)


if __name__ == '__main__':
    main()
