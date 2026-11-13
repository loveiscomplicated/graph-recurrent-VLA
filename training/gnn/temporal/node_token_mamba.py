"""Train and score frame-node-token Mamba on the selected frozen M0 data."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset, InsertionWindowDataset
from data_collection.temporal.node_tokens import NodeWindows, collate_node_windows
from data_collection.temporal.pose_sequences import fit_observation_normalization
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.pose_geometry import geometric_endpoint
from graph.temporal.node_token_mamba import NodeTokenMamba
from training.gnn.temporal.config import resolve_insertion_config
from training.gnn.temporal.frozen_source import prepare_insertion_sequences
from training.gnn.temporal.insertion import configure_training_memory, update_early_stopping
from training.gnn.temporal.node_features import infer_node_data
from training.gnn.temporal.pose_loss import insertion_loss
from training.gnn.trainer import atomic_torch_save, seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def run_epoch(model, windows, effective, device, optimizer=None, generator=None):
    """Run masked windows with the original insertion objective and optimizer."""
    loader = DataLoader(windows, batch_size=effective['training']['episode_batch_size'],
                        shuffle=optimizer is not None, generator=generator,
                        collate_fn=collate_node_windows)
    model.train(optimizer is not None)
    total, frames = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            observations, target, nodes, node_mask, frame_mask = (value.to(device) for value in batch)
            prediction = model.forward_nodes(observations, nodes, node_mask, frame_mask)
            loss, _ = insertion_loss(prediction, target, frame_mask, model.mode, effective['loss'])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), effective['training']['gradient_clip_norm'],
                                              error_if_nonfinite=True)
                optimizer.step()
            count = int(frame_mask.sum())
            total += float(loss.detach()) * count
            frames += count
    if not frames:
        raise ValueError('No training or validation frames')
    return total / frames


@torch.inference_mode()
def evaluate_selected(model, dataset, selection_path, output, device):
    """Replay complete episodes causally, then score the original 960 frame IDs."""
    records = [json.loads(line) for line in Path(selection_path).read_text().splitlines()]
    if len(records) != 960 or len({row['sample_id'] for row in records}) != 960:
        raise ValueError('Expected exactly 960 distinct original validation frames')
    model.eval()
    predictions = {}
    targets = {}
    for index, rows in enumerate(dataset.indices):
        observation, target = dataset[index]
        nodes = [dataset.data['node_features'][row] for row in rows]
        batch = collate_node_windows([(observation, target, nodes)])
        obs, _, node_tensor, node_mask, frame_mask = (value.to(device) for value in batch)
        result = model.forward_nodes(obs, node_tensor, node_mask, frame_mask)[0].cpu()
        for local, row in enumerate(rows):
            sample_id = str(dataset.data['sample_ids'][row])
            predictions[sample_id] = result[local]
            targets[sample_id] = target[local]
    if any(row['sample_id'] not in predictions for row in records):
        raise ValueError('Full episode replay is missing an original validation frame')
    prediction = torch.stack([predictions[row['sample_id']] for row in records])
    target = torch.stack([targets[row['sample_id']] for row in records])
    errors = per_frame_errors({'object21': prediction, 'endpoint_geom': geometric_endpoint(prediction),
                               'endpoint_residual': torch.zeros_like(prediction[:, :3])}, target)
    errors = {key: value for key, value in errors.items() if not key.startswith('endpoint_residual')}
    np.savez_compressed(output / 'selected_errors.npz', **errors)
    np.savez_compressed(output / 'selected_predictions.npz', sample_ids=np.array([r['sample_id'] for r in records]),
                        prediction=prediction.numpy(), target=target.numpy())
    masks = {'all': np.ones(len(records), dtype=bool),
             'physical_both_pass': np.array([row['both_pass'] for row in records], dtype=bool)}
    write_json(output / 'original_validation_21d.json', {
        'selection_sha256': file_sha256(selection_path),
        'counts': {name: int(mask.sum()) for name, mask in masks.items()},
        'metrics': {name: {key: statistics(value[mask]) for key, value in errors.items()}
                    for name, mask in masks.items()},
    })


def should_stop_early(epoch, settings, state):
    """Stop after the minimum epoch and configured consecutive stale epochs."""
    return (settings is not None and epoch >= settings['min_epochs']
            and state['stale_epochs'] >= settings['patience'])


def train_and_evaluate(config):
    """Train a fresh Mamba and evaluate its best development checkpoint."""
    effective = resolve_insertion_config(config)
    device = set_device(effective['device'])
    configure_training_memory(device, effective['training'])
    seed_everything(effective['seed'], device)
    frozen, loaders, identity = prepare_insertion_sequences(effective, device, ('train', 'validation'))
    effective['node_token_contract'] = 'pre_pool_nodes_type_then_world_xyz_frame_token_v1'
    effective['node_token_order_ties'] = 'original detector order'
    repository = Path(__file__).resolve().parents[3]
    for name in ('graph/temporal/node_token_mamba.py', 'data_collection/temporal/node_tokens.py',
                 'training/gnn/temporal/node_token_mamba.py'):
        effective['source_sha256'][name] = file_sha256(repository / name)
    effective['selection_sha256'] = file_sha256(effective['evaluation_selection'])
    with ExperimentLogger(team='graph', model='node_token_mamba', config=effective,
                          seed=effective['seed']) as experiment:
        output = Path(effective['output']['root']) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(effective['checkpoint'], checkpoint=True)
        episodes = {}
        for split, loader in loaders.items():
            print(f'Capturing frozen node tokens and robot observations: {split}', flush=True)
            data = infer_node_data(frozen, loader, device, identity, effective, split)
            episodes[split] = InsertionEpisodeDataset(data)
            print(f'{split}: {len(episodes[split])} episodes; {len(data["frame"])} frames', flush=True)
        mean, std = fit_observation_normalization(episodes['train'])
        effective['normalization'] = {'source_split': 'train', 'mean': mean.tolist(), 'std': std.tolist()}
        widths = {node.shape[1] for dataset in episodes.values() for node in dataset.data['node_features']}
        if len(widths) != 1:
            raise ValueError('Inconsistent frozen GNN node feature widths')
        node_width = widths.pop()
        effective['node_width'] = node_width
        model = NodeTokenMamba(node_width=node_width, mode=effective['mode'],
                               with_robot=effective['with_robot'], **effective['model']).to(device)
        model.set_normalization(mean.to(device), std.to(device))
        windows = {split: NodeWindows(InsertionWindowDataset(dataset, effective['training']['sequence_length']))
                   for split, dataset in episodes.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=effective['training']['lr'])
        generator = torch.Generator().manual_seed(effective['seed'])
        write_json(output / 'effective_config.json', effective)
        experiment.record(dataset_sha256=identity['dataset_sha256'], node_width=node_width,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          test_frames_read=False, context_policy='reset_windows')
        best = float('inf')
        history = []
        early_settings = effective['training'].get('early_stopping')
        early_state = {'reference_loss': float('inf'), 'stale_epochs': 0}
        early_stopped = False
        for epoch in range(1, effective['training']['epochs'] + 1):
            trained = run_epoch(model, windows['train'], effective, device, optimizer, generator)
            validation = run_epoch(model, windows['validation'], effective, device)
            history.append({'epoch': epoch, 'train_loss': trained, 'validation_loss': validation})
            experiment.log(history[-1], step=epoch)
            if early_settings is not None:
                early_state = update_early_stopping(early_state, validation, early_settings)
            if validation < best:
                best = validation
                atomic_torch_save(output / 'best.pt', {'kind': 'node_token_mamba', 'model': model.state_dict(),
                    'config': effective, 'dataset_sha256': identity['dataset_sha256'],
                    'epoch': epoch, 'validation_loss': best})
                experiment.link_output(output / 'best.pt')
            atomic_torch_save(output / 'last.pt', {'kind': 'node_token_mamba_state',
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'config': effective, 'dataset_sha256': identity['dataset_sha256'],
                'early_stopping_state': early_state if early_settings else None})
            write_json(output / 'history.json', history)
            print(f'epoch {epoch}: train={trained:.6f} validation={validation:.6f}', flush=True)
            if should_stop_early(epoch, early_settings, early_state):
                early_stopped = True
                print(f'Early stopping at epoch {epoch}; best validation loss {best:.6f}', flush=True)
                break
        del episodes, windows, loaders
        selected = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(selected['model'], strict=True)
        frozen, test_loaders, test_identity = prepare_insertion_sequences(effective, device, ('test',))
        if test_identity['dataset_sha256'] != identity['dataset_sha256']:
            raise ValueError('Evaluation graph cache differs from training')
        data = infer_node_data(frozen, test_loaders['test'], device, test_identity, effective, 'test')
        evaluate_selected(model, InsertionEpisodeDataset(data), effective['evaluation_selection'], output, device)
        for name in ('last.pt', 'history.json', 'effective_config.json', 'selected_errors.npz',
                     'selected_predictions.npz', 'original_validation_21d.json'):
            experiment.link_output(output / name)
        experiment.record(completed_epoch=epoch, selected_epoch=selected['epoch'],
                          early_stopped=early_stopped, test_frames_read=True)
        return output


def aggregate(config):
    """Combine matched seed errors per frame, as in the baseline leaderboard."""
    roots = [Path(path) for path in config['seed_roots']]
    selections = []
    errors = []
    for root in roots:
        paths = list(root.glob('*/selected_errors.npz'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one completed seed run in {root}, found {len(paths)}')
        with np.load(paths[0], allow_pickle=False) as data:
            errors.append({key: data[key].copy() for key in data.files})
        with np.load(paths[0].parent / 'selected_predictions.npz', allow_pickle=False) as data:
            selections.append(data['sample_ids'].copy())
    if len(errors) != 3 or any(not np.array_equal(selections[0], value) for value in selections[1:]):
        raise ValueError('Three seed runs must use exactly the same selected-frame order')
    records = [json.loads(line) for line in Path(config['evaluation_selection']).read_text().splitlines()]
    if list(selections[0]) != [row['sample_id'] for row in records]:
        raise ValueError('Selected frame IDs differ from the evaluation manifest')
    if any(set(item) != set(errors[0]) for item in errors):
        raise ValueError('Seed error schemas differ')
    masks = {'all': np.ones(len(records), dtype=bool),
             'physical_both_pass': np.array([row['both_pass'] for row in records], dtype=bool)}
    summary = {'counts': {name: int(mask.sum()) for name, mask in masks.items()},
               'metrics': {name: {key: statistics(np.stack([item[key] for item in errors]).mean(0)[mask])
                                 for key in errors[0]} for name, mask in masks.items()}}
    write_json(Path(config['output_path']), summary)
