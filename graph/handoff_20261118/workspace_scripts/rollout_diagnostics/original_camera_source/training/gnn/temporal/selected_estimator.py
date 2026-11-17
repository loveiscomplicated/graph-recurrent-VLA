"""Train and evaluate a temporal estimator on frozen selected-frame M0 checkpoints."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.temporal.insertion import evaluate
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.pose_geometry import geometric_endpoint
from training.gnn.temporal.insertion import train


def train_and_evaluate(config):
    """Train one seed, evaluate complete episodes, then score the 960 saved frames."""
    method = config['model']['architecture']
    if method not in ('mamba', 'transformer', 'gru'):
        raise ValueError('Expected a supported temporal estimator architecture')
    original_hash = file_sha256(config['checkpoint'])
    training_output = train(config, continuation_checkpoint=config.get('continuation_checkpoint'))
    if file_sha256(config['checkpoint']) != original_hash:
        raise ValueError('Frozen M0 checkpoint changed during temporal training')
    return evaluate_existing(config, training_output)


def evaluate_existing(config, training_output):
    """Evaluate a completed training run without retraining its checkpoint."""
    method = config['model']['architecture']
    if method not in ('mamba', 'transformer', 'gru'):
        raise ValueError('Expected a supported temporal estimator architecture')
    training_output = Path(training_output)
    original_hash = file_sha256(config['checkpoint'])
    evaluation_config = deepcopy(config)
    evaluation_config['temporal_checkpoint'] = str(training_output / 'best.pt')
    evaluation_config['evaluation']['split'] = 'test'
    evaluation_config['output']['root'] = config['comparison_output']
    evaluation_output = evaluate(evaluation_config)

    records = [json.loads(line) for line in Path(config['evaluation_selection']).read_text().splitlines()]
    if len(records) != 960 or len({r['sample_id'] for r in records}) != 960:
        raise ValueError('Expected exactly 960 distinct original validation frames')
    with np.load(evaluation_output / 'predictions.npz', allow_pickle=False) as archive:
        lookup = {str(value): index for index, value in enumerate(archive['sample_ids'])}
        if not all(row['sample_id'] in lookup for row in records):
            raise ValueError('Full episode evaluation is missing original validation frames')
        indices = [lookup[row['sample_id']] for row in records]
        target = torch.from_numpy(archive['target'][indices].copy())
        results = {}
        for name, key in (('m0', 'gnn_object21'), (method, 'primary_object21')):
            prediction = torch.from_numpy(archive[key][indices].copy())
            errors = per_frame_errors({'object21': prediction,
                                       'endpoint_geom': geometric_endpoint(prediction),
                                       'endpoint_residual': torch.zeros_like(prediction[:, :3])}, target)
            # The helper also computes residual diagnostics, but those cannot be
            # reconstructed from the final Object21 output alone.
            errors = {key: value for key, value in errors.items()
                      if not key.startswith('endpoint_residual')}
            np.savez_compressed(evaluation_output / f'{name}_selected_errors.npz', **errors)
            results[name] = {}
            for group, mask in (('all', np.ones(len(records), dtype=bool)),
                                ('physical_both_pass', np.array([r['both_pass'] for r in records]))):
                results[name][group] = {key: statistics(value[mask]) for key, value in errors.items()}
    write_json(evaluation_output / 'original_validation_21d.json', {
        'training_output': str(training_output), 'm0_sha256': original_hash,
        'selection_sha256': file_sha256(config['evaluation_selection']),
        'counts': {'all': 960, 'physical_both_pass': sum(r['both_pass'] for r in records)},
        'metrics': results,
    })
    print(f'Original validation 21D metrics: {evaluation_output / "original_validation_21d.json"}', flush=True)
    return evaluation_output


def aggregate(output_directories, selection_path, output_path, method='mamba'):
    """Average three seed errors per frame before calculating distribution metrics."""
    records = [json.loads(line) for line in Path(selection_path).read_text().splitlines()]
    if len(records) != 960 or len(output_directories) != 3:
        raise ValueError('Expected 960 validation frames and three seed results')
    masks = {'all': np.ones(len(records), dtype=bool),
             'physical_both_pass': np.array([row['both_pass'] for row in records])}
    summary = {'counts': {name: int(mask.sum()) for name, mask in masks.items()}, 'metrics': {}}
    if method not in ('mamba', 'transformer', 'gru'):
        raise ValueError('Expected a supported temporal estimator architecture')
    for method in ('m0', method):
        arrays = []
        for directory in output_directories:
            with np.load(Path(directory) / f'{method}_selected_errors.npz', allow_pickle=False) as data:
                arrays.append({key: data[key].copy() for key in data.files})
        if any(set(item) != set(arrays[0]) for item in arrays):
            raise ValueError('Seed metric schemas differ')
        summary['metrics'][method] = {}
        for group, mask in masks.items():
            summary['metrics'][method][group] = {
                key: statistics(np.stack([item[key] for item in arrays]).mean(axis=0)[mask])
                for key in arrays[0]
            }
    write_json(output_path, summary)
