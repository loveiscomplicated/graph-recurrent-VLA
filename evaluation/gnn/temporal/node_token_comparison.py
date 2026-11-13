"""Compare completed node-token and frozen-GNN temporal experiments frame by frame."""

import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import statistics
from utils.experiment_logger import ExperimentLogger


METRICS = (
    'peg_position_error_mm', 'peg_rotation_error_deg', 'relative_3d_error_mm',
    'hole_position_error_mm', 'hole_rotation_error_deg',
    'peg_half_size_x_error_mm', 'peg_half_size_y_error_mm',
    'peg_half_size_z_error_mm', 'hole_radius_error_mm',
)


def one_run(root, filename):
    """Find a unique completed artifact for one seed, rejecting ambiguous runs."""
    paths = list(Path(root).glob(f'*/{filename}'))
    if len(paths) != 1:
        raise ValueError(f'Expected one {filename} under {root}, found {len(paths)}')
    return paths[0]


def load_node_run(root, selected_ids):
    """Load selected-frame errors and world-frame targets from a node-token run."""
    errors_path = one_run(root, 'selected_errors.npz')
    with np.load(errors_path, allow_pickle=False) as archive:
        errors = {name: archive[name].copy() for name in METRICS}
    with np.load(errors_path.parent / 'selected_predictions.npz', allow_pickle=False) as archive:
        if list(archive['sample_ids']) != selected_ids:
            raise ValueError(f'Selected frame order differs: {root}')
        targets = archive['target'].copy()
    config = json.loads((errors_path.parent / 'effective_config.json').read_text())
    return errors, targets, config


def load_baseline_run(root, selected_ids):
    """Join baseline full-episode predictions to its saved selected-frame errors."""
    errors_path = one_run(root, 'mamba_selected_errors.npz')
    with np.load(errors_path, allow_pickle=False) as archive:
        errors = {name: archive[name].copy() for name in METRICS}
    with np.load(errors_path.parent / 'predictions.npz', allow_pickle=False) as archive:
        ids = list(archive['sample_ids'])
        if len(ids) != len(set(ids)):
            raise ValueError('Baseline full-episode sample IDs must be unique')
        lookup = {value: index for index, value in enumerate(ids)}
        if any(value not in lookup for value in selected_ids):
            raise ValueError('Baseline is missing selected frames')
        indices = [lookup[value] for value in selected_ids]
        targets = archive['target'][indices].copy()
        tcp = archive['tcp_world'][indices]
        targets[:, :3] += tcp
        targets[:, 10:13] += tcp
    config = json.loads((errors_path.parent / 'effective_config.json').read_text())
    return errors, targets, config


def compare(config):
    """Report matched 960/151-frame error distributions and paired differences."""
    selection = Path(config['evaluation_selection'])
    rows = [json.loads(line) for line in selection.read_text().splitlines()]
    ids = [row['sample_id'] for row in rows]
    if len(ids) != 960 or len(set(ids)) != 960:
        raise ValueError('Expected exactly 960 unique selected-frame IDs')
    methods = config['methods']
    expected = {'node_early', 'mamba_early', 'node_120'}
    if set(methods) != expected or any(len(value) != 3 for value in methods.values()):
        raise ValueError('Expected three seed roots for node_early, mamba_early and node_120')
    results = {name: [] for name in methods}
    references = None
    for seed in range(3):
        configs = {}
        for name, roots in methods.items():
            loader = load_baseline_run if name == 'mamba_early' else load_node_run
            errors, target, run_config = loader(roots[seed], ids)
            if references is None:
                references = target
            np.testing.assert_allclose(target, references, rtol=1e-6, atol=1e-7)
            if run_config.get('seed') != seed:
                raise ValueError(f'{name} seed does not match its expected index: {seed}')
            configs[name] = run_config
            results[name].append(errors)
        if configs['node_early']['source_checkpoint_sha256'] != configs['mamba_early']['source_checkpoint_sha256']:
            raise ValueError(f'Early-stopped GNN checkpoint differs for seed {seed}')
    masks = {'all': np.ones(960, dtype=bool),
             'physical_both_pass': np.array([row['both_pass'] for row in rows], dtype=bool)}
    if int(masks['physical_both_pass'].sum()) != 151:
        raise ValueError('Expected 151 physical-condition selected frames')
    averaged = {name: {key: np.stack([seed[key] for seed in runs]).mean(axis=0)
                       for key in METRICS} for name, runs in results.items()}
    groups = {}
    for group, mask in masks.items():
        groups[group] = {
            'count': int(mask.sum()),
            'methods': {name: {key: statistics(values[key][mask]) for key in METRICS}
                        for name, values in averaged.items()},
            'paired_node_early_minus_mamba_early': {
                key: statistics((averaged['node_early'][key] - averaged['mamba_early'][key])[mask])
                for key in METRICS},
            'paired_node_early_minus_node_120': {
                key: statistics((averaged['node_early'][key] - averaged['node_120'][key])[mask])
                for key in METRICS},
        }
    with ExperimentLogger(team='graph', model='node_token_comparison', config=config) as experiment:
        output = Path(config['output_root']) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'comparison.json', {
            'selection_sha256': file_sha256(selection), 'seed_count': 3,
            'interpretation': ('node_early versus mamba_early shares the frozen GNN; '
                               'node_120 uses the 60-epoch GNN and a 120-epoch budget.'),
            'groups': groups,
        })
        experiment.link_output(output / 'comparison.json')
        return output
