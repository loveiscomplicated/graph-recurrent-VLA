"""Freeze test seeds and model identities before any test observations are read."""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger


def freeze(config):
    """Draw deterministic seeds outside known training and prior collection inventories."""
    training_path = Path(config['training_exclusions'])
    additional_path = Path(config['additional_exclusions'])
    training = json.loads(training_path.read_text())
    additional = json.loads(additional_path.read_text())
    original = set(training['original_collection_seeds'])
    graph = set(training['graph_development_seeds'])
    if len(original) != 900 or len(graph) != 630 or not graph <= original:
        raise ValueError('Expected Graph development seeds to be a subset of the original 900')
    other = {item['seed'] for item in additional['episodes']}
    if len(other) != additional['episode_count']:
        raise ValueError('Additional source inventory contains duplicate or missing episode seeds')
    excluded = original | other
    for left, right in config['prior_explored_seed_ranges']:
        excluded.update(range(left, right+1))
    start, stop = config['seed_range']
    if not (0 <= start < stop and 0 < config['count'] <= stop-start):
        raise ValueError('seed_range and count must define a valid positive sample')
    rng = np.random.default_rng(config['seed_generator'])
    candidates = np.arange(start, stop, dtype=np.int64)
    pool = np.asarray([value for value in candidates if int(value) not in excluded])
    if len(pool) < config['count']:
        raise ValueError('Not enough seeds remain after exclusions')
    selected = sorted(map(int, rng.choice(pool, size=config['count'], replace=False)))
    models = []
    for item in config['models']:
        checkpoint = torch.load(item['gnn'], map_location='cpu', weights_only=False)
        if checkpoint['config']['seed'] != item['seed']:
            raise ValueError('GNN checkpoint seed mismatch')
        if checkpoint['config']['data']['dataset_sha256'] != training['graph_dataset_sha256']:
            raise ValueError('Graph exclusion inventory does not match GNN training dataset')
        temporal_hash = file_sha256(item['temporal'])
        if temporal_hash != checkpoint['config']['temporal']['sha256']:
            raise ValueError('Temporal checkpoint does not match the GNN provenance')
        models.append(dict(**item, gnn_sha256=file_sha256(item['gnn']), temporal_sha256=temporal_hash))
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ('torch', 'mani_skill', 'sapien', 'gymnasium', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = dict(schema_version=1, role='independent_test', seeds=selected,
                    exclusions=dict(training_inventory_sha256=file_sha256(training_path),
                                    additional_inventory_sha256=file_sha256(additional_path),
                                    original_count=len(original), graph_count=len(graph), additional_count=len(other),
                                    prior_explored_seed_ranges=config['prior_explored_seed_ranges']),
                    generation=dict(seed=config['seed_generator'], seed_range=config['seed_range'], count=config['count']),
                    policy=config['policy'], models=models, versions=versions,
                    limitation='Seed disjointness is checked against the named inventories; unseen external datasets are not certified.')
    path = root / 'protocol.json'
    write_json(path, protocol)
    with ExperimentLogger(team='graph', model='independent_test_protocol', config=config,
                          seed=config['seed_generator'], job_type='collection') as logger:
        logger.record(protocol_sha256=file_sha256(path), test_seeds=len(selected), known_overlap=0)
        logger.link_output(path)
    print(f'Frozen protocol: {path} sha256={file_sha256(path)}', flush=True)
