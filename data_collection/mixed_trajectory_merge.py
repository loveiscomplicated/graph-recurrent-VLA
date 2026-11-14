"""Merge completed pilot partitions without deleting their original data."""

import json
import os
import shutil
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger


def merge(config):
    """Validate disjoint partitions, preserve overlaps, and assemble one pilot manifest."""
    root = Path(config['collection_root'])
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    original = json.loads((root / 'manifest.json').read_text())
    if original['status'] != 'ERROR':
        raise ValueError('Expected an interrupted sequential pilot manifest')
    if file_sha256(protocol_path) != original['protocol_sha256']:
        raise ValueError('Sequential pilot protocol checksum mismatch')
    pilot_seeds = [row['seed'] for row in protocol['records'] if row['split'] == 'pilot']
    cutoff = config['sequential_seed_count']
    kept_seeds = set(pilot_seeds[:cutoff])
    expected = {(seed, mode) for seed in pilot_seeds for mode in protocol['modes']}
    selected = [row for row in original['episodes'] if row['seed'] in kept_seeds]
    if len(selected) != cutoff * len(protocol['modes']):
        raise ValueError('Sequential pilot does not contain the planned complete prefix')
    additions = []
    for worker_path in config['worker_roots']:
        worker = Path(worker_path)
        worker_manifest = json.loads((worker / 'manifest.json').read_text())
        if worker_manifest['status'] != 'COMPLETE':
            raise ValueError(f'Worker is not complete: {worker}')
        if file_sha256(worker / 'protocol.json') != original['protocol_sha256']:
            raise ValueError(f'Worker protocol mismatch: {worker}')
        additions.extend((worker, row) for row in worker_manifest['episodes'])
    actual = {(row['seed'], row['mode']) for row in selected}
    actual.update((row['seed'], row['mode']) for _, row in additions)
    if actual != expected or len(selected) + len(additions) != len(expected):
        raise ValueError('Worker partitions do not exactly cover remaining pilot cases')
    sources = {}
    for name in ('collector_source.py', 'worker_collector_source.py'):
        source = root / name
        if not source.is_file():
            raise ValueError(f'Collector source snapshot is missing: {source}')
        sources[name] = file_sha256(source)
    archive = Path(config['overlap_archive'])
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / 'manifest.json', archive / 'sequential_manifest.json')
    for row in original['episodes']:
        if row['seed'] not in kept_seeds:
            shutil.move(root / row['path'], archive / row['path'])
    for partial in root.glob('*.partial'):
        shutil.move(partial, archive / partial.name)
    for worker, row in additions:
        destination = root / row['path']
        if destination.exists():
            raise ValueError(f'Unexpected existing episode directory: {destination}')
        # Hardlinks preserve the worker originals without doubling image storage.
        shutil.copytree(worker / row['path'], destination, copy_function=os.link)
        selected.append(row)
    order = {seed: index for index, seed in enumerate(pilot_seeds)}
    selected.sort(key=lambda row: (order[row['seed']], protocol['modes'].index(row['mode'])))
    merged = dict(original, status='COMPLETE', episodes=selected,
                  merged_from=config['worker_roots'], overlap_archive=str(archive),
                  collector_source_sha256=sources)
    write_json(root / 'manifest.json', merged)
    with ExperimentLogger(team='graph', model='mixed_trajectory_merge', config=config,
                          seed=protocol['generator'], job_type='collection') as logger:
        logger.record(protocol_sha256=original['protocol_sha256'], episodes=len(selected))
        logger.link_output(root / 'manifest.json')
    print(f'Merged pilot: {len(selected)} episodes in {root}', flush=True)
