"""Stage Moving-Hole episodes in RAM and stream verified archives to disk."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from data_collection.collection_batches import collection_defaults
from data_collection.episode_dataset import file_sha256
from data_collection.episode_staging import stage_batches, staging_root, sync_directory
from data_collection.mixed_trajectory_moving_hole import collect as collect_moving_hole
from data_collection.storage import write_json


def archive_batch(source, target, seed_range):
    """Encode one complete batch and remove its staging files after verification."""
    with (source / 'motion_audit.json').open() as stream:
        audit = json.load(stream)
    expected = 4 * (seed_range[1] - seed_range[0])
    if audit['episodes'] != expected or audit['full_translation'] != expected:
        raise ValueError(f'Incomplete Moving-Hole batch: {source}')
    return archive_verified_batch(source, target, seed_range, expected)


def archive_verified_batch(source, target, seed_range, episodes, *, long_distance=False):
    """Persist a verified collection batch as a checksummed Zstandard archive."""
    pending = target.with_suffix('.partial')
    print(f'Compressing {target.name}', flush=True)
    encoder = ['zstd', '-q', '-3', '-T1']
    if long_distance:
        encoder.append('--long=27')
    with pending.open('wb') as output:
        with subprocess.Popen(['tar', '-C', str(source), '-cf', '-', '.'],
                              stdout=subprocess.PIPE) as tar:
            try:
                subprocess.run(encoder, stdin=tar.stdout, stdout=output, check=True)
            finally:
                tar.stdout.close()
            if tar.wait():
                raise RuntimeError(f'Could not archive {source}')
        output.flush()
        os.fsync(output.fileno())
    subprocess.run(['zstd', '-qt', str(pending)], check=True)
    receipt = dict(file=target.name, sha256=file_sha256(pending), bytes=pending.stat().st_size,
                   record_range=list(seed_range), episodes=episodes)
    pending.rename(target)
    sync_directory(target.parent)
    print(f'Saved {target.name}', flush=True)
    return receipt


def collect(config):
    """Keep at most one completed batch pending while the next batch collects."""
    config = collection_defaults(config)
    if config.get('physics_device', 'cpu') != 'cpu':
        raise ValueError('RAM streaming currently uses the CPU physics collector')
    start, stop = config['record_range']
    batch_size = config.get('batch_seed_count')
    if batch_size is None:
        episodes_per_seed = 4
        if config['batch_episodes'] % episodes_per_seed:
            raise ValueError('Moving-Hole batch_episodes must be a multiple of four')
        batch_size = config['batch_episodes'] // episodes_per_seed
    config['batch_seed_count'] = batch_size
    if type(batch_size) is not int or batch_size < 1 or not 0 <= start < stop:
        raise ValueError('Expected a positive batch_seed_count and a nonempty record_range')
    for program in ('tar', 'zstd'):
        if shutil.which(program) is None:
            raise RuntimeError(f'Install {program} to stream Moving-Hole archives')
    config['ram_root'] = str(staging_root(config))
    target = Path(config['output_root']).resolve()
    target.mkdir(parents=True, exist_ok=False)
    batches = [(begin, min(begin + batch_size, stop)) for begin in range(start, stop, batch_size)]

    def collect_batch(seed_range, staging):
        begin, end = seed_range
        name = f'episodes_{begin:05d}_{end:05d}'
        print(f'Collecting {name}', flush=True)
        collect_moving_hole(dict(config, output_root=str(staging / name), record_range=list(seed_range),
                                 compress_frames=False))
        return name, seed_range

    def persist_batch(staging, result):
        name, seed_range = result
        return archive_batch(staging / name, target / f'{name}.tar.zst', seed_range)

    records = list(stage_batches(batches, collect_batch, persist_batch, config))
    write_json(target / 'manifest.json', dict(status='COMPLETE', effective_config=config,
                                              episodes=sum(row['episodes'] for row in records),
                                              archives=records))
    sync_directory(target)
