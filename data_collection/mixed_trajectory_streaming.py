"""Stage Moving-Hole episodes in RAM and stream verified archives to disk."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from data_collection.collection_batches import collection_defaults
from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_moving_hole import collect as collect_moving_hole
from data_collection.storage import write_json


def archive_batch(source, target, seed_range):
    """Encode one complete batch and remove its staging files after verification."""
    with (source / 'motion_audit.json').open() as stream:
        audit = json.load(stream)
    expected = 4 * (seed_range[1] - seed_range[0])
    if audit['episodes'] != expected or audit['full_translation'] != expected:
        raise ValueError(f'Incomplete Moving-Hole batch: {source}')
    pending = target.with_suffix('.partial')
    print(f'Compressing {target.name}', flush=True)
    with pending.open('wb') as output:
        with subprocess.Popen(['tar', '-C', str(source), '-cf', '-', '.'],
                              stdout=subprocess.PIPE) as tar:
            try:
                subprocess.run(['zstd', '-q', '-3', '-T1'], stdin=tar.stdout, stdout=output, check=True)
            finally:
                tar.stdout.close()
            if tar.wait():
                raise RuntimeError(f'Could not archive {source}')
        output.flush()
        os.fsync(output.fileno())
    subprocess.run(['zstd', '-qt', str(pending)], check=True)
    receipt = dict(file=target.name, sha256=file_sha256(pending), bytes=pending.stat().st_size,
                   record_range=list(seed_range), episodes=expected)
    pending.rename(target)
    sync_directory(target.parent)
    shutil.rmtree(source)
    print(f'Saved {target.name}', flush=True)
    return receipt


def sync_directory(path):
    """Persist directory entries before reporting completion."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    # Linux usually provides tmpfs here. A caller can also supply a mounted
    # RAM disk or a regular temporary directory on another platform.
    default_ram_root = '/dev/shm' if Path('/dev/shm').is_dir() else tempfile.gettempdir()
    config['ram_root'] = str(Path(config.get('ram_root', default_ram_root)).resolve())
    target = Path(config['output_root']).resolve()
    target.mkdir(parents=True, exist_ok=False)
    records = []
    scratch = Path(tempfile.mkdtemp(dir=config['ram_root'], prefix='moving_hole_'))
    print(f'RAM staging directory: {scratch}', flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None
        for begin in range(start, stop, batch_size):
            end = min(begin + batch_size, stop)
            name = f'episodes_{begin:05d}_{end:05d}'
            source = scratch / name
            print(f'Collecting {name}', flush=True)
            collect_moving_hole(dict(config, output_root=str(source), record_range=[begin, end],
                                     compress_frames=False))
            if pending is not None:
                records.append(pending.result())
            pending = executor.submit(archive_batch, source, target / f'{name}.tar.zst', (begin, end))
        records.append(pending.result())
    scratch.rmdir()
    write_json(target / 'manifest.json', dict(status='COMPLETE', effective_config=config,
                                              episodes=sum(row['episodes'] for row in records),
                                              archives=records))
    sync_directory(target)
