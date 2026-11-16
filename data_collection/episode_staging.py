"""Bounded staging and asynchronous persistence of native episode files."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import tempfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json


def staging_root(config):
    """Resolve the shared RAM mount or the platform temporary directory."""
    default = '/dev/shm' if Path('/dev/shm').is_dir() else tempfile.gettempdir()
    return Path(config.get('ram_root', default)).resolve()


def stage_batches(batches, collect_batch, persist_batch, config):
    """Collect the next batch with at most one completed batch being persisted."""
    root = Path(tempfile.mkdtemp(dir=staging_root(config), prefix='episode_staging_'))
    print(f'Episode staging directory: {root}', flush=True)
    if not config.get('overlap_persistence', True):
        # Keep only one batch on staging mounts that cannot fit two raw batches.
        for index, batch in enumerate(batches):
            staging = root / f'batch_{index:06d}'
            staging.mkdir()
            result = collect_batch(batch, staging)
            yield finish_batch(staging, result, persist_batch)
        root.rmdir()
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None
        for index, batch in enumerate(batches):
            staging = root / f'batch_{index:06d}'
            staging.mkdir()
            result = collect_batch(batch, staging)
            if pending is not None:
                yield pending.result()
            pending = executor.submit(finish_batch, staging, result, persist_batch)
        if pending is not None:
            yield pending.result()
    root.rmdir()


def finish_batch(staging, result, persist_batch):
    """Keep the staging files available when persistence fails."""
    saved = persist_batch(staging, result)
    shutil.rmtree(staging)
    return saved


def copy_frame(source, target):
    """Compress stored NPY entries without decoding arrays or changing their data."""
    with ZipFile(source) as raw:
        entries = raw.infolist()
        if any(entry.compress_type == ZIP_STORED for entry in entries):
            with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=1) as compressed:
                for entry in entries:
                    with raw.open(entry) as contents, compressed.open(entry.filename, 'w', force_zip64=True) as output:
                        shutil.copyfileobj(contents, output)
            with ZipFile(target) as saved:
                expected = [(entry.filename, entry.CRC, entry.file_size) for entry in entries]
                observed = [(entry.filename, entry.CRC, entry.file_size) for entry in saved.infolist()]
                if observed != expected or saved.testzip() is not None:
                    raise OSError(f'NPZ verification failed: {target}')
        else:
            shutil.copyfile(source, target)
    sync_file(target)


def persist_episodes(staging, episodes, *, output):
    """Publish compressed NPZs and updated receipts in the original directory layout."""
    for row in episodes:
        if 'file' in row:
            path = output / row['file']
            pending = path.with_suffix('.partial')
            copy_frame(staging / row['file'], pending)
            row['sha256'] = file_sha256(pending)
            pending.rename(path)
            continue
        source = staging / row['path']
        path = output / row['path']
        pending = path.with_suffix('.partial')
        pending.mkdir()
        hashes = []
        for frame in sorted(source.glob('frame_*.npz')):
            target = pending / frame.name
            copy_frame(frame, target)
            if 'frame_sha256' in row:
                hashes.append(file_sha256(target))
        if 'frame_sha256' in row:
            row['frame_sha256'] = hashes
        for item in source.iterdir():
            if item.suffix != '.npz' and item.name != 'episode.json':
                target = pending / item.name
                shutil.copy2(item, target)
                sync_file(target)
        write_json(pending / 'episode.json', row)
        sync_directory(pending)
        pending.rename(path)
    sync_directory(output)
    return episodes


def sync_file(path):
    """Flush a complete file before publishing its final name."""
    with path.open('rb') as stream:
        os.fsync(stream.fileno())


def sync_directory(path):
    """Persist the directory entries before exposing the batch receipt."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def persist_tree(source, output):
    """Publish a staged shard or diagnostic tree as compressed NPZs."""
    pending = output.with_name(output.name + '.partial')

    def copy(source_name, target_name):
        source_path, target_path = Path(source_name), Path(target_name)
        if source_path.suffix == '.npz':
            copy_frame(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)
            sync_file(target_path)

    shutil.copytree(source, pending, copy_function=copy)
    for directory, _, _ in os.walk(pending, topdown=False):
        sync_directory(directory)
    pending.rename(output)
    sync_directory(output.parent)
