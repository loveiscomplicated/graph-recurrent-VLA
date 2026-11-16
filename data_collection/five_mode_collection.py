"""Collect five paired RGB-D policy modes on held-out reset seeds."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, flock
from functools import partial
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from data_collection import mixed_trajectory_collection as pilot
from data_collection import mixed_trajectory_recovery as recovery
from data_collection.collection_batches import collection_defaults, iter_batches
from data_collection.episode_dataset import file_sha256
from data_collection.episode_staging import stage_batches, staging_root, sync_directory
from data_collection.mixed_trajectory_streaming import archive_verified_batch
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger

FIXED_MODES = ('expert', 'expert_random', recovery.MODE, 'random')
MOVING_MODE = 'Moving_Hole_expert'
POLICY_MODES = (*(f'Fixed_Hole_{mode}' for mode in FIXED_MODES), MOVING_MODE)
BASE_EPISODES_PER_SEED = len(pilot.MODES) + 1
DATASET_SPLITS = ('train', 'validation', 'test')


def resolve_config(config):
    """Apply the mixed pilot camera and action defaults and validate the workflow."""
    base = yaml.safe_load(Path(config['base_config']).read_text())
    settings = collection_defaults(dict(base, **config))
    count = settings['seed_count']
    for key in ('seed_count', 'random_steps', 'recovery_steps'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    split_counts = settings.get('dataset_split_counts', dict(train=0, validation=0, test=count))
    if (set(split_counts) != set(DATASET_SPLITS)
            or any(type(value) is not int or value < 0 for value in split_counts.values())
            or sum(split_counts.values()) != count):
        raise ValueError('dataset_split_counts must provide train, validation and test counts totaling seed_count')
    if settings['batch_episodes'] < BASE_EPISODES_PER_SEED:
        raise ValueError(f'batch_episodes must fit at least {BASE_EPISODES_PER_SEED} base episodes')
    if settings.get('physics_device', 'cpu') != 'cpu':
        raise ValueError('Five-mode collection requires CPU physics')
    translation = np.asarray(settings['hole_motion']['translation_m'], dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all() or not translation.any():
        raise ValueError('hole_motion.translation_m must be a nonzero finite three-vector')
    if settings['hole_motion'].get('stop_expert_insert', False):
        raise ValueError('Use a Box motion schedule independent of policy phase')
    if not settings.get('excluded_seed_protocols', {}).get('mixed_trajectory'):
        raise ValueError('Provide the historical mixed_trajectory protocol in excluded_seed_protocols')
    settings.update(pilot_count=count, collect_splits=['pilot'], dataset_split_counts=split_counts)
    pilot.validate_config(settings)
    return settings


def build_dataset_protocol(config):
    """Freeze the per-seed split before any rollouts; pilot is the replay cohort."""
    protocol = pilot.build_protocol(config)
    records = [row for row in protocol['records'] if row['split'] == 'pilot']
    split_counts = config['dataset_split_counts']
    assignments = [split for split in DATASET_SPLITS for _ in range(split_counts[split])]
    for row, split in zip(records, assignments, strict=True):
        row['dataset_split'] = split
    return dict(protocol, records=records, modes=list(POLICY_MODES), dataset_split_counts=split_counts)


def load_dataset_protocol(config):
    """Read the shared protocol, or build one for a standalone collection."""
    source = config.get('protocol_path')
    if source is None:
        return build_dataset_protocol(config)
    protocol = json.loads(Path(source).read_text())
    records = protocol['records']
    seeds = {row['seed'] for row in records}
    splits = {name: sum(row['dataset_split'] == name for row in records) for name in DATASET_SPLITS}
    excluded, hashes = pilot.load_exclusions(config)
    if (len(seeds) != len(records) or len(records) != config['seed_count']
            or seeds & excluded or splits != config['dataset_split_counts']
            or protocol['generator'] != config['seed_generator']
            or protocol['exclusion_sha256'] != hashes
            or protocol['modes'] != list(POLICY_MODES)
            or any(protocol[key] != config[key] for key in
                   ('camera', 'control_mode', 'max_steps', 'visibility_stride'))):
        raise ValueError(f'Frozen protocol does not match the collection settings: {source}')
    return protocol


def collection_ranges(config, record_count):
    """Partition the worker's assigned record interval into archive ranges."""
    begin, end = config.get('record_range', [0, record_count])
    if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= record_count:
        raise ValueError('record_range must be a nonempty [begin, end) within the protocol')
    return [(batch[0], batch[-1] + 1)
            for batch in iter_batches(range(begin, end), config['batch_seed_count'])]


@contextmanager
def collection_lock(target):
    """Prevent concurrent writers from sharing one collection directory."""
    with (target / '.collection.lock').open('a') as lock:
        try:
            flock(lock, LOCK_EX | LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'A collector already owns {target}') from exc
        yield


def resume_collection(target, config, protocol, batches):
    """Keep verified receipts and schedule missing or damaged archives again."""
    protocol_path = target / 'protocol.json'
    manifest_path = target / 'manifest.json'
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    if json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(f'Collection protocol changed: {target}')
    digest = file_sha256(protocol_path)
    manifest = dict(status='RUNNING', effective_config=config, episodes=0, archives=[],
                    protocol_sha256=digest)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['effective_config'] != config or previous['protocol_sha256'] != digest:
            raise ValueError(f'Collection settings changed: {target}')
        for receipt in previous['archives']:
            begin, end = receipt['record_range']
            expected = f'episodes_{begin:05d}_{end:05d}.tar.zst'
            path = target / expected
            if (receipt['file'] == expected and (begin, end) in batches
                    and receipt['episodes'] == len(POLICY_MODES) * (end - begin)
                    and path.is_file() and file_sha256(path) == receipt['sha256']):
                manifest['archives'].append(receipt)
    completed = {tuple(receipt['record_range']) for receipt in manifest['archives']}
    if len(completed) != len(manifest['archives']):
        raise ValueError(f'Duplicate archive receipts: {target}')
    manifest['episodes'] = sum(receipt['episodes'] for receipt in manifest['archives'])
    return manifest, [batch for batch in batches if batch not in completed]


def collect_base_seed(record, config, protocol, root):
    """Collect three fixed trajectories and one moving expert for the same reset."""
    torch.set_num_threads(1)
    fixed = list(pilot.iter_seed_episodes(record, config, protocol, root / 'pilot'))
    moving_protocol = dict(protocol, hole_motion=config['hole_motion'])
    moving, = pilot.iter_seed_episodes(
        record, config, moving_protocol, root / 'moving', modes=('expert',)
    )
    return fixed, moving


def write_pilot_manifest(path, episodes, protocol):
    """Record the exact inputs required by the existing recovery planner."""
    write_json(path / 'protocol.json', protocol)
    write_json(path / 'manifest.json', dict(
        status='COMPLETE', protocol_sha256=file_sha256(path / 'protocol.json'),
        effective_config={'record_range': [0, len(protocol['records'])]}, episodes=episodes,
    ))
    return json.loads((path / 'manifest.json').read_text())


def audit_batch(root, records, fixed, moving, recovered, config):
    """Verify one paired trajectory per mode and record realized Box motion."""
    index = []
    split_by_seed = {row['seed']: row['dataset_split'] for row in records}
    fingerprints = {row['seed']: row['initial_fingerprint'] for row in fixed if row['mode'] == 'expert'}
    translation = np.asarray(config['hole_motion']['translation_m'], dtype=float)
    for cohort, episodes in (('pilot', fixed), ('recovery', recovered), ('moving', moving)):
        for row in episodes:
            seed = row['seed']
            path = root / cohort / row['path']
            if row['initial_fingerprint'] != fingerprints[seed]:
                raise ValueError(f'Initial state differs across policy modes: {path}')
            with np.load(path / 'frame_000000.npz', allow_pickle=False) as first:
                initial = first['ground_truth/box_pose'][0]
            with np.load(path / f'frame_{row["steps"]:06d}.npz', allow_pickle=False) as last:
                offset = last['ground_truth/box_pose'][0] - initial
            if cohort == 'moving':
                progress = float(offset[:3] @ translation / (translation @ translation))
                expected = [*(progress * translation), 0, 0, 0, 0]
                if not (0 < progress <= 1.0001 and np.allclose(offset, expected, rtol=0, atol=1e-6)):
                    raise ValueError(f'Moving-Hole Box did not translate along the configured path: {path}')
                label = MOVING_MODE
            else:
                if not np.allclose(offset, 0, rtol=0, atol=1e-6):
                    raise ValueError(f'Fixed-Hole Box moved: {path}')
                label = f'Fixed_Hole_{row["mode"]}'
            index.append(dict(seed=seed, split=split_by_seed[seed], policy_mode=label,
                              path=f'{cohort}/{row["path"]}', observations=row['observations'],
                              box_offset_m=offset[:3].tolist()))
    expected = {(row['seed'], mode) for row in records for mode in POLICY_MODES}
    if len(index) != len(expected) or {(row['seed'], row['policy_mode']) for row in index} != expected:
        raise ValueError('Expected exactly one episode per seed and policy mode')
    write_json(root / 'episode_index.json', index)


def collect_batch(seed_range, staging, *, config, protocol, executor):
    """Collect four base trajectories per seed, then branch from the fixed parent."""
    begin, end = seed_range
    records = protocol['records'][begin:end]
    fixed_protocol = dict(protocol, records=records, modes=list(pilot.MODES))
    for name in ('pilot', 'moving', 'recovery'):
        (staging / name).mkdir()
    base_job = partial(collect_base_seed, config=config, protocol=fixed_protocol, root=staging)
    fixed, moving = [], []
    for fixed_rows, moving_row in executor.map(base_job, records):
        fixed.extend(fixed_rows)
        moving.append(moving_row)
    manifest = write_pilot_manifest(staging / 'pilot', fixed, fixed_protocol)
    moving_protocol = dict(fixed_protocol, modes=['expert'], hole_motion=config['hole_motion'])
    write_pilot_manifest(staging / 'moving', moving, moving_protocol)
    cases = recovery.plan_cases(staging / 'pilot', manifest, fixed_protocol,
                                config['random_steps'], allow_fixed_subset=True)
    jobs = [
        executor.submit(recovery.collect_case, staging / 'pilot', staging / 'recovery',
                        case, fixed_protocol, config['recovery_steps'],
                        config.get('render_backend') or 'cpu', compress=config['compress_frames'])
        for case in cases
    ]
    recovered = [job.result() for job in jobs]
    write_json(staging / 'recovery' / 'manifest.json', dict(status='COMPLETE', episodes=recovered))
    audit_batch(staging, records, fixed, moving, recovered, config)
    return seed_range


def persist_batch(staging, seed_range, *, target, config):
    """Publish a complete archive while keeping all five modes together per seed."""
    begin, end = seed_range
    name = f'episodes_{begin:05d}_{end:05d}.tar.zst'
    return archive_verified_batch(staging, target / name, seed_range, len(POLICY_MODES) * (end - begin),
                                  long_distance=not config['compress_frames'])


def collect(config):
    """Freeze held-out seed selection and stream verified paired archives."""
    config = resolve_config(config)
    for program in ('tar', 'zstd'):
        if shutil.which(program) is None:
            raise RuntimeError(f'Install {program} before collecting five-mode archives')
    config['ram_root'] = str(staging_root(config))
    config.setdefault('compress_frames', True)
    config['batch_seed_count'] = config['batch_episodes'] // BASE_EPISODES_PER_SEED

    protocol = load_dataset_protocol(config)
    batches = collection_ranges(config, len(protocol['records']))
    target = Path(config['output_root']).resolve()
    target.mkdir(parents=True, exist_ok=config.get('resume', False))
    with collection_lock(target):
        run_collection(config, protocol, target, batches)


def run_collection(config, protocol, target, batches):
    """Collect the outstanding ranges and persist progress after every archive."""
    protocol_path, manifest_path = target / 'protocol.json', target / 'manifest.json'
    manifest, batches = resume_collection(target, config, protocol, batches)
    write_json(manifest_path, manifest)

    size = config['batch_seed_count']
    with ExperimentLogger(team='graph', model='five_mode_collection', config=config,
                          seed=config['seed_generator'], job_type='collection') as logger:
        try:
            workers = min(config['workers'], size, len(protocol['records']))
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
                prepare = partial(collect_batch, config=config, protocol=protocol, executor=executor)
                persist = partial(persist_batch, target=target, config=config)
                for receipt in stage_batches(batches, prepare, persist, config):
                    manifest['archives'].append(receipt)
                    manifest['archives'].sort(key=lambda row: row['record_range'])
                    manifest['episodes'] += receipt['episodes']
                    write_json(manifest_path, manifest)
            manifest['status'] = 'COMPLETE'
        except BaseException:
            manifest['status'] = 'ERROR'
            raise
        finally:
            write_json(manifest_path, manifest)
            sync_directory(target)
            logger.record(status=manifest['status'], episodes=manifest['episodes'],
                          protocol_sha256=manifest['protocol_sha256'])
            logger.link_output(protocol_path)
            logger.link_output(manifest_path)
