"""Align fixed-Hole Expert data and prepare offline corner supervision."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    FORMAT,
    pack_episode,
    validate_packed_episode,
)
from graph.temporal.detected_corner_tracking import match_positions
from utils.experiment_logger import write


def checked_path(path, expected_sha):
    """Check a configured artifact before using its row data."""
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return path


def verify_split(split, rows):
    """Verify that all 400 policy episodes follow the frozen reset-seed assignment."""
    groups = split['reset_seeds']
    sizes = {name: len(seeds) for name, seeds in groups.items()}
    if sizes != {'train': 80, 'validation': 10, 'test': 10}:
        raise ValueError(f'Unexpected reset-seed split: {sizes}')
    owners = {seed: name for name, seeds in groups.items() for seed in seeds}
    if len(owners) != 100 or len(rows) != 400 or len(split['episodes']) != 400:
        raise ValueError('Require 400 episodes and 100 nonoverlapping reset seeds')
    for row, assignment in zip(rows, split['episodes'], strict=True):
        expected = dict(episode_id=row['episode_id'], reset_seed=row['reset_seed'],
                        policy_mode=row['policy_mode'], frames=row['frames'], split=owners[row['reset_seed']])
        if expected != assignment:
            raise ValueError(f'Frozen split does not match cached episode {row["episode_id"]}')
    if Counter(row['split'] for row in split['episodes']) != {'train': 320, 'validation': 40, 'test': 40}:
        raise ValueError('Expected all four policy variants together for each reset seed')


def read_detections(root, row):
    """Pad one episode's ragged detections while leaving measured XYZ intact."""
    path = checked_path(root / row['detector_file'], row['detector_sha256'])
    frames = row['frames']
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['frame_indices'], np.arange(frames))
        offsets, xyz, kinds = data['corner_offsets'], data['corner_world_xyz'], data['corner_object_id']
    counts = np.diff(offsets)
    if (offsets.shape != (frames + 1,) or offsets[0] != 0 or offsets[-1] != len(xyz)
            or np.any(counts < 0) or xyz.shape != (len(kinds), 3) or xyz.dtype != np.float32
            or not np.isfinite(xyz).all() or not np.isin(kinds, (0, 1)).all()):
        raise ValueError(f'Invalid detection archive: {path}')
    nodes = max(1, int(counts.max()))
    padded_xyz = np.zeros((frames, nodes, 3), dtype=np.float32)
    padded_kind = np.full((frames, nodes), -1, dtype=np.int64)
    valid = np.arange(nodes) < counts[:, None]
    padded_xyz[valid] = xyz
    padded_kind[valid] = kinds
    return padded_xyz, padded_kind, valid


def load_expert(config):
    """Join detection, canonical GT, Recurrent targets and observed Robot9 by frame."""
    root = Path(config['cache_root'])
    manifest = json.loads(checked_path(root / 'manifest.json', config['cache_manifest_sha256']).read_text())
    split = json.loads(checked_path(config['split_manifest'], config['split_manifest_sha256']).read_text())
    rows = manifest['episodes']
    if manifest['status'] != 'VERIFIED' or manifest['gt_shape'] != [80645, 24, 3]:
        raise ValueError('Expected the verified 400-episode corner cache')
    if {name: row['raw_manifest_sha256'] for name, row in manifest['sources'].items()} != split['source_manifest_sha256']:
        raise ValueError('The split and corner cache reference different raw collections')
    verify_split(split, rows)
    with np.load(checked_path(root / manifest['gt_file'], manifest['gt_sha256']), allow_pickle=False) as data:
        gt = {name: data[name] for name in ('episode_ids', 'episode_offsets', 'source_episode_receipt_sha256',
                                           'frame_indices', 'reference_xyz', 'reference_kind')}
    with np.load(checked_path(manifest['state36_path'], manifest['state36_sha256']), allow_pickle=False) as data:
        for name in ('episode_ids', 'episode_offsets', 'source_episode_receipt_sha256', 'frame_indices'):
            np.testing.assert_array_equal(gt[name], data[name])
        object21 = data['state36_gt'][:, 15:36].copy()
    np.testing.assert_array_equal(gt['reference_kind'], [0] * 8 + [1] * 16)
    with np.load(checked_path(config['robot_archive'], config['robot_archive_sha256']), allow_pickle=False) as data:
        robot = {name: data[name] for name in data.files}
    if robot['source_manifest_sha256'].item() != split['source_manifest_sha256']['pilot']:
        raise ValueError('Observed Robot9 provenance differs from the fixed-Hole pilot')

    selected = [(i, row, assignment['split'])
                for i, (row, assignment) in enumerate(zip(rows, split['episodes'], strict=True))
                if row['policy_mode'] == 'expert' and assignment['split'] in ('train', 'validation')]
    if Counter(label for _, _, label in selected) != {'train': 80, 'validation': 10}:
        raise ValueError('Expected 80 train and 10 validation Expert episodes')
    np.testing.assert_array_equal(robot['episode_ids'], [row['episode_id'] for _, row, _ in selected])
    episodes = []
    for robot_index, (index, row, label) in enumerate(selected):
        name, frames = row['episode_id'], row['frames']
        start, stop = gt['episode_offsets'][index:index + 2]
        robot_start, robot_stop = robot['episode_offsets'][robot_index:robot_index + 2]
        if (name != gt['episode_ids'][index] or stop - start != frames or robot_stop - robot_start != frames
                or gt['source_episode_receipt_sha256'][index] != robot['source_episode_receipt_sha256'][robot_index]):
            raise ValueError(f'GT and robot alignment mismatch: {name}')
        np.testing.assert_array_equal(gt['frame_indices'][start:stop], np.arange(frames))
        np.testing.assert_array_equal(robot['frame_indices'][robot_start:robot_stop], np.arange(frames))
        state = robot['robot'][robot_start:robot_stop]
        if state.shape != (frames, 9) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError(f'Expected observed float32 Robot9 in {name}')
        xyz, kinds, valid = read_detections(root, row)
        episodes.append(dict(episode=name, parent=str(row['reset_seed']), split=label,
                             xyz=xyz, kind=kinds, valid=valid, robot=state,
                             reference_xyz=gt['reference_xyz'][start:stop], reference_kind=gt['reference_kind'],
                             target=object21[start:stop],
                             sample_ids=np.asarray([f'{name}/frame_{frame:06d}.npz' for frame in range(frames)])))
    return episodes


def write_sequences(episodes, output, tolerance_m, lineage):
    """Use GT only to label same-kind one-to-one matches and missing targets offline."""
    output.mkdir()
    records = []
    for index, episode in enumerate(episodes):
        identities = np.full(episode['valid'].shape, -1, dtype=np.int64)
        for frame, valid in enumerate(episode['valid']):
            points = np.flatnonzero(valid)
            matches = match_positions(episode['reference_xyz'][frame], episode['reference_kind'],
                                      episode['xyz'][frame, points], episode['kind'][frame, points], tolerance_m)
            for vertex, point in matches:
                identities[frame, points[point]] = vertex
        packed, stats = pack_episode(dict(episode, physical_ids=identities))
        if packed is None:
            raise ValueError(f'No geometrically matched corner sequence: {episode["episode"]}')
        validate_packed_episode(packed)
        path = output / f'episode_{index:06d}.npz'
        np.savez_compressed(path, **packed)
        records.append(dict(episode=episode['episode'], parent=episode['parent'], split=episode['split'],
                            file=path.name, sha256=file_sha256(path), sequences=stats['sequences'],
                            unknown_detections=stats['unknown_detections']))
    manifest = dict(format=FORMAT, time_unit='saved_frame', match_tolerance_m=tolerance_m,
                    label_method='same-object geometric one-to-one matching', synthetic_occlusion=False,
                    source=lineage, episodes=records)
    manifest['dataset_sha256'] = canonical_hash(manifest)
    write(output / 'manifest.json', manifest)
    return manifest['dataset_sha256']
