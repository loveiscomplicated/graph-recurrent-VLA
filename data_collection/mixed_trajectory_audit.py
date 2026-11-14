"""Audit every collected frame and report the unfiltered pilot distribution."""

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.visibility import CameraObservation, evaluate
from utils.experiment_logger import ExperimentLogger


def audit_episode(root, row, camera):
    """Verify frame hashes, action alignment, camera and initial-to-final motion."""
    directory = root / row['path']
    receipt = json.loads((directory / 'episode.json').read_text())
    if (receipt != row or row['observations'] != row['steps'] + 1
            or len(row['frame_sha256']) != row['observations']):
        raise ValueError(f'Inconsistent episode receipt: {directory}')
    initial_position = None
    initial_rotation = None
    maximum_displacement = 0.0
    maximum_rotation = 0.0
    peg_visibility = []
    sampled_frames = {sample['frame'] for sample in row['hole_visibility_samples']}
    for index, expected_hash in enumerate(row['frame_sha256']):
        path = directory / f'frame_{index:06d}.npz'
        if file_sha256(path) != expected_hash:
            raise ValueError(f'Frame checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as frame:
            valid = bool(frame['action_valid'])
            if valid != (index < row['steps']) or int(frame['collection/frame']) != index:
                raise ValueError(f'Invalid observation/action alignment: {path}')
            action = frame['action']
            if action.size != 7 or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise ValueError(f'Invalid native action: {path}')
            sensor = 'observation/sensor_data/base_camera/'
            calibration = 'observation/sensor_param/base_camera/'
            rgb, depth = frame[sensor + 'rgb'], frame[sensor + 'depth']
            if rgb.shape != (1, 512, 512, 3) or rgb.dtype != np.uint8:
                raise ValueError(f'Invalid RGB shape or dtype: {path}')
            if depth.shape != (1, 512, 512, 1) or depth.dtype != np.int16:
                raise ValueError(f'Invalid depth shape or dtype: {path}')
            matrix = frame[calibration + 'cam2world_gl'][0]
            if not np.allclose(matrix[:3, 3], camera['eye'], rtol=0, atol=1e-6):
                raise ValueError(f'Wrong recorded camera position: {path}')
            if index == 0:
                digest = hashlib.sha256()
                fields = ('observation/agent/qpos', 'ground_truth/peg_pose',
                          'ground_truth/box_pose', 'ground_truth/hole_pose',
                          'ground_truth/peg_half_sizes', 'ground_truth/hole_radius')
                for field in fields:
                    digest.update(frame[field].tobytes())
                if digest.hexdigest() != row['initial_fingerprint']:
                    raise ValueError(f'Initial state fingerprint mismatch: {path}')
            if index in sampled_frames:
                tcp = frame['ground_truth/ee_pose'][0]
                image = CameraObservation(row['path'], index, 'base_camera', 0.0,
                    rgb[0], depth[0, ..., 0], frame[calibration + 'intrinsic_cv'][0],
                    frame[calibration + 'extrinsic_cv'][0], matrix, tcp)
                visibility, _ = evaluate(image, frame['ground_truth/peg_pose'][0],
                                         frame['ground_truth/peg_half_sizes'][0], tcp)
                peg_visibility.append(visibility['supported_fraction_in_frame'])
            pose = frame['ground_truth/peg_pose'][0]
            if not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError(f'Invalid Peg GT pose: {path}')
            position = pose[:3]
            rotation = pose[3:] / np.linalg.norm(pose[3:])
            if initial_position is None:
                initial_position = position.copy()
                initial_rotation = rotation.copy()
            maximum_displacement = max(maximum_displacement, float(np.linalg.norm(position - initial_position)))
            similarity = np.clip(abs(float(np.dot(rotation, initial_rotation))), 0.0, 1.0)
            maximum_rotation = max(maximum_rotation, float(np.degrees(2 * np.arccos(similarity))))
    return (maximum_displacement, maximum_rotation,
            [value for value in peg_visibility if value is not None])


def audit(config):
    """Audit a complete cohort and save aggregate, non-model-based statistics."""
    root = Path(config['collection_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if manifest['status'] != 'COMPLETE' or file_sha256(root / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Collection must be complete and match its frozen protocol')
    split_by_seed = {row['seed']: row['split'] for row in protocol['records']}
    if len(split_by_seed) != len(protocol['records']):
        raise ValueError('Protocol contains a duplicate initial seed')
    records = {row['seed']: row for row in protocol['records']}
    expected = {(row['seed'], mode) for row in protocol['records']
                if row['split'] in manifest['effective_config']['collect_splits'] for mode in protocol['modes']}
    actual = {(row['seed'], row['mode']) for row in manifest['episodes']}
    if expected != actual or len(actual) != len(manifest['episodes']):
        raise ValueError('Collection cases do not exactly match the frozen protocol')
    fingerprints = {}
    seeds_by_fingerprint = {}
    strength_groups = defaultdict(lambda: dict(episodes=0, displacements=[], rotations=[], successes=0))
    summary = defaultdict(lambda: dict(episodes=0, steps=0, successes=0,
                                       stop_reasons=Counter(), initial_peg_visible=0,
                                       hole_visibility=[], peg_visibility=[], max_displacement_m=[], max_rotation_deg=[],
                                       transition_planned=Counter(), transition_reached=Counter(),
                                       transition_completed=Counter(), elapsed_seconds=0.0))
    episodes = manifest['episodes']
    audited = []
    with ProcessPoolExecutor(max_workers=config.get('workers', 1)) as executor:
        checks = executor.map(audit_episode, repeat(root), episodes, repeat(protocol['camera']))
        for index, check in enumerate(checks):
            audited.append(check)
            if (index + 1) % 10 == 0:
                print(f'Audited episodes {index + 1}/{len(episodes)}', flush=True)
    for row, check in zip(episodes, audited):
        seed = row['seed']
        if row['split'] != split_by_seed[seed]:
            raise ValueError(f'Split does not match frozen assignment for seed {seed}')
        previous = fingerprints.setdefault(seed, row['initial_fingerprint'])
        if previous != row['initial_fingerprint']:
            raise ValueError(f'Initial physical state differs across modes for seed {seed}')
        other = seeds_by_fingerprint.setdefault(row['initial_fingerprint'], seed)
        if other != seed:
            raise ValueError(f'Duplicate initial state across seeds {other} and {seed}')
        distance, rotation, peg_visibility = check
        group = summary[row['mode']]
        group['episodes'] += 1
        group['steps'] += row['steps']
        group['successes'] += int(row['successful'])
        group['stop_reasons'][row['stop_reason']] += 1
        group['initial_peg_visible'] += int(row['initial_peg_visibility']['requested_pass'])
        group['hole_visibility'].extend(item['visible_fraction'] for item in row['hole_visibility_samples'])
        group['peg_visibility'].extend(peg_visibility)
        group['max_displacement_m'].append(distance)
        group['max_rotation_deg'].append(rotation)
        group['elapsed_seconds'] += row['elapsed_seconds']
        if row['mode'] != 'expert':
            strength_key = f'{row["mode"]}:{records[seed]["random_limit"]}'
            strength = strength_groups[strength_key]
            strength['episodes'] += 1
            strength['displacements'].append(distance)
            strength['rotations'].append(rotation)
            strength['successes'] += int(row['successful'])
        if row['mode'] == 'expert_random':
            phase = records[seed]['transition_phase']
            group['transition_planned'][phase] += 1
            group['transition_reached'][phase] += int(row['transition_reached'])
            group['transition_completed'][phase] += int(row['switched'])
    for group in summary.values():
        group['hole_visibility_mean'] = float(np.mean(group.pop('hole_visibility')))
        peg_values = group.pop('peg_visibility')
        group['peg_visibility_mean'] = float(np.mean(peg_values)) if peg_values else None
        displacement = group.pop('max_displacement_m')
        group['max_displacement_m_mean'] = float(np.mean(displacement))
        group['max_displacement_m_p90'] = float(np.quantile(displacement, 0.9))
        rotations = group.pop('max_rotation_deg')
        group['max_rotation_deg_mean'] = float(np.mean(rotations))
        group['max_rotation_deg_p90'] = float(np.quantile(rotations, 0.9))
    for group in strength_groups.values():
        group['max_displacement_m_mean'] = float(np.mean(group.pop('displacements')))
        group['max_rotation_deg_mean'] = float(np.mean(group.pop('rotations')))
    byte_count = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    report = dict(protocol_sha256=manifest['protocol_sha256'], initial_states=len(fingerprints),
                  frames=sum(row['observations'] for row in manifest['episodes']),
                  episodes=len(manifest['episodes']), bytes=byte_count,
                  estimated_main_bytes=byte_count * 10,
                  visibility_note='GT/depth diagnostics: Peg initial gate and sampled depth support, sparse 7x7 Hole samples; not exact continuous visibility.',
                  summaries=summary, random_strength_groups=strength_groups)
    output = Path(config['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'audit.json'
    write_json(path, report)
    with ExperimentLogger(team='graph', model='mixed_trajectory_audit', config=config,
                          seed=protocol['generator'], job_type='evaluation') as logger:
        logger.record(protocol_sha256=manifest['protocol_sha256'], frames=report['frames'], episodes=report['episodes'])
        logger.link_output(path)
    print(f'Audit complete: {path}', flush=True)
