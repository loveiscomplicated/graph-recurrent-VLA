"""Measure current-frame corner detection on stratified pilot trajectories."""

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_collection import camera_sensors
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, score, visible_reference

_DETECTOR = None


def initialize_detector():
    """Create a separate CPU detector for each worker process."""
    global _DETECTOR
    torch.set_num_threads(2)
    _DETECTOR = TorchCornerDetector(device='cpu')


def confirm_actor_ids(root, episodes, protocol):
    """Match every saved reset against live segmentation before using actor IDs."""
    sensors = camera_sensors(protocol['camera'])
    env = make_peg_insertion_env(obs_mode='rgb+depth+segmentation', sensor_configs=sensors,
                                 control_mode=protocol['control_mode'],
                                 max_episode_steps=protocol['max_steps'])
    ids = {}
    try:
        for row in episodes:
            if row['mode'] != 'expert':
                continue
            obs, _ = env.reset(seed=row['seed'])
            task = env.unwrapped
            actor_ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
            path = root / row['path'] / 'frame_000000.npz'
            if file_sha256(path) != row['frame_sha256'][0]:
                raise ValueError(f'Initial frame hash mismatch: {path}')
            state = get_gt_state(env)
            with np.load(path, allow_pickle=False) as frame:
                saved = frame['observation/sensor_data/base_camera/segmentation']
                actual = obs['sensor_data']['base_camera']['segmentation'].cpu().numpy()
                if not np.array_equal(saved, actual):
                    raise ValueError(f'Live reset segmentation differs from saved frame: {path}')
                for key in ('peg_pose', 'box_pose', 'hole_pose', 'qpos'):
                    if not np.array_equal(frame[f'ground_truth/{key}'], getattr(state, key).cpu().numpy()):
                        raise ValueError(f'Live reset {key} differs from saved frame: {path}')
            ids[row['seed']] = actor_ids
    finally:
        env.close()
    return ids


def sample_frames(row):
    """Sample four trajectory quartiles; count each shared reset only once."""
    steps = row['steps']
    frames = {int(round(steps * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)}
    if row['mode'] == 'expert':
        frames.add(0)
    return sorted(frames)


def evaluate_frame(job):
    """Score one frame with RGB-D-only inference and GT-only evaluation."""
    root, row, frame_number, actor_ids = job
    path = Path(root) / row['path'] / f'frame_{frame_number:06d}.npz'
    if file_sha256(path) != row['frame_sha256'][frame_number]:
        raise ValueError(f'Frame checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as frame:
        sensor_prefix = 'observation/sensor_data/base_camera/'
        calibration_prefix = 'observation/sensor_param/base_camera/'
        sensor = {name: frame[sensor_prefix + name][0].copy()
                  for name in ('rgb', 'depth', 'segmentation')}
        sensor.update({name: frame[calibration_prefix + name][0].copy()
                       for name in ('intrinsic_cv', 'extrinsic_cv')})
        gt = {name: frame['ground_truth/' + name][0].copy() for name in
              ('peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes', 'hole_radius')}
        phase = str(frame['collection/phase'])
        switched = bool(frame['collection/switched'])
        grasp = bool(frame['collection/grasp'])
    result = _DETECTOR(sensor['rgb'], sensor['depth'], sensor['intrinsic_cv'], sensor['extrinsic_cv'])
    xyz = result.xyz.cpu().numpy()
    object_ids = result.object_id.cpu().numpy()
    predictions = [dict(xyz=point, object_id=int(kind)) for point, kind in zip(xyz, object_ids)]
    all_refs = corners(gt)
    supported = visible_reference(gt, sensor, actor_ids)
    scores = {str(threshold): score(predictions, supported, all_refs, threshold)
              for threshold in (0.003, 0.005)}
    nearest_all = {}
    for kind in (0, 1):
        points = xyz[object_ids == kind]
        targets = np.asarray([r['xyz'] for r in all_refs if r['object_id'] == kind])
        distances = np.linalg.norm(points[:, None] - targets[None], axis=-1)
        nearest_all[str(kind)] = (distances.min(axis=1) * 1000).tolist() if len(points) else []
    # Store scalar scoring details only; detector results and GT remain separated.
    return dict(seed=row['seed'], mode=row['mode'], frame=frame_number, steps=row['steps'],
                phase=phase, switched=switched, grasp=grasp,
                scores=scores, nearest_all_gt_mm=nearest_all)


def summarize(rows, threshold):
    """Aggregate one-to-one frame matches, preserving object and corner families."""
    counts = {kind: dict(predictions=0, supported=0, matched=0, nearest_all_gt_mm=[])
              for kind in ('peg', 'box')}
    families = {kind: dict(supported=0, matched=0) for kind in ('peg', 'box_outer', 'hole_rim')}
    no_predictions = {kind: 0 for kind in counts}
    threshold_key = str(threshold)
    for row in rows:
        for object_id, kind in enumerate(('peg', 'box')):
            result = row['scores'][threshold_key][object_id]
            target = counts[kind]
            target['predictions'] += result['predictions']
            target['supported'] += result['visible_reference']
            target['matched'] += result['matched_3mm']
            target['nearest_all_gt_mm'].extend(row['nearest_all_gt_mm'][str(object_id)])
            no_predictions[kind] += int(result['predictions'] == 0)
            for family, value in result['by_kind'].items():
                families[family]['supported'] += value['visible']
                families[family]['matched'] += value['matched']
    for kind, entry in counts.items():
        entry['precision'] = entry['matched'] / entry['predictions'] if entry['predictions'] else None
        entry['recall'] = entry['matched'] / entry['supported'] if entry['supported'] else None
        distances = entry.pop('nearest_all_gt_mm')
        entry['nearest_any_gt_mean_mm'] = float(np.mean(distances)) if distances else None
        entry['nearest_any_gt_p90_mm'] = float(np.quantile(distances, 0.9)) if distances else None
        entry['frames_without_detections'] = no_predictions[kind]
    for entry in families.values():
        entry['recall'] = entry['matched'] / entry['supported'] if entry['supported'] else None
    return dict(frames=len(rows), objects=counts, families=families)


def evaluate(config):
    """Evaluate a fixed per-episode sample and save every per-frame result."""
    root = Path(config['dataset'])
    manifest = json.loads((root / 'manifest.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if manifest['status'] != 'COMPLETE' or file_sha256(root / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Pilot manifest must be complete and match its protocol')
    episodes = manifest['episodes']
    expected = {(r['seed'], mode) for r in protocol['records'] if r['split'] == 'pilot'
                for mode in protocol['modes']}
    if {(row['seed'], row['mode']) for row in episodes} != expected or len(episodes) != len(expected):
        raise ValueError('Pilot manifest is missing or duplicates a planned trajectory')
    ids = confirm_actor_ids(root, episodes, protocol)
    if len(ids) != 100:
        raise ValueError('Expected live actor IDs for all 100 pilot initial states')
    jobs = [(str(root), row, frame, ids[row['seed']]) for row in episodes for frame in sample_frames(row)]
    with ExperimentLogger(team='graph', model='pilot_corner_detection', config=config,
                          seed=protocol['generator'], job_type='evaluation') as logger:
        with ProcessPoolExecutor(max_workers=config['workers'], initializer=initialize_detector) as executor:
            rows = []
            for index, row in enumerate(executor.map(evaluate_frame, jobs, chunksize=2)):
                rows.append(row)
                if (index + 1) % 100 == 0:
                    print(f'Corner detection {index + 1}/{len(jobs)} frames', flush=True)
        subsets = {'all': rows}
        for mode in protocol['modes']:
            subsets[mode] = [row for row in rows if row['mode'] == mode]
        subsets['reset'] = [row for row in rows if row['frame'] == 0]
        for grasp in (True, False):
            subsets[f'grasp_{str(grasp).lower()}'] = [row for row in rows if row['grasp'] == grasp]
        summaries = {name: {str(t): summarize(group, t) for t in (0.003, 0.005)}
                     for name, group in subsets.items()}
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        result = dict(config=config, protocol_sha256=manifest['protocol_sha256'],
                      source_manifest_sha256=file_sha256(root / 'manifest.json'),
                      sampling='four trajectory quartiles per episode plus reset once per initial seed',
                      actor_id_reset_checks=len(ids), actor_id_values=sorted({tuple(v.values()) for v in ids.values()}),
                      reference='7x7 pixel same-actor depth within 6 mm of projected GT vertex',
                      note='Precision and recall are relative to approximate supported references, not certified visibility; Hole uses rim corners, not center.',
                      summaries=summaries, rows=rows)
        path = output / 'results.json'
        write_json(path, result)
        logger.record(frames=len(rows), episodes=len(episodes), summary=summaries['all'])
        logger.link_output(path)
        print(f'Pilot detector evaluation: {path}', flush=True)
