"""Evaluate RGB-D corner detection on the restored initial-state audit cohort."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, score, visible_reference


FAMILIES = ('peg', 'box_outer', 'hole_front', 'hole_rear')


def read_rows(path):
    return json.loads(Path(path).read_text())['rows']


def load_cohort(config):
    """Join the restored Peg and Hole measurements without filtering seeds."""
    peg = [row for path in config['peg_audits'] for row in read_rows(path)]
    hole = read_rows(config['hole_audit'])
    expected = list(range(config['seed_start'], config['seed_stop']))
    for name, rows in (('Peg', peg), ('Hole', hole)):
        if sorted(row['seed'] for row in rows) != expected:
            raise ValueError(f'{name} audit must contain every configured seed exactly once')
    return {row['seed']: row for row in peg}, {row['seed']: row for row in hole}


def verify_camera(sensor, calibration, config):
    """Verify actual reset camera intrinsics, origin and viewing direction."""
    width, height = config['resolution']
    eye = np.asarray(config['camera_eye_m'])
    direction = np.asarray(config['camera_target_m']) - eye
    direction /= np.linalg.norm(direction)
    gl = calibration['cam2world_gl']
    focal = height / (2 * np.tan(np.deg2rad(config['vertical_fov_deg']) / 2))
    if (sensor['rgb'].shape != (height, width, 3)
            or not np.allclose(gl[:3, 3], eye, atol=1e-5)
            or not np.allclose(-gl[:3, 2], direction, atol=1e-5)
            or not np.allclose(np.diag(calibration['intrinsic_cv'])[:2], focal, atol=1e-4)):
        raise ValueError('Live base camera differs from the configured evaluation camera')


def frame_score(env, detector, seed, peg_audit, hole_audit, config):
    """Re-render one seed, verify the reset geometry and score detector output."""
    obs, _ = env.reset(seed=seed)
    sensor = {key: value[0].cpu().numpy()
              for key, value in obs['sensor_data']['base_camera'].items()}
    calibration = {key: value[0].cpu().numpy()
                   for key, value in obs['sensor_param']['base_camera'].items()}
    verify_camera(sensor, calibration, config)
    sensor.update(calibration)
    gt = {key: value[0].cpu().numpy() for key, value in vars(get_gt_state(env)).items()}
    for key, audit_key in (('peg_pose', 'gt_peg_pose'), ('peg_half_sizes', 'peg_half_sizes')):
        if not np.allclose(gt[key], peg_audit[audit_key], atol=1e-6, rtol=0):
            raise ValueError(f'Seed {seed}: restored and live {key} differ')

    task = env.unwrapped
    ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    supported = visible_reference(gt, sensor, ids)
    all_refs = corners(gt)
    front = [row for row in all_refs if row['kind'] == 'hole_rim' and row['corner_id'] < 4]
    xyz = np.array([row['xyz'] for row in front])
    ext = sensor['extrinsic_cv']
    camera_xyz = xyz @ ext[:, :3].T + ext[:, 3]
    pixels = camera_xyz @ sensor['intrinsic_cv'].T
    for point, pixel, old in zip(camera_xyz, pixels, hole_audit['corners'], strict=True):
        if (not np.allclose(pixel[:2] / pixel[2], old['uv'], atol=1e-3, rtol=0)
                or not np.isclose(point[2], old['camera_depth_m'], atol=1e-5, rtol=0)):
            raise ValueError(f'Seed {seed}: restored and live Hole projection differ')

    # Front and rear references stay in the same assignment as Box exterior references.
    for row in all_refs + supported:
        if row['kind'] == 'hole_rim':
            row['kind'] = 'hole_front' if row['corner_id'] < 4 else 'hole_rear'

    prediction = detector(sensor['rgb'], sensor['depth'], sensor['intrinsic_cv'],
                          sensor['extrinsic_cv']).as_dict()['corners']
    scores = {}
    for threshold in config['thresholds_mm']:
        objects = score(prediction, supported, all_refs, threshold / 1000)
        families = {name: {'supported': 0, 'matched': 0} for name in FAMILIES}
        for refs in supported:
            families[refs['kind']]['supported'] += 1
        for obj in objects:
            for match in obj['matches']:
                families[match['kind']]['matched'] += 1
        scores[str(threshold)] = {'families': families, 'objects': objects}
    return dict(seed=seed, scores=scores,
                peg_robot_occlusion_fraction=peg_audit['robot_occlusion_fraction'],
                hole_robot_center_corners=hole_audit['robot_center_corners'])


def summarize(rows, thresholds):
    """Report supported recall and counts of frames with zero or multiple matches."""
    result = {}
    for threshold in thresholds:
        scores = [row['scores'][str(threshold)] for row in rows]
        totals = {}
        for name in FAMILIES:
            family = [value['families'][name] for value in scores]
            eligible = [value for value in family if value['supported']]
            histogram = Counter(value['matched'] for value in family)
            supported = sum(value['supported'] for value in family)
            matched = sum(value['matched'] for value in family)
            totals[name] = dict(
                supported=supported, matched=matched, recall=matched / supported if supported else None,
                frames_with_support=len(eligible),
                supported_frames_without_match=sum(value['matched'] == 0 for value in eligible),
                frames_with_at_least_two_matches=sum(n for k, n in histogram.items() if k >= 2),
                match_histogram=dict(histogram),
            )
        result[str(threshold)] = totals
    return dict(frames=len(rows), thresholds_mm=result)


def evaluate(config):
    """Re-render all configured initial states and write the complete evaluation."""
    peg, hole = load_cohort(config)
    torch.set_num_threads(config['torch_threads'])
    detector = TorchCornerDetector(device='cpu')
    env = make_peg_insertion_env(device='cpu', obs_mode='rgb+depth+segmentation')
    try:
        with ExperimentLogger(team='graph', model='initial_corner_detection', config=config,
                              seed=config['seed_start'], job_type='evaluation') as logger:
            rows = []
            for seed in range(config['seed_start'], config['seed_stop']):
                rows.append(frame_score(env, detector, seed, peg[seed], hole[seed], config))
                if len(rows) == 1 or len(rows) % 100 == 0:
                    print(f'Corner detection {len(rows)}/{len(peg)} resets', flush=True)
            groups = {
                'all': rows,
                'peg_robot_occluded': [row for row in rows if row['peg_robot_occlusion_fraction'] >= .01],
                'hole_robot_blocks_two_or_more': [row for row in rows if row['hole_robot_center_corners'] >= 2],
                'hole_robot_blocks_fewer_than_two': [row for row in rows if row['hole_robot_center_corners'] < 2],
            }
            summaries = {key: summarize(value, config['thresholds_mm']) for key, value in groups.items()}
            source_paths = [*config['peg_audits'], config['hole_audit']]
            result = dict(config=config, source_sha256={path: file_sha256(path) for path in source_paths},
                          summaries=summaries, rows=rows)
            output = Path(config['output_root']) / logger.run_id / 'results.json'
            output.parent.mkdir(parents=True, exist_ok=False)
            write_json(output, result)
            logger.record(frames=len(rows), summary=summaries['all'])
            logger.link_output(output)
            print(f'Initial corner detection: {output}', flush=True)
    finally:
        env.close()
