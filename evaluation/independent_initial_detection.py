"""Inspect first-frame Peg detector coverage on a frozen independent test."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.vertex_family_labels import SIGNS, pose_rotation
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def evaluate(config):
    """Compare same-kind detections with GT vertices for diagnosis only."""
    if file_sha256(config['protocol']) != config['protocol_sha256']:
        raise ValueError('Frozen test protocol SHA256 mismatch')
    protocol = json.loads(Path(config['protocol']).read_text())
    root = Path(config['raw_root'])
    manifest = json.loads((root/'manifest.json').read_text())
    visibility = json.loads(Path(config['visibility_audit']).read_text())
    gate = {row['seed']: row['requested_pass'] for row in visibility['rows']}
    episodes = {row['seed']: row for row in manifest['episodes'] if row['mode'] == 'expert'}
    if manifest['status'] != 'COMPLETE' or set(episodes) != set(protocol['seeds']) or set(gate) != set(protocol['seeds']):
        raise ValueError('Dataset or visibility audit does not match the fixed test')
    torch.set_num_threads(2)
    detector = TorchCornerDetector(device=config['detector_device'])
    with ExperimentLogger(team='graph', model='independent_initial_detection', config=config,
                          seed=protocol['generation']['seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for seed in protocol['seeds']:
            receipt = episodes[seed]
            path = root / receipt['path'] / 'frame_000000.npz'
            if file_sha256(path) != receipt['frame_sha256'][0]:
                raise ValueError(f'Initial frame checksum mismatch: {seed}')
            observation, gt = read_frame(path, 'base_camera')
            detection = detector(observation['rgb'], observation['depth'],
                                 observation['intrinsic_cv'], observation['extrinsic_cv'])
            kinds = detection.object_id.cpu().numpy()
            points = detection.xyz.cpu().numpy()[kinds == 0]
            pose = gt['peg_pose'][0]
            corners = SIGNS * gt['peg_half_sizes'][0] @ pose_rotation(pose).T + pose[:3]
            if len(points):
                distances = np.linalg.norm(points[:, None, :] - corners[None, :, :], axis=2)
                nearest = distances.argmin(axis=1)
                close = distances[np.arange(len(points)), nearest] <= config['coordinate_tolerance_m']
                unique = len(set(nearest[close]))
            else:
                unique = 0
            rows.append(dict(seed=seed, initial_gate_pass=gate[seed], peg_detections=len(points),
                             box_detections=int(np.sum(kinds == 1)),
                             peg_gt_nearest_unique_within_3mm=unique))
        result = dict(config=config, rows=rows,
                      caveat='GT nearest-vertex association is diagnostic, not certified identity or visibility.')
        path = output/'initial_detection.json'
        path.write_text(json.dumps(result, indent=2))
        logger.record(cases=len(rows))
        logger.link_output(path)
        print(f'Initial detection audit: {output}', flush=True)
