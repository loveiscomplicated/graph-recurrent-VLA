"""Compare current detector outputs with original cached detections on raw frames."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from data_collection.episode_dataset import file_sha256, read_frame
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def compare_detections(actual_xyz, actual_kind, recorded_xyz, recorded_kind):
    """Match each same-kind point set and retain unmatched counts and distances."""
    distances = []
    matched = 0
    for kind in (0, 1):
        actual = actual_xyz[actual_kind == kind]
        recorded = recorded_xyz[recorded_kind == kind]
        if len(actual) and len(recorded):
            cost = np.linalg.norm(actual[:, None, :] - recorded[None, :, :], axis=-1)
            left, right = linear_sum_assignment(cost)
            distances.extend((cost[left, right] * 1000).tolist())
            matched += len(left)
    return dict(actual_count=len(actual_xyz), recorded_count=len(recorded_xyz), matched=matched,
                unmatched_actual=len(actual_xyz)-matched, unmatched_recorded=len(recorded_xyz)-matched,
                distance_mm=distances)


def evaluate(config):
    """Hash-check each available raw validation frame and compare full point sets."""
    torch.set_num_threads(config['cpu_threads'])
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    if file_sha256(config['cache']) != checkpoint['rollout_cache_sha256']:
        raise ValueError('Rollout cache hash does not match checkpoint')
    arrays = torch.load(config['cache'], map_location='cpu', weights_only=False)['arrays']
    index = {str(sample): i for i, sample in enumerate(arrays['sample_ids']) if arrays['split'][i]}
    manifest = json.loads(Path(config['manifest']).read_text())
    samples = [row for row in manifest['samples'] if row['sample_id'] in index]
    if not samples:
        raise ValueError('No raw frames match the original leaderboard validation IDs')
    detector = TorchCornerDetector(device=config['device'])
    with ExperimentLogger(team='graph', model='leaderboard_detector_audit', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for number, row in enumerate(samples):
            raw = Path(config['raw_root']) / row['raw_path']
            if file_sha256(raw) != row['source_sha256']:
                raise ValueError(f'Raw frame checksum mismatch: {row["sample_id"]}')
            obs, _ = read_frame(raw, 'base_camera')
            detection = detector(obs['rgb'], obs['depth'], obs['intrinsic_cv'], obs['extrinsic_cv'])
            i = index[row['sample_id']]
            mask = arrays['valid'][i] & arrays['observed'][i]
            result = compare_detections(detection.xyz.cpu().numpy(), detection.object_id.cpu().numpy(),
                                        arrays['xyz'][i, mask], arrays['kinds'][i, mask])
            rows.append(dict(sample_id=row['sample_id'], **result))
            if (number+1) % 100 == 0:
                print(f'Detector audit {number+1}/{len(samples)} frames', flush=True)
        distances = [distance for row in rows for distance in row['distance_mm']]
        summary = dict(frames=len(rows), count_equal_frames=sum(row['actual_count']==row['recorded_count'] for row in rows),
                       unmatched_actual=sum(row['unmatched_actual'] for row in rows),
                       unmatched_recorded=sum(row['unmatched_recorded'] for row in rows),
                       matched_points=len(distances), mean_distance_mm=float(np.mean(distances)) if distances else None,
                       p90_distance_mm=float(np.percentile(distances, 90)) if distances else None,
                       max_distance_mm=float(max(distances)) if distances else None,
                       within_0_01mm=sum(x <= .01 for x in distances),
                       exact_frames=sum(row['unmatched_actual']==row['unmatched_recorded']==0 and
                                        all(distance == 0 for distance in row['distance_mm']) for row in rows))
        path = output / 'detector_audit.json'
        path.write_text(json.dumps(dict(config=config, summary=summary, frames=rows,
                                       scope='Subset with saved raw RGB-D; same-kind nearest matching; CPU detector vs cached original detections'), indent=2))
        logger.record(**summary)
        logger.link_output(path)
        print(f'Detector audit: {output}', flush=True)
