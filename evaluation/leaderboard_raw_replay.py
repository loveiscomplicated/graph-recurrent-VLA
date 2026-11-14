"""Re-run detector, tracker and GNN on contiguous original validation episodes."""

import io
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from evaluation.independent_test import predict_frame
from evaluation.random_action_diagnostics import load_models
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.utils.object21_targets import object21_from_gt
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def track_difference(actual, kinds, expected_xyz, expected_kinds, expected_observed):
    """Match same-kind and same-provenance graph nodes without using GT identity."""
    actual_xyz = actual['positions']
    actual_observed = actual['observed']
    distance = []
    matched = 0
    for kind in (0, 1):
        for observed in (False, True):
            left = actual_xyz[(kinds == kind) & (actual_observed == observed)]
            right = expected_xyz[(expected_kinds == kind) & (expected_observed == observed)]
            if len(left) and len(right):
                cost = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
                i, j = linear_sum_assignment(cost)
                distance.extend((cost[i, j] * 1000).tolist())
                matched += len(i)
    return dict(actual_nodes=len(actual_xyz), reference_nodes=len(expected_xyz), matched=matched,
                unmatched_actual=len(actual_xyz)-matched, unmatched_reference=len(expected_xyz)-matched,
                distances_mm=distance)


def archive_index(archive, sample_ids):
    """Map expected safe frame IDs to exactly one regular archive member."""
    expected = set(map(str, sample_ids))
    found = {}
    for member in archive:
        name = member.name.removeprefix('./')
        if name in expected:
            if not member.isfile() or name in found:
                raise ValueError(f'Invalid or duplicate archive member: {name}')
            found[name] = member
    if found.keys() != expected:
        raise ValueError(f'Archive lacks {len(expected - found.keys())} expected frames')
    return found


def load_raw(archive, member):
    """Load a validated frame member without extracting paths onto the filesystem."""
    with archive.extractfile(member) as stream:
        payload = stream.read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as frame:
        prefix = 'observation/sensor_data/base_camera/'
        camera = 'observation/sensor_param/base_camera/'
        rgb = frame[prefix + 'rgb'][0].copy()
        depth = frame[prefix + 'depth'][0].copy()
        intrinsic = frame[camera + 'intrinsic_cv'][0].copy()
        extrinsic = frame[camera + 'extrinsic_cv'][0].copy()
        tcp = frame['observation/extra/tcp_pose'][0].copy()
        fingers = frame['observation/agent/qpos'][0, -2:].copy()
        gt = {key.removeprefix('ground_truth/'): torch.as_tensor(frame[key].copy())
              for key in frame.files if key.startswith('ground_truth/')}
    return rgb, depth, intrinsic, extrinsic, np.concatenate((tcp, fingers)).astype(np.float32), gt


def evaluate(config):
    """Check original source identity and run full causal inference on one shard."""
    if file_sha256(config['raw_shard']) != config['raw_shard_sha256']:
        raise ValueError('Original raw shard checksum mismatch')
    checkpoint = torch.load(config['gnn_checkpoint'], map_location='cpu', weights_only=False)
    if file_sha256(config['cache']) != checkpoint['rollout_cache_sha256']:
        raise ValueError('Original rollout cache checksum mismatch')
    with np.load(config['reference'], allow_pickle=False) as data:
        reference = {key: data[key].copy() for key in data.files}
    selected = np.asarray([str(sample).startswith(config['shard_name'] + '/') for sample in reference['sample_ids']])
    ids = reference['sample_ids'][selected]
    if not len(ids):
        raise ValueError('No original validation frames in selected shard')
    arrays = torch.load(config['cache'], map_location='cpu', weights_only=False)['arrays']
    cache_index = {str(sample): index for index, sample in enumerate(arrays['sample_ids']) if arrays['split'][index]}
    if any(str(sample) not in cache_index for sample in ids):
        raise ValueError('Original validation frame missing from rollout cache')
    device = torch.device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    temporal, gnn = load_models(config, device)
    detector = TorchCornerDetector(device=config['device'])
    with ExperimentLogger(team='graph', model='leaderboard_raw_replay', config=config,
                          seed=config['model_seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        current_episode = None
        with tarfile.open(config['raw_shard'], 'r:') as archive:
            members = archive_index(archive, ids)
            for local_index, sample in enumerate(ids):
                name = str(sample)
                episode = str(Path(name).parent)
                if episode != current_episode:
                    if not name.endswith('frame_000000.npz'):
                        raise ValueError('Every replay episode must start at frame zero')
                    tracker = DetectedCornerTracker(temporal, config['gate_m'], device)
                    current_episode = episode
                    expected_frame = 0
                if Path(name).name != f'frame_{expected_frame:06d}.npz':
                    raise ValueError(f'Original episode has a frame gap or ordering error: {name}')
                expected_frame += 1
                rgb, depth, intrinsic, extrinsic, robot, gt = load_raw(archive, members[name])
                detection = detector(rgb, depth, intrinsic, extrinsic)
                target = object21_from_gt(gt).to(device)
                original_target = reference['target'][selected][local_index]
                if not np.array_equal(target[0].cpu().numpy(), original_target):
                    raise ValueError(f'Original Object21 GT mismatch: {name}')
                output_model, tracked = predict_frame(gnn, tracker, detection, robot, device)
                index = cache_index[name]
                mask = arrays['valid'][index]
                kinds = np.asarray([track.kind for track in tracker.tracks])
                tracking = track_difference(tracked, kinds, arrays['xyz'][index, mask],
                                            arrays['kinds'][index, mask], arrays['observed'][index, mask])
                if output_model is None:
                    row = dict(sample_id=name, available=False, tracking=tracking)
                else:
                    errors = {key: float(value[0]) for key, value in per_frame_errors(output_model, target).items()}
                    prediction = output_model['object21'][0].cpu().numpy()
                    difference = np.abs(prediction-reference['prediction'][selected][local_index])
                    row = dict(sample_id=name, available=True, tracking=tracking, errors=errors,
                               prediction=prediction.tolist(), max_prediction_difference=float(difference.max()))
                rows.append(row)
                if (local_index+1) % 100 == 0:
                    print(f'Original raw replay: {local_index+1}/{len(ids)} frames', flush=True)
        scored = [row for row in rows if row['available']]
        metrics = {key: statistics(np.asarray([row['errors'][key] for row in scored]))
                   for key in scored[0]['errors']} if scored else {}
        result = dict(config=config, frames=len(rows), scored_frames=len(scored),
                      original_raw_shard_sha256=config['raw_shard_sha256'], metrics=metrics,
                      reference_hole_mean_mm=float(reference['hole_position_error_mm'][selected].mean()),
                      reference_peg_mean_mm=float(reference['peg_position_error_mm'][selected].mean()),
                      unmatched_track_frames=sum(row['tracking']['unmatched_actual'] > 0 or
                                                 row['tracking']['unmatched_reference'] > 0 for row in rows),
                      frames_data=rows)
        path = output / 'raw_replay.json'
        path.write_text(json.dumps(result, indent=2))
        logger.record(frames=len(rows), metrics=metrics)
        logger.link_output(path)
        print(f'Original raw replay: {output}', flush=True)
