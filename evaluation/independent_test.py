"""Evaluate frozen Transformer/GNNs on an independently collected RGB-D test set."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from evaluation.random_action_diagnostics import load_models
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.utils.object21_targets import object21_from_gt
from training.detected_corner_gnn import graph_batch
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def read_observation(path):
    """Load native raw camera and robot arrays, keeping GT separate from inputs."""
    with np.load(path, allow_pickle=False) as frame:
        prefix = 'observation/sensor_data/base_camera/'
        camera = 'observation/sensor_param/base_camera/'
        rgb = frame[prefix + 'rgb'][0].copy()
        depth = frame[prefix + 'depth'][0].copy()
        intrinsic = frame[camera + 'intrinsic_cv'][0].copy()
        extrinsic = frame[camera + 'extrinsic_cv'][0].copy()
        tcp = frame['observation/extra/tcp_pose'][0].copy()
        fingers = frame['observation/agent/qpos'][0, -2:].copy()
        ground_truth = {key.removeprefix('ground_truth/'): torch.as_tensor(frame[key].copy())
                        for key in frame.files if key.startswith('ground_truth/')}
        after_lift = int(frame['test_frame/after_lift'])
        grasp = bool(frame['test_frame/grasp'])
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'Expected native uint8 RGB at {path}')
    if depth.shape not in ((512, 512), (512, 512, 1)) or depth.dtype != np.int16:
        raise ValueError(f'Expected native int16 depth at {path}')
    robot = np.concatenate((tcp, fingers)).astype(np.float32)
    return (rgb, depth, intrinsic, extrinsic, robot), ground_truth, after_lift, grasp


def initialize_models(protocol, device, gate_m):
    """Load each protocol-pinned pair using the existing checkpoint validation."""
    models = {}
    for item in protocol['models']:
        if file_sha256(item['gnn']) != item['gnn_sha256'] or file_sha256(item['temporal']) != item['temporal_sha256']:
            raise ValueError(f'Protocol checkpoint hash mismatch for model seed {item["seed"]}')
        settings = dict(model_seed=item['seed'], gnn_checkpoint=item['gnn'], temporal_checkpoint=item['temporal'])
        temporal, gnn = load_models(settings, device)
        models[item['seed']] = dict(temporal=temporal, gnn=gnn, gate_m=gate_m)
    return models


def verify_expert_random_prefix(manifest, root, seeds):
    """Require identical recorded observations and GT through the lift switch."""
    rows = {(item['seed'], item['mode']): item for item in manifest['episodes']}
    fields = ('observation/sensor_data/base_camera/rgb', 'observation/sensor_data/base_camera/depth',
              'observation/sensor_param/base_camera/intrinsic_cv',
              'observation/sensor_param/base_camera/extrinsic_cv', 'observation/agent/qpos',
              'observation/extra/tcp_pose', 'ground_truth/peg_pose', 'ground_truth/hole_pose')
    checked = 0
    for seed in seeds:
        expert = rows[(seed, 'expert')]
        random = rows[(seed, 'random_after_lift')]
        if expert['switch_frame'] != random['switch_frame']:
            raise ValueError(f'Expert and random switch frames differ for seed {seed}')
        end = expert['switch_frame']
        if end is None:
            end = min(expert['observations'], random['observations']) - 1
        for index in range(end + 1):
            name = f'frame_{index:06d}.npz'
            with np.load(root / expert['path'] / name, allow_pickle=False) as left:
                with np.load(root / random['path'] / name, allow_pickle=False) as right:
                    for field in fields:
                        if not np.array_equal(left[field], right[field]):
                            raise ValueError(f'Expert/random prefix differs for seed {seed}, frame {index}, {field}')
                    if index < end and not np.array_equal(left['action'], right['action']):
                        raise ValueError(f'Expert/random prefix action differs for seed {seed}, frame {index}')
            checked += 1
    return checked


def predict_frame(gnn, tracker, detection, robot, device):
    """Feed detector coordinates through the unchanged tracker and graph/GNN."""
    tracked = tracker.step(detection.xyz.cpu().numpy(), detection.object_id.cpu().numpy(), robot)
    count = len(tracked['positions'])
    if not count:
        return None, tracked
    xyz = torch.as_tensor(tracked['positions'], dtype=torch.float32, device=device).unsqueeze(0)
    kinds = torch.as_tensor([track.kind for track in tracker.tracks], dtype=torch.long, device=device).unsqueeze(0)
    valid = torch.ones((1, count), dtype=torch.bool, device=device)
    observed = torch.as_tensor(tracked['observed'], device=device).unsqueeze(0)
    tcp = torch.as_tensor(robot[:3], dtype=torch.float32, device=device).unsqueeze(0)
    graph, features = graph_batch(xyz, kinds, valid, observed, tcp)
    with torch.inference_mode():
        return gnn(graph, features), tracked


def evaluate_episode(receipt, root, models, detector, device, output):
    """Detect once per saved frame and evaluate all temporal/GNN seeds on it."""
    trackers = {seed: DetectedCornerTracker(model['temporal'], model['gate_m'], device)
                for seed, model in models.items()}
    frames = []
    arrays = {seed: {'prediction': [], 'target': [], 'available': []} for seed in models}
    for frame_number, expected_sha in enumerate(receipt['frame_sha256']):
        path = root / receipt['path'] / f'frame_{frame_number:06d}.npz'
        if file_sha256(path) != expected_sha:
            raise ValueError(f'Raw frame checksum mismatch: {path}')
        obs, gt, after_lift, grasp = read_observation(path)
        rgb, depth, intrinsic, extrinsic, robot = obs
        detection = detector(rgb, depth, intrinsic, extrinsic)
        target = object21_from_gt(gt).to(device)
        row = dict(frame=frame_number, after_lift=after_lift, grasp=grasp,
                   detections=len(detection.xyz), models={})
        for seed, model in models.items():
            prediction, tracked = predict_frame(model['gnn'], trackers[seed], detection, robot, device)
            arrays[seed]['target'].append(target[0].cpu().numpy())
            arrays[seed]['available'].append(prediction is not None)
            if prediction is None:
                arrays[seed]['prediction'].append(np.full(21, np.nan, dtype=np.float32))
                row['models'][str(seed)] = dict(available=False, tracks=0, observed=0, errors=None)
                continue
            values = per_frame_errors(prediction, target)
            errors = {name: float(value[0]) for name, value in values.items()}
            arrays[seed]['prediction'].append(prediction['object21'][0].cpu().numpy())
            row['models'][str(seed)] = dict(available=True, tracks=len(tracked['positions']),
                                           observed=int(tracked['observed'].sum()), errors=errors)
        frames.append(row)
    target_path = output / f'{receipt["mode"]}_seed_{receipt["seed"]}.npz'
    packed = {}
    for seed, values in arrays.items():
        for name, rows in values.items():
            packed[f'{name}_{seed}'] = np.asarray(rows)
    np.savez_compressed(target_path, **packed)
    return dict(seed=receipt['seed'], mode=receipt['mode'], switched=receipt['switched'],
                stop_reason=receipt['stop_reason'], frames=frames, arrays=str(target_path))


def summarize(episodes, model_seeds):
    """Summarize all frames and post-lift frames without hiding missing outputs."""
    result = {}
    for mode in ('expert', 'random_after_lift'):
        group = [episode for episode in episodes if episode['mode'] == mode]
        result[mode] = {}
        for period in ('all', 'post_lift'):
            frames = [frame for episode in group for frame in episode['frames']
                      if period == 'all' or frame['after_lift'] > 0]
            metrics = {}
            valid = [frame for frame in frames if all(frame['models'][str(seed)]['available'] for seed in model_seeds)]
            metric_names = list(valid[0]['models'][str(model_seeds[0])]['errors']) if valid else []
            for name in metric_names:
                values = np.asarray([np.mean([frame['models'][str(seed)]['errors'][name]
                                             for seed in model_seeds]) for frame in valid])
                metrics[name] = statistics(values)
            result[mode][period] = dict(frames=len(frames), scored_frames=len(valid),
                                        missing_any_model=len(frames)-len(valid), metrics=metrics)
    return result


def evaluate(config):
    """Verify the frozen dataset and run one detector with all three model pairs."""
    if file_sha256(config['protocol']) != config['protocol_sha256']:
        raise ValueError('Frozen test protocol SHA256 mismatch')
    protocol = json.loads(Path(config['protocol']).read_text())
    root = Path(config['raw_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE' or manifest['protocol_sha256'] != config['protocol_sha256']:
        raise ValueError('Test dataset is incomplete or has a different protocol')
    expected = {(seed, mode) for seed in protocol['seeds'] for mode in protocol['policy']['modes']}
    actual = {(row['seed'], row['mode']) for row in manifest['episodes']}
    if expected != actual or len(actual) != len(manifest['episodes']):
        raise ValueError('Test dataset is missing planned cases or has duplicates')
    prefix_frames = verify_expert_random_prefix(manifest, root, protocol['seeds'])
    device = torch.device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    models = initialize_models(protocol, device, config['gate_m'])
    detector = TorchCornerDetector(device=config['detector_device'])
    with ExperimentLogger(team='graph', model='independent_generalization_test', config=config,
                          seed=protocol['generation']['seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for receipt in manifest['episodes']:
            episode = evaluate_episode(receipt, root, models, detector, device, output)
            episodes.append(episode)
            print(f'Test inference {len(episodes)}/{len(manifest["episodes"])}: '
                  f'{receipt["mode"]} seed={receipt["seed"]}', flush=True)
        result = dict(config=config, protocol_sha256=config['protocol_sha256'],
                      raw_manifest_sha256=file_sha256(root / 'manifest.json'),
                      paired_prefix_frames_verified=prefix_frames,
                      aggregation='per-frame mean of three model-seed errors, then frame-weighted statistics',
                      summary=summarize(episodes, list(models)), episodes=episodes)
        path = output / 'results.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False))
        logger.record(protocol_sha256=config['protocol_sha256'], episodes=len(episodes), summary=result['summary'])
        logger.link_output(path)
        for episode in episodes:
            logger.link_output(episode['arrays'])
        print(f'Independent test: {output}', flush=True)
