"""Visualize guarded Hole motion with detector-only tracking."""

import importlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from mani_skill.utils.structs import Pose

from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import HoleMotion
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.fixed_id_corner_video import project_world
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.detected_corner_gnn import graph_batch
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def load_models(config, device):
    """Load the selected GNN and its exact frozen temporal source."""
    gnn_path = Path(config['gnn_checkpoint']).resolve()
    temporal_path = Path(config['temporal_checkpoint']).resolve()
    checkpoint = torch.load(gnn_path, map_location='cpu', weights_only=False)
    source = checkpoint['config']
    if source['seed'] != config['seed'] or source['temporal']['architecture'] != 'transformer':
        raise ValueError('GNN checkpoint must match the configured seed and Transformer architecture')
    source['temporal']['checkpoint'] = str(temporal_path)
    temporal, _ = load_temporal(source, device)
    model = VertexEdgeGatedMPNN(**source['gnn']).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    return temporal, model, file_sha256(gnn_path), file_sha256(temporal_path)


def infer_gnn(model, tracked, tracker, tcp, device):
    """Build the same detector-only graph used for leaderboard training."""
    count = len(tracked['positions'])
    if not count:
        return None
    xyz = torch.as_tensor(tracked['positions'], device=device).unsqueeze(0)
    kinds = torch.tensor([[track.kind for track in tracker.tracks]], device=device)
    valid = torch.ones((1, count), dtype=torch.bool, device=device)
    observed = torch.as_tensor(tracked['observed'], device=device).unsqueeze(0)
    tcp_tensor = torch.as_tensor(tcp, device=device).unsqueeze(0)
    graph, source = graph_batch(xyz, kinds, valid, observed, tcp_tensor)
    with torch.inference_mode():
        return model(graph, source)['object21'][0].float().cpu().numpy()


def make_policy(config):
    """Load an optional module:factory policy; default is a zero-action smoke run."""
    target = config.get('policy')
    if target is None:
        return None
    module_name, separator, factory_name = target.partition(':')
    if not separator or not module_name or not factory_name:
        raise ValueError('policy must be null or module:factory')
    return getattr(importlib.import_module(module_name), factory_name)()

def visualize(config):
    """Move a kinematic box, track detected corners, and save an annotated video."""
    frames = int(config['frames'])
    if frames < 2:
        raise ValueError('frames must be at least 2')
    motion = HoleMotion(config['motion'])
    policy = make_policy(config)
    if config['fps'] <= 0 or config['gate_m'] <= 0:
        raise ValueError('fps and gate_m must be positive')
    device = torch.device(config['device'])
    temporal, gnn, gnn_hash, temporal_hash = load_models(config, device)
    tracker = DetectedCornerTracker(temporal, config['gate_m'], device)
    detector = TorchCornerDetector(device=config['detector_device'])
    sensors = {name: {'width': 512, 'height': 512} for name in ('base_camera', 'hand_camera')}
    env = make_peg_insertion_env(seed=config['seed'], obs_mode='rgb+depth', sensor_configs=sensors,
                                 control_mode=config.get('control_mode'), max_episode_steps=frames)
    with ExperimentLogger(team='graph', model='moving_hole_visualization', config=config,
                          seed=config['seed'], job_type='visualization') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        video = output / 'moving_hole.mp4'
        process = None
        rows = []
        try:
            obs, _ = env.reset(seed=config['seed'])
            task = env.unwrapped
            initial = task.box.pose.raw_pose.clone()
            if policy is not None:
                policy.reset(env, config['seed'])
            if config['motion'].get('stop_expert_insert', False) and not hasattr(policy, 'phase'):
                raise ValueError('stop_expert_insert requires a policy exposing its current phase')
            zero_action = torch.zeros(env.action_space.shape, device=task.device)
            episode_ended = False
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                       '-pixel_format', 'bgr24', '-video_size', '512x600', '-framerate', str(config['fps']),
                       '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            for frame in range(frames):
                seconds = frame / task.control_freq
                tip = task.peg_head_pose.p[0].detach().cpu().numpy()
                hole = task.box_hole_pose.p[0].detach().cpu().numpy()
                distance_before = float(np.linalg.norm(tip - hole))
                if (not episode_ended and config['motion'].get('stop_expert_insert', False)
                        and getattr(policy, 'phase', None) == 'insert'):
                    motion.stopped = True
                    motion.status = 'blocked: expert insert phase'
                offset = motion.offset.copy() if episode_ended else motion.update(seconds, tip, hole)
                pose = initial.clone()
                pose[0, :3] += torch.as_tensor(offset, dtype=pose.dtype, device=pose.device)
                task.box.set_pose(Pose.create_from_pq(pose[:, :3], pose[:, 3:]))
                obs = task.get_obs()
                camera = obs['sensor_data']['base_camera']
                params = obs['sensor_param']['base_camera']
                rgb = camera['rgb'][0].cpu().numpy()
                depth = camera['depth'][0].cpu().numpy()
                intrinsic = params['intrinsic_cv'][0].cpu().numpy()
                extrinsic = params['extrinsic_cv'][0].cpu().numpy()
                detected = detector(rgb, depth, intrinsic, extrinsic)
                robot = np.concatenate((obs['extra']['tcp_pose'][0].cpu().numpy(),
                                        obs['agent']['qpos'][0, -2:].cpu().numpy())).astype(np.float32)
                tracked = tracker.step(detected.xyz.cpu().numpy(),
                                       detected.object_id.cpu().numpy(), robot)
                prediction = infer_gnn(gnn, tracked, tracker, robot[:3], device)
                truth = get_gt_state(env).hole_pose[0, :3].cpu().numpy()
                predicted_hole = prediction[10:13] + robot[:3] if prediction is not None else None
                error = float(np.linalg.norm(predicted_hole - truth) * 1000) if predicted_hole is not None else None
                canvas = np.zeros((600, 512, 3), dtype=np.uint8)
                canvas[:512] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for index, (point, observed) in enumerate(zip(tracked['positions'], tracked['observed'])):
                    pixel = project_world(point, intrinsic, extrinsic)
                    if pixel is not None:
                        color = (0, 210, 255) if observed else (255, 180, 0)
                        cv2.circle(canvas, pixel, 3, color, -1)
                        cv2.putText(canvas, str(index), (pixel[0] + 3, pixel[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
                for point, color in ((truth, (0, 255, 0)), (predicted_hole, (255, 0, 255))):
                    if point is not None:
                        pixel = project_world(point, intrinsic, extrinsic)
                        if pixel is not None:
                            cv2.drawMarker(canvas, pixel, color, cv2.MARKER_CROSS, 15, 2)
                lines = [f'{seconds:.2f}s  {motion.status}  shift {np.linalg.norm(offset) * 1000:.1f} mm',
                         'Green: GT hole  Magenta: predicted hole',
                         f'Tracks: yellow detected / blue predicted  expert: {getattr(policy, "phase", "none")}',
                         f'Hole error: {error:.1f} mm  tracks: {len(tracker.tracks)}  new: {tracked["new_tracks"]}' if error is not None else 'No prediction']
                for index, line in enumerate(lines):
                    cv2.putText(canvas, line, (7, 531 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (235, 235, 235), 1, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
                rows.append({'frame': frame, 'hole_gt': truth.tolist(), 'hole_prediction': predicted_hole.tolist() if predicted_hole is not None else None,
                             'hole_error_mm': error, 'new_tracks': int(tracked['new_tracks']), 'tracks': len(tracker.tracks),
                             'time_seconds': seconds, 'peg_tip': tip.tolist(), 'pre_move_distance_m': distance_before,
                             'offset_m': offset.tolist(), 'motion_status': motion.status,
                             'expert_phase': getattr(policy, 'phase', None),
                             'success': bool(get_gt_state(env).success[0])})
                if frame % 10 == 0:
                    print(f'frame {frame + 1}/{frames}, tracks {len(tracker.tracks)}, hole error {error}', flush=True)
                if episode_ended:
                    print('Episode ended; terminal frame recorded.', flush=True)
                    break
                if frame + 1 < frames:
                    action = zero_action if policy is None else policy.act(obs)
                    action = torch.as_tensor(action, dtype=torch.float32, device=task.device)
                    if action.shape != zero_action.shape or not torch.isfinite(action).all():
                        raise ValueError(f'policy must return finite actions with shape {tuple(zero_action.shape)}')
                    obs, _, terminated, truncated, _ = env.step(action)
                    episode_ended = bool(terminated.any() or truncated.any())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('ffmpeg video encoding failed')
            process = None
            summary = {'config': config, 'gnn_sha256': gnn_hash, 'temporal_sha256': temporal_hash,
                       'sim_backend': str(task.backend), 'control_freq_hz': task.control_freq,
                       'policy': config.get('policy') or 'zero_actions', 'frames': rows}
            summary_path = output / 'results.json'
            summary_path.write_text(json.dumps(summary, indent=2))
            logger.record(gnn_sha256=gnn_hash, temporal_sha256=temporal_hash, frame_count=len(rows))
            logger.link_output(video)
            logger.link_output(summary_path)
            print(f'Video: {video}', flush=True)
        finally:
            if process is not None:
                process.stdin.close()
                process.wait()
            env.close()
