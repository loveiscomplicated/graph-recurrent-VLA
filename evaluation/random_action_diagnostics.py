"""Paired expert and random-action diagnostics for a frozen corner/GNN pipeline."""

import json
from pathlib import Path
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mani_skill.utils import sapien_utils

from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.vertex_family_labels import SIGNS, physical_references, pose_rotation
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.fixed_id_corner_video import project_world
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.detected_corner_gnn import graph_batch
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def load_models(config, device):
    """Verify the frozen temporal identity recorded by the selected GNN."""
    gnn_path = Path(config['gnn_checkpoint'])
    checkpoint = torch.load(gnn_path, map_location='cpu', weights_only=False)
    source = checkpoint['config']
    if source['seed'] != config['model_seed'] or source['temporal']['architecture'] != 'transformer':
        raise ValueError('GNN checkpoint must match model_seed and use Transformer')
    source['temporal']['checkpoint'] = config['temporal_checkpoint']
    temporal, _ = load_temporal(source, device)
    gnn = VertexEdgeGatedMPNN(**source['gnn']).to(device).eval()
    gnn.load_state_dict(checkpoint['model'], strict=True)
    return temporal, gnn


def references(state):
    """Return the 8 peg and 16 box/aperture GT vertices, for diagnostics only."""
    truth = {name: getattr(state, name)[0].cpu().numpy() for name in
             ('peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes', 'hole_radius')}
    peg = SIGNS * truth['peg_half_sizes'] @ pose_rotation(truth['peg_pose']).T + truth['peg_pose'][:3]
    box = np.asarray([item['world'] for item in physical_references(truth)])
    return np.concatenate((peg, box)).astype(np.float32)


def nearest_references(positions, kinds, truth):
    """Find nearest same-kind reference independently; duplicates remain visible."""
    identities = []
    distances = []
    for position, kind in zip(positions, kinds):
        indices = np.arange(0, 8) if kind == 0 else np.arange(8, 24)
        local = np.linalg.norm(truth[indices] - position, axis=1)
        index = int(np.argmin(local))
        identities.append(int(indices[index]))
        distances.append(float(local[index] * 1000))
    return identities, distances


def predict(gnn, positions, kinds, observed, tcp, device):
    """Run the unchanged GNN on supplied nodes with their original provenance."""
    if len(positions) == 0:
        return None
    xyz = torch.as_tensor(positions, dtype=torch.float32, device=device).unsqueeze(0)
    kinds = torch.as_tensor(kinds, dtype=torch.long, device=device).unsqueeze(0)
    valid = torch.ones_like(kinds, dtype=torch.bool)
    observed = torch.as_tensor(observed, dtype=torch.bool, device=device).unsqueeze(0)
    tcp = torch.as_tensor(tcp, dtype=torch.float32, device=device).unsqueeze(0)
    graph, provenance = graph_batch(xyz, kinds, valid, observed, tcp)
    with torch.inference_mode():
        return gnn(graph, provenance)['object21'][0].float().cpu().numpy()


def errors(prediction, tcp, state):
    """Calculate Peg and Hole position errors in world coordinates."""
    if prediction is None:
        return {'peg': None, 'hole': None}
    peg = state.peg_pose[0, :3].cpu().numpy()
    hole = state.hole_pose[0, :3].cpu().numpy()
    return {'peg': float(np.linalg.norm(prediction[:3] + tcp - peg) * 1000),
            'hole': float(np.linalg.norm(prediction[10:13] + tcp - hole) * 1000)}


def annotate(rgb, frame, phase, tracked, intrinsics, extrinsics, result):
    """Draw the unmodified pipeline's point provenance and position errors."""
    image = np.zeros((590, 512, 3), dtype=np.uint8)
    image[:512] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for point, observed in zip(tracked['positions'], tracked['observed']):
        pixel = project_world(point, intrinsics, extrinsics)
        if pixel is not None:
            cv2.circle(image, pixel, 3, (0, 210, 255) if observed else (255, 180, 0), -1)
    for point, color in ((result['peg_gt'], (0, 255, 0)), (result['peg_prediction'], (255, 0, 255)),
                         (result['hole_gt'], (0, 180, 0)), (result['hole_prediction'], (200, 0, 200))):
        if point is not None:
            pixel = project_world(np.asarray(point), intrinsics, extrinsics)
            if pixel is not None:
                cv2.drawMarker(image, pixel, color, cv2.MARKER_CROSS, 15, 2)
    lines = [f'frame={frame}  {phase}  grasp={result["grasp"]}',
             'Green=GT / Magenta=estimate; yellow=detected / blue=predicted',
             f'Peg={result["errors"]["original"]["peg"]:.1f}mm  Hole={result["errors"]["original"]["hole"]:.1f}mm',
             f'tracks={result["tracks"]}  observed={result["detections"]}  new={result["new_tracks"]}']
    for index, line in enumerate(lines):
        cv2.putText(image, line, (6, 530 + index * 17), cv2.FONT_HERSHEY_SIMPLEX,
                    .39, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def make_sensor_configs(camera_view):
    """Build the standard sensors, optionally placing the base camera in world coordinates."""
    sensors = {name: {'width': 512, 'height': 512} for name in ('base_camera', 'hand_camera')}
    if camera_view is None:
        return sensors
    for name in ('eye', 'target'):
        value = np.asarray(camera_view.get(name), dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f'camera_view.{name} must be a finite world-coordinate 3-vector')
    if np.allclose(camera_view['eye'], camera_view['target']):
        raise ValueError('camera_view.eye and camera_view.target must differ')
    sensors['base_camera']['pose'] = sapien_utils.look_at(
        eye=camera_view['eye'], target=camera_view['target'],
    )
    return sensors


def run_episode(config, env_seed, action_seed, temporal, gnn, detector, output, source_episode=None):
    """Replay the expert prefix, then continue with expert or random actions."""
    device = torch.device(config['device'])
    tracker = DetectedCornerTracker(temporal, config['gate_m'], device)
    policy = ScriptedPegPolicy()
    rng = np.random.default_rng(action_seed) if action_seed is not None else None
    sensors = make_sensor_configs(config.get('camera_view'))
    env = make_peg_insertion_env(seed=env_seed, obs_mode='rgb+depth', sensor_configs=sensors,
                                 control_mode=config['control_mode'], max_episode_steps=config['max_frames'])
    label = f'env_{env_seed}_' + (f'random_{action_seed}' if rng else 'expert')
    video = output / f'{label}.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pixel_format', 'bgr24', '-video_size', '512x590', '-framerate', str(config['fps']),
               '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    rows = []
    track_ids = []
    switched = False
    steps_after = 0
    try:
        obs, _ = env.reset(seed=env_seed)
        policy.reset(env, env_seed)
        task = env.unwrapped
        initial_z = float(get_gt_state(env).peg_pose[0, 2])
        for frame in range(config['max_frames']):
            state = get_gt_state(env)
            grasp = bool(task.agent.is_grasping(task.peg)[0].item())
            lift = float(state.peg_pose[0, 2]) - initial_z
            if source_episode is not None:
                expected = source_episode['frames'][frame]
                peg = state.peg_pose[0, :3].cpu().numpy()
                hole = state.hole_pose[0, :3].cpu().numpy()
                if not np.allclose(peg, expected['peg_gt'], rtol=0, atol=1e-6):
                    raise ValueError(f'Peg GT differs from source at seed {env_seed}, frame {frame}')
                if not np.allclose(hole, expected['hole_gt'], rtol=0, atol=1e-6):
                    raise ValueError(f'Hole GT differs from source at seed {env_seed}, frame {frame}')
                should_switch = frame == source_episode['switch_frame']
                if should_switch and not (grasp and lift > config['minimum_lift_m']):
                    raise ValueError(f'Grasp/lift differs from source at seed {env_seed}, frame {frame}')
            else:
                should_switch = policy.phase == 'align' and grasp and lift > config['minimum_lift_m']
            if not switched and should_switch:
                switched = True
                switch_frame = frame
            sensor = obs['sensor_data']['base_camera']
            params = obs['sensor_param']['base_camera']
            rgb = sensor['rgb'][0].cpu().numpy()
            depth = sensor['depth'][0].cpu().numpy()
            intrinsic = params['intrinsic_cv'][0].cpu().numpy()
            extrinsic = params['extrinsic_cv'][0].cpu().numpy()
            detected = detector(rgb, depth, intrinsic, extrinsic)
            detection_xyz = detected.xyz.cpu().numpy()
            detection_kinds = detected.object_id.cpu().numpy()
            robot = np.concatenate((obs['extra']['tcp_pose'][0].cpu().numpy(),
                                    obs['agent']['qpos'][0, -2:].cpu().numpy())).astype(np.float32)
            tracked = tracker.step(detection_xyz, detection_kinds, robot)
            kinds = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
            truth = references(state)
            nearest_ids, detection_distances = nearest_references(detection_xyz, detection_kinds, truth)
            positions = tracked['positions']
            for index in range(len(track_ids), len(positions)):
                ids, distances = nearest_references(positions[index:index + 1], kinds[index:index + 1], truth)
                track_ids.append(ids[0] if distances[0] <= config['identity_birth_gate_m'] * 1000 else None)
            fixed_distances = [None if identity is None else float(np.linalg.norm(position - truth[identity]) * 1000)
                               for position, identity in zip(positions, track_ids)]
            _, nearest_track_distances = nearest_references(positions, kinds, truth)
            observed = tracked['observed']
            variants = {'original': positions}
            for name in ('correct_observed', 'correct_predicted', 'correct_all'):
                corrected = positions.copy()
                for index, identity in enumerate(track_ids):
                    selected = name == 'correct_all' or (name == 'correct_observed' and observed[index]) or (name == 'correct_predicted' and not observed[index])
                    if selected and identity is not None:
                        corrected[index] = truth[identity]
                variants[name] = corrected
            predictions = {name: predict(gnn, value, kinds, observed, robot[:3], device)
                           for name, value in variants.items()}
            assigned = np.asarray([identity is not None for identity in track_ids], dtype=bool)
            predictions['assigned_gt_only'] = predict(
                gnn, variants['correct_all'][assigned], kinds[assigned], observed[assigned], robot[:3], device,
            )
            predictions['assigned_gt_as_observed'] = predict(
                gnn, variants['correct_all'][assigned], kinds[assigned],
                np.ones(int(assigned.sum()), dtype=bool), robot[:3], device,
            )
            # This idealized graph also reveals hidden vertices; it is out of the
            # training distribution and cannot be interpreted as deployable input.
            predictions['complete_gt'] = predict(
                gnn, truth, np.asarray([0] * 8 + [1] * 16), np.ones(24, dtype=bool), robot[:3], device,
            )
            metrics = {name: errors(value, robot[:3], state) for name, value in predictions.items()}
            prediction = predictions['original']
            phase = ('random' if rng else 'expert_control') if switched else policy.phase
            if source_episode is not None:
                phase = source_episode['frames'][frame]['phase']
            result = dict(frame=frame, relative_step=steps_after if switched else None, phase=phase,
                          grasp=grasp, lift_m=lift, tracks=len(positions), detections=int(observed.sum()),
                          new_tracks=int(tracked['new_tracks']), detection_nearest_mm=detection_distances,
                          detection_reference_ids=nearest_ids, track_fixed_mm=fixed_distances,
                          track_nearest_mm=nearest_track_distances, track_reference_ids=track_ids.copy(),
                          track_observed=observed.tolist(), errors=metrics,
                          peg_gt=state.peg_pose[0,:3].cpu().numpy().tolist(),
                          hole_gt=state.hole_pose[0,:3].cpu().numpy().tolist(),
                          peg_prediction=(prediction[:3] + robot[:3]).tolist() if prediction is not None else None,
                          hole_prediction=(prediction[10:13] + robot[:3]).tolist() if prediction is not None else None)
            rows.append(result)
            if prediction is not None:
                process.stdin.write(annotate(rgb, frame, phase, tracked, intrinsic, extrinsic, result).tobytes())
            else:
                process.stdin.write(np.zeros((590, 512, 3), dtype=np.uint8).tobytes())
            if switched and steps_after >= config['random_steps']:
                break
            if source_episode is not None:
                action = torch.as_tensor(source_episode['frames'][frame]['action'], dtype=torch.float32,
                                         device=task.device).reshape(env.action_space.shape)
            elif switched and rng is not None:
                action = np.concatenate((rng.uniform(-config['random_arm_limit'], config['random_arm_limit'], 6), [-1])).astype(np.float32)
                action = torch.as_tensor(action, device=task.device).reshape(env.action_space.shape)
            else:
                action = policy.act(obs)
            result['action'] = action.cpu().numpy().reshape(-1).tolist()
            obs, _, terminated, truncated, _ = env.step(action)
            if switched:
                steps_after += 1
            if bool(terminated.any() or truncated.any()):
                break
    finally:
        process.stdin.close()
        status = process.wait()
        env.close()
    if status:
        raise RuntimeError(f'ffmpeg failed for {label}')
    print(f'{label}: switched={switched}, steps={steps_after}, frames={len(rows)}', flush=True)
    return dict(environment_seed=env_seed, action_seed=action_seed, switched=switched,
                switch_frame=switch_frame if switched else None, steps_after=steps_after,
                video=str(video), frames=rows)


def valid_frames(episode):
    """Exclude the common switch frame and retain the requested post-action frames."""
    return [frame for frame in episode['frames'] if frame['relative_step'] is not None
            and frame['relative_step'] > 0]


def episode_summary(episode):
    """Compute frame-averaged GNN errors and node-weighted diagnostic errors."""
    frames = valid_frames(episode)
    result = {'frames': len(frames), 'grasp_frames': sum(frame['grasp'] for frame in frames)}
    for variant in ('original', 'correct_observed', 'correct_predicted', 'correct_all',
                    'assigned_gt_only', 'assigned_gt_as_observed', 'complete_gt'):
        for object_name in ('peg', 'hole'):
            values = [frame['errors'][variant][object_name] for frame in frames
                      if frame['errors'][variant][object_name] is not None]
            result[f'{variant}_{object_name}_mean_mm'] = float(np.mean(values)) if values else None
    for kind in ('detected', 'predicted'):
        values = [distance for frame in frames
                  for distance, observed in zip(frame['track_fixed_mm'], frame['track_observed'])
                  if distance is not None and observed == (kind == 'detected')]
        result[f'{kind}_fixed_rmse_mm'] = float(np.sqrt(np.mean(np.square(values)))) if values else None
        result[f'{kind}_fixed_count'] = len(values)
    values = [distance for frame in frames for distance in frame['detection_nearest_mm']]
    result['detection_nearest_rmse_mm'] = float(np.sqrt(np.mean(np.square(values)))) if values else None
    result['detection_count'] = len(values)
    result['unassigned_track_nodes'] = sum(sum(value is None for value in frame['track_reference_ids']) for frame in frames)
    result['track_nodes'] = sum(frame['tracks'] for frame in frames)
    return result


def plot_results(episodes, output):
    """Plot per-episode effects and per-step position error for both policies."""
    complete = [episode for episode in episodes if episode['switched'] and episode['steps_after'] > 0]
    random = [episode for episode in complete if episode['action_seed'] is not None]
    expert = [episode for episode in complete if episode['action_seed'] is None]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for group, color, label in ((random, '#d65043', 'Random'), (expert, '#2474b7', 'Expert')):
        for object_name, axis in (('peg', axes[0, 0]), ('hole', axes[0, 1])):
            for episode in group:
                frames = valid_frames(episode)
                axis.plot([frame['relative_step'] for frame in frames],
                          [frame['errors']['original'][object_name] for frame in frames],
                          color=color, alpha=.24, linewidth=1)
            steps = sorted({frame['relative_step'] for episode in group for frame in valid_frames(episode)})
            means = [np.mean([frame['errors']['original'][object_name] for episode in group
                              for frame in valid_frames(episode) if frame['relative_step'] == step
                              and frame['errors']['original'][object_name] is not None]) for step in steps]
            axis.plot(steps, means, color=color, linewidth=2.5, label=label)
            axis.set_title(f'{object_name.title()} position: actual pipeline')
            axis.set_ylabel('3D position error (mm)')
            axis.set_xlabel('Steps after lift')
            axis.grid(alpha=.2)
    axes[0, 0].legend()
    labels = ['Original', 'GT detected', 'GT predicted', 'GT both']
    keys = ['original', 'correct_observed', 'correct_predicted', 'correct_all']
    for object_name, axis in (('peg', axes[1, 0]), ('hole', axes[1, 1])):
        for episode in random:
            summary = episode['summary']
            values = [summary[f'{key}_{object_name}_mean_mm'] for key in keys]
            axis.plot(range(4), values, color='#d65043', alpha=.35, marker='o')
        if random:
            mean = [np.mean([episode['summary'][f'{key}_{object_name}_mean_mm'] for episode in random]) for key in keys]
            axis.plot(range(4), mean, color='#a11b12', marker='o', linewidth=3, label='Episode mean')
        axis.set_xticks(range(4), labels, rotation=12)
        axis.set_title(f'{object_name.title()}: GT coordinate substitutions')
        axis.set_ylabel('Mean 3D position error (mm)')
        axis.grid(alpha=.2)
    figure.suptitle('Random-action diagnostics: thin lines are individual episodes')
    path = output / 'diagnostics.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def evaluate(config):
    """Run paired trajectories, retain failures, and save diagnostics and videos."""
    if not config['environment_seeds'] or not config['action_seeds']:
        raise ValueError('environment_seeds and action_seeds must be nonempty')
    if config['random_steps'] < 1 or config['identity_birth_gate_m'] <= 0:
        raise ValueError('random_steps and identity_birth_gate_m must be positive')
    torch.set_num_threads(2)
    device = torch.device(config['device'])
    temporal, gnn = load_models(config, device)
    detector = TorchCornerDetector(device=config['detector_device'])
    source_episodes = None
    if config.get('trajectory_source'):
        source = json.loads(Path(config['trajectory_source']).read_text())
        source_episodes = {(episode['environment_seed'], episode['action_seed']): episode
                           for episode in source['episodes']}
    with ExperimentLogger(team='graph', model='random_action_diagnostics', config=config,
                          seed=config['model_seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for env_seed in config['environment_seeds']:
            action_seeds = list(config['action_seeds'])
            if config.get('include_expert_control', True):
                action_seeds.insert(0, None)
            for action_seed in action_seeds:
                source_episode = None
                if source_episodes is not None:
                    source_episode = source_episodes.get((env_seed, action_seed))
                    if source_episode is None or not source_episode['switched']:
                        raise ValueError(f'No lifted source trajectory for seed {env_seed}, action {action_seed}')
                    if source_episode['random_steps'] < config['random_steps']:
                        raise ValueError(f'Source trajectory is too short for seed {env_seed}, action {action_seed}')
                episode = run_episode(config, env_seed, action_seed, temporal, gnn, detector, output, source_episode)
                episode['summary'] = episode_summary(episode)
                episodes.append(episode)
                # Persist progress so interrupted experiments retain completed episodes.
                (output / 'episodes.json').write_text(json.dumps(episodes, indent=2))
        figure = plot_results(episodes, output)
        stage_figure = plot_stage_results(episodes, output)
        result = dict(config=config, episodes=episodes,
                      gnn_sha256=file_sha256(config['gnn_checkpoint']),
                      temporal_sha256=file_sha256(config['temporal_checkpoint']),
                      description='GT is used only for diagnosis, switch eligibility and substitutions; not for the original model input.')
        result_path = output / 'results.json'
        result_path.write_text(json.dumps(result, indent=2))
        logger.record(episodes=len(episodes), gnn_sha256=result['gnn_sha256'], temporal_sha256=result['temporal_sha256'])
        for path in (result_path, figure, stage_figure):
            logger.link_output(path)
        for episode in episodes:
            logger.link_output(episode['video'], kind='rollouts')
        print(f'Diagnostics: {output}', flush=True)


def plot_stage_results(episodes, output):
    """Show the corner diagnostics and GT substitutions per random rollout."""
    random = [episode for episode in episodes if episode['action_seed'] is not None
              and episode['summary']['frames'] > 0]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    corner_keys = ['detection_nearest_rmse_mm', 'detected_fixed_rmse_mm', 'predicted_fixed_rmse_mm']
    for episode in random:
        values = [episode['summary'][key] for key in corner_keys]
        axes[0].plot(range(3), [np.nan if value is None else value for value in values],
                     color='#3175ab', alpha=.38, marker='o')
    axes[0].set_xticks(range(3), ['Detector → nearest GT', 'Observed → birth GT', 'Predicted → birth GT'], rotation=12)
    axes[0].set_title('Node-weighted 3D corner RMSE per episode')
    axes[0].set_ylabel('RMSE (mm)')
    axes[0].grid(alpha=.2)
    keys = ['original', 'correct_predicted', 'correct_all', 'assigned_gt_only', 'complete_gt']
    for episode in random:
        values = [episode['summary'][f'{key}_peg_mean_mm'] for key in keys]
        axes[1].plot(range(5), [np.nan if value is None else value for value in values],
                     color='#cf5148', alpha=.38, marker='o')
    axes[1].set_xticks(range(5), ['Original', 'GT predicted', 'GT assigned', 'GT assigned only', 'GT all 24'], rotation=15)
    axes[1].set_title('Peg error under GT interventions (not deployment)')
    axes[1].set_ylabel('Mean 3D position error (mm)')
    axes[1].grid(alpha=.2)
    path = output / 'stage_diagnostics.png'
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
