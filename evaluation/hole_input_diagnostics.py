"""Replay one trajectory and inspect the exact detector-to-GNN hole inputs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.fixed_id_corner_video import project_world
from evaluation.random_action_diagnostics import load_models, make_sensor_configs, nearest_references, references
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.detected_corner_gnn import graph_batch
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def run_graph(gnn, xyz, kinds, observed, tcp):
    """Run the unmodified GNN and extract its target anchor and pooling weights."""
    if not len(xyz):
        return None
    xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32).unsqueeze(0)
    kind_tensor = torch.as_tensor(kinds, dtype=torch.long).unsqueeze(0)
    mask = torch.ones_like(kind_tensor, dtype=torch.bool)
    observed_tensor = torch.as_tensor(observed, dtype=torch.bool).unsqueeze(0)
    tcp_tensor = torch.as_tensor(tcp, dtype=torch.float32).unsqueeze(0)
    graph, provenance = graph_batch(xyz_tensor, kind_tensor, mask, observed_tensor, tcp_tensor)
    attention_logits = []
    handle = gnn.hole_pool.attention.register_forward_hook(
        lambda module, inputs, output: attention_logits.append(output.detach().clone())
    )
    try:
        with torch.inference_mode():
            output = gnn(graph, provenance)
    finally:
        handle.remove()
    prediction = output['object21'][0].cpu().numpy()
    anchor = output['hole_anchor'][0].cpu().numpy() + tcp
    hole = prediction[10:13] + tcp
    target = kind_tensor[0] == 1
    logits = attention_logits[0][0, :, 0].masked_fill(~target, -torch.inf)
    weights = torch.softmax(logits, dim=0) if target.any() else torch.zeros_like(logits)
    return dict(hole=hole.tolist(), target_anchor=anchor.tolist(),
                head_offset=(hole-anchor).tolist(), attention=weights.cpu().numpy().tolist(),
                edge_count=int(graph.edge_index.shape[1]))


def run_counterfactuals(gnn, xyz, kinds, observed, tcp, truth, nearest, distances):
    """Probe input sensitivity; changes to graph support are not causal isolation."""
    target = kinds == 1
    close = target & (np.asarray(distances) <= 10.0)
    corrected = xyz.copy()
    corrected[close] = truth[np.asarray(nearest, dtype=int)[close]]
    rim = truth[16:24]
    all_xyz = np.concatenate((xyz, rim))
    all_kinds = np.concatenate((kinds, np.ones(len(rim), dtype=np.int64)))
    all_observed = np.concatenate((observed, np.ones(len(rim), dtype=bool)))
    return {
        'original': run_graph(gnn, xyz, kinds, observed, tcp),
        'target_only': run_graph(gnn, xyz[target], kinds[target], observed[target], tcp),
        'detected_only': run_graph(gnn, xyz[observed], kinds[observed], observed[observed], tcp),
        'target_near_gt': run_graph(gnn, corrected, kinds, observed, tcp),
        'add_gt_rim': run_graph(gnn, all_xyz, all_kinds, all_observed, tcp),
    }


def probe_aperture_centers(truth, depth, intrinsic, extrinsic):
    """Compare measured depth with the two GT aperture-plane center projections."""
    results = []
    for start in (16, 20):
        center = truth[start:start + 4].mean(axis=0)
        camera = extrinsic[:, :3] @ center + extrinsic[:, 3]
        pixel = project_world(center, intrinsic, extrinsic)
        raw = int(depth[pixel[1], pixel[0]].item()) if pixel is not None else None
        measured = raw * 0.001 if raw is not None and 0 < raw < 4000 else None
        results.append(dict(reference_indices=list(range(start, start + 4)),
                            world=center.tolist(), pixel=list(pixel) if pixel else None,
                            expected_depth_m=float(camera[2]), measured_depth_m=measured,
                            measured_minus_expected_m=measured - float(camera[2]) if measured is not None else None))
    return results


def aperture_geometry(truth, intrinsic, extrinsic):
    """Measure the near aperture's projected size and angle from the camera."""
    face = truth[16:20]
    camera_points = face @ extrinsic[:, :3].T + extrinsic[:, 3]
    homogeneous = camera_points @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    near_center = face.mean(axis=0)
    axis = truth[20:24].mean(axis=0) - near_center
    axis /= np.linalg.norm(axis)
    eye = -extrinsic[:, :3].T @ extrinsic[:, 3]
    ray = eye - near_center
    ray /= np.linalg.norm(ray)
    cosine = np.clip(abs(float(axis @ ray)), 0, 1)
    return dict(near_rim_pixels=pixels.tolist(),
                bounding_box_px=np.ptp(pixels, axis=0).tolist(),
                normal_view_angle_degrees=float(np.degrees(np.arccos(cosine))))


def build_snapshot(frame, obs, state, detector, tracker, gnn):
    """Capture detector results, graph inputs and independent GT references."""
    cam = obs['sensor_data']['base_camera']
    param = obs['sensor_param']['base_camera']
    rgb = cam['rgb'][0].cpu().numpy()
    depth = cam['depth'][0].cpu().numpy()
    intrinsic = param['intrinsic_cv'][0].cpu().numpy()
    extrinsic = param['extrinsic_cv'][0].cpu().numpy()
    detection = detector(rgb, depth, intrinsic, extrinsic)
    xyz = detection.xyz.cpu().numpy()
    object_ids = detection.object_id.cpu().numpy()
    robot = np.concatenate((obs['extra']['tcp_pose'][0].cpu().numpy(),
                            obs['agent']['qpos'][0, -2:].cpu().numpy())).astype(np.float32)
    tracked = tracker.step(xyz, object_ids, robot)
    kinds = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
    truth = references(state)
    near_detection, det_distance = nearest_references(xyz, object_ids, truth)
    near_tracks, node_distance = nearest_references(tracked['positions'], kinds, truth)
    outcomes = run_counterfactuals(gnn, tracked['positions'], kinds, tracked['observed'], robot[:3],
                                  truth, near_tracks, node_distance)
    result = dict(frame=frame, robot=robot.tolist(), peg_gt=state.peg_pose[0,:3].cpu().numpy().tolist(),
                  hole_gt=state.hole_pose[0,:3].cpu().numpy().tolist(), reference_xyz=truth.tolist(),
                  detection_xyz=xyz.tolist(), detection_kinds=object_ids.tolist(),
                  detection_uv=detection.uv.cpu().numpy().tolist(), detection_nearest_ids=near_detection,
                  detection_nearest_mm=det_distance, diagnostics=detection.diagnostics,
                  track_xyz=tracked['positions'].tolist(), track_kinds=kinds.tolist(),
                  track_observed=tracked['observed'].tolist(), track_nearest_ids=near_tracks,
                  track_nearest_mm=node_distance, detection_tracks=tracked['detection_tracks'].tolist(),
                  outcomes=outcomes, intrinsic=intrinsic.tolist(), extrinsic=extrinsic.tolist(),
                  aperture_center_depth=probe_aperture_centers(truth, depth, intrinsic, extrinsic),
                  aperture_geometry=aperture_geometry(truth, intrinsic, extrinsic),
                  gt_aperture_is_diagnostic_only=True)
    return result, rgb


def draw_snapshot(snapshot, rgb, camera_name, output):
    """Plot RGB projections, exact graph nodes, and target readout attention."""
    xyz = np.asarray(snapshot['track_xyz'])
    kinds = np.asarray(snapshot['track_kinds'])
    observed = np.asarray(snapshot['track_observed'])
    ref = np.asarray(snapshot['reference_xyz'])
    intrinsic = np.asarray(snapshot['intrinsic'])
    extrinsic = np.asarray(snapshot['extrinsic'])
    output_graph = snapshot['outcomes']['original']
    gt_hole = np.asarray(snapshot['hole_gt'])
    predicted = np.asarray(output_graph['hole'])
    fig = plt.figure(figsize=(16, 8), layout='constrained')
    image_axis = fig.add_subplot(1, 3, 1)
    image_axis.imshow(rgb)
    for index, (point, kind, visible) in enumerate(zip(xyz, kinds, observed)):
        pixel = project_world(point, intrinsic, extrinsic)
        if pixel is not None:
            color = 'orange' if kind == 1 else 'cyan'
            image_axis.scatter(*pixel, s=30, facecolors=color if visible else 'none', edgecolors=color)
            image_axis.text(pixel[0] + 3, pixel[1] - 3, str(index), color=color, fontsize=7, weight='bold')
    for point, label, color in ((gt_hole, 'GT', 'lime'), (predicted, 'Pred', 'magenta')):
        pixel = project_world(point, intrinsic, extrinsic)
        if pixel is not None:
            image_axis.scatter(*pixel, marker='+', s=180, color=color, linewidths=2)
            image_axis.text(pixel[0] + 4, pixel[1] + 9, label, color=color, fontsize=9)
    image_axis.set_title(f'{camera_name}: image / GNN nodes\norange=box, cyan=peg, hollow=predicted')
    image_axis.set_xlim(0,512); image_axis.set_ylim(512,0)
    graph_axis = fig.add_subplot(1, 3, 2, projection='3d')
    target_ids = np.flatnonzero(kinds == 1)
    for i in target_ids:
        for j in target_ids:
            if i < j:
                graph_axis.plot(*np.stack((xyz[i], xyz[j])).T, color='gray', alpha=.22, linewidth=.6)
    for index in range(len(xyz)):
        color = 'darkorange' if kinds[index] == 1 else 'deepskyblue'
        graph_axis.scatter(*xyz[index], c=color, marker='o' if observed[index] else '^', s=35)
        if kinds[index] == 1:
            graph_axis.text(*xyz[index], str(index), fontsize=7)
    graph_axis.scatter(*ref[16:24].T, marker='x', color='limegreen', s=36, label='GT rim corners')
    graph_axis.scatter(*gt_hole, marker='*', color='green', s=130, label='GT hole center')
    graph_axis.scatter(*predicted, marker='*', color='magenta', s=130, label='Predicted hole')
    graph_axis.set_xlabel('world X (m)'); graph_axis.set_ylabel('world Y (m)'); graph_axis.set_zlabel('world Z (m)')
    graph_axis.set_title(f'GNN graph: {len(xyz)} nodes / {output_graph["edge_count"]} directed edges\nOnly box-box edges drawn; triangles=predicted')
    graph_axis.legend(fontsize=7, loc='lower left')
    bar_axis = fig.add_subplot(1, 3, 3)
    weights = np.asarray(output_graph['attention'])
    ids = target_ids.tolist()
    bar_axis.barh([str(i) for i in ids], weights[ids], color=['darkorange' if observed[i] else 'royalblue' for i in ids])
    bar_axis.invert_yaxis()
    bar_axis.set_xlabel('Target pooling attention weight')
    bar_axis.set_ylabel('Box track index')
    error = np.linalg.norm(predicted - gt_hole) * 1000
    bar_axis.set_title(f'Box nodes: {len(ids)}  |  Hole error: {error:.1f} mm\nOrange=detected, blue=predicted')
    path = output / f'{camera_name}_frame_{snapshot["frame"]}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def draw_aperture_zoom(snapshot, rgb, camera_name, output):
    """Enlarge the GT mouth projection and overlay actual detector candidates."""
    corners = np.asarray(snapshot['aperture_geometry']['near_rim_pixels'])
    center = corners.mean(axis=0)
    fig, axis = plt.subplots(figsize=(7, 7), layout='constrained')
    axis.imshow(rgb)
    closed = corners[[0, 1, 3, 2, 0]]
    axis.plot(closed[:, 0], closed[:, 1], color='lime', linewidth=1.5, label='GT aperture outline')
    axis.scatter(corners[:, 0], corners[:, 1], marker='x', color='lime', s=36)
    for index, (uv, kind) in enumerate(zip(snapshot['detection_uv'], snapshot['detection_kinds'])):
        if kind == 1:
            axis.scatter(*uv, marker='o', facecolors='none', edgecolors='orange', s=50)
            axis.text(uv[0] + 2, uv[1] - 2, str(index), color='orange', fontsize=7)
    axis.set_xlim(center[0] - 65, center[0] + 65)
    axis.set_ylim(center[1] + 65, center[1] - 65)
    size = snapshot['aperture_geometry']['bounding_box_px']
    angle = snapshot['aperture_geometry']['normal_view_angle_degrees']
    axis.set_title(f'{camera_name}: frame {snapshot["frame"]}\n'
                   f'GT mouth bbox {size[0]:.1f} × {size[1]:.1f} pixels; angle {angle:.1f}°')
    axis.legend(loc='upper right', fontsize=8)
    path = output / f'{camera_name}_aperture_frame_{snapshot["frame"]}.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def replay(config, camera_name, expected_episode, temporal, gnn, detector, output):
    """Replay saved actions and reject any mismatch with saved GT or prediction."""
    camera = config['camera_view'] if camera_name == 'selected' else None
    sensors = make_sensor_configs(camera)
    env = make_peg_insertion_env(seed=config['environment_seed'], obs_mode='rgb+depth',
                                 sensor_configs=sensors, control_mode=config['control_mode'],
                                 max_episode_steps=config['max_frames'])
    tracker = DetectedCornerTracker(temporal, config['gate_m'], torch.device('cpu'))
    captured = []
    try:
        observation, _ = env.reset(seed=config['environment_seed'])
        for frame in range(max(config['capture_frames']) + 1):
            state = get_gt_state(env)
            snapshot, rgb = build_snapshot(frame, observation, state, detector, tracker, gnn)
            expected = expected_episode['frames'][frame]
            actual_hole = np.asarray(snapshot['outcomes']['original']['hole'])
            for name, actual, recorded in (
                ('peg_gt', snapshot['peg_gt'], expected['peg_gt']),
                ('hole_gt', snapshot['hole_gt'], expected['hole_gt']),
                ('hole_prediction', actual_hole, expected['hole_prediction']),
            ):
                if not np.allclose(actual, recorded, rtol=0, atol=1e-6):
                    raise ValueError(f'{camera_name} frame {frame}: {name} differs from recorded run')
            if frame in config['capture_frames']:
                snapshot['camera'] = camera_name
                snapshot['image_path'] = str(output / f'{camera_name}_frame_{frame}.png')
                snapshot['aperture_image_path'] = str(output / f'{camera_name}_aperture_frame_{frame}.png')
                draw_snapshot(snapshot, rgb, camera_name, output)
                draw_aperture_zoom(snapshot, rgb, camera_name, output)
                captured.append(snapshot)
                print(f'{camera_name} frame {frame}: hole error '
                      f'{np.linalg.norm(actual_hole - np.asarray(snapshot["hole_gt"])) * 1000:.2f} mm', flush=True)
            if frame < max(config['capture_frames']):
                action = torch.as_tensor(expected['action'], dtype=torch.float32, device=env.unwrapped.device)
                observation, _, terminated, truncated, _ = env.step(action.reshape(env.action_space.shape))
                if bool(terminated.any() or truncated.any()):
                    raise ValueError(f'{camera_name}: environment ended before requested frame')
    finally:
        env.close()
    return captured


def evaluate(config):
    """Recreate both camera views, compare recorded predictions, and save evidence."""
    if config['device'] != 'cpu' or config['detector_device'] != 'cpu':
        raise ValueError('This diagnostic currently supports CPU model and detector execution only')
    if not config['capture_frames'] or min(config['capture_frames']) < 0:
        raise ValueError('capture_frames must contain nonnegative frame indices')
    torch.set_num_threads(2)
    temporal, gnn = load_models(config, torch.device('cpu'))
    detector = TorchCornerDetector(device='cpu')
    expected = {}
    for camera in ('default', 'selected'):
        result = json.loads(Path(config[f'{camera}_results']).read_text())
        if result['gnn_sha256'] != file_sha256(config['gnn_checkpoint']):
            raise ValueError(f'{camera} source GNN checkpoint hash differs')
        if result['temporal_sha256'] != file_sha256(config['temporal_checkpoint']):
            raise ValueError(f'{camera} source temporal checkpoint hash differs')
        matches = [episode for episode in result['episodes']
                   if episode['environment_seed'] == config['environment_seed']
                   and episode['action_seed'] == config['action_seed']]
        if len(matches) != 1:
            raise ValueError(f'{camera} requires exactly one source episode')
        if matches[0]['switch_frame'] != config['switch_frame']:
            raise ValueError(f'{camera} source episode has a different action switch frame')
        source_camera = result['config'].get('camera_view')
        expected_camera = config['camera_view'] if camera == 'selected' else None
        if source_camera != expected_camera:
            raise ValueError(f'{camera} source episode has a different camera configuration')
        expected[camera] = matches[0]
    with ExperimentLogger(team='graph', model='hole_input_diagnostics', config=config,
                          seed=config['model_seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        snapshots = []
        for camera in ('default', 'selected'):
            snapshots.extend(replay(config, camera, expected[camera], temporal, gnn, detector, output))
        result_path = output / 'snapshots.json'
        result_path.write_text(json.dumps({'config': config, 'snapshots': snapshots}, indent=2))
        logger.link_output(result_path)
        for snapshot in snapshots:
            logger.link_output(snapshot['image_path'])
            logger.link_output(snapshot['aperture_image_path'])
        logger.record(gnn_sha256=file_sha256(config['gnn_checkpoint']),
                      temporal_sha256=file_sha256(config['temporal_checkpoint']))
        print(f'Hole diagnostics: {output}', flush=True)
