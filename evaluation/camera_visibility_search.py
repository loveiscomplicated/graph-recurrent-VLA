"""Rank fixed camera poses on sampled, action-matched simulator states."""

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.random_action_diagnostics import make_sensor_configs
from utils.experiment_logger import ExperimentLogger


def record_states(config, source_episodes, seed):
    """Replay saved actions and collect simulator states at predeclared frames."""
    env = make_peg_insertion_env(seed=seed, obs_mode='state', control_mode=config['control_mode'],
                                 max_episode_steps=config['max_frames'])
    snapshots = []
    try:
        for episode in source_episodes:
            obs, _ = env.reset(seed=seed)
            switch = episode['switch_frame']
            if 'sample_frames' in episode:
                selected = set(episode['sample_frames'])
            else:
                selected = {switch + step for step in config['steps_after_lift']}
                if config['include_initial_frame']:
                    selected.add(0)
            if max(selected) >= len(episode['frames']):
                raise ValueError(f'Source episode lacks required frames for seed {seed}')
            for frame in range(max(selected) + 1):
                saved = episode['frames'][frame]
                if frame in selected:
                    gt = get_gt_state(env)
                    peg = gt.peg_pose[0, :3].cpu().numpy()
                    hole = gt.hole_pose[0, :3].cpu().numpy()
                    peg_matches = np.allclose(peg, saved['peg_gt'], rtol=0, atol=1e-6)
                    hole_matches = np.allclose(hole, saved['hole_gt'], rtol=0, atol=1e-6)
                    if not (peg_matches and hole_matches):
                        raise ValueError(f'Source trajectory mismatch at seed {seed}, frame {frame}')
                    snapshots.append(dict(seed=seed, action_seed=episode['action_seed'], frame=frame,
                                          relative_step=frame-switch if switch is not None else None,
                                          phase=saved.get('phase'), peg=peg.tolist(), hole=hole.tolist(),
                                          state=copy.deepcopy(env.unwrapped.get_state_dict())))
                if frame < max(selected):
                    action = torch.tensor(saved['action'], dtype=torch.float32, device=env.unwrapped.device)
                    obs, _, terminated, truncated, _ = env.step(action.reshape(env.action_space.shape))
                    if bool(terminated.any() or truncated.any()):
                        raise ValueError('Source episode ended before the selected sample frames')
    finally:
        env.close()
    return snapshots


def score_frame(obs, task, sample_count, tolerance):
    """Measure visible, robot-blocked and peg-blocked front-opening samples."""
    sensor = obs['sensor_data']['base_camera']
    param = obs['sensor_param']['base_camera']
    depth = sensor['depth'][0, :, :, 0].cpu().numpy()
    segmentation = sensor['segmentation'][0, :, :, 0].cpu().numpy()
    intrinsic = param['intrinsic_cv'][0].cpu().numpy()
    extrinsic = param['extrinsic_cv'][0].cpu().numpy()
    eye = -extrinsic[:, :3].T @ extrinsic[:, 3]
    transform = task.box_hole_pose.to_transformation_matrix()[0].cpu().numpy()
    half_length = float(task.peg_half_sizes[0, 0])
    radius = float(task.box_hole_radii[0])
    normal = -transform[:3, 0]
    center = transform[:3, 3] - half_length * transform[:3, 0]
    front_facing = bool((eye - center) @ normal > 0)
    robot_ids = {int(link.per_scene_id[0].item()) for link in task.agent.robot.links}
    peg_id = int(task.peg.per_scene_id[0].item())
    visible = robot_blocked = peg_blocked = 0
    for y in np.linspace(-.7, .7, sample_count):
        for z in np.linspace(-.7, .7, sample_count):
            point = center + radius * (y * transform[:3, 1] + z * transform[:3, 2])
            camera = extrinsic[:, :3] @ point + extrinsic[:, 3]
            if camera[2] <= 0 or not front_facing:
                continue
            projected = intrinsic @ camera
            u, v = np.rint(projected[:2] / projected[2]).astype(int)
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            raw = int(depth[v, u])
            valid = 0 < raw < 4000
            is_visible = valid and raw * .001 >= camera[2] - tolerance
            visible += int(is_visible)
            if valid and not is_visible:
                robot_blocked += int(int(segmentation[v, u]) in robot_ids)
                peg_blocked += int(int(segmentation[v, u]) == peg_id)
    total = sample_count * sample_count
    corners = np.asarray([center + radius * (y * transform[:3, 1] + z * transform[:3, 2])
                          for y in (-1, 1) for z in (-1, 1)])
    camera_corners = corners @ extrinsic[:, :3].T + extrinsic[:, 3]
    projected = camera_corners @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    angle_cos = abs(float(np.dot((eye-center)/np.linalg.norm(eye-center), normal)))
    return dict(visible_fraction=visible/total, robot_blocked_fraction=robot_blocked/total,
                peg_blocked_fraction=peg_blocked/total, front_facing=front_facing,
                aperture_bbox_px=np.ptp(pixels, axis=0).tolist(),
                angle_degrees=float(np.degrees(np.arccos(np.clip(angle_cos, 0, 1)))))


def scan_candidate(config, candidate, seed, snapshots):
    """Render identical saved physical states from one candidate camera."""
    view = None if candidate['eye'] is None else {'eye': candidate['eye'], 'target': config['camera_target']}
    sensors = make_sensor_configs(view)
    env = make_peg_insertion_env(seed=seed, obs_mode='rgb+depth+segmentation',
                                 control_mode=config['control_mode'], sensor_configs=sensors,
                                 max_episode_steps=config['max_frames'])
    rows = []
    try:
        env.reset(seed=seed)
        for snapshot in snapshots:
            env.unwrapped.set_state_dict(snapshot['state'])
            gt = get_gt_state(env)
            if not np.allclose(gt.peg_pose[0,:3].cpu().numpy(), snapshot['peg'], rtol=0, atol=1e-6):
                raise ValueError(f'Restored Peg state differs for seed {seed}, frame {snapshot["frame"]}')
            if not np.allclose(gt.hole_pose[0,:3].cpu().numpy(), snapshot['hole'], rtol=0, atol=1e-6):
                raise ValueError(f'Restored Hole state differs for seed {seed}, frame {snapshot["frame"]}')
            obs = env.unwrapped.get_obs()
            result = score_frame(obs, env.unwrapped, config['surface_sample_count'], config['depth_tolerance_m'])
            rows.append(dict(seed=seed, action_seed=snapshot['action_seed'], frame=snapshot['frame'],
                             relative_step=snapshot['relative_step'], phase=snapshot['phase'], **result))
    finally:
        env.close()
    return rows


def rank_candidates(records, selection_seeds, ranking_rule='worst_seed'):
    """Prioritize worst-seed visibility, overall visibility, clearance and projected width."""
    for candidate in records:
        rows = candidate['frames']
        means = [float(np.mean([row['visible_fraction'] for row in rows if row['seed']==seed]))
                 for seed in selection_seeds]
        candidate['seed_mean_visibility'] = dict(zip(map(str, selection_seeds), means))
        candidate['worst_seed_mean_visibility'] = min(means)
        candidate['p10_seed_mean_visibility'] = float(np.percentile(means, 10))
        candidate['mean_visibility'] = float(np.mean([row['visible_fraction'] for row in rows]))
        candidate['mean_robot_blocked'] = float(np.mean([row['robot_blocked_fraction'] for row in rows]))
        candidate['min_projected_bbox_width_px'] = float(min(row['aperture_bbox_px'][0] for row in rows))
    if ranking_rule == 'mean_then_p10':
        return sorted(records, key=lambda row: (row['mean_visibility'], row['p10_seed_mean_visibility'],
                                                -row['mean_robot_blocked'], row['min_projected_bbox_width_px']), reverse=True)
    if ranking_rule != 'worst_seed':
        raise ValueError('ranking_rule must be worst_seed or mean_then_p10')
    return sorted(records, key=lambda row: (row['worst_seed_mean_visibility'], row['mean_visibility'],
                                            -row['mean_robot_blocked'], row['min_projected_bbox_width_px']), reverse=True)


def search(config):
    """Capture action-matched states, score candidate views, and save rankings."""
    source = json.loads(Path(config['source_results']).read_text())
    candidates = copy.deepcopy(config['candidates'])
    for candidate in candidates:
        candidate['frames'] = []
    with ExperimentLogger(team='graph', model='camera_visibility_search', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        for seed in config['selection_environment_seeds']:
            episodes = [e for e in source['episodes'] if e['environment_seed'] == seed
                        and e['action_seed'] in config['action_seeds']]
            if len(episodes) != len(config['action_seeds']):
                raise ValueError(f'Missing source trajectories for seed {seed}')
            snapshots = record_states(config, episodes, seed)
            for candidate in candidates:
                candidate['frames'].extend(scan_candidate(config, candidate, seed, snapshots))
                print(f'seed={seed} camera={candidate["name"]} samples={len(snapshots)}', flush=True)
            (output / 'progress.json').write_text(json.dumps(candidates, indent=2))
        rule = config.get('ranking_rule', 'worst_seed')
        ranked = rank_candidates(candidates, config['selection_environment_seeds'], rule)
        result = {'config': config, 'ranked_candidates': ranked, 'selected': ranked[0]['name'],
                  'selection_rule': rule}
        path = output / 'search_results.json'
        path.write_text(json.dumps(result, indent=2))
        fig, axis = plt.subplots(figsize=(9,6), layout='constrained')
        names = [row['name'] for row in ranked]
        score = 'mean_visibility' if rule == 'mean_then_p10' else 'worst_seed_mean_visibility'
        axis.barh(names, [row[score] for row in ranked], color='#3175ab')
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xlabel('Mean sampled aperture visibility' if rule == 'mean_then_p10'
                        else 'Worst seed: mean sampled aperture visibility')
        axis.set_title('Camera selection: selection seeds only')
        figure = output / 'camera_ranking.png'
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        logger.record(selected=result['selected'])
        logger.link_output(path)
        logger.link_output(figure)
        print(f'Camera search: {output}', flush=True)
