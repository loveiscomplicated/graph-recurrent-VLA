"""Collect every preregistered raw RGB-D test trajectory without filtering."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.storage import save_frame, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger


def observation_subset(observation):
    """Retain the native base-camera images, calibration and observed robot state."""
    return dict(sensor_data={'base_camera': observation['sensor_data']['base_camera']},
                sensor_param={'base_camera': observation['sensor_param']['base_camera']},
                agent={'qpos': observation['agent']['qpos'], 'qvel': observation['agent']['qvel']},
                extra={'tcp_pose': observation['extra']['tcp_pose']})


def collect_episode(protocol, seed, mode, root):
    """Save one complete prefix and post-lift sequence, including unsuccessful cases."""
    settings = protocol['policy']
    sensors = {name: {'width': settings['resolution'], 'height': settings['resolution']}
               for name in ('base_camera', 'hand_camera')}
    env = make_peg_insertion_env(seed=seed, obs_mode='rgb+depth', sensor_configs=sensors,
                                 control_mode=settings['control_mode'], max_episode_steps=settings['max_frames'])
    policy = ScriptedPegPolicy()
    rng = np.random.default_rng(settings['random_generator_base_seed'] + seed)
    pending = root / f'{mode}_seed_{seed}.partial'
    pending.mkdir()
    switched = False
    after = 0
    frame_hashes = []
    stop_reason = None
    try:
        obs, _ = env.reset(seed=seed)
        selection = protocol.get('selection')
        if selection is not None:
            from data_collection.matched_visibility_test import fingerprint
            expected = selection['initial_fingerprints'][str(seed)]
            if fingerprint(obs, vars(get_gt_state(env))) != expected:
                raise ValueError(f'Initial frame differs from the frozen gate selection for seed {seed}')
        policy.reset(env, seed)
        task = env.unwrapped
        initial_height = float(get_gt_state(env).peg_pose[0, 2])
        for frame in range(settings['max_frames']):
            gt = get_gt_state(env)
            grasp = bool(task.agent.is_grasping(task.peg)[0])
            lift = float(gt.peg_pose[0, 2]) - initial_height
            if not switched and policy.phase == 'align' and grasp and lift > settings['minimum_lift_m']:
                switched = True
                switch_frame = frame
            phase = ('random' if mode == 'random_after_lift' else 'expert_control') if switched else policy.phase
            if switched and after >= settings['random_steps']:
                stop_reason = 'post_lift_horizon'
                action = None
            elif frame + 1 >= settings['max_frames']:
                stop_reason = 'frame_limit'
                action = None
            elif switched and mode == 'random_after_lift':
                values = np.concatenate((rng.uniform(-settings['random_arm_limit'], settings['random_arm_limit'], 6), [-1]))
                action = torch.as_tensor(values, dtype=torch.float32, device=task.device).reshape(env.action_space.shape)
            else:
                action = policy.act(obs)
            native_action = action if action is not None else torch.zeros(env.action_space.shape, dtype=torch.float32)
            path = pending / f'frame_{frame:06d}.npz'
            save_frame(path, observation_subset(obs), ground_truth=vars(gt),
                       action=native_action, action_valid=np.asarray(action is not None),
                       test_frame=dict(frame=frame, phase=phase, grasp=grasp,
                                       after_lift=after if switched else -1))
            frame_hashes.append(file_sha256(path))
            if action is None:
                break
            obs, _, terminated, truncated, _ = env.step(action)
            if switched:
                after += 1
            if bool(terminated.any() or truncated.any()):
                stop_reason = 'terminated' if bool(terminated.any()) else 'truncated'
                break
    finally:
        env.close()
    receipt = dict(seed=seed, mode=mode, path=f'{mode}_seed_{seed}', observations=len(frame_hashes),
                   switched=switched, switch_frame=switch_frame if switched else None,
                   post_lift_actions=after, stop_reason=stop_reason, frame_sha256=frame_hashes)
    write_json(pending / 'episode.json', receipt)
    pending.rename(root / receipt['path'])
    return receipt


def collect(config):
    """Enforce the frozen protocol and save all planned seed/mode cases."""
    if file_sha256(config['protocol']) != config['protocol_sha256']:
        raise ValueError('Frozen test protocol SHA256 mismatch')
    protocol = json.loads(Path(config['protocol']).read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with ExperimentLogger(team='graph', model='independent_test_collection', config=config,
                          seed=protocol['generation']['seed'], job_type='collection') as logger:
        episodes = []
        manifest = dict(status='RUNNING', protocol_sha256=config['protocol_sha256'], episodes=episodes)
        try:
            for seed in protocol['seeds']:
                for mode in protocol['policy']['modes']:
                    episode = collect_episode(protocol, seed, mode, root)
                    episodes.append(episode)
                    write_json(root / 'manifest.json', manifest)
                    print(f'{len(episodes)}/{len(protocol["seeds"])*2}: seed={seed}, mode={mode}, '
                          f'lift={episode["switched"]}, frames={episode["observations"]}', flush=True)
            manifest['status'] = 'COMPLETE'
        except BaseException:
            manifest['status'] = 'ERROR'
            raise
        finally:
            write_json(root / 'manifest.json', manifest)
        logger.record(protocol_sha256=config['protocol_sha256'], episodes=len(episodes),
                      manifest_sha256=file_sha256(root / 'manifest.json'))
        logger.link_output(root / 'manifest.json')
        print(f'Fixed raw test dataset: {root}', flush=True)
