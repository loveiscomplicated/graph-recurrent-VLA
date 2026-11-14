"""Collect reproducible expert-prefix and random-action camera study trajectories."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.scripted_policy import ScriptedPegPolicy
from envs.maniskill import get_gt_state, make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger


def collect_episode(config, seed):
    """Record all actions and preserve failures instead of filtering by success."""
    env = make_peg_insertion_env(seed=seed, obs_mode='state', control_mode=config['control_mode'],
                                 max_episode_steps=config['max_frames'])
    policy = ScriptedPegPolicy()
    rng = np.random.default_rng(config['random_seed'] + seed)
    rows = []
    switched = False
    random_steps = 0
    try:
        observation, _ = env.reset(seed=seed)
        policy.reset(env, seed)
        task = env.unwrapped
        initial_z = float(get_gt_state(env).peg_pose[0, 2])
        for frame in range(config['max_frames']):
            state = get_gt_state(env)
            grasp = bool(task.agent.is_grasping(task.peg)[0])
            lift = float(state.peg_pose[0, 2]) - initial_z
            if not switched and policy.phase == 'align' and grasp and lift > config['minimum_lift_m']:
                switched = True
                switch_frame = frame
            row = dict(frame=frame, phase='random' if switched else policy.phase,
                       grasp=grasp, lift_m=lift,
                       peg_gt=state.peg_pose[0, :3].cpu().numpy().tolist(),
                       hole_gt=state.hole_pose[0, :3].cpu().numpy().tolist(),
                       hole_quaternion=state.hole_pose[0, 3:].cpu().numpy().tolist(),
                       box_gt=state.box_pose[0, :3].cpu().numpy().tolist())
            rows.append(row)
            if switched and random_steps >= config['random_steps']:
                break
            if not switched and frame >= config['expert_step_limit']:
                break
            if switched:
                values = np.concatenate((rng.uniform(-config['random_arm_limit'],
                                                       config['random_arm_limit'], 6), [-1]))
                action = torch.as_tensor(values, dtype=torch.float32, device=task.device).reshape(env.action_space.shape)
            else:
                action = policy.act(observation)
            row['action'] = action.cpu().numpy().reshape(-1).tolist()
            observation, _, terminated, truncated, _ = env.step(action)
            if switched:
                random_steps += 1
            if bool(terminated.any() or truncated.any()):
                break
    finally:
        env.close()
    if switched:
        samples = [0, round(switch_frame * .25), round(switch_frame * .5),
                   round(switch_frame * .75), switch_frame,
                   switch_frame + 10, switch_frame + 20, switch_frame + 35]
    else:
        last = len(rows) - 1
        samples = [round(last * fraction) for fraction in (0, .2, .4, .6, .8, 1)]
    samples = sorted({frame for frame in samples if frame < len(rows)})
    return dict(environment_seed=seed, action_seed=config['random_seed'], switched=switched,
                switch_frame=switch_frame if switched else None, random_steps=random_steps,
                sample_frames=samples, frames=rows)


def collect(config):
    """Save a camera-independent episode manifest with effective settings."""
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must contain unique environment seed values')
    torch.set_num_threads(2)
    with ExperimentLogger(team='graph', model='camera_visibility_trajectories', config=config,
                          seed=config['random_seed'], job_type='collection') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for seed in config['seeds']:
            episode = collect_episode(config, seed)
            episodes.append(episode)
            (output / 'progress.json').write_text(json.dumps(episodes, indent=2))
            print(f'seed {seed}: switch={episode["switched"]}, samples={len(episode["sample_frames"])}', flush=True)
        path = output / 'trajectories.json'
        path.write_text(json.dumps({'config': config, 'episodes': episodes}, indent=2))
        logger.record(episodes=len(episodes), switched=sum(e['switched'] for e in episodes))
        logger.link_output(path)
        print(f'Trajectories: {output}', flush=True)
