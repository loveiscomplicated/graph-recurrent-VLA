"""Task-local renderer selection, gated on exact complete capture equality."""
from dataclasses import asdict
from pathlib import Path
from graph.plan_a_bc.common import read
from graph.integration_baseline.rollout_stop_clock import controller_semantics


def gpu_render_environment(root):
    import gymnasium as gym
    from mani_skill.utils.registration import TimeLimitWrapper
    import envs.maniskill
    env=TimeLimitWrapper(gym.make('PegInsertionSide-v1',num_envs=1,sim_backend='physx_cpu',
        render_backend='gpu',obs_mode='rgb+depth',control_mode='pd_ee_delta_pose').unwrapped,250)
    task=env.unwrapped;ref=read(Path(root)/'inputs/task08_reference_conditions.json')
    actual={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()}
    assert controller_semantics(actual)==controller_semantics(ref['controller_config'])
    assert (task.control_freq,task.sim_freq,task._sim_steps_per_control)==(20,100,5)
    return env


def environment(root):
    gate=Path(root)/'renderer_gpu_gate.json'
    if gate.exists() and read(gate).get('adopted'):
        return gpu_render_environment(root)
    from graph.plan_a_bc.rollout import environment as original
    return original(root)
