"""Opt-in RoArm-M2-S agent and CPU/CUDA tabletop bench."""

from __future__ import annotations

import os
from pathlib import Path

from roarm.assets import HOME_QPOS, JOINT_NAMES, URDF_PATH
from roarm.scene import (
    MOUNT_POSITION,
    TABLE_HALF_SIZE,
    TABLE_POSITION,
    validate_obstacles,
)

from ..observation_device import ObservationDevice
from ..peg_insertion import _configure_macos_vulkan

_configure_macos_vulkan()

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_agent()
class RoArmM2S(BaseAgent):
    """Three arm axes plus one jaw axis; TCP is the official fixed hand_tcp."""

    uid = "roarm_m2_s"
    urdf_path = str(URDF_PATH)
    fix_root_link = True
    disable_self_collisions = False
    keyframes = {"home": Keyframe(qpos=HOME_QPOS.copy(), pose=sapien.Pose())}

    @property
    def _controller_configs(self):
        # Modest simulation-only force drives. Explicitly keep gravity enabled:
        # ManiSkill otherwise disables it to emulate passive-force compensation.
        common = dict(
            joint_names=JOINT_NAMES,
            stiffness=[80, 80, 60, 5],
            damping=[8, 8, 6, 0.5],
            force_limit=[5, 5, 5, 0.5],
        )
        absolute = PDJointPosControllerConfig(
            **common, lower=None, upper=None, normalize_action=False
        )
        delta = PDJointPosControllerConfig(
            **common, lower=-0.05, upper=0.05, use_delta=True, normalize_action=True
        )
        # Optional simulation gains for slow guarded trajectory tracking. Existing
        # position/delta modes keep their original gains and action conventions.
        guarded = PDJointPosControllerConfig(
            joint_names=JOINT_NAMES,
            lower=None,
            upper=None,
            normalize_action=False,
            stiffness=[160, 160, 120, 10],
            damping=[2, 2, 1.5, 0.2],
            force_limit=[5, 5, 5, 0.5],
        )
        return {
            "pd_joint_pos": {"joints": absolute, "balance_passive_force": False},
            "pd_joint_delta_pos": {"joints": delta, "balance_passive_force": False},
            "pd_joint_pos_guarded": {"joints": guarded, "balance_passive_force": False},
        }

    def _after_loading_articulation(self):
        self.tcp = self.robot.links_map["hand_tcp"]

    @property
    def tcp_pose(self):
        return self.tcp.pose


@register_env("RoArmM2Bench-v1", max_episode_steps=2000)
class RoArmM2Bench(EmptyEnv):
    """Fixed base on a solid tabletop; no manipulation rewards or policy changes."""

    SUPPORTED_ROBOTS = ["roarm_m2_s"]

    def __init__(self, *args, collision_probe=False, obstacles=(), **kwargs):
        self.obstacles = validate_obstacles(obstacles)
        self.collision_probe = collision_probe
        super().__init__(*args, robot_uids="roarm_m2_s", **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.05, -1.05, 1.15], [0.12, 0, 0.75])
        return CameraConfig("render_camera", pose, 640, 480, 1, 0.01, 10)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.8, -0.8, 1.1], [0.12, 0, 0.75])
        return [CameraConfig("base_camera", pose, 128, 128, 1, 0.01, 10)]

    def _load_scene(self, options):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=TABLE_HALF_SIZE)
        builder.add_box_visual(
            half_size=TABLE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=[0.4, 0.45, 0.5, 1]),
        )
        builder.initial_pose = sapien.Pose(TABLE_POSITION)
        self.table = builder.build_static(name="table")
        for index, item in enumerate(self.obstacles):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=item["half_size"])
            builder.add_box_visual(
                half_size=item["half_size"],
                material=sapien.render.RenderMaterial(base_color=[0.85, 0.2, 0.1, 1]),
            )
            builder.initial_pose = sapien.Pose(
                item["center"] + np.array(MOUNT_POSITION)
            )
            builder.build_static(name=f"guard_obstacle_{index}")
        if self.collision_probe:
            builder = self.scene.create_actor_builder()
            builder.add_sphere_collision(radius=0.015, density=100)
            builder.initial_pose = sapien.Pose([3, 0, 1])
            self.probe = builder.build(name="collision_probe")

    def _initialize_episode(self, env_idx, options):
        self.agent.robot.set_pose(sapien.Pose(MOUNT_POSITION))
        qpos = torch.as_tensor(HOME_QPOS, device=self.device).repeat(len(env_idx), 1)
        self.agent.reset(qpos)
        if self.collision_probe:
            self.probe.set_pose(sapien.Pose([3, 0, 1]))
            self.probe.set_linear_velocity(
                torch.zeros((len(env_idx), 3), device=self.device)
            )
            self.probe.set_angular_velocity(
                torch.zeros((len(env_idx), 3), device=self.device)
            )

    def _get_obs_extra(self, info):
        return {"tcp_pose": self.agent.tcp_pose.raw_pose}


def make_roarm_env(
    *,
    control_mode="pd_joint_pos",
    render_mode=None,
    collision_probe=False,
    device="cpu",
    num_envs=1,
    render_backend=None,
    obs_mode="state_dict",
    observation_device=None,
    sensor_configs=None,
    obstacles=(),
):
    """Create the bench, with optional transfer of raw reset/step observations."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if num_envs < 1 or (device == "cpu" and num_envs != 1):
        raise ValueError("num_envs must be positive; CPU requires num_envs=1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation")
    output_device = None
    if observation_device is not None:
        output_device = torch.device(observation_device)
        if output_device.type not in {"cpu", "mps", "cuda"}:
            raise ValueError("observation_device must be 'cpu', 'mps', or 'cuda'")
        if output_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS observation_device requested but MPS is unavailable")
        if output_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA observation_device requested but CUDA is unavailable")

    kwargs = dict(
        num_envs=num_envs,
        sim_backend="physx_cuda" if device == "cuda" else "physx_cpu",
        render_backend=render_backend or device,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        reward_mode="none",
        collision_probe=collision_probe,
        obstacles=obstacles,
    )
    if sensor_configs is not None:
        kwargs["sensor_configs"] = sensor_configs
    original_cwd = Path.cwd()
    try:
        env = gym.make("RoArmM2Bench-v1", **kwargs)
    finally:
        os.chdir(original_cwd)

    if output_device is not None:
        env = ObservationDevice(env, output_device)
    return env
