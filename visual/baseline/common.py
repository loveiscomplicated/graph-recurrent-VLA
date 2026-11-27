"""Shared environment, robot-state and image helpers for the baseline policies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from envs.maniskill import get_gt_state, make_peg_insertion_env
from visual.baseline.state import ROBOT_DIM, robot_state

# Same simulator settings as data_collection/configs/visible_peg.yaml.
RESOLUTION = 512
HORIZON = 250
CONTROL_MODE = "pd_ee_delta_pose"
CAMERAS = ("base_camera", "hand_camera")

__all__ = ["CAMERAS", "CONTROL_MODE", "HORIZON", "RESOLUTION", "ROBOT_DIM",
           "live_robot_state", "make_env", "resize_depth", "resize_rgb", "robot_state"]


def make_env(render_backend: str = "cuda", *, observation_device=None, horizon=HORIZON):
    """Create the 512×512 RGB-D environment used to collect the expert dataset.

    Only the resolution is overridden here, so ``base_camera`` keeps the shared
    pose set in ``envs/maniskill/peg_insertion.py`` (eye [0.65, -0.30, 0.65],
    target [0.065, 0.297, 0.11], since commit ec73248 on 2026-11-13).
    """
    return make_peg_insertion_env(
        device="cpu",
        render_backend=render_backend,
        observation_device=observation_device,
        obs_mode="rgb+depth",
        control_mode=CONTROL_MODE,
        sensor_configs={
            name: dict(width=RESOLUTION, height=RESOLUTION) for name in CAMERAS
        },
        max_episode_steps=horizon,
    )


def live_robot_state(env) -> np.ndarray:
    """Read the current 17D robot state (see state.py) of environment 0."""
    gt = get_gt_state(env)
    return robot_state(
        gt.ee_pose[0].cpu().numpy(),
        gt.left_fingertip_pos[0].cpu().numpy(),
        gt.right_fingertip_pos[0].cpu().numpy(),
        gt.gripper_qpos[0].cpu().numpy(),
    )


def resize_rgb(rgb, size: int) -> torch.Tensor:
    """Area-resize uint8 HWC (or NHWC) RGB to ``size`` and return uint8 NCHW."""
    images = torch.as_tensor(rgb)
    if images.ndim == 3:
        images = images[None]
    images = images.permute(0, 3, 1, 2).float()
    if images.shape[-1] != size:
        images = F.interpolate(images, size=(size, size), mode="area")
    return images.round().clamp(0, 255).to(torch.uint8)


def resize_depth(depth, size: int) -> torch.Tensor:
    """Resize signed int16 mm depth (HW1 or NHW1) by nearest sampling to int16 N1HW."""
    values = torch.as_tensor(depth)
    if values.ndim == 3:
        values = values[None]
    values = values.permute(0, 3, 1, 2).float()
    if values.shape[-1] != size:
        # Nearest keeps invalid (nonpositive) pixels from blending into valid depth.
        values = F.interpolate(values, size=(size, size), mode="nearest")
    return values.to(torch.int16)
