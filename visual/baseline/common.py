"""Shared environment, robot-state and image helpers for the baseline policies."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from envs.maniskill import get_gt_state, make_peg_insertion_env

# Same simulator settings as data_collection/configs/visible_peg.yaml.
RESOLUTION = 512
HORIZON = 250
CONTROL_MODE = "pd_ee_delta_pose"
CAMERAS = ("base_camera", "hand_camera")
ROBOT_DIM = 15


def make_env(render_backend: str = "cuda"):
    """Create the 512×512 RGB-D environment used to collect the expert dataset."""
    return make_peg_insertion_env(
        device="cpu",
        render_backend=render_backend,
        obs_mode="rgb+depth",
        control_mode=CONTROL_MODE,
        sensor_configs={
            name: dict(width=RESOLUTION, height=RESOLUTION) for name in CAMERAS
        },
        max_episode_steps=HORIZON,
    )


def canonical_quat(quat: np.ndarray) -> np.ndarray:
    """Flip wxyz quaternions to a non-negative w, matching recurrent/features.py."""
    return np.where(quat[..., :1] < 0, -quat, quat)


def robot15(ee_pose, left_tip, right_tip, gripper_qpos) -> np.ndarray:
    """Build the measured robot state used by the Recurrent controller.

    Layout (recurrent/features.py, first 15 dims): tcp_pos 3, tcp_quat 4,
    left_tip - tcp 3, right_tip - tcp 3, gripper_qpos 2. Inputs may carry any
    leading batch/time dimensions.
    """
    ee_pose = np.asarray(ee_pose, dtype=np.float64)
    tcp_position = ee_pose[..., :3]
    parts = [
        tcp_position,
        canonical_quat(ee_pose[..., 3:]),
        np.asarray(left_tip) - tcp_position,
        np.asarray(right_tip) - tcp_position,
        np.asarray(gripper_qpos),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def live_robot15(env) -> np.ndarray:
    """Read the current robot15 state of environment 0 from the simulator."""
    gt = get_gt_state(env)
    return robot15(
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
