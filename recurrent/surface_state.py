"""Canonical Robot15, Object21 and State36 assembly for Surface-Mamba workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from recurrent.surface_mamba import quaternion_rotate


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Choose the nonnegative-w representative of a wxyz quaternion."""
    quaternion = np.asarray(quaternion, dtype=np.float32)
    return np.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def robot15_from_values(
    tcp_pose: np.ndarray,
    left_fingertip: np.ndarray,
    right_fingertip: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Assemble the recurrent Robot15 prefix from simulator GT values."""
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32).reshape(7)
    tcp_position = tcp_pose[:3]
    robot = np.concatenate(
        (
            tcp_position,
            canonical_quaternion(tcp_pose[3:]),
            np.asarray(left_fingertip, dtype=np.float32).reshape(3) - tcp_position,
            np.asarray(right_fingertip, dtype=np.float32).reshape(3) - tcp_position,
            np.asarray(gripper_qpos, dtype=np.float32).reshape(2),
        )
    )
    if robot.shape != (15,) or not np.isfinite(robot).all():
        raise ValueError("Robot15 assembly failed")
    return robot.astype(np.float32)


def object21_from_values(
    tcp_pose: np.ndarray,
    peg_pose: np.ndarray,
    hole_pose: np.ndarray,
    peg_half_sizes: np.ndarray,
    hole_radius: np.ndarray,
) -> np.ndarray:
    """Assemble the privileged Object21 target using the established layout."""
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32).reshape(7)
    peg_pose = np.asarray(peg_pose, dtype=np.float32).reshape(7)
    hole_pose = np.asarray(hole_pose, dtype=np.float32).reshape(7)
    half_sizes = np.asarray(peg_half_sizes, dtype=np.float32).reshape(3)
    peg_quaternion = canonical_quaternion(peg_pose[3:])
    hole_quaternion = canonical_quaternion(hole_pose[3:])
    peg_tip = peg_pose[:3] + quaternion_rotate(
        peg_quaternion, np.asarray([half_sizes[0], 0, 0], dtype=np.float32)
    )
    object21 = np.concatenate(
        (
            peg_pose[:3] - tcp_pose[:3],
            peg_quaternion,
            peg_tip - hole_pose[:3],
            hole_pose[:3] - tcp_pose[:3],
            hole_quaternion,
            half_sizes,
            np.asarray(hole_radius, dtype=np.float32).reshape(1),
        )
    )
    if object21.shape != (21,) or not np.isfinite(object21).all():
        raise ValueError("Object21 assembly failed")
    return object21.astype(np.float32)


def state36(robot15: np.ndarray, object21: np.ndarray) -> np.ndarray:
    """Join canonical Robot15 and Object21 arrays."""
    result = np.concatenate(
        (
            np.asarray(robot15, dtype=np.float32).reshape(15),
            np.asarray(object21, dtype=np.float32).reshape(21),
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("State36 contains nonfinite values")
    return result.astype(np.float32)


def from_gt_state(gt) -> tuple[np.ndarray, np.ndarray]:
    """Read Robot15 and privileged Object21 from ``envs.maniskill.GTState``."""
    tcp = _numpy(gt.ee_pose)[0]
    robot = robot15_from_values(
        tcp,
        _numpy(gt.left_fingertip_pos)[0],
        _numpy(gt.right_fingertip_pos)[0],
        _numpy(gt.gripper_qpos)[0],
    )
    object21 = object21_from_values(
        tcp,
        _numpy(gt.peg_pose)[0],
        _numpy(gt.hole_pose)[0],
        _numpy(gt.peg_half_sizes)[0],
        _numpy(gt.hole_radius)[0],
    )
    return robot, object21


def from_frame_archive(frame) -> tuple[np.ndarray, np.ndarray]:
    """Read Robot15 and Object21 from one raw frame NPZ archive."""
    prefix = "ground_truth/"
    tcp = frame[prefix + "ee_pose"][0]
    return (
        robot15_from_values(
            tcp,
            frame[prefix + "left_fingertip_pos"][0],
            frame[prefix + "right_fingertip_pos"][0],
            frame[prefix + "gripper_qpos"][0],
        ),
        object21_from_values(
            tcp,
            frame[prefix + "peg_pose"][0],
            frame[prefix + "hole_pose"][0],
            frame[prefix + "peg_half_sizes"][0],
            frame[prefix + "hole_radius"][0],
        ),
    )
