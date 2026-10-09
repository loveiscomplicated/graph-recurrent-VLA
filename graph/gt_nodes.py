"""Extract world-frame point coordinates from PegInsertionSide simulator GT."""

from __future__ import annotations

from itertools import product

import torch
from torch import Tensor

from envs.maniskill import get_gt_state


def _quat_wxyz_to_matrix(quat: Tensor) -> Tensor:
    """Convert batched wxyz quaternions to rotation matrices."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    rows = (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).reshape(quat.shape[:-1] + (3, 3))


def _local_to_world(local_points: Tensor, pose: Tensor) -> Tensor:
    """Transform batched local points using poses [B, xyz, qwxyz]."""
    rotation = _quat_wxyz_to_matrix(pose[..., 3:])
    return torch.matmul(local_points, rotation.transpose(-1, -2)) + pose[..., None, :3]


def _world_to_tcp(points_world: Tensor, tcp_pose: Tensor) -> Tensor:
    """Express batched world points in the TCP frame."""
    tcp_rotation = _quat_wxyz_to_matrix(tcp_pose[..., 3:])
    offset_world = points_world - tcp_pose[..., None, :3]
    return torch.matmul(offset_world, tcp_rotation)


def get_gt_node_positions(env) -> dict[str, Tensor]:
    """Return simulator-derived point positions relative to the gripper TCP.

    Returned tensors retain the simulator's leading environment dimension and
    device. Coordinates are in metres, with origin and axes aligned to the TCP
    frame. Single points have shape [B, 3], peg corners [B, 8, 3], and square
    hole-aperture corners [B, 4, 3]. The gripper reference point is the task's
    TCP, not a separately queried wrist link; its returned coordinate is zero.
    """
    state = get_gt_state(env)
    device, dtype = state.peg_half_sizes.device, state.peg_half_sizes.dtype

    signs = torch.tensor(
        list(product((-1.0, 1.0), repeat=3)), device=device, dtype=dtype
    )
    peg_local_corners = signs.unsqueeze(0) * state.peg_half_sizes.unsqueeze(1)
    peg_corners = _local_to_world(peg_local_corners, state.peg_pose)

    hole_signs = torch.tensor(
        list(product((-1.0, 1.0), repeat=2)), device=device, dtype=dtype
    )
    hole_local_corners = torch.zeros((state.hole_radius.shape[0], 4, 3), device=device, dtype=dtype)
    hole_local_corners[..., 1:] = (
        hole_signs.unsqueeze(0) * state.hole_radius[:, None, None]
    )
    hole_corners = _local_to_world(hole_local_corners, state.hole_pose)

    tcp_position = state.ee_pose[:, :3]
    return {
        "gripper_tcp": torch.zeros_like(tcp_position),
        "left_fingertip": _world_to_tcp(state.left_fingertip_pos.unsqueeze(1), state.ee_pose).squeeze(1),
        "right_fingertip": _world_to_tcp(state.right_fingertip_pos.unsqueeze(1), state.ee_pose).squeeze(1),
        "peg_corners": _world_to_tcp(peg_corners, state.ee_pose),
        "hole_corners": _world_to_tcp(hole_corners, state.ee_pose),
    }
