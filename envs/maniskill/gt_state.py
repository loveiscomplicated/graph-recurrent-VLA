"""Model-independent access to live PegInsertionSide simulator state."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .wrappers import peg_task


# The task uses panda_wristcam / panda_v3.urdf in ManiSkill 3.0.1. Its rubber
# fingertip collision boxes are centered at these finger-link local coordinates.
_LEFT_TIP_LOCAL = (0.0, 0.00758, 0.04525)
_RIGHT_TIP_LOCAL = (0.0, -0.00758, 0.04525)


@dataclass(frozen=True)
class GTState:
    """Detached simulator tensors with leading environment dimension ``B``.

    Poses are ``[B, 7]`` in world coordinates: xyz metres followed by wxyz
    quaternion. Positions are world-frame metres. Joint values use the robot's
    active joint order; see ``envs/maniskill/README.md`` for every field.
    """

    qpos: Tensor
    qvel: Tensor
    ee_pose: Tensor
    left_finger_pose: Tensor
    right_finger_pose: Tensor
    left_fingertip_pos: Tensor
    right_fingertip_pos: Tensor
    peg_pose: Tensor
    box_pose: Tensor
    hole_pose: Tensor
    gripper_qpos: Tensor
    peg_half_sizes: Tensor
    hole_radius: Tensor
    success: Tensor


def _finger_pad_origin(link, offset) -> Tensor:
    transform = link.pose.to_transformation_matrix()
    from torch import as_tensor

    local = as_tensor(offset, device=transform.device, dtype=transform.dtype)
    return transform[..., :3, 3] + transform[..., :3, :3] @ local


def get_gt_state(env) -> GTState:
    """Read GT after reset/step without constructing model features."""
    task = peg_task(env)
    robot = task.agent.robot
    qpos = robot.get_qpos()

    def clone(tensor):
        return tensor.detach().clone()

    return GTState(
        qpos=clone(qpos),
        qvel=clone(robot.get_qvel()),
        ee_pose=clone(task.agent.tcp.pose.raw_pose),
        left_finger_pose=clone(task.agent.finger1_link.pose.raw_pose),
        right_finger_pose=clone(task.agent.finger2_link.pose.raw_pose),
        left_fingertip_pos=clone(_finger_pad_origin(task.agent.finger1_link, _LEFT_TIP_LOCAL)),
        right_fingertip_pos=clone(_finger_pad_origin(task.agent.finger2_link, _RIGHT_TIP_LOCAL)),
        peg_pose=clone(task.peg.pose.raw_pose),
        box_pose=clone(task.box.pose.raw_pose),
        hole_pose=clone(task.box_hole_pose.raw_pose),
        gripper_qpos=clone(qpos[..., -2:]),
        peg_half_sizes=clone(task.peg_half_sizes),
        hole_radius=clone(task.box_hole_radii),
        success=clone(task.evaluate()["success"]),
    )
