"""GT-guided peg expert with the collector's reset/act policy interface.

The control law is extracted from recurrent/expert.py without importing a team
experiment. It requires ManiSkill 3.0.1's pd_ee_delta_pose controller. Its world
rotation command uses that controller's negative rotational scaling convention.
"""

import numpy as np
import sapien
import torch
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def to_pose(pose):
    """Convert a single batched ManiSkill world pose to a SAPIEN pose."""
    return sapien.Pose(pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy())


def rotation_error(current, target):
    """Return the shortest world-frame rotation vector from current to target."""
    difference = qmult(target, qinverse(current))
    if difference[0] < 0:
        difference = -difference
    axis, angle = quat2axangle(difference)
    return np.asarray(axis) * angle


class ScriptedPegPolicy:
    """Reach, grasp, lift, align and insert using live simulator ground truth."""

    def reset(self, env, seed):
        """Choose a grasp in the accepted state without resetting or stepping."""
        self.env = env
        task = env.unwrapped
        if task.control_mode != "pd_ee_delta_pose":
            raise ValueError("ScriptedPegPolicy requires control_mode='pd_ee_delta_pose'")
        closing_hint = task.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        grasp_info = compute_grasp_info_by_obb(
            get_actor_obb(task.peg),
            approaching=np.array([0, 0, -1]),
            target_closing=closing_hint.cpu().numpy(),
            depth=0.025,
        )
        grasp = task.agent.build_grasp_pose(
            np.array([0, 0, -1]), grasp_info["closing"], grasp_info["center"]
        )
        self.half_length = task.peg_half_sizes[0, 0].item()
        grasp_offset = max(0.05, self.half_length / 2 + 0.01)
        self.grasp = grasp * sapien.Pose([-grasp_offset, 0, 0])
        self.phase = "reach"
        self.counter = 0
        self.lift_target = None

    def tcp(self):
        """Read the current world-frame tool-center pose."""
        return to_pose(self.env.unwrapped.agent.tcp.pose)

    def servo(self, target, translation_limit=0.02):
        """Convert a target world pose to the installed normalized EE action."""
        current = self.tcp()
        translation = (target.p - current.p) * 3.0
        translation_norm = np.linalg.norm(translation)
        if translation_norm > translation_limit:
            translation *= translation_limit / translation_norm
        rotation = rotation_error(current.q, target.q)
        rotation_norm = np.linalg.norm(rotation)
        if rotation_norm > 0.15:
            rotation *= 0.15 / rotation_norm
        return np.concatenate([translation / 0.1, -rotation / 0.1])

    def tcp_error(self, target):
        """Measure translation and rotation errors at the tool center."""
        current = self.tcp()
        return (
            np.linalg.norm(target.p - current.p),
            np.linalg.norm(rotation_error(current.q, target.q)),
        )

    def tcp_for_peg(self, peg_target):
        """Preserve the measured peg-to-tool transform for a desired peg pose."""
        peg = to_pose(self.env.unwrapped.peg.pose)
        return peg_target * peg.inv() * self.tcp()

    def act(self, observation):
        """Return one finite batched native action for the current expert phase."""
        task = self.env.unwrapped
        goal = to_pose(task.goal_pose)
        gripper = 1.0

        if self.phase == "reach":
            target = self.grasp * sapien.Pose([0, 0, -0.05])
            motion = self.servo(target)
            position_error, angle_error = self.tcp_error(target)
            if position_error < 0.01 and angle_error < 0.05:
                self.phase = "descend"
        elif self.phase == "descend":
            motion = self.servo(self.grasp, translation_limit=0.01)
            position_error, angle_error = self.tcp_error(self.grasp)
            if position_error < 0.004 and angle_error < 0.03:
                self.phase = "close"
        elif self.phase == "close":
            motion = np.zeros(6)
            gripper = -1.0
            self.counter += 1
            if self.counter >= 8:
                self.phase = "lift"
                current = self.tcp()
                self.lift_target = sapien.Pose(
                    current.p + np.array([0, 0, 0.08]), current.q
                )
        elif self.phase == "lift":
            gripper = -1.0
            motion = self.servo(self.lift_target)
            if self.tcp_error(self.lift_target)[0] < 0.015:
                self.phase = "align"
                self.counter = 0
        elif self.phase == "align":
            gripper = -1.0
            peg_target = goal * sapien.Pose([-0.02 - self.half_length, 0, 0])
            motion = self.servo(self.tcp_for_peg(peg_target))
            peg = to_pose(task.peg.pose)
            position_error = np.linalg.norm(peg_target.p - peg.p)
            angle_error = np.linalg.norm(rotation_error(peg.q, peg_target.q))
            if position_error < 0.002 and angle_error < 0.02:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter >= 3:
                self.phase = "insert"
        else:
            gripper = -1.0
            relative_peg = goal.inv() * to_pose(task.peg.pose)
            lateral_error = np.linalg.norm(relative_peg.p[1:])
            advance = 0.006 if lateral_error < 0.0015 else 0.0
            peg_target = goal * sapien.Pose([relative_peg.p[0] + advance, 0, 0])
            motion = self.servo(self.tcp_for_peg(peg_target))

        action = np.concatenate([motion, [gripper]]).astype(np.float32)
        native_action = torch.as_tensor(action, device=task.device).clamp(-1.0, 1.0)
        return native_action.reshape(self.env.action_space.shape)


def make_policy():
    """Factory for --policy data_collection.scripted_policy:make_policy."""
    return ScriptedPegPolicy()
