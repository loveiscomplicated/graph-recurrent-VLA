from __future__ import annotations

import numpy as np
import sapien
from transforms3d.quaternions import qinverse, qmult, quat2axangle

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from recurrent.common import peg_task, to_sapien

OPEN = 1.0
CLOSED = -1.0


def rotvec_between(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    dq = qmult(q_to, qinverse(q_from))
    if dq[0] < 0:
        dq = -dq
    axis, angle = quat2axangle(dq)
    return np.asarray(axis) * angle


class ScriptedExpert:
    def __init__(self, env, max_trans=0.02, max_rot=0.15, insert_speed=0.006, gain=3.0):
        self.env = env
        self.gain = gain
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.insert_speed = insert_speed
        self.reset()

    def reset(self):
        task = peg_task(self.env)
        obb = get_actor_obb(task.peg)
        closing_hint = task.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        info = compute_grasp_info_by_obb(
            obb, approaching=np.array([0, 0, -1]), target_closing=closing_hint, depth=0.025
        )
        grasp = task.agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
        half_len = task.peg_half_sizes[0, 0].item()
        self.grasp = grasp * sapien.Pose([-max(0.05, half_len / 2 + 0.01), 0, 0])
        self.half_len = half_len
        self.phase = "reach"
        self.counter = 0
        self.lift_target = None

    def _tcp(self):
        return to_sapien(peg_task(self.env).agent.tcp.pose)

    def _servo(self, target: sapien.Pose, max_trans: float) -> np.ndarray:
        tcp = self._tcp()
        dp = (target.p - tcp.p) * self.gain
        n = np.linalg.norm(dp)
        if n > max_trans:
            dp = dp / n * max_trans
        rv = rotvec_between(tcp.q, target.q)
        n = np.linalg.norm(rv)
        if n > self.max_rot:
            rv = rv / n * self.max_rot
        return np.concatenate([dp / 0.1, -rv / 0.1])

    def _err(self, target: sapien.Pose):
        tcp = self._tcp()
        return np.linalg.norm(target.p - tcp.p), np.linalg.norm(rotvec_between(tcp.q, target.q))

    def _tcp_for_peg(self, peg_target: sapien.Pose) -> sapien.Pose:
        task = peg_task(self.env)
        return peg_target * to_sapien(task.peg.pose).inv() * self._tcp()

    def _peg_err(self, peg_target: sapien.Pose):
        peg = to_sapien(peg_task(self.env).peg.pose)
        return np.linalg.norm(peg_target.p - peg.p), np.linalg.norm(rotvec_between(peg.q, peg_target.q))

    def act(self) -> np.ndarray:
        task = peg_task(self.env)
        goal = to_sapien(task.goal_pose)
        grip = OPEN
        if self.phase == "reach":
            target = self.grasp * sapien.Pose([0, 0, -0.05])
            move = self._servo(target, self.max_trans)
            dp, dr = self._err(target)
            if dp < 0.01 and dr < 0.05:
                self.phase = "descend"
        elif self.phase == "descend":
            target = self.grasp
            move = self._servo(target, 0.01)
            dp, dr = self._err(target)
            if dp < 0.004 and dr < 0.03:
                self.phase = "close"
        elif self.phase == "close":
            move = np.zeros(6)
            grip = CLOSED
            self.counter += 1
            if self.counter >= 8:
                self.phase = "lift"
                tcp = self._tcp()
                self.lift_target = sapien.Pose(tcp.p + np.array([0, 0, 0.08]), tcp.q)
        elif self.phase == "lift":
            grip = CLOSED
            move = self._servo(self.lift_target, self.max_trans)
            if self._err(self.lift_target)[0] < 0.015:
                self.phase = "align"
                self.counter = 0
        elif self.phase == "align":
            grip = CLOSED
            peg_target = goal * sapien.Pose([-0.02 - self.half_len, 0, 0])
            move = self._servo(self._tcp_for_peg(peg_target), self.max_trans)
            dp, dr = self._peg_err(peg_target)
            self.counter = self.counter + 1 if (dp < 0.002 and dr < 0.02) else 0
            if self.counter >= 3:
                self.phase = "insert"
        else:
            grip = CLOSED
            rel = goal.inv() * to_sapien(task.peg.pose)
            lateral = np.linalg.norm(rel.p[1:])
            advance = self.insert_speed if lateral < 0.0015 else 0.0
            peg_target = goal * sapien.Pose([rel.p[0] + advance, 0, 0])
            move = self._servo(self._tcp_for_peg(peg_target), self.max_trans)
        return np.concatenate([move, [grip]]).astype(np.float32)
