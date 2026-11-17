from __future__ import annotations

import numpy as np
import sapien

from recurrent.common import is_success, peg_task, to_sapien

THRESHOLDS = {
    "retention_steps": 5,
    "lift_height_m": 0.03,
    "alignment_transverse_m": 0.01,
    "alignment_axis_rad": float(np.deg2rad(10)),
    "alignment_entry_distance_m": 0.04,
}

FAILURE_ORDER = ("acquisition", "retention", "transport", "alignment", "insertion")


def snapshot(env) -> dict:
    task = peg_task(env)
    peg = to_sapien(task.peg.pose)
    hole = to_sapien(task.box_hole_pose)
    half = task.peg_half_sizes[0].cpu().numpy()
    radius = float(task.box_hole_radii.reshape(-1)[0])
    depth = float(half[0])
    head = peg * sapien.Pose([depth, 0.0, 0.0])
    rel = (hole.inv() * head).p
    peg_x = peg.to_transformation_matrix()[:3, 0]
    hole_x = hole.to_transformation_matrix()[:3, 0]
    axis = float(np.arccos(np.clip(np.dot(peg_x, hole_x), -1.0, 1.0)))
    transverse = float(np.linalg.norm(rel[1:]))
    return {
        "grasped": bool(task.agent.is_grasping(task.peg).reshape(-1)[0].item()),
        "lifted": float(peg.p[2] - half[2]) > THRESHOLDS["lift_height_m"],
        "transverse": transverse,
        "progress": float(np.clip((rel[0] + depth) / (2 * depth), 0.0, 1.0)),
        "alignment": transverse <= THRESHOLDS["alignment_transverse_m"]
        and axis <= THRESHOLDS["alignment_axis_rad"]
        and abs(float(rel[0]) + depth) <= THRESHOLDS["alignment_entry_distance_m"],
        "insertion": float(rel[0]) >= -depth and abs(float(rel[1])) <= radius and abs(float(rel[2])) <= radius,
    }


class StageTracker:
    def __init__(self):
        self.run = 0
        self.acquisition = False
        self.transport = False
        self.alignment = False
        self.insertion = False
        self.success = False
        self.losses = 0
        self.previous = False
        self.last_grasped = False
        self.transport_step = None
        self.min_transverse = float("inf")
        self.max_progress = 0.0
        self.t = 0

    def update(self, env, info=None) -> dict:
        s = snapshot(env)
        self.run = self.run + 1 if s["grasped"] else 0
        self.acquisition |= self.run >= THRESHOLDS["retention_steps"]
        moved = self.run >= THRESHOLDS["retention_steps"] and s["lifted"]
        if moved and not self.transport:
            self.transport_step = self.t
        self.transport |= moved
        self.alignment |= s["alignment"]
        self.insertion |= s["insertion"]
        self.losses += int(self.previous and not s["grasped"])
        self.previous = s["grasped"]
        self.last_grasped = s["grasped"]
        self.min_transverse = min(self.min_transverse, s["transverse"])
        self.max_progress = max(self.max_progress, s["progress"])
        if info is not None:
            self.success |= is_success(info)
        self.t += 1
        return s

    def failure_stage(self):
        if self.success:
            return None
        if not self.acquisition:
            return "acquisition"
        if not self.last_grasped:
            return "retention"
        if not self.transport:
            return "transport"
        return "insertion" if self.alignment else "alignment"

    def summary(self) -> dict:
        return {
            "acquisition": self.acquisition,
            "transport": self.transport,
            "alignment": self.alignment,
            "insertion": self.insertion,
            "grasp_losses": self.losses,
            "failure_stage": self.failure_stage(),
            "transport_step": self.transport_step,
            "min_transverse_m": None if np.isinf(self.min_transverse) else self.min_transverse,
            "max_insertion_progress": self.max_progress,
        }
