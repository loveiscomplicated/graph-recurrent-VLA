"""Plan and time RoArm motions in the known static simulation bench."""

from __future__ import annotations

import time

import numpy as np

from .assets import HOME_QPOS
from .collision import MotionRejected, RoArmCollisionModel


class RoArmMotionPlanner:
    """Direct path, home detour, then bounded seeded RRT with certified edges."""

    def __init__(self, collision_model=None):
        self.collision = collision_model or RoArmCollisionModel()

    def plan_joints(self, start, goal, *, seed=0, max_nodes=600, timeout=5.0):
        start, goal = (
            self.collision.validate_qpos(start),
            self.collision.validate_qpos(goal),
        )
        for label, qpos in (("start", start), ("goal", goal)):
            reason = self.collision.rejection(qpos)
            if reason:
                raise MotionRejected(f"Unsafe {label}: {reason}")
        if self.collision.edge_valid(start, goal):
            return np.array([start, goal])
        if (
            self.collision.rejection(HOME_QPOS) is None
            and self.collision.edge_valid(start, HOME_QPOS)
            and self.collision.edge_valid(HOME_QPOS, goal)
        ):
            return np.array([start, HOME_QPOS, goal])
        rng = np.random.default_rng(seed)
        limits = self.collision.kinematics.limits
        nodes = [start]
        parents = [-1]
        deadline = time.monotonic() + timeout
        for _ in range(max_nodes):
            if time.monotonic() > deadline:
                break
            sample = (
                goal
                if rng.random() < 0.25
                else rng.uniform(limits[:, 0] + 0.02, limits[:, 1] - 0.02)
            )
            # Do not open/close the jaw beyond what the endpoints request.
            sample = np.array(sample, copy=True)
            sample[3] = np.clip(
                sample[3], min(start[3], goal[3]), max(start[3], goal[3])
            )
            nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - sample, axis=1)))
            delta = sample - nodes[nearest]
            candidate = nodes[nearest] + delta * min(
                1, 0.25 / max(np.linalg.norm(delta), 1e-12)
            )
            if self.collision.rejection(candidate) or not self.collision.edge_valid(
                nodes[nearest], candidate
            ):
                continue
            nodes.append(candidate)
            parents.append(nearest)
            if self.collision.edge_valid(candidate, goal):
                path = [goal, candidate]
                parent = nearest
                while parent != -1:
                    path.append(nodes[parent])
                    parent = parents[parent]
                return np.asarray(path[::-1])
        raise MotionRejected("No certified route found within the planning budget")

    def plan_position(self, start, target):
        """Try several bounded IK branches; accept only a certified route."""
        last_error = None
        for seed in (
            start,
            HOME_QPOS,
            [0, -0.6, 1.2, start[3]],
            [0, 0.6, 2.0, start[3]],
        ):
            try:
                goal = self.collision.kinematics.solve_position(target, seed)
                goal[3] = start[3]
                return self.plan_joints(start, goal)
            except (MotionRejected, ValueError) as exc:
                last_error = exc
        raise MotionRejected(f"No acceptable IK/path: {last_error}")

    def trajectory(self, path, dt):
        """Quintic time scaling, with zero speed/acceleration at each waypoint."""
        path = np.asarray(path, dtype=float)
        if (
            path.ndim != 2
            or path.shape[1] != 4
            or len(path) < 2
            or not np.isfinite(path).all()
        ):
            raise MotionRejected(
                "Expected a finite path with at least two four-angle waypoints"
            )
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        settings = self.collision.settings
        output = [path[0]]
        for start, goal in zip(path[:-1], path[1:]):
            if (
                self.collision.rejection(start)
                or self.collision.rejection(goal)
                or not self.collision.edge_valid(start, goal)
            ):
                raise MotionRejected("Uncertified path segment")
            distance = np.abs(goal - start).max()
            duration = max(
                dt,
                1.875 * distance / settings.command_speed_rad_s,
                np.sqrt(5.774 * distance / settings.command_acceleration_rad_s2),
            )
            count = int(np.ceil(duration / dt))
            progress = np.arange(1, count + 1) / count
            # Quintic smoothstep has zero velocity/acceleration at both ends.
            scale = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
            output.extend(start + scale[:, None] * (goal - start))
        return np.asarray(output)
