"""Policy-independent, world-space box trajectories for controller rollouts."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MotionPlan:
    """A hashed sequence of XYZ offsets indexed by observation step."""

    offsets: np.ndarray

    @property
    def sha256(self) -> str:
        canonical_offsets = self.offsets.astype("<f8", copy=False)
        return hashlib.sha256(canonical_offsets.tobytes()).hexdigest()


def make_motion_plan(
    seed: int,
    settings: Mapping[str, Any],
    horizon: int,
) -> MotionPlan:
    """Sample a direction from a dedicated RNG and interpolate by control step."""
    start = settings["start_step"]
    end = settings["end_step"]
    distance = float(settings["distance_m"])
    if not (type(start) is int and type(end) is int and 0 < start < end < horizon):
        raise ValueError("Expected 0 < motion.start_step < motion.end_step < horizon")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Expected a finite positive motion.distance_m")
    generator = np.random.default_rng([settings["seed_salt"], seed])
    angle = generator.uniform(0.0, 2.0 * np.pi)
    translation = distance * np.array([np.cos(angle), np.sin(angle), 0.0])
    progress = np.clip(
        (np.arange(horizon + 1) - start) / (end - start),
        0.0,
        1.0,
    )
    return MotionPlan(progress[:, None] * translation)


class MovingSurfaceEpisode:
    """Apply the next scripted offset before physics and sensor readout."""

    def __init__(self, env, motion, horizon):
        self.env = env
        self.motion = motion
        self.horizon = horizon
        self.unwrapped = env.unwrapped
        if self.unwrapped.num_envs != 1 or self.unwrapped.gpu_sim_enabled:
            raise ValueError(
                "MovingSurfaceEpisode requires one CPU physics environment"
            )

    def reset(self, *, seed):
        result = self.env.reset(seed=seed)
        self.plan = make_motion_plan(seed, self.motion, self.horizon)
        self.initial_pose = self.unwrapped.box.pose.raw_pose.clone()
        self.step_index = 0
        return result

    def step(self, action):
        from mani_skill.utils.structs import Pose

        index = self.step_index + 1
        offset = self.plan.offsets[index]
        moved = not np.array_equal(offset, self.plan.offsets[self.step_index])
        if moved:
            pose = self.initial_pose.clone()
            pose[0, :3] += pose.new_tensor(offset)
            self.unwrapped.box.set_pose(Pose.create_from_pq(pose[:, :3], pose[:, 3:]))
        result = self.env.step(action)
        self.step_index = index
        if moved:
            self._verify_offset()
        return result

    def _verify_offset(self):
        actual = (self.unwrapped.box.pose.raw_pose - self.initial_pose)[0, :3]
        expected = self.plan.offsets[self.step_index]
        actual = actual.detach().cpu().numpy()
        if not np.allclose(actual, expected, rtol=0, atol=2e-6):
            raise RuntimeError("Simulator box pose differs from the scripted trajectory")
        return actual

    def motion_receipt(self):
        actual = self._verify_offset()
        return {
            "trajectory_sha256": self.plan.sha256,
            "planned_translation_m": self.plan.offsets[-1].tolist(),
            "realized_translation_m": actual.tolist(),
            "motion_completed": self.step_index >= self.motion["end_step"],
            "control_frequency_hz": float(self.unwrapped.control_freq),
        }

    def close(self):
        self.env.close()
