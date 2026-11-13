"""Motion schedule shared by collection and moving-Hole visualization."""

import numpy as np
import torch
from mani_skill.utils.structs import Pose


class HoleMotion:
    """Schedule a single translation and latch the insertion-proximity guard."""

    def __init__(self, settings):
        self.mode = settings['mode']
        self.start = float(settings['start_seconds'])
        self.end = float(settings['end_seconds'])
        self.duration = float(settings['duration_seconds'])
        self.minimum_distance = float(settings['minimum_distance_m'])
        self.translation = np.asarray(settings['translation_m'], dtype=np.float64)
        if self.mode not in ('continuous', 'jump'):
            raise ValueError("motion.mode must be 'continuous' or 'jump'")
        if not (0 <= self.start < self.end and self.duration > 0 and self.minimum_distance > 0):
            raise ValueError('motion needs 0 <= start_seconds < end_seconds, positive duration and minimum distance')
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError('motion.translation_m must be a finite three-vector')
        self.offset = np.zeros(3, dtype=np.float64)
        self.stopped = False
        self.status = 'waiting'

    def update(self, seconds, peg_tip, hole):
        """Advance only when the full proposed displacement stays outside the guard."""
        peg_tip = np.asarray(peg_tip, dtype=np.float64)
        hole = np.asarray(hole, dtype=np.float64)
        if np.linalg.norm(peg_tip - hole) <= self.minimum_distance:
            self.stopped = True
            self.status = 'blocked: peg near hole'
        if self.stopped:
            return self.offset.copy()
        if seconds < self.start:
            self.status = 'waiting'
            return self.offset.copy()
        if seconds >= self.end:
            self.stopped = True
            self.status = 'window closed'
            return self.offset.copy()
        progress = 1.0 if self.mode == 'jump' else min((seconds - self.start) / self.duration, 1.0)
        target = self.translation * progress
        delta = target - self.offset
        # Conservatively reject the entire move if its path intersects the guard.
        denominator = float(delta @ delta)
        fraction = 0.0 if denominator == 0 else np.clip((peg_tip - hole) @ delta / denominator, 0.0, 1.0)
        closest = hole + fraction * delta
        if np.linalg.norm(peg_tip - closest) <= self.minimum_distance:
            self.stopped = True
            self.status = 'blocked: proposed path near peg'
            return self.offset.copy()
        self.offset = target
        self.status = 'complete' if progress >= 1.0 else 'moving'
        return self.offset.copy()


class EpisodeHoleMotion:
    """Apply a motion schedule to one CPU-physics episode's kinematic box."""

    def __init__(self, env, settings, policy):
        self.task = env.unwrapped
        if self.task.num_envs != 1 or self.task.gpu_sim_enabled:
            raise ValueError("hole motion collection requires one CPU physics environment")
        self.motion = HoleMotion(settings)
        if self.motion.start <= 0:
            raise ValueError("collection motion.start_seconds must be greater than zero")
        self.stop_on_insert = settings.get("stop_expert_insert", False)
        if not isinstance(self.stop_on_insert, bool):
            raise ValueError("motion.stop_expert_insert must be a boolean")
        if self.stop_on_insert and not hasattr(policy, "phase"):
            raise ValueError("stop_expert_insert requires a policy exposing its current phase")
        self.initial_pose = self.task.box.pose.raw_pose.clone()
        self.advance(0, policy)

    def advance(self, step, policy):
        """Update the box after a nonterminal step; return whether its pose changed."""
        if self.stop_on_insert and policy.phase == "insert":
            self.motion.stopped = True
            self.motion.status = "blocked: expert insert phase"
        task = self.task
        tip = task.peg_head_pose.p[0].detach().cpu().numpy()
        hole = task.box_hole_pose.p[0].detach().cpu().numpy()
        previous = self.motion.offset.copy()
        offset = self.motion.update(step / task.control_freq, tip, hole)
        if np.array_equal(previous, offset):
            return False
        pose = self.initial_pose.clone()
        pose[0, :3] += torch.as_tensor(offset, dtype=pose.dtype, device=pose.device)
        task.box.set_pose(Pose.create_from_pq(pose[:, :3], pose[:, 3:]))
        return True

    def frame_data(self):
        """Return the offset and guard status aligned with the saved observation."""
        return {"offset_m": self.motion.offset.copy(), "status": self.motion.status}


