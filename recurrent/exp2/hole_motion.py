import numpy as np
import sapien


def add_args(p):
    p.add_argument("--motion", default="none", choices=["none", "continuous", "jump"])
    p.add_argument("--motion-frac", type=float, default=1.0)
    p.add_argument("--motion-dist", default="0.06,0.12")
    p.add_argument("--motion-dir", default="x", choices=["x", "y", "xy"])
    p.add_argument("--motion-start", type=float, default=0.5)
    p.add_argument("--motion-end", type=float, default=2.5)
    p.add_argument("--motion-duration", type=float, default=1.5)
    p.add_argument("--motion-guard", type=float, default=0.08)


def sample_translation(seed, cfg):
    rng = np.random.default_rng([seed, 7])
    if cfg.motion == "none" or rng.random() >= cfg.motion_frac:
        return None
    lo, hi = (float(v) for v in cfg.motion_dist.split(","))
    dist = rng.uniform(lo, hi)
    if cfg.motion_dir == "xy":
        ang = rng.uniform(0.0, 2.0 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
    else:
        d = np.zeros(3)
        d[0 if cfg.motion_dir == "x" else 1] = rng.choice([-1.0, 1.0])
    return d * dist


class HoleMotion:
    def __init__(self, cfg, translation):
        self.mode, self.translation = cfg.motion, translation
        self.start, self.end, self.duration = cfg.motion_start, cfg.motion_end, cfg.motion_duration
        self.guard = cfg.motion_guard
        self.offset = np.zeros(3)
        self.stopped = translation is None
        self.status = "none" if translation is None else "waiting"
        self.first_step, self.last_step = None, None

    def update(self, t, seconds, tip, hole, phase=None):
        if self.stopped:
            return self.offset.copy()
        if phase == "insert":
            self.stopped, self.status = True, "blocked: insert phase"
            return self.offset.copy()
        if np.linalg.norm(tip - hole) <= self.guard:
            self.stopped, self.status = True, "blocked: peg near hole"
            return self.offset.copy()
        if seconds < self.start:
            return self.offset.copy()
        if seconds >= self.end:
            self.stopped, self.status = True, "window closed"
            return self.offset.copy()
        progress = 1.0 if self.mode == "jump" else min((seconds - self.start) / self.duration, 1.0)
        target = self.translation * progress
        delta = target - self.offset
        den = float(delta @ delta)
        frac = 0.0 if den == 0 else float(np.clip((tip - hole) @ delta / den, 0.0, 1.0))
        if np.linalg.norm(tip - (hole + frac * delta)) <= self.guard:
            self.stopped, self.status = True, "blocked: path near peg"
            return self.offset.copy()
        if self.first_step is None:
            self.first_step = t
        self.offset = target
        self.last_step = t
        self.status = "complete" if progress >= 1.0 else "moving"
        if progress >= 1.0:
            self.stopped = True
        return self.offset.copy()

    def summary(self):
        return {"translation_mm": None if self.translation is None else [round(float(v) * 1000, 1) for v in self.translation],
                "moved_mm": round(float(np.linalg.norm(self.offset)) * 1000, 1), "status": self.status,
                "first_step": self.first_step, "last_step": self.last_step}


class BoxMover:
    def __init__(self, task):
        self.task = task
        p = task.box.pose
        self.p0, self.q0 = p.p[0].cpu().numpy().copy(), p.q[0].cpu().numpy().copy()
        self.applied = np.zeros(3)

    def apply(self, offset):
        if np.array_equal(offset, self.applied):
            return False
        self.task.box.set_pose(sapien.Pose(self.p0 + offset, self.q0))
        self.applied = offset.copy()
        return True
