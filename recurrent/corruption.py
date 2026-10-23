from __future__ import annotations

from collections import deque

import numpy as np

from recurrent.common import load_config
from recurrent.features import quat_mul, rotvec_to_quat

OBJECT_KEYS = ("peg_pose", "hole_pose")


def level_config(level: str, path="corruption.yaml") -> dict:
    cfg = load_config(path)
    if level not in cfg["levels"]:
        raise ValueError(f"unknown level {level}, choose from {list(cfg['levels'])}")
    out = dict(cfg["levels"][level])
    out["fill_mode"] = cfg.get("fill_mode", "hold")
    out["name"] = level
    return out


class Corruptor:
    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.cfg["delay"] + 1)
        self.last = None
        self.drop_left = 0
        self.age = 0
        self.t = 0

    def force_drop(self, steps: int) -> None:
        if self.t > 0 and steps > 0:
            self.drop_left = max(self.drop_left, int(steps))

    def _noisy(self, pose: np.ndarray, scale: float) -> np.ndarray:
        c = self.cfg
        out = pose.copy()
        if c["pos_noise"] > 0:
            out[:3] += self.rng.normal(0.0, c["pos_noise"] * scale, 3)
        if c["rot_noise"] > 0:
            dq = rotvec_to_quat(self.rng.normal(0.0, c["rot_noise"] * scale, 3))
            out[3:] = quat_mul(dq, out[3:])
        return out

    def _zeros(self, value):
        return np.zeros_like(value)

    def step(self, raw: dict):
        c = self.cfg
        self.history.append({k: raw[k] for k in OBJECT_KEYS})
        observed = self.history[0]
        dropped = False
        if self.t > 0:
            if self.drop_left > 0:
                self.drop_left -= 1
                dropped = True
            elif c["drop_prob"] > 0 and self.rng.random() < c["drop_prob"]:
                lo, hi = c["drop_len"]
                self.drop_left = int(self.rng.integers(lo, hi + 1)) - 1
                dropped = True
        self.t += 1
        out = dict(raw)
        if dropped:
            self.age += 1
            for k in OBJECT_KEYS:
                out[k] = self.last[k] if c["fill_mode"] == "hold" else self._zeros(raw[k])
            return out, 0.0, float(self.age)
        scale = c["burst_scale"] if (c["burst_prob"] > 0 and self.rng.random() < c["burst_prob"]) else 1.0
        for k in OBJECT_KEYS:
            out[k] = self._noisy(observed[k], scale)
        self.last = {k: out[k] for k in OBJECT_KEYS}
        self.age = 0
        return out, 1.0, 0.0


def corrupt_sequence(raw_seq: dict, cfg: dict, rng: np.random.Generator):
    n = len(raw_seq["tcp_pose"])
    corr = Corruptor(cfg, rng)
    out = {k: v.copy() for k, v in raw_seq.items()}
    fresh = np.ones(n, np.float32)
    age = np.zeros(n, np.float32)
    for t in range(n):
        step, fresh[t], age[t] = corr.step({k: v[t] for k, v in raw_seq.items()})
        for k in OBJECT_KEYS:
            out[k][t] = step[k]
    return out, fresh, age
