import numpy as np

from kin_filter import q_conj, q_rot, unit


def servo(delta_world, limit=0.02):
    t = np.asarray(delta_world, np.float64) * 3.0
    n = np.linalg.norm(t)
    if n > limit:
        t *= limit / n
    return t / 0.1


class SpiralSearch:
    def __init__(self, near=0.15, window=10, stall=0.001, forward=0.02, press=0.002, found=0.003,
                 pitch=0.002, spacing=0.0015, r_max=0.009, push=0.006, jam=15):
        self.near, self.window, self.stall, self.forward = near, window, stall, forward
        self.press, self.found, self.pitch, self.spacing, self.r_max = press, found, pitch, spacing, r_max
        self.push, self.jam = push, jam
        self.reset()

    def reset(self):
        self.mode, self.hist, self.theta, self.origin, self.x0 = "dec", [], 0.0, None, None
        self.searches, self.pushes, self.push_hist, self.fwd = 0, 0, [], []

    def start(self, x, lat):
        self.mode, self.theta, self.origin, self.x0 = "search", 0.0, lat.copy(), x
        self.searches += 1

    def step(self, a_dec, f):
        hq = unit(np.asarray(f[28:32], np.float64))
        rel = q_rot(q_conj(hq), np.asarray(f[22:25], np.float64))
        x, lat = rel[0], rel[1:]
        self.hist.append(x)
        if self.mode == "dec":
            axis = q_rot(hq, np.array([1.0, 0.0, 0.0]))
            self.fwd.append(float(np.dot(a_dec[:3], axis)) > self.forward and a_dec[6] < 0)
            w = self.window
            if (np.linalg.norm(rel) < self.near and len(self.hist) > w and all(self.fwd[-w:])
                    and self.hist[-1] - self.hist[-1 - w] < self.stall):
                self.start(x, lat)
            else:
                return a_dec
        if self.mode == "push":
            self.push_hist.append(x)
            if len(self.push_hist) > self.jam and self.push_hist[-1] - self.push_hist[-1 - self.jam] < self.stall:
                self.start(x, lat)
            else:
                return np.concatenate([servo(q_rot(hq, np.array([self.push, 0.0, 0.0]))), a_dec[3:6], [-1.0]]).astype(np.float32)
        if x - self.x0 > self.found:
            self.mode, self.push_hist = "push", []
            self.pushes += 1
            return np.concatenate([servo(q_rot(hq, np.array([self.push, 0.0, 0.0]))), a_dec[3:6], [-1.0]]).astype(np.float32)
        r = self.pitch * self.theta / (2 * np.pi)
        if r > self.r_max:
            self.theta = 0.0
        self.theta += self.spacing / max(self.pitch * self.theta / (2 * np.pi), 0.001)
        r = self.pitch * self.theta / (2 * np.pi)
        target = self.origin + r * np.array([np.cos(self.theta), np.sin(self.theta)])
        d = np.array([self.x0 + self.press - x, target[0] - lat[0], target[1] - lat[1]])
        return np.concatenate([servo(q_rot(hq, d)), a_dec[3:6], [-1.0]]).astype(np.float32)
