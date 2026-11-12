import argparse
import itertools
import json
from pathlib import Path

import numpy as np

BUCKETS = [(0, 40), (40, 80), (80, 120), (120, 180)]


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def q_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def q_rot(q, v):
    w, xyz = q[0], q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def unit(q):
    q = q / max(np.linalg.norm(q), 1e-9)
    return q if q[0] >= 0 else -q


def hole_features(p, q):
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([1.0, p[0], p[1], p[2], np.sin(yaw), np.cos(yaw)])


class Pool:
    def __init__(self, robust):
        self.robust, self.p, self.q = robust, [], []

    def add(self, p, q):
        if self.q and np.dot(q, self.q[0]) < 0:
            q = -q
        self.p.append(p)
        self.q.append(q)

    def get(self):
        p = np.median(self.p, axis=0) if self.robust else np.mean(self.p, axis=0)
        return p, unit(np.mean(self.q, axis=0))


class KinFilter:
    def __init__(self, min_box=0, min_peg=0, robust=False, close_ratio=0.75, stable=5e-4, prior=None, hole_corr=None, hole_frames=0, peg_frames=0):
        self.min_box, self.min_peg, self.robust = min_box, min_peg, robust
        self.prior = None if prior is None else (np.asarray(prior["coef"], np.float64), np.asarray(prior["use"], bool))
        self.hole_corr = None if hole_corr is None else np.asarray(hole_corr, np.float64)
        self.close_ratio, self.stable = close_ratio, stable
        self.hole_frames, self.peg_frames = hole_frames, peg_frames
        self.reset()

    def reset(self):
        self.hole_good, self.hole_all = Pool(self.robust), Pool(self.robust)
        self.peg_good, self.peg_all = Pool(self.robust), Pool(self.robust)
        self.shape = []
        self.gmax, self.gprev = 0.0, None

    def step(self, s, counts=None):
        s = np.asarray(s, np.float64)
        tcp, tq, grip, o = s[0:3], unit(s[3:7]), float(s[13] + s[14]), s[15:36]
        n_peg, n_box = (99, 99) if counts is None else (int(counts[0]), int(counts[1]))
        hole_w, hq = o[10:13] + tcp, unit(o[13:17])
        if not self.hole_frames or len(self.hole_all.p) < self.hole_frames:
            self.hole_all.add(hole_w, hq)
        if n_box >= self.min_box and (not self.hole_frames or len(self.hole_good.p) < self.hole_frames):
            self.hole_good.add(hole_w, hq)
        hole_p, hole_q = (self.hole_good if self.hole_good.p else self.hole_all).get()
        if self.hole_corr is not None:
            hole_p = hole_p + hole_features(hole_p, hole_q) @ self.hole_corr
        self.shape.append(o[17:21])
        shape = np.mean(self.shape, axis=0)
        self.gmax = max(self.gmax, grip)
        held = grip < self.close_ratio * self.gmax and self.gprev is not None and abs(grip - self.gprev) < self.stable
        self.gprev = grip
        peg_rel, peg_q = o[0:3], unit(o[3:7])
        if held:
            local, lq = q_rot(q_conj(tq), peg_rel), unit(q_mul(q_conj(tq), peg_q))
            if not self.peg_frames or len(self.peg_all.p) < self.peg_frames:
                self.peg_all.add(local, lq)
            if n_peg >= self.min_peg and (not self.peg_frames or len(self.peg_good.p) < self.peg_frames):
                self.peg_good.add(local, lq)
            lp, lq = (self.peg_good if self.peg_good.p else self.peg_all).get()
            peg_rel, peg_q = q_rot(tq, lp), unit(q_mul(tq, lq))
        elif self.peg_all.p:
            self.peg_good, self.peg_all = Pool(self.robust), Pool(self.robust)
        tip = tcp + peg_rel + q_rot(peg_q, np.array([shape[0], 0.0, 0.0]))
        if held and self.prior is not None:
            coef, use = self.prior
            loc = q_rot(q_conj(tq), tip - tcp)
            loc = np.where(use, np.array([1.0, shape[0]]) @ coef, loc)
            tip = tcp + q_rot(tq, loc)
        out = np.concatenate([peg_rel, peg_q, tip - hole_p, hole_p - tcp, hole_q, shape])
        return np.concatenate([s[:15], out]).astype(np.float32)


def filter_episodes(x, off, counts=None, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    out = np.empty_like(x)
    for a, b in zip(off[:-1], off[1:]):
        f = KinFilter(**cfg)
        for t in range(a, b):
            out[t] = f.step(x[t], None if counts is None else counts[t])
    return out


def errors(x, gt, step, sl):
    e = np.linalg.norm(x[:, sl] - gt[:, sl], axis=1) * 1000
    return float(e.mean()), [round(float(e[(step >= lo) & (step < hi)].mean()), 2) for lo, hi in BUCKETS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default="/workspace/kin_best.json")
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    n = min(a.episodes, len(off) - 1)
    end = off[n]
    g, gt, counts, off = g[:end], gt[:end], counts[:end], off[:n + 1]
    step = np.concatenate([np.arange(e - s) for s, e in zip(off[:-1], off[1:])])
    late = step >= 80
    for k, name in ((0, "peg"), (1, "box")):
        print(f"observed_{name} by_steps " + str([round(float(counts[(step >= lo) & (step < hi), k].mean()), 1) for lo, hi in BUCKETS]), flush=True)
    for name, sl in (("tip_hole", slice(22, 25)), ("peg_pos", slice(15, 18)), ("hole_pos", slice(25, 28))):
        m, per = errors(g, gt, step, sl)
        print(f"raw {name} mean {m:.2f} by_steps {per}", flush=True)
    results = []
    for min_box, min_peg, robust in itertools.product([0, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5], [False, True]):
        cfg = {"min_box": min_box, "min_peg": min_peg, "robust": robust}
        f = filter_episodes(g, off, counts, cfg)
        e = np.linalg.norm(f[:, 22:25] - gt[:, 22:25], axis=1) * 1000
        results.append((float(e[late].mean()), float(e.mean()), cfg, f))
    results.sort(key=lambda r: r[0])
    for late_e, all_e, cfg, _ in results[:5]:
        print(f"cfg {json.dumps(cfg)} tip_hole_late {late_e:.2f} all {all_e:.2f}", flush=True)
    base = [r for r in results if r[2] == {"min_box": 0, "min_peg": 0, "robust": False}][0]
    print(f"plain_filter tip_hole_late {base[0]:.2f} all {base[1]:.2f}", flush=True)
    best = results[0]
    for name, sl in (("tip_hole", slice(22, 25)), ("peg_pos", slice(15, 18)), ("hole_pos", slice(25, 28))):
        m, per = errors(best[3], gt, step, sl)
        print(f"best {name} mean {m:.2f} by_steps {per}", flush=True)
    Path(a.out).write_text(json.dumps(best[2]))
    print(f"saved {a.out} {json.dumps(best[2])}", flush=True)


if __name__ == "__main__":
    main()
