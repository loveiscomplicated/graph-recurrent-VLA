import argparse
import json
from pathlib import Path

import numpy as np

from grasp_diag import held_mask, local_tip
from kin_filter import filter_episodes, q_conj, q_rot, unit

BUCKETS = [(0, 40), (40, 80), (80, 120), (120, 180)]


def fit(gt, off, held, episodes):
    loc, half = [], []
    for i in episodes:
        s, e = off[i], off[i + 1]
        idx = np.arange(s, e)[held[s:e]]
        if len(idx) < 10:
            continue
        loc.append(np.mean([local_tip(gt[t]) for t in idx], axis=0))
        half.append(gt[idx[0], 32])
    X = np.c_[np.ones(len(half)), half]
    coef, *_ = np.linalg.lstsq(X, np.array(loc), rcond=None)
    return coef


def score(x, gt, off, episodes):
    step, lat_end, rows = [], [], []
    for i in episodes:
        s, e = off[i], off[i + 1]
        rows += list(range(s, e))
        step += list(range(e - s))
        late = [t for t in range(s, e) if t - s >= 80]
        if len(late) >= 5:
            hq = gt[late[-1], 28:32]
            v = np.mean([q_rot(q_conj(hq), x[t, 22:25] - gt[t, 22:25]) for t in late], axis=0)
            lat_end.append(np.linalg.norm(v[1:]) * 1000)
    rows, step = np.array(rows), np.array(step)
    e = np.linalg.norm(x[rows, 22:25] - gt[rows, 22:25], axis=1) * 1000
    per = [round(float(e[(step >= lo) & (step < hi)].mean()), 2) for lo, hi in BUCKETS]
    lat_end = np.array(lat_end)
    return f"tip_hole mean {e.mean():.2f} by_steps {per} lateral_median {np.median(lat_end):.2f} under_1.5mm {100 * np.mean(lat_end < 1.5):.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--cfg", default="/workspace/kin_best.json")
    p.add_argument("--out", default="/workspace/kin_prior.json")
    p.add_argument("--use", default="1,1,0")
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    n = len(off) - 1
    held = held_mask(gt, off)
    base = json.loads(Path(a.cfg).read_text())
    train, test = range(0, n // 2), range(n // 2, n)
    use = [bool(int(v)) for v in a.use.split(",")]
    coef = fit(gt, off, held, train)
    print("coef_mm " + json.dumps(np.round(coef * 1000, 3).tolist()), flush=True)
    sub = np.concatenate([np.arange(off[i], off[i + 1]) for i in test])
    sub_off = np.concatenate([[0], np.cumsum([off[i + 1] - off[i] for i in test])])
    gs, gts, cs = g[sub], gt[sub], counts[sub]
    old = filter_episodes(gs, sub_off, cs, base)
    new = filter_episodes(gs, sub_off, cs, dict(base, prior={"coef": coef.tolist(), "use": use}))
    eps = range(len(test))
    print("heldout raw      " + score(gs, gts, sub_off, eps), flush=True)
    print("heldout filter   " + score(old, gts, sub_off, eps), flush=True)
    print("heldout +prior   " + score(new, gts, sub_off, eps), flush=True)
    full = fit(gt, off, held, range(n))
    Path(a.out).write_text(json.dumps(dict(base, prior={"coef": full.tolist(), "use": use})))
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
