import argparse
import json
from pathlib import Path

import numpy as np

from grasp_prior import score
from kin_filter import filter_episodes, hole_features


def fit(f, gt, off, episodes, lam):
    X, Y = [], []
    for i in episodes:
        t = off[i + 1] - 1
        tcp = gt[t, 0:3]
        hp = f[t, 25:28] + tcp
        X.append(hole_features(hp, f[t, 28:32]))
        Y.append(gt[t, 25:28] + tcp - hp)
    X, Y = np.array(X), np.array(Y)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--cfg", default="/workspace/kin_prior.json")
    p.add_argument("--out", default="/workspace/kin_final.json")
    p.add_argument("--min-gain", type=float, default=2.0)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    cfg = json.loads(Path(a.cfg).read_text())
    n = len(off) - 1
    f = filter_episodes(g, off, counts, cfg)
    train, test = range(0, n // 2), range(n // 2, n)
    sub = np.concatenate([np.arange(off[i], off[i + 1]) for i in test])
    sub_off = np.concatenate([[0], np.cumsum([off[i + 1] - off[i] for i in test])])
    base = score(f[sub], gt[sub], sub_off, range(len(test)))
    print("heldout prior      " + base, flush=True)
    best = (float(base.split("under_1.5mm ")[1].rstrip("%")), None, None)
    for lam in (1e-4, 1e-3, 1e-2):
        W = fit(f, gt, off, train, lam)
        fc = filter_episodes(g[sub], sub_off, counts[sub], dict(cfg, hole_corr=W.tolist()))
        s = score(fc, gt[sub], sub_off, range(len(test)))
        u = float(s.split("under_1.5mm ")[1].rstrip("%"))
        print(f"heldout +hole lam {lam:g} " + s, flush=True)
        if u > best[0] + (a.min_gain if best[1] is None else 0):
            best = (u, lam, W)
    if best[1] is None:
        Path(a.out).write_text(json.dumps(cfg))
        print(f"hole correction not used, saved {a.out}", flush=True)
    else:
        W = fit(f, gt, off, range(n), best[1])
        Path(a.out).write_text(json.dumps(dict(cfg, hole_corr=W.tolist())))
        print(f"hole correction used lam {best[1]:g} heldout_under_1.5mm {best[0]:.1f}%, saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
