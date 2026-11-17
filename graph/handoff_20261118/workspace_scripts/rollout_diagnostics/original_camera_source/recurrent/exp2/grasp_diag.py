import argparse
import json
from pathlib import Path

import numpy as np

from kin_filter import KinFilter, filter_episodes, q_conj, q_mul, q_rot, unit


def held_mask(x, off):
    m = np.zeros(len(x), bool)
    for s, e in zip(off[:-1], off[1:]):
        g = x[s:e, 13] + x[s:e, 14]
        gmax, prev = 0.0, None
        for t in range(e - s):
            gmax = max(gmax, g[t])
            m[s + t] = prev is not None and g[t] < 0.75 * gmax and abs(g[t] - prev) < 5e-4
            prev = g[t]
    return m


def local_tip(row):
    tcp, tq = row[0:3], unit(row[3:7])
    tip_rel_tcp = row[22:25] + row[25:28]
    return q_rot(q_conj(tq), tip_rel_tcp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--cfg", default="/workspace/kin_best.json")
    p.add_argument("--episodes", type=int, default=600)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    n = min(a.episodes, len(off) - 1)
    g, gt, counts, off = g[:off[n]], gt[:off[n]], counts[:off[n]], off[:n + 1]
    held = held_mask(gt, off)
    f = filter_episodes(g, off, counts, json.loads(Path(a.cfg).read_text()))
    gt_loc, f_loc, half = [], [], []
    within = []
    for s, e in zip(off[:-1], off[1:]):
        idx = np.arange(s, e)[held[s:e]]
        if len(idx) < 10:
            continue
        L = np.array([local_tip(gt[t]) for t in idx])
        within.append(L.std(0))
        gt_loc.append(L.mean(0))
        f_loc.append(local_tip(f[idx[-1]]))
        half.append(gt[idx[0], 32])
    gt_loc, f_loc, half = np.array(gt_loc) * 1000, np.array(f_loc) * 1000, np.array(half) * 1000
    print(f"episodes {len(gt_loc)} peg_half_x_mm min {half.min():.1f} max {half.max():.1f}", flush=True)
    print(f"GT tip_in_hand within_episode_sd_mm {np.round(np.mean(within, axis=0) * 1000, 2).tolist()}", flush=True)
    print(f"GT tip_in_hand across_episode_sd_mm {np.round(gt_loc.std(0), 2).tolist()} mean_mm {np.round(gt_loc.mean(0), 1).tolist()}", flush=True)
    X = np.c_[np.ones(len(half)), half]
    tr, te = slice(0, len(half) // 2), slice(len(half) // 2, None)
    coef, *_ = np.linalg.lstsq(X[tr], gt_loc[tr], rcond=None)
    pred = X[te] @ coef
    prior_err = np.abs(pred - gt_loc[te])
    graph_err = np.abs(f_loc[te] - gt_loc[te])
    print(f"prior_from_half heldout_abs_err_mm mean {np.round(prior_err.mean(0), 2).tolist()} p90 {np.round(np.percentile(prior_err, 90, axis=0), 2).tolist()}", flush=True)
    print(f"filtered_graph heldout_abs_err_mm mean {np.round(graph_err.mean(0), 2).tolist()} p90 {np.round(np.percentile(graph_err, 90, axis=0), 2).tolist()}", flush=True)
    for k, name in enumerate("xyz"):
        r = np.corrcoef(half, gt_loc[:, k])[0, 1]
        print(f"corr half_x vs tip_in_hand_{name} {r:.3f}", flush=True)


if __name__ == "__main__":
    main()
