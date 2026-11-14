import argparse
import json
from pathlib import Path

import numpy as np

from kin_filter import filter_episodes, q_conj, q_rot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--cfg", default="/workspace/kin_best.json")
    p.add_argument("--episodes", type=int, default=400)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    n = min(a.episodes, len(off) - 1)
    g, gt, counts, off = g[:off[n]], gt[:off[n]], counts[:off[n]], off[:n + 1]
    f = filter_episodes(g, off, counts, json.loads(Path(a.cfg).read_text()))
    half = n // 2
    rows = {"tip_hole": [], "hole": [], "peg_tip_local": []}
    for i, (s, e) in enumerate(zip(off[:-1], off[1:])):
        late = np.arange(s, e)[np.arange(e - s) >= 80]
        if len(late) < 5:
            continue
        hq = gt[late[-1], 28:32]
        th = np.mean([q_rot(q_conj(hq), f[t, 22:25] - gt[t, 22:25]) for t in late], axis=0) * 1000
        ho = np.mean([q_rot(q_conj(hq), f[t, 25:28] - gt[t, 25:28]) for t in late], axis=0) * 1000
        tq = gt[late[-1], 3:7]
        pt = np.mean([q_rot(q_conj(tq), (f[t, 22:25] + f[t, 25:28]) - (gt[t, 22:25] + gt[t, 25:28])) for t in late], axis=0) * 1000
        rows["tip_hole"].append(th)
        rows["hole"].append(ho)
        rows["peg_tip_local"].append(pt)
    for name, v in rows.items():
        v = np.array(v)
        a_, b_ = v[:len(v) // 2], v[len(v) // 2:]
        lat = np.linalg.norm(v[:, 1:], axis=1)
        print(f"{name} episodes {len(v)} bias_xyz {np.round(v.mean(0), 2).tolist()} sd_xyz {np.round(v.std(0), 2).tolist()} "
              f"bias_first_half {np.round(a_.mean(0), 2).tolist()} bias_second_half {np.round(b_.mean(0), 2).tolist()}", flush=True)
        if name == "tip_hole":
            c = v - a_.mean(0)
            lat_c = np.linalg.norm(c[len(c) // 2:, 1:], axis=1)
            lat_h = lat[len(lat) // 2:]
            print(f"tip_hole lateral_mm median {np.median(lat_h):.2f} under_1.5mm {100 * np.mean(lat_h < 1.5):.1f}% "
                  f"after_bias_correction median {np.median(lat_c):.2f} under_1.5mm {100 * np.mean(lat_c < 1.5):.1f}%", flush=True)


if __name__ == "__main__":
    main()
