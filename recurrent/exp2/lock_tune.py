import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from grasp_prior import score
from kin_filter import filter_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data_kin/train.npz")
    p.add_argument("--cfg", default="/workspace/kin_final.json")
    p.add_argument("--out", default="/workspace/kin_lock.json")
    p.add_argument("--episodes", type=int, default=300)
    a = p.parse_args()
    with np.load(a.data, allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
    n = min(a.episodes, len(off) - 1)
    g, gt, counts, off = g[:off[n]], gt[:off[n]], counts[:off[n]], off[:n + 1]
    step = np.concatenate([np.arange(e - s) for s, e in zip(off[:-1], off[1:])])
    late = step >= 120
    base = json.loads(Path(a.cfg).read_text())
    res = []
    for hf, pf in itertools.product([0, 20, 40, 60, 80], [0, 10, 20, 40]):
        cfg = dict(base, hole_frames=hf, peg_frames=pf)
        f = filter_episodes(g, off, counts, cfg)
        e = np.linalg.norm(f[:, 22:25] - gt[:, 22:25], axis=1) * 1000
        s = score(f, gt, off, range(n))
        res.append((float(e[late].mean()), hf, pf, s))
        print(f"hole_frames {hf} peg_frames {pf} tip_hole_120plus {e[late].mean():.2f} {s}", flush=True)
    res.sort()
    _, hf, pf, s = res[0]
    Path(a.out).write_text(json.dumps(dict(base, hole_frames=hf, peg_frames=pf)))
    print(f"best hole_frames {hf} peg_frames {pf} saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
