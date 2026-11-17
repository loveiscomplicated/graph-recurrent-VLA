import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env
from exp2_eval import gt_object21, robot15


def work(args):
    data_dir, split, items = args
    env = make_peg_insertion_env(device="cpu", obs_mode="state", control_mode="pd_ee_delta_pose", max_episode_steps=1000)
    with np.load(Path(data_dir) / f"{split}.npz", allow_pickle=False) as d:
        actions = d["expert_action"]
    out = []
    for i, seed, s, e in items:
        env.reset(seed=int(seed))
        rows = []
        for t in range(s, e):
            rows.append(np.concatenate([robot15(env), gt_object21(env)]))
            env.step(torch.as_tensor(actions[t][None], device=env.unwrapped.device))
        out.append((i, np.stack(rows).astype(np.float32)))
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data")
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    for split in ("train", "validation", "test"):
        with np.load(Path(a.data) / f"{split}.npz", allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        off, seeds = arrays["episode_offsets"], arrays["reset_seed"]
        items = [(i, seeds[i], off[i], off[i + 1]) for i in range(len(seeds))]
        chunks = [(a.data, split, items[k::a.workers]) for k in range(a.workers)]
        with ctx.Pool(a.workers) as pool:
            parts = [x for r in pool.map(work, chunks) for x in r]
        parts.sort(key=lambda x: x[0])
        gt = np.concatenate([x[1] for x in parts])
        diff = np.abs(gt[:, :15] - arrays["state36_transformer"][:, :15]).max()
        graph_err = np.linalg.norm(gt[:, 22:25] - arrays["state36_transformer"][:, 22:25], axis=1) * 1000
        print(f"{split} rows {len(gt)} robot15_max_diff {diff:.2e} graph_tip_err_mean_mm {graph_err.mean():.2f} p90 {np.percentile(graph_err, 90):.2f}", flush=True)
        arrays["state36_gt"] = gt
        np.savez(Path(a.data) / f"{split}.npz", **arrays)
    print("DONE")


if __name__ == "__main__":
    main()
