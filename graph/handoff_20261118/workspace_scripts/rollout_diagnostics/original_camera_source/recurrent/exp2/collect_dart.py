import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from data_collection.scripted_policy import make_policy
from envs.maniskill import make_peg_insertion_env
from exp2_eval import gt_object21, robot15, shift_box, geometry, is_success, to_np


def make(graph_mode):
    if graph_mode:
        from bench_exp2 import make_env
        return make_env(16)
    return make_peg_insertion_env(device="cpu", obs_mode="state", control_mode="pd_ee_delta_pose", max_episode_steps=1000)


def episode(env, policy, graph, seed, cfg):
    rng = np.random.default_rng([seed, 2])
    plan_rng = np.random.default_rng([seed, 1])
    plan = None
    if plan_rng.random() < cfg.perturb_frac:
        plan = (int(plan_rng.integers(40, 81)), float(plan_rng.choice([-1.0, 1.0]) * plan_rng.uniform(0.01, 0.025)))
    obs, _ = env.reset(seed=seed)
    policy.reset(env, seed)
    if graph is not None:
        graph.reset()
    states, acts, gts, counts, ok = [], [], [], [], False
    for t in range(cfg.horizon):
        if plan is not None and t == plan[0]:
            shift_box(env, geometry(env)[1][3:], abs(plan[1]), np.sign(plan[1]))
        gt = gt_object21(env)
        o21 = graph.step(obs)[0] if graph is not None else gt
        r15 = robot15(env)
        states.append(np.concatenate([r15, o21]))
        gts.append(np.concatenate([r15, gt]))
        counts.append(graph.counts if graph is not None else (0, 0))
        a = np.asarray(to_np(policy.act(obs)), np.float32).reshape(-1)
        acts.append(a.copy())
        a[:6] += rng.normal(0.0, cfg.noise, 6).astype(np.float32)
        obs, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
        if is_success(info):
            ok = True
            break
    return ok, np.stack(states).astype(np.float32), np.stack(acts).astype(np.float32), np.stack(gts).astype(np.float32), np.array(counts, np.int16)


def work(args):
    cfg, seeds = args
    torch.set_num_threads(1)
    env = make(cfg.graph)
    graph = None
    if cfg.graph:
        import json
        from bench_exp2 import Graph
        paths = json.load(open(cfg.paths))
        graph = Graph(paths["gnn"], paths["corner"], "cuda")
    policy = make_policy()
    out = []
    for s in seeds:
        out.append((int(s),) + episode(env, policy, graph, int(s), cfg))
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/data")
    p.add_argument("--out", required=True)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--paths", default="/workspace/graph_ckpt/paths.json")
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--perturb-frac", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--new", type=int, default=0)
    p.add_argument("--plus-dataset", action="store_true")
    p.add_argument("--seed-start", type=int, default=800000)
    cfg = p.parse_args()
    seeds = []
    if not cfg.new or cfg.plus_dataset:
        for split in ("train", "validation", "test"):
            with np.load(Path(cfg.data) / f"{split}.npz", allow_pickle=False) as d:
                seeds += d["reset_seed"].tolist()
    if cfg.new:
        from exp2_eval import accepted_seeds
        Path(cfg.out).mkdir(parents=True, exist_ok=True)
        seeds += accepted_seeds(cfg.new, cfg.seed_start, Path(cfg.out) / "seeds.json")
    if cfg.limit:
        seeds = seeds[:cfg.limit]
    chunks = [(cfg, seeds[k::cfg.workers]) for k in range(cfg.workers)]
    with mp.get_context("spawn").Pool(cfg.workers) as pool:
        rows = [r for part in pool.map(work, chunks) for r in part]
    rows.sort(key=lambda r: r[0])
    kept = [r for r in rows if r[1]]
    off = np.cumsum([0] + [len(r[2]) for r in kept])
    key = "state36_transformer" if cfg.graph else "state36_gt"
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate([r[2] for r in kept]), "expert_action": np.concatenate([r[3] for r in kept]),
              "episode_offsets": off, "reset_seed": np.array([r[0] for r in kept])}
    if cfg.graph:
        arrays["state36_gt"] = np.concatenate([r[4] for r in kept])
        arrays["counts"] = np.concatenate([r[5] for r in kept])
    np.savez(Path(cfg.out) / "train.npz", **arrays)
    print(f"episodes {len(rows)} kept {len(kept)} rows {off[-1]} mean_len {off[-1] / max(len(kept), 1):.1f}", flush=True)


if __name__ == "__main__":
    main()
