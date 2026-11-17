import argparse
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from data_collection.object21_bc import predict_graphs
from data_collection.scripted_policy import make_policy
from envs.maniskill import make_peg_insertion_env
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.fixed_id_gnn import load_temporal
from visual.corner_detection import TorchCornerDetector


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Graph:
    def __init__(self, gnn_path, corner_path, device):
        self.device = torch.device(device)
        ck = torch.load(gnn_path, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(ck["config"])
        cfg["temporal"]["checkpoint"] = corner_path
        self.temporal, _ = load_temporal(cfg, self.device)
        self.gnn = VertexEdgeGatedMPNN(**cfg["gnn"]).to(self.device).eval().requires_grad_(False)
        self.gnn.load_state_dict(ck["model"], strict=True)
        self.arch = cfg["temporal"]["architecture"]
        self.detector = TorchCornerDetector(device)
        self.reset()

    def reset(self):
        self.tracker = DetectedCornerTracker(self.temporal, 0.030, self.device)

    @torch.no_grad()
    def step(self, obs):
        img, par = obs["sensor_data"]["base_camera"], obs["sensor_param"]["base_camera"]
        rgb, depth = to_np(img["rgb"])[0], to_np(img["depth"])[0]
        robot = np.concatenate((to_np(obs["extra"]["tcp_pose"])[0], to_np(obs["agent"]["qpos"])[0, -2:])).astype(np.float32)
        t0 = time.perf_counter()
        res = self.detector(rgb, depth, to_np(par["intrinsic_cv"])[0], to_np(par["extrinsic_cv"])[0])
        points, kinds = res.xyz.cpu().numpy(), res.object_id.cpu().numpy()
        sync()
        t1 = time.perf_counter()
        r = self.tracker.step(points, kinds, robot)
        tk = np.array([t.kind for t in self.tracker.tracks])
        ob = np.asarray(r["observed"], bool)
        self.counts = (int(((tk == 0) & ob).sum()), int(((tk == 1) & ob).sum()))
        sync()
        t2 = time.perf_counter()
        frame = (r["positions"], [t.kind for t in self.tracker.tracks], r["observed"])
        o21 = predict_graphs(self.gnn, [frame], robot[None, :3], 1, self.device)[0]
        sync()
        t3 = time.perf_counter()
        return o21, robot, (t1 - t0, t2 - t1, t3 - t2), len(self.tracker.tracks)


def make_env(hand):
    return make_peg_insertion_env(device="cpu", render_backend="cuda", obs_mode="rgb+depth",
                                  control_mode="pd_ee_delta_pose", max_episode_steps=1000,
                                  sensor_configs={"base_camera": dict(width=512, height=512),
                                                  "hand_camera": dict(width=hand, height=hand)})


def act(env, policy, obs):
    return torch.as_tensor(policy.act(obs), device=env.unwrapped.device)


def save_policy(policy):
    out = {}
    for k, v in vars(policy).items():
        if k == "env":
            continue
        try:
            out[k] = copy.deepcopy(v)
        except Exception:
            out[k] = v
    return out


def run_episode(env, policy, graph, seed, steps, branch_at=None, branch_len=30):
    obs, _ = env.reset(seed=seed)
    policy.reset(env, seed)
    graph.reset()
    rows, branch = [], None
    for t in range(steps):
        if branch_at is not None and t == branch_at:
            saved = (env.unwrapped.get_state_dict(), save_policy(policy), copy.deepcopy(graph.tracker.tracks), obs)
            ref = []
            for _ in range(branch_len):
                obs, *_ = env.step(act(env, policy, obs))
                o21, robot, _, _ = graph.step(obs)
                ref.append(np.concatenate([robot[:3], o21]))
            state, pol, tracks, obs = saved
            env.unwrapped.set_state_dict(state)
            policy.__dict__.update(pol)
            graph.tracker.tracks = copy.deepcopy(tracks)
            rep = []
            for _ in range(branch_len):
                obs, *_ = env.step(act(env, policy, obs))
                o21, robot, _, _ = graph.step(obs)
                rep.append(np.concatenate([robot[:3], o21]))
            ref, rep = np.array(ref), np.array(rep)
            branch = {"tcp_max_diff_m": float(np.abs(ref[:, :3] - rep[:, :3]).max()),
                      "object21_max_diff": float(np.abs(ref[:, 3:] - rep[:, 3:]).max())}
            break
        a = act(env, policy, obs)
        s0 = time.perf_counter()
        obs, _, _, _, info = env.step(a)
        s1 = time.perf_counter()
        _, _, (d, k, g), n = graph.step(obs)
        rows.append((s1 - s0, d, k, g, n))
    return np.array(rows), branch


def replay(env, graph, data_dir, count):
    from pathlib import Path
    with np.load(Path(data_dir) / "test.npz", allow_pickle=False) as d:
        seeds, off = d["reset_seed"], d["episode_offsets"]
        o21 = d["object21_transformer"] if "object21_transformer" in d.files else d["state36_transformer"][:, 15:]
        actions = d["expert_action"]
    out = []
    for i in range(count):
        s, e = off[i], off[i + 1]
        obs, _ = env.reset(seed=int(seeds[i]))
        graph.reset()
        live = []
        for t in range(e - s):
            live.append(graph.step(obs)[0])
            obs, *_ = env.step(torch.as_tensor(actions[s + t][None], device=env.unwrapped.device))
        live, ref = np.array(live), o21[s:e]
        tip = np.linalg.norm(live[:, 7:10] - ref[:, 7:10], axis=-1) * 1000
        hole = np.linalg.norm(live[:, 10:13] - ref[:, 10:13], axis=-1) * 1000
        out.append({"seed": int(seeds[i]), "steps": int(e - s), "tip_mean_mm": round(float(tip.mean()), 3),
                    "tip_max_mm": round(float(tip.max()), 3), "hole_mean_mm": round(float(hole.mean()), 3),
                    "hole_max_mm": round(float(hole.max()), 3)})
    return out


def worker(args, seed, queue):
    torch.set_num_threads(1)
    env = make_env(args.hand)
    graph = Graph(args.gnn, args.corner, "cuda")
    policy = make_policy()
    run_episode(env, policy, graph, seed, 5)
    start = time.perf_counter()
    rows, _ = run_episode(env, policy, graph, seed, args.steps)
    queue.put((time.perf_counter() - start, float(rows[:, :4].sum(1).mean())))
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paths", default="/workspace/graph_ckpt/paths.json")
    p.add_argument("--data", default="")
    p.add_argument("--replay", type=int, default=3)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--hand", type=int, default=512)
    p.add_argument("--seed", type=int, default=500000)
    p.add_argument("--workers", default="1,4,8,12,16")
    p.add_argument("--episodes", type=int, default=5000)
    a = p.parse_args()
    paths = json.load(open(a.paths))
    a.gnn, a.corner = paths["gnn"], paths["corner"]
    torch.set_num_threads(4)
    env = make_env(a.hand)
    graph = Graph(a.gnn, a.corner, "cuda")
    policy = make_policy()
    print(f"graph {graph.arch} hand_camera {a.hand}", flush=True)
    if a.data:
        rows = replay(env, graph, a.data, a.replay)
        ok = all(r["tip_mean_mm"] < 0.5 and r["hole_mean_mm"] < 0.5 for r in rows)
        print(("PASS" if ok else "FAIL") + " replay " + json.dumps(rows), flush=True)
    run_episode(env, policy, graph, a.seed, 5)
    rows, _ = run_episode(env, policy, graph, a.seed, a.steps)
    print("steps  sim+render_ms  detector_ms  tracker_ms  gnn_ms  total_ms  tracks", flush=True)
    for lo in range(0, a.steps, 50):
        r = rows[lo:lo + 50]
        m = r.mean(0) * 1000
        print(f"{lo:3d}-{lo + len(r) - 1:3d}  {m[0]:8.1f}  {m[1]:8.1f}  {m[2]:8.1f}  {m[3]:8.1f}  {m[:4].sum():8.1f}  {r[:, 4].mean():6.1f}", flush=True)
    single = rows[:, :4].sum(1)
    print(f"episode_s_150 {single[:150].sum():.1f} episode_s_250 {single.sum():.1f}", flush=True)
    _, branch = run_episode(env, policy, graph, a.seed + 1, 200, branch_at=60)
    ok = branch["tcp_max_diff_m"] < 1e-4
    print(("PASS" if ok else "FAIL") + " restore " + json.dumps(branch), flush=True)
    env.close()
    ctx = mp.get_context("spawn")
    best = None
    for w in [int(x) for x in a.workers.split(",")]:
        queue = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(a, a.seed + 10 + i, queue)) for i in range(w)]
        start = time.perf_counter()
        for pr in procs:
            pr.start()
        res = [queue.get() for _ in procs]
        for pr in procs:
            pr.join()
        wall = time.perf_counter() - start
        sps = w * a.steps / max(r[0] for r in res)
        mean_step = np.mean([r[1] for r in res]) * 1000
        print(f"workers {w:2d} steps_per_s {sps:7.1f} mean_step_ms {mean_step:7.1f} wall_s {wall:6.1f}", flush=True)
        if best is None or sps > best[1]:
            best = (w, sps)
    for factor, name in ((1.0, "no_branch"), (0.6, "branch")):
        hours = a.episodes * 150 * factor / best[1] / 3600
        print(f"estimate {name} episodes {a.episodes} avg_steps 150 workers {best[0]} hours {hours:.2f}", flush=True)


if __name__ == "__main__":
    main()
