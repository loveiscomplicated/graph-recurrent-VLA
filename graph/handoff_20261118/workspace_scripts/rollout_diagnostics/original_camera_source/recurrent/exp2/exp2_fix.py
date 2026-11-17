import argparse
import gc
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bench_exp2 import Graph, make_env, to_np
from data_collection.scripted_policy import make_policy, to_pose
from exp2_eval import Decoder, geometry, is_success, robot15, shift_box

HERE = Path(__file__).resolve().parent
BUCKETS = [(0, 40), (40, 80), (80, 120), (120, 180)]
_W = {}


def say(text):
    print("FIX " + text, flush=True)


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def load_rows(out_dir, cand, seeds):
    rows = {}
    for f in sorted(Path(out_dir).glob("results_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "error" in r or r["cand"] != cand or r["train_seed"] not in seeds:
                continue
            rows[(r["train_seed"], r["env_seed"])] = r
    return list(rows.values())


def rates(rows):
    def pct(values):
        return round(100.0 * float(np.mean(values)), 1) if len(values) else None
    none = [r["none"]["success"] for r in rows]
    move = [r["move"]["success"] if r["move"].get("triggered") else r["none"]["success"] for r in rows]
    reach = [r["none"]["triggered"] for r in rows]
    lifted = [r["none"]["lifted"] for r in rows if "lifted" in r["none"]]
    per_seed = {}
    for r in rows:
        per_seed.setdefault(r["train_seed"], []).append(r["none"]["success"])
    return {"episodes": len(rows), "none": pct(none), "move": pct(move), "reach": pct(reach), "lifted": pct(lifted),
            "none_per_seed": [pct(v) for _, v in sorted(per_seed.items())]}


def buckets(rows, key):
    out = {}
    for name, flag in (("success", True), ("fail", False)):
        eps = [r["none"][key] for r in rows if r["none"]["success"] == flag and r["none"].get(key)]
        vals = []
        for lo, hi in BUCKETS:
            m = [float(np.mean(e[lo:hi])) for e in eps if len(e) > lo]
            vals.append(None if not m else round(float(np.median(m)), 1))
        out[name] = {"n": len(eps), "median": vals}
    return out


def summarize(tag, out_dir, cand, seeds):
    rows = load_rows(out_dir, cand, seeds)
    if not rows:
        say(f"{tag} no rows in {out_dir}")
        return None
    r = rates(rows)
    say(f"{tag} RATES " + json.dumps(r))
    for key, label in (("err_mm", "tip_hole_err"), ("epeg_mm", "peg_pos_err"), ("ehole_mm", "hole_pos_err"), ("ntr", "tracks")):
        b = buckets(rows, key)
        if b["success"]["n"] or b["fail"]["n"]:
            say(f"{tag} {label} steps{BUCKETS} success n={b['success']['n']} {b['success']['median']} "
                f"fail n={b['fail']['n']} {b['fail']['median']}")
    return r


def shifted(seed):
    return bool(np.random.default_rng([int(seed), 1]).random() < 0.5)


def filter_dart(src, dst):
    with np.load(src, allow_pickle=False) as d:
        x, a, off, seeds = d["state36_transformer"], d["expert_action"], d["episode_offsets"], d["reset_seed"]
    keep = [i for i, s in enumerate(seeds) if not shifted(s)]
    rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in keep])
    lens = np.array([off[i + 1] - off[i] for i in keep])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, state36_transformer=x[rows], expert_action=a[rows],
             episode_offsets=np.concatenate([[0], np.cumsum(lens)]), reset_seed=seeds[keep])
    say(f"DART noshift kept {len(keep)}/{len(seeds)} episodes rows {len(rows)}")


def dart_graph_error(graph_file, gt_file):
    if not (Path(graph_file).is_file() and Path(gt_file).is_file()):
        say("DART files missing, skipped")
        return
    with np.load(graph_file, allow_pickle=False) as d:
        g, off, seeds = d["state36_transformer"], d["episode_offsets"], d["reset_seed"]
    with np.load(gt_file, allow_pickle=False) as d:
        t, off_t = d["state36_gt"], d["episode_offsets"]
    if g.shape != t.shape or not np.array_equal(off, off_t):
        say(f"DART files differ in shape {g.shape} {t.shape}, skipped")
        return
    step = np.concatenate([np.arange(b - a) for a, b in zip(off[:-1], off[1:])])
    say(f"DART robot15_max_diff {float(np.abs(g[:, :15] - t[:, :15]).max()):.2e} rows {len(g)}")
    for name, sl in (("peg_pos_err", slice(15, 18)), ("tip_hole_err", slice(22, 25)), ("hole_pos_err", slice(25, 28))):
        e = np.linalg.norm(g[:, sl] - t[:, sl], axis=1) * 1000
        per = [round(float(e[(step >= lo) & (step < hi)].mean()), 2) if ((step >= lo) & (step < hi)).any() else None
               for lo, hi in BUCKETS]
        say(f"DART {name} mean {e.mean():.2f} p90 {np.percentile(e, 90):.2f} by_steps{BUCKETS} {per}")
    moved = np.concatenate([np.full(b - a, shifted(sd)) for a, b, sd in zip(off[:-1], off[1:], seeds)])
    for name, sl in (("tip_hole_err", slice(22, 25)), ("hole_pos_err", slice(25, 28))):
        e = np.linalg.norm(g[:, sl] - t[:, sl], axis=1) * 1000
        for label, m in (("hole_moved", moved), ("hole_fixed", ~moved)):
            per = [round(float(e[m & (step >= lo) & (step < hi)].mean()), 2) if (m & (step >= lo) & (step < hi)).any() else None
                   for lo, hi in BUCKETS]
            say(f"DART {label} {name} mean {e[m].mean():.2f} by_steps {per}")


def sync_phase(expert, env):
    task = env.unwrapped
    rel = to_pose(task.goal_pose).inv() * to_pose(task.peg.pose)
    lateral = float(np.linalg.norm(rel.p[1:]))
    near = rel.p[0] > -0.02 - expert.half_length - 0.003
    expert.phase = "insert" if (near and lateral < 0.003) else "align"


def peg_lost(env):
    task = env.unwrapped
    return float(np.linalg.norm(to_pose(task.peg.pose).p - to_pose(task.agent.tcp.pose).p)) > 0.12


def dagger_episode(env, graph, expert, dec, seed, rnd, cfg):
    rng = np.random.default_rng([seed, rnd, 7])
    shift = None
    if rng.random() < cfg.shift_frac:
        shift = (int(rng.integers(0, 40)), float(rng.uniform(0.005, 0.02)), float(rng.choice([-1.0, 1.0])))
    obs, _ = env.reset(seed=seed)
    expert.reset(env, seed)
    graph.reset()
    dec.reset()
    states, labels = [], []
    handover, ok, lost = None, False, False
    for t in range(cfg.max_steps):
        driving = handover is not None and t >= handover
        if driving:
            if shift is not None and t == handover + shift[0]:
                shift_box(env, geometry(env)[1][3:], shift[1], shift[2])
            sync_phase(expert, env)
        o21 = graph.step(obs)[0]
        s36 = np.concatenate([robot15(env), o21]).astype(np.float32)
        a_exp = np.asarray(to_np(expert.act(obs)), np.float32).reshape(-1)
        a_dec = dec.act(s36)
        if handover is None and expert.phase in ("align", "insert"):
            handover = t + 1
        states.append(s36)
        labels.append(a_exp)
        a = a_dec if driving else a_exp
        obs, _, _, _, info = env.step(torch.as_tensor(a[None], device=env.unwrapped.device))
        if is_success(info):
            ok = True
            break
        if driving and peg_lost(env):
            lost = True
            break
    return {"seed": seed, "ok": ok, "lost": lost, "handover": handover, "steps": len(states),
            "states": np.stack(states), "labels": np.stack(labels)}


def init_worker(cfg):
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    paths = json.load(open(cfg.paths))
    _W.update(env=make_env(16), graph=Graph(paths["gnn"], paths["corner"], "cuda"), expert=make_policy(),
              cfg=cfg, dec={})


def task(job):
    tseed, eseed, runs, rnd = job
    cfg, graph = _W["cfg"], _W["graph"]
    if tseed not in _W["dec"]:
        ck = torch.load(Path(runs) / f"{cfg.type}_s{tseed}" / "last.pt", map_location="cpu", weights_only=False)
        _W["dec"][tseed] = Decoder(ck, cfg.mode)
    result = {"seed": eseed, "error": "not run"}
    for attempt in range(4):
        try:
            result = dagger_episode(_W["env"], graph, _W["expert"], _W["dec"][tseed], eseed, rnd, cfg)
            break
        except torch.cuda.OutOfMemoryError as e:
            result = {"seed": eseed, "error": repr(e)[:200]}
            graph.reset()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5 + 10 * attempt)
        except Exception as e:
            result = {"seed": eseed, "error": repr(e)[:300]}
            break
    graph.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def collect(cfg, rnd, runs, seeds, pool_file):
    jobs = [(cfg.seeds[i % len(cfg.seeds)], int(s), str(runs), rnd) for i, s in enumerate(seeds)]
    rows, errors, t0 = [], [], time.time()
    with mp.get_context("spawn").Pool(cfg.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        results = pool.imap_unordered(task, jobs)
        for i in range(1, len(jobs) + 1):
            try:
                r = results.next(timeout=1800)
            except mp.TimeoutError:
                raise SystemExit("FIX FAIL collection stalled for 30 minutes")
            (errors if "error" in r else rows).append(r)
            if i % 25 == 0 or i == len(jobs):
                say(f"round {rnd} collect {i}/{len(jobs)} errors {len(errors)} min {(time.time() - t0) / 60:.1f}")
    time.sleep(5)
    if errors:
        say(f"round {rnd} first_error {errors[0]['error']}")
    rows.sort(key=lambda r: r["seed"])
    kept = [r for r in rows if r["handover"] is not None]
    if not kept:
        raise SystemExit("FIX FAIL no episodes with handover")
    off = np.cumsum([0] + [r["steps"] for r in kept])
    np.savez(pool_file, state36_transformer=np.concatenate([r["states"] for r in kept]),
             expert_action=np.concatenate([r["labels"] for r in kept]), episode_offsets=off,
             reset_seed=np.array([r["seed"] for r in kept]), student_ok=np.array([r["ok"] for r in kept]),
             handover=np.array([r["handover"] for r in kept]))
    say(f"round {rnd} POOL episodes {len(rows)} kept {len(kept)} rows {int(off[-1])} "
        f"student_success_after_expert_grasp {100 * np.mean([r['ok'] for r in kept]):.1f}% "
        f"peg_lost {100 * np.mean([r['lost'] for r in kept]):.1f}% handover_step_median {np.median([r['handover'] for r in kept]):.0f}")


def aggregate(dart_file, pool_files, out_file):
    xs, acts, lens, seeds = [], [], [], []
    for f in [dart_file] + pool_files:
        with np.load(f, allow_pickle=False) as d:
            xs.append(d["state36_transformer"])
            acts.append(d["expert_action"])
            lens.append(np.diff(d["episode_offsets"]))
            seeds.append(d["reset_seed"])
    lens = np.concatenate(lens)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, state36_transformer=np.concatenate(xs), expert_action=np.concatenate(acts),
             episode_offsets=np.concatenate([[0], np.cumsum(lens)]), reset_seed=np.concatenate(seeds))
    say(f"aggregate {out_file} episodes {len(lens)} rows {int(lens.sum())}")


def sh(args):
    say("RUN " + " ".join(str(a) for a in args))
    subprocess.run([sys.executable] + [str(a) for a in args], cwd=HERE, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "analyze"])
    p.add_argument("--type", default="gru_step")
    p.add_argument("--mode", default="step")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--per-round", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=180)
    p.add_argument("--shift-frac", type=float, default=0.0)
    p.add_argument("--dart-keep", default="noshift", choices=["all", "noshift"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--paths", default="/workspace/graph_ckpt/paths.json")
    p.add_argument("--base-eval", default="/workspace/eval/stage1")
    p.add_argument("--dart", default="/workspace/data_dart_graph/train.npz")
    p.add_argument("--dart-gt", default="/workspace/data_dart_gt/train.npz")
    p.add_argument("--seed-file", default="/workspace/data/train.npz")
    p.add_argument("--work", default="/workspace/dagger2")
    cfg = p.parse_args()
    cfg.seeds = [int(s) for s in cfg.seeds.split(",")]
    cand = f"{cfg.type}@{cfg.mode}"
    seed_arg = ",".join(str(s) for s in cfg.seeds)
    work = Path(cfg.work)
    work.mkdir(parents=True, exist_ok=True)

    dart_graph_error(cfg.dart, cfg.dart_gt)
    table = {"baseline": summarize("baseline", cfg.base_eval, cand, cfg.seeds)}
    if cfg.cmd == "analyze":
        return
    used = gpu_used_mib()
    if used > 2000:
        raise SystemExit(f"FIX FAIL gpu already uses {used} MiB, stop other runs first")
    with np.load(cfg.seed_file, allow_pickle=False) as d:
        all_seeds = d["reset_seed"].tolist()
    dart = Path(cfg.dart)
    if cfg.dart_keep == "noshift":
        dart = work / "dart_noshift.npz"
        filter_dart(cfg.dart, dart)
    runs, pools = None, []
    for rnd in range(0, cfg.rounds + 1):
        if rnd > 0:
            pool_file = work / f"pool_r{rnd}.npz"
            if not pool_file.is_file():
                seeds = all_seeds[(rnd - 1) * cfg.per_round: rnd * cfg.per_round]
                collect(cfg, rnd, runs, seeds, pool_file)
            pools.append(pool_file)
        data_dir = work / f"data_r{rnd}"
        aggregate(dart, pools, data_dir / "train.npz")
        runs = work / f"runs_r{rnd}"
        sh(["exp2_train.py", "run", "--types", ":".join(cfg.type.rsplit("_", 1)), "--seeds", seed_arg,
            "--input", "transformer", "--splits", "train", "--data-dir", data_dir, "--out", runs,
            "--jobs", len(cfg.seeds)])
        eval_dir = work / f"eval_r{rnd}"
        sh(["exp2_eval.py", "run", "--cands", cand, "--train-seeds", seed_arg, "--episodes", cfg.episodes,
            "--workers", cfg.workers, "--runs", runs, "--out", eval_dir])
        table[f"round{rnd}"] = summarize(f"round{rnd}", eval_dir, cand, cfg.seeds)
    say("SUMMARY " + json.dumps(table))
    say("DONE")


if __name__ == "__main__":
    main()
