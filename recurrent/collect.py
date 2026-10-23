from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from recurrent.common import (
    DATA_DIR,
    HORIZON,
    RAW_KEYS,
    environment_record,
    is_success,
    make_env,
    raw_gt,
    resolve,
    shift_hole,
    to_action_tensor,
    versions,
)
from recurrent.expert import ScriptedExpert
from recurrent.execution import writer_lock
from recurrent.integrity import validate_demos

_ENV = None


def perturb_plan(seed: int, frac: float):
    rng = np.random.default_rng([seed, 1])
    if rng.random() >= frac:
        return None
    return int(rng.integers(40, 81)), float(rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.025))


def run_episode(env, seed: int, noise: float, frac: float, horizon: int):
    plan = perturb_plan(seed, frac)
    rng = np.random.default_rng([seed, 2])
    env.reset(seed=seed)
    expert = ScriptedExpert(env)
    obs, act = {k: [] for k in RAW_KEYS}, []
    ok = False
    for t in range(horizon):
        if plan is not None and t == plan[0]:
            shift_hole(env, plan[1])
        for k, v in raw_gt(env).items():
            obs[k].append(v)
        a = expert.act()
        act.append(a)
        executed = a.copy()
        if noise > 0:
            executed[:6] += rng.normal(0.0, noise, 6).astype(np.float32)
        _, _, terminated, truncated, info = env.step(to_action_tensor(executed, env))
        if is_success(info):
            ok = True
            break
        if bool(truncated.reshape(-1)[0]) or bool(terminated.reshape(-1)[0]):
            break
    obs = {k: np.stack(v).astype(np.float32) for k, v in obs.items()}
    return {"seed": seed, "ok": ok, "obs": obs, "act": np.stack(act).astype(np.float32), "plan": plan}


def _init(horizon: int):
    global _ENV
    torch.set_num_threads(1)
    _ENV = make_env(horizon)


def _task(args):
    seed, noise, frac, horizon = args
    return run_episode(_ENV, seed, noise, frac, horizon)


def collect(num_demos, start_seed, noise, frac, workers, horizon, log=print):
    kept, tried = [], 0
    seed = start_seed
    batch = max(1, workers) * 2
    if workers <= 1:
        _init(horizon)
        pool = None
    else:
        # Fail in the caller before Pool repeatedly respawns workers whose
        # initializer cannot create the simulator.
        probe = make_env(horizon)
        probe.close()
        pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(horizon,))
    try:
        while len(kept) < num_demos:
            jobs = [(s, noise, frac, horizon) for s in range(seed, seed + batch)]
            results = [_task(j) for j in jobs] if pool is None else pool.map(_task, jobs)
            for r in sorted(results, key=lambda r: r["seed"]):
                if len(kept) >= num_demos:
                    break
                tried += 1
                if r["ok"]:
                    kept.append(r)
                log(f"seed {r['seed']} steps {len(r['act'])} {'PASS' if r['ok'] else 'FAIL'} kept {len(kept)}/{num_demos}")
            seed += batch
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        elif _ENV is not None:
            _ENV.close()
    return kept, tried


def save(path: Path, kept, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        **{f"raw_{k}": np.concatenate([r["obs"][k] for r in kept]) for k in RAW_KEYS},
        act=np.concatenate([r["act"] for r in kept]),
        lengths=np.array([len(r["act"]) for r in kept]),
        seeds=np.array([r["seed"] for r in kept]),
        perturbed=np.array([r.get("plan") is not None for r in kept]),
        meta=np.array(json.dumps(meta)),
    )
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos", type=int, default=100)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--perturb-frac", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--out", default=str(DATA_DIR / "expert_demos.npz"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = resolve(args.out)
    with writer_lock(out.with_name(out.name + ".lock")):
        collect_output(args, out)


def collect_output(args, out):
    if out.exists() and not args.force:
        validate_demos(out, num_demos=args.num_demos, start_seed=args.start_seed,
                       noise=args.noise, frac=args.perturb_frac, horizon=args.horizon)
        print(f"validated existing demos {out}")
        return
    t0 = time.time()
    kept, tried = collect(args.num_demos, args.start_seed, args.noise, args.perturb_frac, args.workers, args.horizon)
    meta = {
        "source": "expert",
        "args": vars(args),
        "environment": environment_record(args.horizon),
        "versions": versions(),
        "expert_success": [len(kept), tried],
        "shift_plans": {str(r["seed"]): r["plan"] for r in kept},
    }
    save(out, kept, meta)
    validate_demos(out, num_demos=args.num_demos, start_seed=args.start_seed,
                   noise=args.noise, frac=args.perturb_frac, horizon=args.horizon)
    print(f"expert success {len(kept)}/{tried}")
    print(f"saved {out} frames {sum(len(r['act']) for r in kept)} time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
