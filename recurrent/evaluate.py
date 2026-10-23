from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from recurrent.common import (
    ACTION_DIM,
    HORIZON,
    environment_record,
    experiment_logger,
    is_success,
    make_env,
    raw_gt,
    resolve,
    shift_hole,
    to_action_tensor,
    versions,
)
from recurrent.corruption import Corruptor, level_config
from recurrent.expert import ScriptedExpert
from recurrent.features import build_features
from recurrent.models import build
from recurrent.stages import FAILURE_ORDER, THRESHOLDS, StageTracker
from recurrent.train import normalize_obs

LEVEL_INDEX = {"none": 0, "low": 1, "mid": 2, "high": 3}


class LearnedPolicy:
    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.ckpt = ckpt
        self.model = build(ckpt["kind"], ckpt["obs_dim"], ACTION_DIM, ckpt["cfg"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.stats = {k: np.asarray(v, np.float32) for k, v in ckpt["stats"].items()}
        self.name = ckpt["kind"]
        self.state = None

    def reset(self, env):
        self.state = None

    def act(self, env, raw, fresh, age):
        obs = normalize_obs(
            build_features(raw, np.float32(fresh), np.float32(age)), self.stats
        )
        x = torch.from_numpy(obs).view(1, 1, -1)
        with torch.no_grad():
            out, self.state = self.model(x, self.state)
        out = out[0, 0].numpy()
        move = out[:6] * self.stats["act_std"] + self.stats["act_mean"]
        move = np.clip(move, -1.0, 1.0)
        return np.concatenate([move, [1.0 if out[6] > 0 else -1.0]]).astype(np.float32)


class ExpertPolicy:
    name = "expert"
    ckpt = None

    def reset(self, env):
        self.expert = ScriptedExpert(env)

    def act(self, env, raw, fresh, age):
        return self.expert.act()


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_episode(env, policy, seed, level_cfg, cond, horizon, fixtures=None):
    reset_start = time.perf_counter()
    if fixtures is None:
        env.reset(seed=seed)
    if fixtures is not None:
        from recurrent.initial_states import restore

        restore(env, fixtures, [seed])
    reset_s = time.perf_counter() - reset_start
    rollout_start = time.perf_counter()
    policy.reset(env)
    corruptor = Corruptor(
        level_cfg,
        np.random.default_rng([seed, LEVEL_INDEX.get(level_cfg["name"], 9), 7]),
    )
    tracker = StageTracker()
    tracker.update(env)
    sign = 1.0 if seed % 2 == 0 else -1.0
    shift_t = blackout_t = None
    prev = None
    deltas, infer = [], []
    success, steps = False, horizon
    for t in range(horizon):
        if (
            cond.get("shift_dx")
            and shift_t is None
            and tracker.transport_step is not None
            and t >= tracker.transport_step + cond.get("shift_delay", 10)
        ):
            shift_hole(env, sign * cond["shift_dx"])
            shift_t = t
        raw = raw_gt(env)
        if (
            cond.get("blackout")
            and blackout_t is None
            and tracker.transport_step is not None
        ):
            corruptor.force_drop(cond["blackout"])
            blackout_t = t
        craw, fresh, age = corruptor.step(raw)
        t0 = time.perf_counter()
        a = policy.act(env, craw, fresh, age)
        infer.append((time.perf_counter() - t0) * 1000)
        if prev is not None:
            deltas.append(float(np.linalg.norm(a[:6] - prev[:6])))
        prev = a
        _, _, terminated, truncated, info = env.step(to_action_tensor(a, env))
        tracker.update(env, info)
        if is_success(info):
            success, steps = True, t + 1
            break
        if bool(truncated.reshape(-1)[0]) or bool(terminated.reshape(-1)[0]):
            steps = t + 1
            break
    row = {
        "reset_s": reset_s,
        "rollout_s": time.perf_counter() - rollout_start,
        "episode_seed": seed,
        "success": success,
        "steps": steps,
        "shift_applied": shift_t is not None,
        "shift_step": shift_t,
        "steps_after_shift": (steps - shift_t)
        if (success and shift_t is not None)
        else None,
        "blackout_applied": blackout_t is not None,
        "blackout_step": blackout_t,
        "mean_action_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_infer_ms": float(np.mean(infer)) if infer else 0.0,
    }
    row.update(tracker.summary())
    return row


def condition_jobs(
    train_level, conditions, stress_lengths, stress_levels, extra_levels=()
):
    jobs = []
    for level in [train_level, *[l for l in extra_levels if l != train_level]]:
        for name, cond in conditions.items():
            jobs.append((name, level, dict(cond or {})))
    if train_level in stress_levels:
        for n in stress_lengths:
            jobs.append((f"blackout_{n}", "none", {"blackout": int(n)}))
    return jobs


def done_keys(path: Path):
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["condition"], r["eval_level"], r["episode_seed"]))
    return keys


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["eval_level"])].append(r)
    out = {}
    for (cond, level), rs in sorted(groups.items()):
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        steps = [r["steps"] for r in rs if r["success"]]
        fail = Counter(r["failure_stage"] for r in rs if r["failure_stage"])
        shifted = [r for r in rs if r["shift_applied"]]
        out[f"{cond}/{level}"] = {
            "successes": k,
            "episodes": n,
            "success_rate": k / n if n else 0.0,
            "success_ci_low": lo,
            "success_ci_high": hi,
            "mean_steps_success": float(np.mean(steps)) if steps else None,
            "mean_action_delta": float(np.mean([r["mean_action_delta"] for r in rs])),
            "mean_infer_ms": (
                float(np.mean([r["mean_infer_ms"] for r in rs]))
                if all(r["mean_infer_ms"] is not None for r in rs)
                else None
            ),
            "acquisition_rate": float(np.mean([r["acquisition"] for r in rs])),
            "transport_rate": float(np.mean([r["transport"] for r in rs])),
            "alignment_rate": float(np.mean([r["alignment"] for r in rs])),
            "shift_applied": len(shifted),
            "success_after_shift": sum(r["success"] for r in shifted),
            "failures": {s: fail.get(s, 0) for s in FAILURE_ORDER},
        }
    return out


def evaluate(
    ckpt,
    jobs,
    episodes,
    start_seed,
    horizon,
    out_path: Path,
    name=None,
    corruption="corruption.yaml",
    log=print,
    sim_backend="physx_cpu",
    num_envs=1,
    initial_states=None,
):
    if sim_backend not in {"physx_cpu", "physx_cuda"}:
        raise ValueError("Unknown simulation backend")
    if num_envs < 1 or (sim_backend == "physx_cpu" and num_envs != 1):
        raise ValueError(
            "CPU evaluation requires num_envs=1; num_envs must be positive"
        )
    if sim_backend == "physx_cuda":
        if ckpt == "expert":
            raise ValueError("GPU expert evaluation is unsupported")
        if horizon != HORIZON:
            raise ValueError("GPU evaluation requires horizon 250")
        if initial_states is None:
            raise ValueError("GPU evaluation requires --initial-states CPU fixture")
        return evaluate_gpu(
            ckpt,
            jobs,
            episodes,
            start_seed,
            horizon,
            out_path,
            name,
            corruption,
            log,
            num_envs,
            initial_states,
        )
    torch.set_num_threads(1)
    policy = ExpertPolicy() if ckpt == "expert" else LearnedPolicy(ckpt)
    meta = policy.ckpt or {}
    name = name or (Path(ckpt).parent.name if ckpt != "expert" else "expert")
    base = {
        "ckpt": name,
        "model": policy.name,
        "train_level": meta.get("train_level", "none" if ckpt == "expert" else None),
        "train_seed": meta.get("seed"),
        "params": meta.get("params"),
        "data_source": meta.get("data_source"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    identity = verify_resume(
        out_path,
        ckpt,
        jobs,
        episodes,
        start_seed,
        horizon,
        corruption,
        sim_backend,
        initial_states,
    )
    fixtures = (
        torch.load(initial_states, weights_only=False) if initial_states else None
    )
    done = done_keys(out_path)
    config = {
        "checkpoint": str(ckpt),
        "jobs": [[n, l, c] for n, l, c in jobs],
        "episodes": episodes,
        "start_seed": start_seed,
        "horizon": horizon,
        "stage_thresholds": THRESHOLDS,
        "corruption": corruption,
        "evaluation_identity": identity,
    }
    creation_start = time.perf_counter()
    env = make_env(horizon)
    creation_s = time.perf_counter() - creation_start
    t_start = time.time()
    save_s = 0.0
    with experiment_logger(
        config=config, seed=meta.get("seed"), model=policy.name, job_type="evaluate"
    ) as logger:
        logger.record(
            environment={
                **environment_record(horizon),
                "device": str(env.unwrapped.device),
                "render_backend": env.unwrapped.backend.render_backend,
                "num_envs": 1,
            },
            versions=versions(),
            initial_state_source=str(initial_states)
            if initial_states
            else "legacy reset(seed)",
        )
        if ckpt != "expert":
            logger.link_output(ckpt, checkpoint=True)
        try:
            with out_path.open("a") as f:
                for cond_name, level, cond in jobs:
                    level_cfg = level_config(level, corruption)
                    t0 = time.time()
                    ok = 0
                    for i in range(episodes):
                        seed = start_seed + i
                        if (cond_name, level, seed) in done:
                            continue
                        row = {
                            **base,
                            "condition": cond_name,
                            "eval_level": level,
                            **run_episode(
                                env, policy, seed, level_cfg, cond, horizon, fixtures
                            ),
                        }
                        ok += row["success"]
                        save_start = time.perf_counter()
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        save_s += time.perf_counter() - save_start
                        done.add((cond_name, level, seed))
                    log(
                        f"{name} {cond_name} {level} {ok}/{episodes} {time.time() - t0:.0f}s"
                    )
        finally:
            env.close()
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        summary = summarize(rows)
        summary_path = out_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(
                {
                    **base,
                    "summary": summary,
                    "elapsed_s": time.time() - t_start,
                    "performance": {
                        "environment_creation_s": creation_s,
                        "save_s": save_s,
                        "reset_s": sum(r.get("reset_s", 0) for r in rows),
                        "rollout_s": sum(r.get("rollout_s", 0) for r in rows),
                        "active_env_steps": sum(r["steps"] for r in rows),
                        "elapsed_s": time.time() - t_start + creation_s,
                        "episodes_per_s": len(rows)
                        / (time.time() - t_start + creation_s),
                    },
                },
                indent=2,
            )
        )
        logger.log({"eval": {k.replace("/", "_"): s for k, s in summary.items()}})
        logger.link_output(out_path, kind="metrics")
        logger.link_output(summary_path, kind="metrics")
    return summary


def verify_resume(
    path, ckpt, jobs, episodes, seed, horizon, corruption, backend, fixtures
):
    from recurrent.common import sha256

    identity = {
        "checkpoint": "expert" if ckpt == "expert" else sha256(ckpt),
        "jobs": jobs,
        "episodes": episodes,
        "start_seed": seed,
        "horizon": horizon,
        "corruption": level_config("none", corruption),
        "corruption_levels": {l: level_config(l, corruption) for _, l, _ in jobs},
        "backend": backend,
        "initial_states": sha256(fixtures) if fixtures else "legacy reset(seed)",
        "settings": {
            "stage_thresholds": THRESHOLDS,
            "control_mode": "pd_ee_delta_pose",
            "obs_mode": "state",
            "render_backend": "none" if backend == "physx_cuda" else "sapien_cpu",
        },
    }
    identity = json.loads(json.dumps(identity))
    target = path.with_suffix(".identity.json")
    if path.exists() and path.stat().st_size and not target.exists():
        raise ValueError(
            "Existing evaluation has no resume identity; choose a new output"
        )
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError("Evaluation resume identity mismatch")
    target.write_text(json.dumps(identity, indent=2))
    return identity


def evaluate_gpu(
    ckpt,
    jobs,
    episodes,
    start_seed,
    horizon,
    out_path,
    name,
    corruption,
    log,
    num_envs,
    initial_states,
):
    from mani_skill.utils.registration import TimeLimitWrapper

    from envs.maniskill import make_peg_insertion_env
    from recurrent.gpu_eval import TensorPolicy, run_batch

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    start = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    identity = verify_resume(
        out_path,
        ckpt,
        jobs,
        episodes,
        start_seed,
        horizon,
        corruption,
        "physx_cuda",
        initial_states,
    )
    fixtures = torch.load(initial_states, weights_only=False)
    learned = LearnedPolicy(ckpt)
    policy = TensorPolicy(learned, "cuda")
    name = name or Path(ckpt).parent.name
    meta = learned.ckpt
    base = {
        "ckpt": name,
        "model": learned.name,
        "train_level": meta.get("train_level"),
        "train_seed": meta.get("seed"),
        "params": meta.get("params"),
        "data_source": meta.get("data_source"),
    }
    done = done_keys(out_path)
    timings = []
    env = None
    size = None
    creation_s = 0.0
    save_s = 0.0
    torch.cuda.reset_peak_memory_stats()
    with experiment_logger(
        config={**identity, "num_envs": num_envs},
        seed=meta.get("seed"),
        model=learned.name,
        job_type="evaluate",
    ) as logger:
        logger.link_output(ckpt, checkpoint=True)
        logger.record(versions=versions(), initial_state_source=str(initial_states))
        try:
            with out_path.open("a") as f:
                for cond_name, level, cond in jobs:
                    pending = [
                        start_seed + i
                        for i in range(episodes)
                        if (cond_name, level, start_seed + i) not in done
                    ]
                    for offset in range(0, len(pending), num_envs):
                        seeds = pending[offset : offset + num_envs]
                        if size != len(seeds):
                            if env is not None:
                                env.close()
                            tick = time.perf_counter()
                            env = TimeLimitWrapper(
                                make_peg_insertion_env(
                                    device="cuda",
                                    num_envs=len(seeds),
                                    obs_mode="state",
                                    control_mode="pd_ee_delta_pose",
                                    render_backend="none",
                                ).unwrapped,
                                horizon,
                            )
                            creation_s += time.perf_counter() - tick
                            size = len(seeds)
                            logger.record(
                                environment={
                                    **environment_record(horizon),
                                    "sim_backend": env.unwrapped.backend.sim_backend,
                                    "render_backend": env.unwrapped.backend.render_backend,
                                    "device": str(env.unwrapped.device),
                                    "num_envs": size,
                                }
                            )
                        rows, timing = run_batch(
                            env,
                            policy,
                            seeds,
                            level_config(level, corruption),
                            cond,
                            horizon,
                            fixtures,
                        )
                        timings.append(
                            dict(
                                condition=cond_name, level=level, seeds=seeds, **timing
                            )
                        )
                        tick = time.perf_counter()
                        for row in rows:
                            f.write(
                                json.dumps(
                                    {
                                        **base,
                                        "condition": cond_name,
                                        "eval_level": level,
                                        **row,
                                    }
                                )
                                + "\n"
                            )
                            done.add((cond_name, level, row["episode_seed"]))
                        f.flush()
                        save_s += time.perf_counter() - tick
                        log(f"{name} {cond_name}/{level} seeds={seeds[0]}..{seeds[-1]}")
        finally:
            if env is not None:
                env.close()
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        summary = summarize(rows)
        elapsed = time.perf_counter() - start
        perf = {
            "elapsed_s": elapsed,
            "environment_creation_s": creation_s,
            "save_s": save_s,
            "batches": timings,
            "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(),
            "episodes_per_s": sum(len(t["seeds"]) for t in timings) / elapsed,
        }
        target = out_path.with_suffix(".summary.json")
        if not timings and target.exists():
            perf = json.loads(target.read_text()).get("performance", perf)
        target.write_text(
            json.dumps({**base, "summary": summary, "performance": perf}, indent=2)
        )
        logger.log({"eval": {k.replace("/", "_"): v for k, v in summary.items()}})
        logger.link_output(out_path, kind="metrics")
        logger.link_output(target, kind="metrics")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sim-backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu"
    )
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--initial-states", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--conditions", nargs="*", default=["normal", "hole_shift"])
    p.add_argument("--levels", nargs="*", default=None)
    p.add_argument("--shift-dx", type=float, default=0.02)
    p.add_argument("--shift-delay", type=int, default=10)
    p.add_argument("--blackout", nargs="*", type=int, default=[])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--start-seed", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--corruption", default="corruption.yaml")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    specs = {
        "normal": {},
        "hole_shift": {"shift_dx": args.shift_dx, "shift_delay": args.shift_delay},
    }
    conditions = {c: specs[c] for c in args.conditions}
    if args.ckpt == "expert":
        train_level = "none"
    else:
        train_level = torch.load(
            resolve(args.ckpt), map_location="cpu", weights_only=False
        )["train_level"]
    levels = args.levels or [train_level]
    jobs = [(n, l, c) for l in levels for n, c in conditions.items()]
    jobs += [(f"blackout_{b}", "none", {"blackout": b}) for b in args.blackout]
    ckpt = args.ckpt if args.ckpt == "expert" else resolve(args.ckpt)
    name = args.name or ("expert" if args.ckpt == "expert" else Path(ckpt).parent.name)
    out = (
        resolve(args.out)
        if args.out
        else resolve(
            f"outputs/recurrent/{'eval_gpu' if args.sim_backend == 'physx_cuda' else 'eval'}/{name}.jsonl"
        )
    )
    summary = evaluate(
        ckpt,
        jobs,
        args.episodes,
        args.start_seed,
        args.horizon,
        out,
        name,
        args.corruption,
        sim_backend=args.sim_backend,
        num_envs=args.num_envs,
        initial_states=args.initial_states,
    )
    for k, s in summary.items():
        print(f"{name} {k} {s['successes']}/{s['episodes']}")


if __name__ == "__main__":
    main()
