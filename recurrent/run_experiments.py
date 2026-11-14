from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import torch

from recurrent.common import ROOT, load_config, resolve, sha256


def auto_workers(par, cuda):
    cpus = os.cpu_count() or 2
    tw = par.get("train_workers", "auto")
    ew = par.get("eval_workers", "auto")
    if tw == "auto":
        tw = min(4, max(1, cpus // 4)) if cuda else max(1, cpus // 4)
    if ew == "auto":
        ew = max(1, cpus - 2 * int(tw))
    return int(tw), int(ew)


def run_name(model, level, seed):
    return f"{model}_{level}_s{seed}"


def experiment_jobs(cfg, shard_index=0, shard_count=1):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard range")
    if any(len(cfg[k]) != len(set(cfg[k])) for k in ("models", "levels", "seeds")):
        raise ValueError("Duplicate experiment configuration entries")
    jobs = ((run_name(model, level, seed), model, level, seed)
            for level in cfg["levels"] for model in cfg["models"] for seed in cfg["seeds"])
    return [job for index, job in enumerate(jobs) if index % shard_count == shard_index]


def train_cmd(cfg, model, level, seed, out_dir):
    tr = cfg["train"]
    cmd = [sys.executable, "-m", "recurrent.train", "--model", model, "--level", level, "--seed", str(seed),
           "--data", str(resolve(cfg["data"])), "--corruption", cfg.get("corruption", "corruption.yaml"),
           "--steps", str(tr["steps"]), "--batch", str(tr["batch"]), "--lr", str(tr["lr"]),
           "--val-frac", str(tr.get("val_frac", 0.1)), "--split-seed", str(tr.get("split_seed", 0)),
           "--device", str(tr.get("device", "auto")), "--threads", str(tr.get("threads", 2)),
           "--log-every", str(tr.get("log_every", 250)), "--out-dir", str(out_dir)]
    if tr.get("num_demos"):
        cmd += ["--num-demos", str(tr["num_demos"])]
    if cfg.get("model_overrides"):
        cmd += ["--set", *cfg["model_overrides"]]
    return cmd


def eval_cmd(cfg, ckpt, name, level, out, sim_backend="physx_cpu", num_envs=1, initial_states=None):
    ev = cfg["eval"]
    conds = ev.get("conditions", {"normal": {}})
    cmd = [sys.executable, "-m", "recurrent.evaluate", "--ckpt", str(ckpt), "--name", name,
           "--conditions", *conds.keys(), "--episodes", str(ev["episodes"]), "--start-seed", str(ev["start_seed"]),
           "--horizon", str(ev.get("horizon", 250)), "--corruption", cfg.get("corruption", "corruption.yaml"), "--out", str(out)]
    shift = conds.get("hole_shift") or {}
    cmd += ["--shift-dx", str(shift.get("shift_dx", 0.02)), "--shift-delay", str(shift.get("shift_delay", 10))]
    if ckpt != "expert":
        levels = [level, *[l for l in cfg["levels"] if l != level]] if (ev.get("cross_eval_clean") and level == "none") else [level]
        cmd += ["--levels", *levels]
        stress = ev.get("stress") or {}
        if level in stress.get("train_levels", []):
            cmd += ["--blackout", *[str(b) for b in stress.get("blackout_lengths", [])]]
    cmd += ["--sim-backend", sim_backend, "--num-envs", str(num_envs)]
    if initial_states:
        cmd += ["--initial-states", str(initial_states)]
    return cmd


def expected_rows(cfg, level, expert=False):
    ev = cfg["eval"]
    n_cond = len(ev.get("conditions", {"normal": {}}))
    if expert:
        return n_cond * ev["episodes"]
    levels = len(cfg["levels"]) if (ev.get("cross_eval_clean") and level == "none") else 1
    stress = ev.get("stress") or {}
    n_stress = len(stress.get("blackout_lengths", [])) if level in stress.get("train_levels", []) else 0
    return (n_cond * levels + n_stress) * ev["episodes"]


def eval_jobs(cfg, level, expert=False):
    from recurrent.evaluate import condition_jobs
    ev = cfg["eval"]
    conditions = ev.get("conditions", {"normal": {}})
    if set(conditions) - {"normal", "hole_shift"}:
        raise ValueError("Unsupported evaluation condition")
    specs = {"normal": {}, "hole_shift": {"shift_dx": 0.02, "shift_delay": 10}}
    for name, cond in conditions.items():
        if set(cond or {}) - set(specs[name]):
            raise ValueError(f"Unsupported condition settings: {name}")
    conditions = {n: {**specs[n], **(c or {})} for n, c in conditions.items()}
    stress = ev.get("stress") or {}
    extra = cfg["levels"] if ev.get("cross_eval_clean") and level == "none" and not expert else []
    jobs = condition_jobs(level, conditions, [] if expert else stress.get("blackout_lengths", []),
                          [] if expert else stress.get("train_levels", []), extra)
    if not jobs or ev["episodes"] < 1 or len({(n, l) for n, l, _ in jobs}) != len(jobs):
        raise ValueError("Evaluation requires unique jobs and positive episodes")
    return jobs


def job_identity(cfg, model, level, seed, data_hash=None):
    from recurrent.train import get_parser
    from recurrent.integrity import training_identity
    args = get_parser().parse_args(train_cmd(cfg, model, level, seed, "unused")[3:])
    return training_identity(args, data_hash)


def experiment_identity(cfg):
    from recurrent.integrity import canonical, package_versions
    data_hash = sha256(resolve(cfg["data"]))
    return canonical({"schema": 1, "data_sha256": data_hash,
                      "training": {n: job_identity(cfg, m, l, s, data_hash) for n, m, l, s in experiment_jobs(cfg)},
                      "evaluation": cfg["eval"],
                      "evaluation_jobs": {l: eval_jobs(cfg, l) for l in cfg["levels"]},
                      "versions": package_versions()})


def bind_experiment(cfg):
    from recurrent.execution import atomic_json, writer_lock
    from recurrent.integrity import bind_identity
    exp = resolve(cfg["out_dir"])
    identity = experiment_identity(cfg)
    with writer_lock(exp / ".identity.lock", blocking=True):
        legacy = [p for parent in (exp / "runs", exp / "eval") if parent.exists() for p in parent.iterdir()]
        bind_identity(exp / "identity.json", identity, existing=legacy)
        atomic_json(exp / "applied_config.json", {"config": cfg, "identity": identity})
    return identity


def eval_identity(cfg, ckpt, level, backend="physx_cpu", initial_states=None):
    from recurrent.evaluate import evaluation_identity
    ev = cfg["eval"]
    return evaluation_identity(ckpt, eval_jobs(cfg, level, ckpt == "expert"), ev["episodes"],
                               ev["start_seed"], ev.get("horizon", 250),
                               cfg.get("corruption", "corruption.yaml"), backend, initial_states)


def eval_done(out, identity):
    from recurrent.integrity import evaluation_complete
    from recurrent.execution import writer_lock
    with writer_lock(out.with_suffix(".writer.lock")):
        return evaluation_complete(out, identity, repair=True)


def validate_results(cfg, backend="physx_cpu", initial_states=None, *, training_only=False, require_cuda=False):
    from recurrent.integrity import training_complete
    exp = resolve(cfg["out_dir"])
    manifest = experiment_identity(cfg)
    target = exp / "identity.json"
    if not target.exists() or json.loads(target.read_text()) != manifest:
        raise ValueError("Experiment identity mismatch or missing")
    policy_rows = expert_rows = 0
    for name, model, level, seed in experiment_jobs(cfg):
        run = exp / "runs" / name
        if not training_complete(run, manifest["training"][name]):
            raise ValueError(f"Incomplete training: {name}")
        if require_cuda:
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
            if not last["config"]["device"].startswith("cuda"):
                raise ValueError(f"Training did not use CUDA: {name}")
        if not training_only:
            ckpt = run / (cfg["eval"].get("checkpoint", "best") + ".pt")
            identity = eval_identity(cfg, ckpt, level, backend, initial_states)
            if not eval_done(exp / "eval" / f"{name}.jsonl", identity):
                raise ValueError(f"Incomplete evaluation: {name}")
            from recurrent.integrity import read_eval_rows
            rows = read_eval_rows(exp / "eval" / f"{name}.jsonl", identity)
            if any((r["ckpt"], r["model"], r["train_level"], r["train_seed"]) != (name, model, level, seed) for r in rows):
                raise ValueError(f"Evaluation model/seed labels mismatch: {name}")
            policy_rows += expected_rows(cfg, level)
    if not training_only and cfg["eval"].get("with_expert", True):
        identity = eval_identity(cfg, "expert", "none", "physx_cpu", initial_states)
        if not eval_done(exp / "eval" / "expert.jsonl", identity):
            raise ValueError("Incomplete expert evaluation")
        from recurrent.integrity import read_eval_rows
        if any((r["ckpt"], r["model"], r["train_level"], r["train_seed"]) != ("expert", "expert", "none", None)
               for r in read_eval_rows(exp / "eval" / "expert.jsonl", identity)):
            raise ValueError("Expert evaluation labels mismatch")
        expert_rows = expected_rows(cfg, "none", expert=True)
    return {"training_jobs": len(manifest["training"]), "policy_episodes": policy_rows, "expert_episodes": expert_rows}


def execute_queue(tasks, workers, children, exp, kind, verify):
    running = []
    tasks = deque(tasks)
    try:
        while tasks or running:
            while tasks and len(running) < workers:
                name, cmd, env = tasks.popleft()
                stream = (exp / "logs" / f"{kind}_{name}.log").open("w")
                try:
                    proc = children.start(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                running.append((name, proc, stream))
            for name, proc, stream in list(running):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                running.remove((name, proc, stream))
                if code:
                    raise RuntimeError(f"{kind} {name} FAIL (exit {code}); see {exp / 'logs'}")
                verify(name)
                print(f"{kind} {name} PASS", flush=True)
            if running:
                time.sleep(0.1)
    finally:
        # Children context performs termination and reaping on exceptions.
        for _, _, stream in running:
            stream.close()


def main():
    from recurrent.execution import Children, atomic_json, writer_lock, lock_environment
    from recurrent.integrity import training_complete
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiment.yaml")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--levels", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--train-workers", type=int, default=None)
    p.add_argument("--eval-workers", type=int, default=None)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--train-only", action="store_true")
    modes.add_argument("--eval-only", action="store_true")
    p.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"], default="physx_cpu")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--initial-states", default=None)
    p.add_argument("--train-device", default=None)
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        p.error("require --shard-count >= 1 and 0 <= --shard-index < --shard-count")
    if args.num_envs < 1 or (args.sim_backend == "physx_cpu" and args.num_envs != 1):
        p.error("CPU requires --num-envs 1; num-envs must be positive")
    if args.sim_backend == "physx_cuda" and not args.train_only and (not args.initial_states or args.eval_workers not in (None, 1)):
        p.error("CUDA evaluation requires --initial-states and one eval worker")
    cfg = load_config(args.config)
    for key in ("models", "levels", "seeds"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.train_device:
        cfg["train"]["device"] = args.train_device
    if not resolve(cfg["data"]).exists():
        p.error(f"missing data: {cfg['data']}")
    initial = str(resolve(args.initial_states)) if args.initial_states else None
    exp = resolve(cfg["out_dir"])
    tw, ew = auto_workers(cfg.get("parallel", {}), torch.cuda.is_available())
    tw = args.train_workers if args.train_workers is not None else tw
    ew = args.eval_workers if args.eval_workers is not None else (1 if args.sim_backend == "physx_cuda" else ew)
    if min(tw, ew) < 1:
        p.error("workers must be positive")
    jobs = experiment_jobs(cfg, args.shard_index, args.shard_count)
    if not jobs and args.shard_index == 0:
        p.error("No experiment jobs")
    with writer_lock(exp / f".shard-{args.shard_index}.lock") as fd:
        manifest = bind_experiment(cfg)
        for folder in ("runs", "eval", "logs"):
            (exp / folder).mkdir(parents=True, exist_ok=True)
        env = lock_environment(fd, OMP_NUM_THREADS=str(cfg["train"].get("threads", 2)))
        plan_name = f"plan.shard-{args.shard_index}-of-{args.shard_count}.json"
        atomic_json(exp / plan_name, {"config": cfg, "train_workers": tw, "eval_workers": ew,
                    "sim_backend": args.sim_backend, "num_envs": args.num_envs, "initial_states": initial,
                    "shard_index": args.shard_index, "shard_count": args.shard_count,
                    "jobs": [n for n, _, _, _ in jobs]})
        train_tasks = []
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_ids = visible.split(",") if visible and visible != "all" else [str(i) for i in range(gpu_count)]
        for name, model, level, seed in jobs:
            run = exp / "runs" / name
            if not training_complete(run, manifest["training"][name]):
                if args.eval_only:
                    raise ValueError(f"Evaluation-only requires completed training: {name}")
                train_env = dict(env)
                if gpu_count > 1:
                    train_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[len(train_tasks) % gpu_count]
                train_tasks.append((name, train_cmd(cfg, model, level, seed, run), train_env))
        def verify_train(name):
            if not training_complete(exp / "runs" / name, manifest["training"][name]):
                raise ValueError(f"Incomplete training after child success: {name}")
        with Children(groups=os.environ.get("RECURRENT_SHARD_GROUP") != "1") as children:
            execute_queue(train_tasks, tw, children, exp, "train", verify_train)
            if args.train_only:
                return
            eval_tasks, identities = [], {}
            which = cfg["eval"].get("checkpoint", "best")
            entries = [(n, exp / "runs" / n / f"{which}.pt", l, args.sim_backend, args.num_envs) for n, _, l, _ in jobs]
            if cfg["eval"].get("with_expert", True) and args.shard_index == 0:
                entries.append(("expert", "expert", "none", "physx_cpu", 1))
            for name, ckpt, level, backend, size in entries:
                out = exp / "eval" / f"{name}.jsonl"
                identity = eval_identity(cfg, ckpt, level, backend, initial)
                identities[name] = identity
                if not eval_done(out, identity):
                    cmd = eval_cmd(cfg, ckpt, name, level, out, backend, size, initial)
                    eval_tasks.append((name, cmd, dict(env, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")))
            def verify_eval(name):
                if not eval_done(exp / "eval" / f"{name}.jsonl", identities[name]):
                    raise ValueError(f"Incomplete evaluation after child success: {name}")
            execute_queue(eval_tasks, ew, children, exp, "eval", verify_eval)
            if not args.no_report and args.shard_count == 1:
                cmd = [sys.executable, "-m", "recurrent.report", "--config", str(args.config), "--sim-backend", args.sim_backend]
                if initial:
                    cmd += ["--initial-states", initial]
                children.run(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
