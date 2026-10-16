from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import torch

from recurrent.common import ROOT, load_config, resolve


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


def eval_cmd(cfg, ckpt, name, level, out):
    ev = cfg["eval"]
    conds = ev.get("conditions", {"normal": {}})
    cmd = [sys.executable, "-m", "recurrent.evaluate", "--ckpt", str(ckpt), "--name", name,
           "--conditions", *conds.keys(), "--episodes", str(ev["episodes"]), "--start-seed", str(ev["start_seed"]),
           "--horizon", str(ev.get("horizon", 250)), "--corruption", cfg.get("corruption", "corruption.yaml"), "--out", str(out)]
    shift = conds.get("hole_shift") or {}
    cmd += ["--shift-dx", str(shift.get("shift_dx", 0.02)), "--shift-delay", str(shift.get("shift_delay", 10))]
    if ckpt != "expert":
        levels = list(cfg["levels"]) if (ev.get("cross_eval_clean") and level == "none") else [level]
        cmd += ["--levels", *levels]
        stress = ev.get("stress") or {}
        if level in stress.get("train_levels", []):
            cmd += ["--blackout", *[str(b) for b in stress.get("blackout_lengths", [])]]
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


def eval_done(out, expected):
    summary = out.with_suffix(".summary.json")
    if not summary.exists() or not out.exists():
        return False
    return sum(1 for l in out.read_text().splitlines() if l.strip()) >= expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiment.yaml")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--levels", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--train-workers", type=int, default=None)
    p.add_argument("--eval-workers", type=int, default=None)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--no-report", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.models:
        cfg["models"] = args.models
    if args.levels:
        cfg["levels"] = args.levels
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    data = resolve(cfg["data"])
    if not data.exists():
        raise SystemExit(f"missing {data}; run python -m recurrent.collect first")
    exp = resolve(cfg["out_dir"])
    for d in ("runs", "eval", "logs"):
        (exp / d).mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    gpus = torch.cuda.device_count() if cuda else 0
    tw, ew = auto_workers(cfg.get("parallel", {}), cuda)
    tw = args.train_workers or tw
    ew = args.eval_workers or ew
    (exp / "plan.json").write_text(json.dumps({"config": cfg, "train_workers": tw, "eval_workers": ew, "gpus": gpus}, indent=2))
    print(f"exp {exp} train_workers {tw} eval_workers {ew} gpus {gpus}")

    which = cfg["eval"].get("checkpoint", "best")
    train_q, eval_q = deque(), deque()
    if cfg["eval"].get("with_expert", True) and not args.train_only:
        out = exp / "eval" / "expert.jsonl"
        if not eval_done(out, expected_rows(cfg, "none", expert=True)):
            eval_q.append(("expert", "expert", "none", out))
    for level in cfg["levels"]:
        for model in cfg["models"]:
            for seed in cfg["seeds"]:
                name = run_name(model, level, seed)
                run_dir = exp / "runs" / name
                trained = (run_dir / "last.pt").exists() and (run_dir / "history.json").exists()
                if not trained:
                    train_q.append((name, model, level, seed, run_dir))
                elif not args.train_only:
                    out = exp / "eval" / f"{name}.jsonl"
                    if not eval_done(out, expected_rows(cfg, level)):
                        eval_q.append((name, run_dir / f"{which}.pt", level, out))
    n_train, n_eval = len(train_q), len(eval_q) + (0 if args.train_only else len(train_q))
    print(f"train jobs {n_train} eval jobs {n_eval}")

    base_env = dict(os.environ)
    running = []
    failures = []
    done_train = done_eval = 0
    gpu_turn = 0
    t_start = time.time()
    while train_q or eval_q or running:
        while train_q and sum(r[0] == "train" for r in running) < tw:
            name, model, level, seed, run_dir = train_q.popleft()
            env = dict(base_env, OMP_NUM_THREADS=str(cfg["train"].get("threads", 2)))
            if gpus > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_turn % gpus)
                gpu_turn += 1
            log = (exp / "logs" / f"train_{name}.log").open("w")
            proc = subprocess.Popen(train_cmd(cfg, model, level, seed, run_dir), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append(("train", name, proc, log, time.time(), (model, level, seed, run_dir)))
        while eval_q and sum(r[0] == "eval" for r in running) < ew:
            name, ckpt, level, out = eval_q.popleft()
            env = dict(base_env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            log = (exp / "logs" / f"eval_{name}.log").open("w")
            proc = subprocess.Popen(eval_cmd(cfg, ckpt, name, level, out), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append(("eval", name, proc, log, time.time(), (ckpt, level, out)))
        time.sleep(1.0)
        still = []
        for kind, name, proc, log, t0, info in running:
            code = proc.poll()
            if code is None:
                still.append((kind, name, proc, log, t0, info))
                continue
            log.close()
            status = "PASS" if code == 0 else "FAIL"
            if kind == "train":
                done_train += 1
                if code == 0 and not args.train_only:
                    model, level, seed, run_dir = info
                    eval_q.append((name, run_dir / f"{which}.pt", level, exp / "eval" / f"{name}.jsonl"))
            else:
                done_eval += 1
            if code != 0:
                failures.append(f"{kind} {name}")
            print(f"{kind} {name} {status} {time.time() - t0:.0f}s train {done_train}/{n_train} eval {done_eval}/{n_eval} elapsed {(time.time() - t_start) / 60:.1f}m", flush=True)
        running = still

    if failures:
        print("FAILED " + ", ".join(failures))
        print(f"logs {exp / 'logs'}")
    if not args.no_report and not args.train_only:
        subprocess.run([sys.executable, "-m", "recurrent.report", "--config", str(args.config)], cwd=ROOT, check=False)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
