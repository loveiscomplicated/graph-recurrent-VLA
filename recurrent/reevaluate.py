"""Evaluate restored, completed runs in a separate current-environment directory.

Without --execute, only validate the source and write the command plan. This
entrypoint never trains, changes the source experiment, or publishes artifacts.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from recurrent.common import ROOT, load_config, resolve, sha256
from recurrent.execution import Children, atomic_json, lock_environment, writer_lock
from recurrent.integrity import bind_identity, canonical, read_eval_rows, training_complete
from recurrent.run_experiments import (
    eval_cmd, eval_done, eval_identity, expected_rows, experiment_identity,
    experiment_jobs,
)


def prepare(cfg, source_root, out, num_envs):
    """Check original training contracts; record version changes as reevaluation."""
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if out.is_relative_to(source_root) or source_root.is_relative_to(out):
        raise ValueError("Reevaluation output must be separate from the source tree")
    if Path(cfg["out_dir"]).is_absolute() or Path(cfg["data"]).is_absolute():
        raise ValueError("Source config paths must be relative to --source-root")
    source = source_root / cfg["out_dir"]
    original = json.loads((source / "identity.json").read_text())
    restored = copy.deepcopy(cfg)
    restored["data"] = str(source_root / cfg["data"])
    current = experiment_identity(restored)
    # Versions are intentionally different for this new evaluation, never
    # rewritten in the original identity. Every other source contract must match.
    if {k: v for k, v in original.items() if k != "versions"} != {
        k: v for k, v in current.items() if k != "versions"
    }:
        raise ValueError("Restored training/data/evaluation configuration mismatch")
    fixture = out / "initial_states.pt"
    entries, checkpoint_hashes = [], {}
    for name, model, level, seed in experiment_jobs(cfg):
        run = source / "runs" / name
        if not training_complete(run, original["training"][name]):
            raise ValueError(f"Incomplete source training: {name}")
        ckpt = run / (cfg["eval"].get("checkpoint", "best") + ".pt")
        checkpoint_hashes[name] = sha256(ckpt)
        entries.append((name, ckpt, level, "physx_cuda", num_envs, model, seed))
    if cfg["eval"].get("with_expert", True):
        entries.append(("expert", "expert", "none", "physx_cpu", 1, "expert", None))
    tasks = []
    for name, ckpt, level, backend, batch, model, seed in entries:
        target = out / "eval" / f"{name}.jsonl"
        tasks.append(dict(name=name, checkpoint=str(ckpt), level=level,
                          backend=backend, model=model, train_seed=seed,
                          output=str(target), expected_rows=expected_rows(
                              cfg, level, expert=ckpt == "expert"),
                          command=eval_cmd(cfg, ckpt, name, level, target,
                                           backend, batch, str(fixture))))
    identity = canonical(dict(
        schema=1, purpose="restored_checkpoint_reevaluation",
        source_root=str(source_root), source_identity_sha256=sha256(source / "identity.json"),
        source_versions=original["versions"], evaluation_versions=current["versions"],
        config=cfg, checkpoint_sha256=checkpoint_hashes,
        policy_backend="physx_cuda", expert_backend="physx_cpu",
        rkn_covariance_update="stable_v1",
    ))
    return identity, tasks, fixture


def verify_task(cfg, task, fixture):
    identity = eval_identity(cfg, task["checkpoint"], task["level"], task["backend"], str(fixture))
    target = Path(task["output"])
    if not eval_done(target, identity):
        return False
    rows = read_eval_rows(target, identity)
    if len(rows) != task["expected_rows"] or any(
        (r["ckpt"], r["model"], r["train_level"], r["train_seed"])
        != (task["name"], task["model"], task["level"], task["train_seed"])
        for r in rows
    ):
        raise ValueError(f"Evaluation labels/count mismatch: {target}")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiment.yaml")
    p.add_argument("--source-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-envs", type=int, default=30)
    p.add_argument("--execute", action="store_true", help="run evaluation only; default writes plan")
    args = p.parse_args()
    torch.set_num_threads(1)
    cfg, out = load_config(args.config), resolve(args.out_dir).resolve()
    identity, tasks, fixture = prepare(cfg, args.source_root, out, args.num_envs)
    with writer_lock(out / ".reevaluation.lock") as fd:
        bind_identity(out / "reevaluation.identity.json", identity,
                      existing=[*out.glob("eval/*"), *out.glob("initial_states.pt")])
        (out / "logs").mkdir(exist_ok=True)
        plan = dict(identity=identity, num_envs=args.num_envs, tasks=tasks,
                    policy_episodes=sum(t["expected_rows"] for t in tasks if t["name"] != "expert"),
                    expert_episodes=sum(t["expected_rows"] for t in tasks if t["name"] == "expert"))
        atomic_json(out / "plan.json", plan)
        print(f"Validated {len(identity['checkpoint_sha256'])} completed source checkpoints; "
              f"policy={plan['policy_episodes']}, expert={plan['expert_episodes']} episodes", flush=True)
        print(f"Plan: {out / 'plan.json'}", flush=True)
        if not args.execute:
            print("Plan only. No fixture generation or evaluation was started.")
            return
        env = lock_environment(fd, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.setdefault("WANDB_MODE", "disabled")
        env["EXPERIMENTS_DIR"] = str(out / "experiments")
        ev = cfg["eval"]
        # CPU fixture collection and all GPU/CPU evaluations are separate processes.
        fixture_cmd = [sys.executable, "-m", "recurrent.initial_states", "--out", str(fixture),
                       "--episodes", str(ev["episodes"]), "--start-seed", str(ev["start_seed"]),
                       "--horizon", str(ev.get("horizon", 250))]
        with Children() as children:
            with (out / "logs" / "fixture.log").open("a") as stream:
                children.run(fixture_cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            for task in tasks:
                if task["checkpoint"] != "expert" and sha256(task["checkpoint"]) != identity["checkpoint_sha256"][task["name"]]:
                    raise ValueError(f"Source checkpoint changed: {task['name']}")
                if verify_task(cfg, task, fixture):
                    print(f"SKIP verified {task['name']}", flush=True)
                    continue
                print(f"RUN {shlex.join(task['command'])}", flush=True)
                with (out / "logs" / f"eval_{task['name']}.log").open("a") as stream:
                    children.run(task["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                if not verify_task(cfg, task, fixture):
                    raise ValueError(f"Incomplete evaluation: {task['name']}")
                print(f"PASS {task['name']}", flush=True)
        atomic_json(out / "reevaluation.complete.json", dict(
            source_identity_sha256=identity["source_identity_sha256"],
            fixture_sha256=sha256(fixture), policy_episodes=plan["policy_episodes"],
            expert_episodes=plan["expert_episodes"],
            completions={t["name"]: sha256(Path(t["output"]).with_suffix(".complete.json")) for t in tasks},
        ))
        print("Reevaluation PASS", flush=True)


if __name__ == "__main__":
    main()
