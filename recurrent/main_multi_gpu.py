"""Two-phase main launcher. No simulator is initialized in this process."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch

from recurrent.common import ROOT, load_config, resolve
from recurrent.execution import Children, atomic_json, lock_environment, writer_lock
from recurrent.integrity import validate_demos, identity_hash
from recurrent.run_experiments import bind_experiment, validate_results


def run_phase(children, python, config, cfg, gpu_ids, mode, env, num_envs, initial):
    exp = resolve(cfg["out_dir"])
    running, streams = {}, []
    try:
        for index, gpu in enumerate(gpu_ids):
            cmd = [python, "-m", "recurrent.run_experiments", "--config", str(config),
                   "--shard-count", str(len(gpu_ids)), "--shard-index", str(index),
                   "--train-workers", "1", "--eval-workers", "1", "--no-report", "--" + mode,
                   "--train-device", "cuda"]
            if mode == "eval-only":
                cmd += ["--sim-backend", "physx_cuda", "--num-envs", str(num_envs),
                        "--initial-states", str(initial)]
            stream = (exp / f"{mode}-shard-{index}.log").open("w")
            streams.append(stream)
            proc = children.start(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu,
                                   RECURRENT_SHARD_GROUP="1"), stdout=stream, stderr=subprocess.STDOUT)
            running[proc] = index
        while running:
            for proc, index in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[proc]
                if code:
                    raise RuntimeError(f"{mode} shard {index} FAIL (exit {code}); see {exp}")
                print(f"{mode} shard {index} PASS", flush=True)
            if running:
                time.sleep(0.1)
    finally:
        for stream in streams:
            stream.close()


def run_pipeline(config, gpu_ids, num_envs=30, python=sys.executable):
    cfg = load_config(config)
    exp = resolve(cfg["out_dir"])
    initial = exp / "initial_states.pt"
    with writer_lock(exp / ".main.lock") as fd:
        env = lock_environment(fd)
        cpu_env = dict(env, CUDA_VISIBLE_DEVICES=gpu_ids[0], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with Children() as children:
            # collect validates an existing file; mismatches never trigger recollection.
            children.run([python, "-m", "recurrent.collect", "--num-demos", "100",
                          "--out", str(resolve(cfg["data"]))], cwd=ROOT, env=cpu_env)
            validate_demos(resolve(cfg["data"]), num_demos=100)
            manifest = bind_experiment(cfg)
            ev = cfg["eval"]
            children.run([python, "-m", "recurrent.initial_states", "--out", str(initial),
                          "--episodes", str(ev["episodes"]), "--start-seed", str(ev["start_seed"]),
                          "--horizon", str(ev.get("horizon", 250))], cwd=ROOT, env=cpu_env)
            overrides = ["--set", *cfg["model_overrides"]] if cfg.get("model_overrides") else []
            children.run([python, "-m", "recurrent.count_params", "--models", *cfg["models"], *overrides], cwd=ROOT, env=cpu_env)
            children.run([python, "-m", "recurrent.latency", "--models", *cfg["models"], *overrides,
                          "--out", str(exp / "latency.json")], cwd=ROOT, env=cpu_env)
            run_phase(children, python, config, cfg, gpu_ids, "train-only", env, num_envs, initial)
            validate_results(cfg, training_only=True, require_cuda=True)
            run_phase(children, python, config, cfg, gpu_ids, "eval-only", env, num_envs, initial)
            counts = validate_results(cfg, "physx_cuda", str(initial), require_cuda=True)
            atomic_json(exp / "verified_results.json", {**counts, "experiment_identity": identity_hash(manifest), "policy_backend": "physx_cuda",
                        "expert_backend": "physx_cpu", "initial_states": str(initial)})
            children.run([python, "-m", "recurrent.report", "--config", str(config),
                          "--sim-backend", "physx_cuda", "--initial-states", str(initial),
                          "--latency", str(exp / "latency.json")], cwd=ROOT, env=cpu_env)
            print(f"Verified {counts['training_jobs']} training jobs, {counts['policy_episodes']} policy episodes, "
                  f"{counts['expert_episodes']} expert episodes", flush=True)
    print(f"Main multi-GPU run PASS: {exp / 'report' / 'report.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.environ.get("MAIN_CONFIG", "recurrent/configs/experiment.yaml"))
    args = p.parse_args()
    if os.environ.get("EVAL_WORKERS", "1") != "1":
        p.error("EVAL_WORKERS must be 1 for the main multi-GPU run")
    batch = os.environ.get("EVAL_NUM_ENVS", "30")
    if not batch.isdecimal() or int(batch) < 1:
        p.error("EVAL_NUM_ENVS must be a positive integer")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < 2:
        p.error(f"Need at least two CUDA GPUs; Torch sees {count}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    ids = visible.split(",") if visible and visible != "all" else [str(i) for i in range(count)]
    if len(ids) != count or len(ids) != len(set(ids)) or any(not x for x in ids):
        p.error("CUDA_VISIBLE_DEVICES does not match Torch's visible GPU count")
    run_pipeline(args.config, ids, int(batch))


if __name__ == "__main__":
    main()
