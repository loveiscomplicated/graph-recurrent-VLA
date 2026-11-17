"""Fix the closed-loop evaluation seeds and measure the scripted expert on them.

Seeds are scanned consecutively from ``--start-seed`` (disjoint from the training
collection, which starts at 100000) and accepted with the same initial
``ends_and_grasp`` visibility gate as the training dataset. The expert's success
on these seeds is the reference upper bound for every learned policy.

Example:
    python -m visual.baseline.eval_seeds --output outputs/visual_baseline/eval_seeds.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch

from data_collection.collect_episodes import initial_visibility
from data_collection.scripted_policy import make_policy
from visual.baseline.common import HORIZON, make_env


def scan_seeds(seed_range: tuple[int, int]) -> list[int]:
    """Return the seeds in ``[start, stop)`` whose first frame passes the gate."""
    start, stop = seed_range
    env = make_env()
    accepted = []
    try:
        for seed in range(start, stop):
            observation, _ = env.reset(seed=seed)
            visibility, _ = initial_visibility(env, observation)
            if visibility["requested_pass"]:
                accepted.append(seed)
    finally:
        env.close()
    return accepted


def run_expert(seeds: list[int]) -> list[dict]:
    """Roll out the scripted expert once per seed and report success and length."""
    env = make_env()
    policy = make_policy()
    results = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            policy.reset(env, seed)
            success_step = None
            for step in range(HORIZON):
                action = torch.as_tensor(policy.act(observation), device=env.unwrapped.device)
                observation, _, terminated, truncated, info = env.step(action)
                if bool(info["success"].any()):
                    success_step = step + 1
                    break
                if bool(terminated.any() or truncated.any()):
                    break
            results.append(dict(seed=seed, success=success_step is not None, steps=success_step))
    finally:
        env.close()
    return results


def chunks(values: list, count: int) -> list[list]:
    """Split ``values`` into ``count`` nearly equal contiguous lists."""
    size = -(-len(values) // count)
    return [values[i:i + size] for i in range(0, len(values), size)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=500000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--scan", type=int, default=600, help="seeds to scan")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new output path")

    step = -(-args.scan // args.workers)
    ranges = [
        (args.start_seed + i * step, min(args.start_seed + (i + 1) * step, args.start_seed + args.scan))
        for i in range(args.workers)
    ]
    context = mp.get_context("spawn")
    with context.Pool(args.workers) as pool:
        accepted = sorted(seed for part in pool.map(scan_seeds, ranges) for seed in part)
    if len(accepted) < args.num_seeds:
        raise SystemExit(f"only {len(accepted)} seeds passed; increase --scan")
    seeds = accepted[: args.num_seeds]

    with context.Pool(args.workers) as pool:
        expert = [row for part in pool.map(run_expert, chunks(seeds, args.workers)) for row in part]

    successes = sum(row["success"] for row in expert)
    record = dict(
        start_seed=args.start_seed,
        scanned=args.scan,
        gate="ends_and_grasp (requested_pass)",
        horizon=HORIZON,
        seeds=seeds,
        last_seed_needed=seeds[-1],
        expert_success=successes,
        expert_success_rate=successes / len(seeds),
        expert=expert,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(f"seeds={len(seeds)} expert_success={successes}/{len(seeds)}")


if __name__ == "__main__":
    main()
