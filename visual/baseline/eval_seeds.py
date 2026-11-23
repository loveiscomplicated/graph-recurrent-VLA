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


def scan_new_seeds(context, start_seed: int, scan: int, num_seeds: int, workers: int) -> list[int]:
    """Scan ``scan`` consecutive seeds in parallel and keep the first that pass the gate."""
    step = -(-scan // workers)
    ranges = [
        (start_seed + i * step, min(start_seed + (i + 1) * step, start_seed + scan))
        for i in range(workers)
    ]
    with context.Pool(workers) as pool:
        accepted = sorted(seed for part in pool.map(scan_seeds, ranges) for seed in part)
    if len(accepted) < num_seeds:
        raise SystemExit(f"only {len(accepted)} seeds passed; increase --scan")
    return accepted[:num_seeds]


def training_seeds(index_path: Path, num_seeds: int) -> list[int]:
    """Reset seeds of the first training episodes, in the file order train.py uses.

    train.py holds out the *last* sorted files for validation, so the first
    files are always training episodes as long as num_seeds is below the
    training count.
    """
    index = json.loads(index_path.read_text())
    rows = sorted(index["rows"], key=lambda row: row["file"])
    if len(rows) < num_seeds:
        raise SystemExit(f"{index_path} lists {len(rows)} episodes, fewer than --num-seeds {num_seeds}")
    return [int(row["seed"]) for row in rows[:num_seeds]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=500000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--scan", type=int, default=600, help="seeds to scan")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--from-prepared-index", type=Path, default=None,
        help="take the first --num-seeds training-episode seeds from a prepare_data index.json "
             "instead of scanning new seeds (measures the train-vs-new-scene gap)",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new output path")

    context = mp.get_context("spawn")
    if args.from_prepared_index is not None:
        seeds = training_seeds(args.from_prepared_index, args.num_seeds)
        gate = f"training episodes of {args.from_prepared_index}"
    else:
        seeds = scan_new_seeds(context, args.start_seed, args.scan, args.num_seeds, args.workers)
        gate = "ends_and_grasp (requested_pass)"

    with context.Pool(args.workers) as pool:
        expert = [row for part in pool.map(run_expert, chunks(seeds, args.workers)) for row in part]

    successes = sum(row["success"] for row in expert)
    record = dict(
        # Scan settings are meaningless when seeds come from a prepared index.
        start_seed=None if args.from_prepared_index else args.start_seed,
        scanned=None if args.from_prepared_index else args.scan,
        gate=gate,
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
