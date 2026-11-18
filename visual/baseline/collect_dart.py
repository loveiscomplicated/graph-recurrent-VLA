"""Collect DART-style expert data: execute noisy actions, label with the clean expert action.

The plain expert never leaves its nominal path, so the policy never sees how to
recover from lateral misalignment (the dominant closed-loop failure). Here the
executed action is ``expert + Gaussian noise`` while the stored label is the clean
expert action at the visited (perturbed) state. Episodes are written directly in
the prepared format of prepare_data.py (256 px RGB, 128 px depth, 17D robot state, GT).

Seeds start at 300000 (disjoint from the shared-camera training collection at
200000-229999 and from evaluation at 500000+) and
use the same ends_and_grasp visibility gate.

Example:
    python -m visual.baseline.collect_dart --output data/visual_baseline/episodes_dart --episodes 600
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from data_collection.collect_episodes import initial_visibility
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state
from visual.baseline.common import HORIZON, make_env, resize_depth, resize_rgb, robot_state
from visual.baseline.prepare_data import CAMERA_KEYS, DEPTH_SIZE, GT_KEYS, RGB_SIZE

# Noise std in native action units: translation saturates at 0.2, rotation at 1.0.
TRANSLATION_NOISE = 0.08
ROTATION_NOISE = 0.1
# Noise comes in bursts so the expert recovers in between and still finishes the
# insertion (constant noise made the 1.5 mm insert gate unreachable: 0/4 success).
BURST_PROBABILITY = 0.08
BURST_STEPS = 5
NOISY_PHASES = ("reach", "lift", "align")  # not while grasping or inserting


def record_observation(env, observation, buffers) -> None:
    """Append resized images, GT and robot state of the current state."""
    for name, camera in CAMERA_KEYS.items():
        image = observation["sensor_data"][camera]
        buffers[f"{name}_rgb"].append(resize_rgb(image["rgb"].cpu(), RGB_SIZE).numpy()[0])
        buffers[f"{name}_depth"].append(resize_depth(image["depth"].cpu(), DEPTH_SIZE).numpy()[0])
    gt = get_gt_state(env)
    for key in GT_KEYS:
        buffers[f"gt_{key}"].append(getattr(gt, key)[0].reshape(-1).cpu().numpy().astype(np.float32))


def collect_episode(env, expert, seed, rng):
    """Roll out one noisy-execution episode; return arrays and success flag."""
    observation, _ = env.reset(seed=seed)
    expert.reset(env, seed)
    buffers = {key: [] for key in [f"{n}_{k}" for n in CAMERA_KEYS for k in ("rgb", "depth")]
               + [f"gt_{k}" for k in GT_KEYS] + ["actions"]}
    success = False
    burst = 0
    noisy_steps = 0
    for _ in range(HORIZON):
        record_observation(env, observation, buffers)
        clean = torch.as_tensor(expert.act(observation)).reshape(-1).float()
        buffers["actions"].append(clean.numpy())
        executed = clean.clone()
        if expert.phase in NOISY_PHASES:
            if burst == 0 and rng.random() < BURST_PROBABILITY:
                burst = BURST_STEPS
        else:
            burst = 0
        if burst > 0:
            executed[:3] += torch.from_numpy(rng.normal(0, TRANSLATION_NOISE, 3)).float()
            executed[3:6] += torch.from_numpy(rng.normal(0, ROTATION_NOISE, 3)).float()
            burst -= 1
            noisy_steps += 1
        executed = executed.clamp(-1, 1).reshape(env.action_space.shape)
        observation, _, terminated, truncated, info = env.step(executed)
        if bool(info["success"].any()):
            success = True
            break
        if bool(terminated.any() or truncated.any()):
            break

    arrays = {key: np.stack(values) for key, values in buffers.items()}
    arrays["robot_state"] = robot_state(arrays["gt_ee_pose"], arrays["gt_left_fingertip_pos"],
                                arrays["gt_right_fingertip_pos"], arrays["gt_gripper_qpos"])
    assert len(arrays["actions"]) == len(arrays["robot_state"])
    return arrays, success, noisy_steps


def collect_worker(task) -> list[dict]:
    """Collect episodes for a contiguous seed range until ``quota`` are accepted."""
    worker_id, first_seed, last_seed, quota, output = task
    torch.set_num_threads(2)
    env = make_env()
    expert = make_policy()
    rng = np.random.default_rng(worker_id)
    rows = []
    try:
        for seed in range(first_seed, last_seed):
            if len(rows) >= quota:
                break
            observation, _ = env.reset(seed=seed)
            visibility, _ = initial_visibility(env, observation)
            if not visibility["requested_pass"]:
                continue
            arrays, success, noisy_steps = collect_episode(env, expert, seed, rng)
            name = f"dart_{seed}.npz"
            np.savez(Path(output) / name, **arrays)
            rows.append(dict(episode=name[:-4], file=name, seed=seed, steps=len(arrays["actions"]),
                             success=success, noisy_steps=noisy_steps))
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--start-seed", type=int, default=300000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    quota = -(-args.episodes // args.workers)
    span = quota * 8  # ~25% of seeds pass the visibility gate
    tasks = [(i, args.start_seed + i * span, args.start_seed + (i + 1) * span, quota, str(args.output))
             for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [row for part in pool.map(collect_worker, tasks) for row in part]

    index = dict(rgb_size=RGB_SIZE, depth_size=DEPTH_SIZE, episodes=len(rows),
                 steps=sum(r["steps"] for r in rows), translation_noise=TRANSLATION_NOISE,
                 rotation_noise=ROTATION_NOISE, burst_probability=BURST_PROBABILITY,
                 burst_steps=BURST_STEPS, noisy_phases=NOISY_PHASES,
                 success=sum(r["success"] for r in rows), rows=rows)
    (args.output / "index.json").write_text(json.dumps(index, indent=1))
    print(f"episodes={len(rows)} steps={index['steps']} expert_success={index['success']}")


if __name__ == "__main__":
    main()
