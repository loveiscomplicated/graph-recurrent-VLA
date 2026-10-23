"""Check whether state-only PegInsertionSide can run GPU physics on this host.

Run this in a fresh process. SAPIEN render-device selection is process-global.
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - register the task
import torch

from envs.maniskill import get_gt_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    if not torch.cuda.is_available():
        parser.error("Torch cannot see a CUDA device")

    env = gym.make(
        "PegInsertionSide-v1",
        num_envs=args.num_envs,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cuda",
        render_backend="none",
    )
    try:
        obs, _ = env.reset(seed=0)
        gt = get_gt_state(env)
        action = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=env.unwrapped.device)
        for _ in range(5):
            env.step(action)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(action)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"backend={env.unwrapped.backend.sim_backend} render={env.unwrapped.backend.render_backend}")
        print(f"device={env.unwrapped.device} obs={tuple(obs.shape)} gt_qpos={tuple(gt.qpos.shape)}")
        print(f"env_steps_per_s={args.num_envs * args.steps / elapsed:.1f} ({args.num_envs} envs, {args.steps} steps, {elapsed:.2f}s)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
