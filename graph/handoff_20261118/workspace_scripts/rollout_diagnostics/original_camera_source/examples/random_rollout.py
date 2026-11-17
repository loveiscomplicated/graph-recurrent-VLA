"""Run a short, stable zero-action rollout to check multi-step simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from envs.maniskill import make_peg_insertion_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    env = make_peg_insertion_env(obs_mode="state")
    try:
        env.reset(seed=0)
        action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
        total_reward = 0.0
        for step in range(1, args.steps + 1):
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward[0])
            if bool(terminated[0]) or bool(truncated[0]):
                break
        print(f"Episode steps: {step}")
        print(f"Reward sum: {total_reward:.4f}")
        print(f"Terminated: {bool(terminated[0])}; truncated: {bool(truncated[0])}")
        print(f"Success: {bool(info['success'][0])}")
        print("Action: zeros (stable smoke rollout, not a policy)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
