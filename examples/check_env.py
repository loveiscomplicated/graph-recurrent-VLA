"""One-command installation and simulator smoke check."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

try:
    from envs.maniskill import make_peg_insertion_env
except ImportError as exc:
    raise SystemExit(
        f"ManiSkill imports failed: {exc}. Install dependencies with "
        "`.venv/bin/python -m pip install -r requirements.txt`."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default=None)
    args = parser.parse_args()

    for package in ("mani-skill", "sapien", "gymnasium", "torch"):
        print(f"{package}: {importlib.metadata.version(package)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Environment: PegInsertionSide-v1")

    try:
        env = make_peg_insertion_env(
            device=args.device, num_envs=args.num_envs,
            obs_mode=args.obs_mode, control_mode=args.control_mode,
        )
        try:
            task = env.unwrapped
            print(f"Simulation backend: {task.backend.sim_backend}")
            print(f"Render backend: {task.backend.render_backend}")
            print(f"Device: {task.device}")
            print(f"Observation mode: {task.obs_mode}")
            print(f"Control mode: {task.control_mode}")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
            print(f"Action bounds: low={env.action_space.low}, high={env.action_space.high}")
            obs, info = env.reset(seed=0)
            print(f"Reset: OK ({type(obs).__name__}; info keys={list(info)})")
            action = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=task.device)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: OK (reward={reward.tolist()}, terminated={terminated.tolist()}, truncated={truncated.tolist()})")
        finally:
            env.close()
    except Exception as exc:
        raise SystemExit(f"Environment check failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
