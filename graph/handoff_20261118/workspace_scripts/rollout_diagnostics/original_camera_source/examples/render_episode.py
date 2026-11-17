"""Inspect RGB sensors and save a short frame sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch

from envs.maniskill import make_peg_insertion_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--output", type=Path, default=Path("outputs/render_test"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output_dir = args.output.resolve()
    env = make_peg_insertion_env(obs_mode="rgb", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        cameras = list(obs["sensor_data"])
        print(f"Cameras: {cameras}")
        if args.camera not in cameras:
            parser.error(f"unknown camera {args.camera!r}; choose from {cameras}")
        action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(args.steps):
            rgb = obs["sensor_data"][args.camera]["rgb"]
            print(f"Frame {frame}: shape={tuple(rgb.shape)} dtype={rgb.dtype} range=[{int(rgb.min())}, {int(rgb.max())}]")
            Image.fromarray(rgb[0].detach().cpu().numpy()).save(output_dir / f"frame_{frame:03d}.png")
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                break
        render = env.render()
        print(f"Human render: shape={tuple(render.shape)} dtype={render.dtype}")
        Image.fromarray(render[0].detach().cpu().numpy()).save(output_dir / "render_camera.png")
        print(f"Saved frames to {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
