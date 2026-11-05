"""Verify native CUDA RGB-D before the output wrapper, on reset and step."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from envs.maniskill import make_peg_insertion_env
from envs.maniskill.observation_device import ObservationDevice


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; run this check on a Linux CUDA/Vulkan host")
    original = ObservationDevice.observation

    def check(wrapper, observation):
        for name in ("base_camera", "hand_camera"):
            for channel in ("rgb", "depth"):
                image = observation["sensor_data"][name][channel]
                if not image.is_cuda:
                    raise RuntimeError(
                        f"{name}/{channel} was on CPU before output transfer"
                    )
                print(
                    f"Native {name}/{channel}: {image.device}, {image.dtype}, {tuple(image.shape)}"
                )
        return original(wrapper, observation)

    with patch.object(ObservationDevice, "observation", check):
        env = make_peg_insertion_env(
            device="cpu",
            render_backend="cuda",
            observation_device="cuda",
            obs_mode="rgb+depth",
        )
        try:
            print(
                f"Physics: {env.unwrapped.backend.sim_backend}; renderer: {env.unwrapped.backend.render_backend}"
            )
            observation, _ = env.reset(seed=0)
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            observation, *_ = env.step(action)
            assert observation["agent"]["qpos"].is_cuda
            print(
                "PASS: reset/step RGB-D were CUDA before wrapping; qpos output is CUDA"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
