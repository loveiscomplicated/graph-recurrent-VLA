"""Run one real CUDA detector--TAPNext++--tracker--GNN frontend step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from data_collection.collect_episodes import initial_visibility
from envs.maniskill import get_gt_state, make_peg_insertion_env
from graph.surface_context_frontend import SurfaceContextFrontend


def _numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    """Find one accepted reset and execute the complete frozen CUDA frontend."""
    config_path = Path("recurrent/configs/surface_mamba/export_distributed.yaml")
    config = yaml.safe_load(config_path.read_text())
    frontend = SurfaceContextFrontend(config["frontend"])
    env = make_peg_insertion_env(
        device="cpu",
        render_backend="cuda",
        observation_device="cuda",
        obs_mode="rgb+depth",
        control_mode="pd_ee_delta_pose",
        sensor_configs={
            name: {"width": 512, "height": 512}
            for name in ("base_camera", "hand_camera")
        },
        max_episode_steps=250,
    )
    try:
        for seed in range(4_000_000, 4_000_100):
            observation, _ = env.reset(seed=seed)
            visibility, _ = initial_visibility(env, observation)
            if visibility["requested_pass"]:
                break
        else:
            raise RuntimeError("No visibility-approved reset found in the smoke range")

        camera = observation["sensor_data"]["base_camera"]
        calibration = observation["sensor_param"]["base_camera"]
        tcp = _numpy(get_gt_state(env).ee_pose)[0, :3]
        result = frontend.step_arrays(
            _numpy(camera["rgb"])[0],
            _numpy(camera["depth"])[0].squeeze(),
            _numpy(calibration["intrinsic_cv"])[0],
            _numpy(calibration["extrinsic_cv"])[0],
            tcp,
        )
        if result.object21.shape != (21,) or not np.isfinite(result.object21).all():
            raise RuntimeError("Surface frontend smoke returned an invalid Object21")
        print(f"surface frontend CUDA PASS seed={seed}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
