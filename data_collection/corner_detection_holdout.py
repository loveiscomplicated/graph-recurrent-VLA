"""Collect matched policy rollouts and freeze their sampled base-camera RGB-D."""

from functools import partial
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from data_collection.collection_batches import collection_defaults, iter_episode_jobs
from data_collection.episode_dataset import file_sha256
from data_collection.mixed_trajectory_collection import (
    MODES, initial_fingerprint, random_action, validate_camera,
)
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger

IMAGES = ("rgb", "depth", "segmentation")
CALIBRATION = ("intrinsic_cv", "extrinsic_cv")
GROUND_TRUTH = ("peg_pose", "box_pose", "hole_pose", "peg_half_sizes", "hole_radius")


def frame_snapshot(observation, state):
    """Keep native arrays for replay and independent visibility scoring."""
    camera = observation["sensor_data"]["base_camera"]
    calibration = observation["sensor_param"]["base_camera"]
    arrays = {key: camera[key][0].numpy().copy() for key in IMAGES}
    arrays.update({key: calibration[key][0].numpy().copy() for key in CALIBRATION})
    arrays.update({key: getattr(state, key)[0].numpy().copy() for key in GROUND_TRUTH})
    return arrays


def rollout(env, seed, mode, config, output, switch_step):
    """Replay a seed through termination, retaining the action trace and sampled frames."""
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    validate_camera(obs, config["camera"])
    state = get_gt_state(env)
    initial = initial_fingerprint(obs, state)
    initial_position = state.peg_pose[0, :3].numpy().copy()
    initial_rgbd = hashlib.sha256()
    for key in IMAGES:
        initial_rgbd.update(obs["sensor_data"]["base_camera"][key].numpy().tobytes())

    task = env.unwrapped
    actor_ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    expert = ScriptedPegPolicy()
    if mode != "random":
        expert.reset(env, seed)

    mode_index = MODES.index(mode)
    rng = np.random.default_rng([config["protocol_seed"], seed, mode_index])
    random_start = 0 if mode == "random" else switch_step
    actions, frames, frame_info = [], [], []
    phases = []
    terminated = truncated = False
    displacement_m = 0.0
    gripper = 1.0
    while True:
        step = len(actions)
        state = get_gt_state(env)
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        moved = np.linalg.norm(state.peg_pose[0, :3].numpy() - initial_position)
        displacement_m = max(displacement_m, float(moved))
        in_random = random_start is not None and step >= random_start
        phase = "random" if in_random else expert.phase
        if step % config["frame_stride"] == 0 or terminated or truncated or (
            switch_step is not None and abs(step - switch_step) <= 1
        ):
            frames.append(frame_snapshot(obs, state))
            frame_info.append({"step": step, "phase": phase, "grasp": grasp,
                               "after_switch": mode == "expert_random" and step > switch_step})
        if terminated or truncated:
            break

        phases.append(phase)
        if in_random:
            if (step - random_start) % config["gripper_interval"] == 0:
                gripper = float(rng.choice([-1, 1]))
            action = random_action(rng, config["random_limit"], gripper, task)
        else:
            action = expert.act(obs)
        actions.append(action.numpy().copy())
        obs, _, terminated_flag, truncated_flag, _ = env.step(action)
        terminated = bool(terminated_flag.any())
        truncated = bool(truncated_flag.any())

    arrays = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    arrays["actions"] = np.stack(actions)
    name = f"{mode}_{seed}.npz"
    path = output / name
    np.savez_compressed(path, **arrays)
    receipt = {
        "seed": seed, "mode": mode, "file": name, "sha256": file_sha256(path),
        "actor_ids": actor_ids, "initial_state_sha256": initial,
        "initial_rgbd_sha256": initial_rgbd.hexdigest(),
        "steps": len(actions), "frames": frame_info, "switch_step": switch_step,
        "success": bool(state.success[0]), "terminated": terminated, "truncated": truncated,
        "max_peg_displacement_mm": displacement_m * 1000,
    }
    return receipt, arrays["actions"], phases


def collect_seed(seed, config, output):
    """Pair the three rollouts and verify the mixed run's expert prefix."""
    torch.set_num_threads(config["threads"])
    env = make_peg_insertion_env(
        obs_mode="rgb+depth+segmentation",
        control_mode=config["control_mode"],
        max_episode_steps=config["max_steps"],
    )
    try:
        expert, expert_actions, phases = rollout(env, seed, "expert", config, output, None)
        rng = np.random.default_rng([config["protocol_seed"], seed])
        switch_step = int(rng.integers(1, expert["steps"]))
        random, _, _ = rollout(env, seed, "random", config, output, None)
        mixed, mixed_actions, _ = rollout(env, seed, "expert_random", config, output, switch_step)
    finally:
        env.close()

    episodes = [expert, random, mixed]
    for key in ("initial_state_sha256", "initial_rgbd_sha256"):
        if len({episode[key] for episode in episodes}) != 1:
            raise ValueError(f"Reset {key} differed between policies for seed {seed}")
    if len(mixed_actions) <= switch_step:
        raise ValueError(f"Seed {seed} terminated before its first random action")
    np.testing.assert_allclose(mixed_actions[:switch_step], expert_actions[:switch_step],
                               atol=1e-6, rtol=0)
    mixed["expert_phase_at_switch"] = phases[switch_step]
    print(f"{seed}: steps {[row['steps'] for row in episodes]}, "
          f"switch {switch_step} ({phases[switch_step]})", flush=True)
    return episodes


def collect(config):
    """Create a fresh, checksummed holdout cohort for both accelerators."""
    config = collection_defaults(config)
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence of distinct reset seeds")
    if config["max_steps"] < 2 or config["frame_stride"] < 1:
        raise ValueError("max_steps must be at least 2 and frame_stride must be positive")
    if not 0 < config["random_limit"] <= 1 or config["gripper_interval"] < 1:
        raise ValueError("Expected random_limit in (0, 1] and a positive gripper_interval")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("mani_skill", "sapien", "torch", "numpy")
    }
    with ExperimentLogger(team="visual", model="corner_detection_holdout", config=config,
                          job_type="collection", seed=config["protocol_seed"]) as logger:
        output = Path(config["output_root"])
        output.mkdir(parents=True, exist_ok=False)
        logger.record(environment={"env_id": "PegInsertionSide-v1", "physics": "physx_cpu",
                                   "renderer": "sapien_cpu", "versions": versions})
        worker = partial(collect_seed, config=config, output=output.resolve())
        batches = iter_episode_jobs(worker, seeds, config, episodes_per_job=len(MODES))
        episodes = [episode for batch in batches for episode in batch]
        path = output / "manifest.json"
        write_json(path, {"config": config, "versions": versions, "episodes": episodes})
        logger.link_output(path, kind="episode_metadata")
        print(path, flush=True)
