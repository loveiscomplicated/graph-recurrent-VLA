"""Collect compact, causal State36 expert episodes under scripted hole motion."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.collect_episodes import initial_visibility
from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import make_policy
from data_collection.storage import to_numpy, write_json
from data_collection.surface_mamba_dataset import _environment, _write_npz, apply_dart_noise
from data_collection.surface_mamba_motion import MovingSurfaceEpisode
from envs.maniskill import get_gt_state
from graph.surface_context_frontend import SurfaceContextFrontend
from recurrent.surface_state import from_gt_state, state36
from utils.experiment_logger import ExperimentLogger


EPISODE_FIELDS = ("state36_gt", "state36_graph", "action_target", "action_executed")


def _expert_episode(environment, frontend, seed, dart_noise):
    """Align each cached Graph/GT frame with its clean expert action."""
    observation, _ = environment.reset(seed=seed)
    visibility, _ = initial_visibility(environment.env, observation)
    if not visibility["requested_pass"]:
        return "visibility", None, None
    expert = make_policy()
    expert.reset(environment.env, seed)
    frontend.reset()
    rng = np.random.default_rng([seed, 23])
    frames = {name: [] for name in EPISODE_FIELDS}
    for _ in range(environment.horizon):
        robot, gt = from_gt_state(get_gt_state(environment.env))
        graph = frontend.step(observation, robot[:3]).object21
        clean = to_numpy(expert.act(observation)).reshape(7).astype(np.float32)
        executed = apply_dart_noise(clean, rng, dart_noise) if dart_noise else clean
        frames["state36_gt"].append(state36(robot, gt))
        frames["state36_graph"].append(state36(robot, graph))
        frames["action_target"].append(clean)
        frames["action_executed"].append(executed)
        action = torch.as_tensor(executed[None], device=environment.unwrapped.device)
        observation, _, terminated, truncated, info = environment.step(action)
        if bool(to_numpy(info["success"]).any()):
            receipt = environment.motion_receipt()
            if not receipt["motion_completed"]:
                return "motion_incomplete", None, receipt
            return "accepted", {name: np.stack(values) for name, values in frames.items()}, receipt
        if bool(to_numpy(terminated).any() or to_numpy(truncated).any()):
            break
    return "expert_failed", None, environment.motion_receipt()


def _collect_bucket(output, config, bucket, environment, frontend):
    bucket_path = output / bucket["name"]
    bucket_path.mkdir(exist_ok=True)
    accepted = []
    stop = bucket["seed_start"] + bucket["max_attempts"]
    for seed in range(bucket["seed_start"], stop):
        if seed in config["excluded_seeds"]:
            continue
        receipt_path = bucket_path / f"{seed}.json"
        archive = bucket_path / f"{seed}.npz"
        if receipt_path.exists():
            row = json.loads(receipt_path.read_text())
        else:
            status, arrays, receipt = _expert_episode(environment, frontend, seed, bucket["dart_noise"])
            row = {"seed": seed, "status": status, "motion": receipt}
            if status == "accepted":
                _write_npz(archive, **arrays)
                row.update(file=str(archive), sha256=file_sha256(archive), rows=len(arrays["action_target"]))
            write_json(receipt_path, row)
            print(f"{bucket['name']} seed={seed} {status}", flush=True)
        if row["status"] == "accepted":
            if row["seed"] != seed or row["sha256"] != file_sha256(archive):
                raise ValueError(f"Moving episode receipt differs: {archive}")
            accepted.append(row)
        if len(accepted) == bucket["count"]:
            return accepted
    raise RuntimeError(f"Exhausted attempts for {bucket['name']}: {len(accepted)} accepted")


def collect(config):
    """Persist successful, fully moved episodes and atomic attempt receipts."""
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    effective_path = output / "effective_config.json"
    if effective_path.exists() and json.loads(effective_path.read_text()) != config:
        raise ValueError("Moving collection resume requires identical configuration")
    write_json(effective_path, config)
    summary = output / "collection_summary.json"
    if summary.exists():
        return
    frontend = SurfaceContextFrontend(config["frontend"])
    env = _environment(config["environment"])
    wrapped = MovingSurfaceEpisode(env, config["motion"], config["horizon"])
    try:
        with ExperimentLogger(
            team="recurrent", model="surface_mamba_moving", job_type="collect",
            config=config, root=output / "experiments",
        ) as logger:
            splits = {"train": [], "validation": []}
            for bucket in config["buckets"]:
                splits[bucket["split"]].extend(_collect_bucket(output, config, bucket, wrapped, frontend))
            write_json(summary, {"motion": config["motion"], "splits": splits})
            logger.link_output(summary)
    finally:
        wrapped.close()


def merge_collections(sources, output, config):
    """Validate worker receipts and pack disjoint train/validation splits."""
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "validation": []}
    for source in sources:
        document = json.loads(source.read_text())
        if document["motion"] != config["motion"]:
            raise ValueError(f"Motion settings differ: {source}")
        for key in splits:
            splits[key].extend(document["splits"][key])
    manifest = {
        "format": "surface_mamba_moving_state36_v1",
        "motion": config["motion"],
        "frontend": config["frontend"],
        "splits": {},
    }
    seen = set(config["excluded_seeds"])
    for split, records in splits.items():
        columns = {name: [] for name in EPISODE_FIELDS}
        offsets = [0]
        seeds = []
        for row in records:
            seed = row["seed"]
            source = Path(row["file"])
            if seed in seen or file_sha256(source) != row["sha256"]:
                raise ValueError(f"Seed collision or corrupted moving trajectory: {source}")
            seen.add(seed)
            seeds.append(seed)
            with np.load(source, allow_pickle=False) as episode:
                for key in columns:
                    columns[key].append(episode[key])
            offsets.append(offsets[-1] + row["rows"])
        path = output / f"{split}.npz"
        _write_npz(
            path,
            **{name: np.concatenate(values) for name, values in columns.items()},
            reset_seed=np.asarray(seeds, dtype=np.int64),
            episode_offsets=np.asarray(offsets, dtype=np.int64),
        )
        manifest["splits"][split] = {
            "sha256": file_sha256(path), "episodes": len(seeds), "rows": offsets[-1],
        }
    write_json(output / "dataset_manifest.json", manifest)
