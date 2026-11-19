"""Collect DART expert rollouts and export frozen Surface-Graph State36 pairs."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from typing import Any

import numpy as np
import torch

from data_collection.collect_episodes import initial_visibility
from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import make_policy
from data_collection.storage import append_attempt, save_frame, to_numpy, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from graph.surface_context_frontend import SurfaceContextFrontend
from recurrent.surface_state import from_frame_archive, state36
from utils.experiment_logger import ExperimentLogger


DATASET_FORMAT = "surface_mamba_raw_v1"
SHARDED_DATASET_FORMAT = "surface_mamba_raw_shard_v1"
DERIVED_FORMAT = "surface_mamba_state36_v1"
SHARDED_DERIVED_FORMAT = "surface_mamba_state36_shard_v1"
SHARD_PACKAGE_FORMAT = "surface_mamba_shard_package_v1"
EXPECTED_BUCKETS = {
    "train_clean": ("train", False, 360),
    "train_dart": ("train", True, 360),
    "validation_clean": ("validation", False, 90),
    "test_clean": ("test", False, 90),
}


def apply_dart_noise(
    clean_action: np.ndarray,
    generator: np.random.Generator,
    standard_deviation: float,
) -> np.ndarray:
    """Perturb only six motion axes, leaving the gripper command unchanged."""
    clean_action = np.asarray(clean_action, dtype=np.float32)
    if clean_action.shape[-1] != 7 or not np.isfinite(clean_action).all():
        raise ValueError("Clean expert action must be finite with final dimension 7")
    if not np.isfinite(standard_deviation) or standard_deviation < 0:
        raise ValueError("DART standard deviation must be finite and nonnegative")
    executed = clean_action.copy()
    executed[..., :6] += generator.normal(
        0.0, standard_deviation, size=executed[..., :6].shape
    ).astype(np.float32)
    executed[..., :6] = np.clip(executed[..., :6], -1, 1)
    np.testing.assert_array_equal(executed[..., 6], clean_action[..., 6])
    return executed


def _source_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return {
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("mani-skill", "sapien", "gymnasium", "torch", "numpy")
        },
    }


def _validate_buckets(config: dict[str, Any]) -> list[dict[str, Any]]:
    buckets = config.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != len(EXPECTED_BUCKETS):
        raise ValueError("collection.buckets must define the four fixed dataset buckets")
    by_name = {bucket.get("name"): bucket for bucket in buckets}
    if set(by_name) != set(EXPECTED_BUCKETS) or len(by_name) != len(buckets):
        raise ValueError(f"Bucket names must be exactly {sorted(EXPECTED_BUCKETS)}")
    occupied = []
    for name, (split, dart, count) in EXPECTED_BUCKETS.items():
        bucket = by_name[name]
        expected = {"split": split, "dart": dart, "count": count}
        for key, value in expected.items():
            if bucket.get(key) != value:
                raise ValueError(f"Bucket {name} requires {key}={value!r}")
        for key in ("start_seed", "max_attempts"):
            if type(bucket.get(key)) is not int or bucket[key] < 0:
                raise ValueError(f"Bucket {name}.{key} must be a nonnegative integer")
        if bucket["max_attempts"] < count:
            raise ValueError(f"Bucket {name} attempt budget is smaller than its quota")
        occupied.append(
            range(bucket["start_seed"], bucket["start_seed"] + bucket["max_attempts"])
        )
    for index, left in enumerate(occupied):
        for right in occupied[index + 1 :]:
            if max(left.start, right.start) < min(left.stop, right.stop):
                raise ValueError("Collection bucket seed ranges overlap")
    return [by_name[name] for name in EXPECTED_BUCKETS]


def _validate_shard(config: dict[str, Any]) -> dict[str, int]:
    """Return one validated distributed shard assignment.

    Omitting ``shard`` preserves the original single-machine behavior. A
    distributed run uses zero-based ``index`` and a shared positive ``count``.
    Every worker must receive the same count and a different index.
    """
    assignment = config.get("shard")
    if assignment is None:
        return {"index": 0, "count": 1}
    if not isinstance(assignment, dict):
        raise ValueError("shard must be a mapping with integer index and count")
    index = assignment.get("index")
    count = assignment.get("count")
    if type(index) is not int or type(count) is not int or count < 1:
        raise ValueError("shard.index and shard.count must be integers with count >= 1")
    if index < 0 or index >= count:
        raise ValueError("shard.index must satisfy 0 <= index < shard.count")
    return {"index": index, "count": count}


def _partition_size(total: int, shard_count: int, shard_index: int) -> tuple[int, int]:
    """Return the contiguous offset and size assigned to one shard."""
    base, remainder = divmod(total, shard_count)
    size = base + int(shard_index < remainder)
    offset = shard_index * base + min(shard_index, remainder)
    return offset, size


def _partition_buckets(
    buckets: list[dict[str, Any]],
    assignment: dict[str, int],
) -> list[dict[str, Any]]:
    """Split bucket quotas and attempt ranges without overlapping reset seeds."""
    shard_index = assignment["index"]
    shard_count = assignment["count"]
    partitioned = []
    for bucket in buckets:
        episode_offset, episode_count = _partition_size(
            bucket["count"], shard_count, shard_index
        )
        attempt_offset, attempt_count = _partition_size(
            bucket["max_attempts"], shard_count, shard_index
        )
        if episode_count and attempt_count < episode_count:
            raise ValueError(
                f"Shard attempt range is smaller than its quota for {bucket['name']}"
            )
        local = dict(bucket)
        local.update(
            count=episode_count,
            start_seed=bucket["start_seed"] + attempt_offset,
            max_attempts=attempt_count,
            global_count=bucket["count"],
            global_start_seed=bucket["start_seed"],
            global_max_attempts=bucket["max_attempts"],
            global_episode_offset=episode_offset,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        partitioned.append(local)
    return partitioned


def _load_excluded_seeds(path: str | None) -> set[int]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())

    def integer_lists(value):
        if isinstance(value, list) and all(type(item) is int for item in value):
            yield from value
        elif isinstance(value, dict):
            for item in value.values():
                yield from integer_lists(item)

    return set(integer_lists(payload))


def create_seed_plan(config: dict[str, Any]) -> None:
    """Reserve the 300 evaluation seeds and merge historical collection exclusions."""
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation_seeds.json", "excluded_and_evaluation_seeds.json"):
        if (output / name).exists():
            raise FileExistsError(f"Seed-plan output already exists: {output / name}")
    historical = set()
    sources = []
    for source in config.get("historical_exclusion_files", []):
        path = Path(source)
        historical.update(_load_excluded_seeds(str(path)))
        sources.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    start = config["evaluation_start_seed"]
    count = config.get("evaluation_seed_count", 300)
    if type(start) is not int or start < 0 or count != 300:
        raise ValueError("Seed plan requires a nonnegative start and exactly 300 evaluation seeds")
    evaluation = []
    seed = start
    while len(evaluation) < count:
        if seed not in historical:
            evaluation.append(seed)
        seed += 1
    evaluation_path = output / "evaluation_seeds.json"
    excluded_path = output / "excluded_and_evaluation_seeds.json"
    write_json(evaluation_path, {"seeds": evaluation, "purpose": "paired held-out closed-loop evaluation"})
    write_json(
        excluded_path,
        {
            "historical_seeds": sorted(historical),
            "evaluation_seeds": evaluation,
            "sources": sources,
        },
    )
    print(f"Reserved {len(evaluation)} evaluation seeds in {output}", flush=True)


def _environment(settings: dict[str, Any]):
    required = {
        "device",
        "render_backend",
        "observation_device",
        "resolution",
        "control_mode",
        "max_episode_steps",
    }
    missing = required - set(settings)
    if missing:
        raise ValueError(f"Missing collection environment settings: {sorted(missing)}")
    if settings["resolution"] != 512 or settings["control_mode"] != "pd_ee_delta_pose":
        raise ValueError("Surface collection requires 512 RGB-D and pd_ee_delta_pose")
    return make_peg_insertion_env(
        device=settings["device"],
        render_backend=settings["render_backend"],
        observation_device=settings["observation_device"],
        obs_mode="rgb+depth",
        control_mode=settings["control_mode"],
        sensor_configs={
            name: {"width": 512, "height": 512}
            for name in ("base_camera", "hand_camera")
        },
        max_episode_steps=settings["max_episode_steps"],
    )


def _resume_bucket(directory: Path, bucket: dict[str, Any]) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not directory.exists():
        directory.mkdir(parents=True)
        manifest = {
            "status": "RUNNING",
            "bucket": bucket,
            "attempted": 0,
            "visibility_passed": 0,
            "expert_attempted": 0,
            "expert_failed": 0,
            "episodes": [],
        }
        write_json(manifest_path, manifest)
        return manifest
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bucket") != bucket:
        raise ValueError(f"Existing bucket configuration differs: {directory}")
    if manifest.get("status") == "COMPLETE":
        return manifest
    for partial in directory.glob("*.partial"):
        partial.rename(directory / f"{partial.name}.abandoned-{int(time.time())}")
    manifest["status"] = "RUNNING"
    write_json(manifest_path, manifest)
    return manifest


def _collect_bucket(
    env,
    root: Path,
    bucket: dict[str, Any],
    collection: dict[str, Any],
    excluded: set[int],
) -> dict[str, Any]:
    """Collect exactly one deterministic split/condition bucket with resume."""
    directory = root / "buckets" / bucket["name"]
    manifest = _resume_bucket(directory, bucket)
    if manifest["status"] == "COMPLETE":
        return manifest
    if bucket["count"] == 0:
        manifest["status"] = "COMPLETE"
        write_json(directory / "manifest.json", manifest)
        return manifest
    accepted_seeds = {row["seed"] for row in manifest["episodes"]}
    next_seed = bucket["start_seed"] + manifest["attempted"]
    final_seed = bucket["start_seed"] + bucket["max_attempts"]
    noise_standard_deviation = float(collection["dart_noise_std"])
    visibility_key = {
        "ends_and_grasp": "requested_pass",
        "full_surface": "strict_100_pass",
    }.get(collection["visibility_mode"])
    if visibility_key is None:
        raise ValueError("Unknown collection visibility_mode")

    for seed in range(next_seed, final_seed):
        manifest["attempted"] += 1
        if seed in excluded or seed in accepted_seeds:
            append_attempt(directory, {"seed": seed, "accepted": False, "reason": "excluded"})
            write_json(directory / "manifest.json", manifest)
            continue
        observation, info = env.reset(seed=seed)
        visibility, initial_gt = initial_visibility(env, observation)
        visible = bool(visibility[visibility_key])
        append_attempt(
            directory,
            {"seed": seed, "accepted": visible, "visibility": visibility},
        )
        if not visible:
            write_json(directory / "manifest.json", manifest)
            continue
        manifest["visibility_passed"] += 1
        manifest["expert_attempted"] += 1
        policy = make_policy()
        policy.reset(env, seed)
        generator = np.random.default_rng(collection["noise_seed"] + seed)
        name = f"episode_{len(manifest['episodes']):06d}"
        pending = directory / f"{name}.partial"
        pending.mkdir()
        write_json(pending / "visibility.json", visibility)
        save_frame(
            pending / "frame_000000.npz",
            observation,
            info=info,
            ground_truth=initial_gt,
        )
        success = False
        terminated = truncated = False
        steps = 0
        for step in range(collection["steps"]):
            clean = to_numpy(policy.act(observation)).astype(np.float32)
            executed = (
                apply_dart_noise(clean, generator, noise_standard_deviation)
                if bucket["dart"]
                else clean.copy()
            )
            action = torch.as_tensor(executed, device=env.unwrapped.device)
            if tuple(action.shape) != tuple(env.action_space.shape):
                raise ValueError("Scripted policy action does not match the environment action space")
            observation, reward, term, trunc, info = env.step(action)
            steps = step + 1
            terminated = bool(to_numpy(term).any())
            truncated = bool(to_numpy(trunc).any())
            success = success or bool(to_numpy(info.get("success", False)).any())
            save_frame(
                pending / f"frame_{steps:06d}.npz",
                observation,
                action=executed,
                action_target=clean,
                reward=reward,
                terminated=term,
                truncated=trunc,
                info=info,
                ground_truth=vars(get_gt_state(env)),
            )
            if terminated or truncated or success:
                break
        receipt = {
            "seed": seed,
            "path": name,
            "steps": steps,
            "observations": steps + 1,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "split": bucket["split"],
            "dart": bucket["dart"],
        }
        if not success:
            manifest["expert_failed"] += 1
            shutil.rmtree(pending)
            append_attempt(directory, {"seed": seed, "accepted": False, "reason": "expert_failed"})
            write_json(directory / "manifest.json", manifest)
            continue
        receipt["frame_sha256"] = [
            file_sha256(pending / f"frame_{index:06d}.npz")
            for index in range(steps + 1)
        ]
        write_json(pending / "episode.json", receipt)
        pending.rename(directory / name)
        manifest["episodes"].append(receipt)
        accepted_seeds.add(seed)
        print(
            f"{bucket['name']}: {len(manifest['episodes'])}/{bucket['count']} seed={seed}",
            flush=True,
        )
        if len(manifest["episodes"]) == bucket["count"]:
            manifest["status"] = "COMPLETE"
            write_json(directory / "manifest.json", manifest)
            return manifest
        write_json(directory / "manifest.json", manifest)
    manifest["status"] = "EXHAUSTED"
    write_json(directory / "manifest.json", manifest)
    raise RuntimeError(f"Attempt budget exhausted for {bucket['name']}")


def collect(config: dict[str, Any]) -> None:
    """Collect 720 training, 90 validation and 90 test successful episodes."""
    collection = config["collection"]
    global_buckets = _validate_buckets(collection)
    assignment = _validate_shard(config)
    buckets = _partition_buckets(global_buckets, assignment)
    expected_episode_count = sum(bucket["count"] for bucket in buckets)
    if expected_episode_count < 1:
        raise ValueError("This shard has no assigned episodes")
    excluded = _load_excluded_seeds(config.get("excluded_seeds"))
    output = Path(config["output"]).resolve()
    effective = {key: value for key, value in config.items() if key != "trainer"}
    if output.exists():
        if not config.get("resume"):
            raise FileExistsError(f"Collection output already exists: {output}")
        if json.loads((output / "effective_config.json").read_text()) != effective:
            raise ValueError("Resume requires an identical effective collection config")
        completed_manifest = output / "dataset_manifest.json"
        if completed_manifest.exists():
            completed = json.loads(completed_manifest.read_text())
            expected_format = (
                DATASET_FORMAT if assignment["count"] == 1 else SHARDED_DATASET_FORMAT
            )
            if (
                completed.get("format") != expected_format
                or completed.get("episode_count") != expected_episode_count
                or completed.get("shard") != assignment
            ):
                raise ValueError("Existing collection manifest is not a complete Surface-Mamba release")
            print(f"Reusing completed raw dataset at {output}", flush=True)
            return
    else:
        required_free = float(collection.get("required_free_gib", 120))
        free = shutil.disk_usage(output.parent).free / 2**30
        if free < required_free:
            raise OSError(
                f"Surface dataset collection requires {required_free:.1f} GiB free; found {free:.1f} GiB"
            )
        (output / "buckets").mkdir(parents=True)
        write_json(output / "effective_config.json", effective)
        write_json(output / "source.json", _source_metadata())

    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_raw_dataset",
        job_type="collection",
        seed=collection["noise_seed"],
        config=effective,
    ) as logger:
        env = _environment(collection["environment"])
        try:
            manifests = [
                _collect_bucket(env, output, bucket, collection, excluded)
                for bucket in buckets
            ]
        finally:
            env.close()
        episodes = []
        for manifest in manifests:
            if manifest["status"] != "COMPLETE":
                raise RuntimeError("All collection buckets must complete before publication")
            bucket_name = manifest["bucket"]["name"]
            episodes.extend(
                dict(row, path=f"buckets/{bucket_name}/{row['path']}")
                for row in manifest["episodes"]
            )
        seeds = [row["seed"] for row in episodes]
        if (
            len(episodes) != expected_episode_count
            or len(set(seeds)) != expected_episode_count
            or set(seeds) & excluded
        ):
            raise ValueError("Final raw dataset count, uniqueness or exclusion check failed")
        manifest_path = output / "dataset_manifest.json"
        local_counts = {
            bucket["name"]: bucket["count"] for bucket in buckets
        }
        write_json(
            manifest_path,
            {
                "format": (
                    DATASET_FORMAT
                    if assignment["count"] == 1
                    else SHARDED_DATASET_FORMAT
                ),
                "run_id": logger.run_id,
                "shard": assignment,
                "episodes": episodes,
                "episode_count": len(episodes),
                "counts": local_counts,
                "global_counts": {
                    name: expected[2] for name, expected in EXPECTED_BUCKETS.items()
                },
                "global_buckets": global_buckets,
                "bucket_assignments": buckets,
                "action_alignment": "frame_t+1 stores clean/executed action applied to observation_t",
                "config": effective,
            },
        )
        logger.link_output(manifest_path, kind="episode_metadata")
        logger.log({"episodes": len(episodes)})


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _encode_episode(
    raw_root: Path,
    record: dict[str, Any],
    frontend: SurfaceContextFrontend,
) -> dict[str, np.ndarray]:
    directory = raw_root / record["path"]
    frames = sorted(directory.glob("frame_*.npz"))
    if len(frames) != record["observations"] or len(frames) != record["steps"] + 1:
        raise ValueError(f"Raw episode frame count differs: {directory}")
    if len(record.get("frame_sha256", [])) != len(frames):
        raise ValueError(f"Raw episode lacks complete frame checksums: {directory}")
    for path, expected_hash in zip(frames, record["frame_sha256"], strict=True):
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Raw frame checksum differs: {path}")
    frontend.reset()
    rows: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "robot15",
            "object21_gt",
            "object21_graph",
            "state36_gt",
            "state36_graph",
            "action_target",
            "action_executed",
            "detected",
            "observed",
            "surface_xyz",
            "surface_valid",
            "node_valid",
            "endpoint_residual",
            "counts",
            "frontend_timing_ms",
        )
    }
    for index, path in enumerate(frames[:-1]):
        with np.load(path, allow_pickle=False) as frame:
            robot, object21_gt = from_frame_archive(frame)
            image = "observation/sensor_data/base_camera/"
            camera = "observation/sensor_param/base_camera/"
            result = frontend.step_arrays(
                frame[image + "rgb"][0],
                frame[image + "depth"][0].squeeze(),
                frame[camera + "intrinsic_cv"][0],
                frame[camera + "extrinsic_cv"][0],
                robot[:3],
            )
        with np.load(frames[index + 1], allow_pickle=False) as next_frame:
            clean = next_frame["action_target"].reshape(7).astype(np.float32)
            executed = next_frame["action"].reshape(7).astype(np.float32)
        values = {
            "robot15": robot,
            "object21_gt": object21_gt,
            "object21_graph": result.object21,
            "state36_gt": state36(robot, object21_gt),
            "state36_graph": state36(robot, result.object21),
            "action_target": clean,
            "action_executed": executed,
            "detected": result.detected,
            "observed": result.observed,
            "surface_xyz": result.surface_xyz,
            "surface_valid": result.surface_valid,
            "node_valid": result.node_valid,
            "endpoint_residual": result.endpoint_residual,
            "counts": result.counts,
            "frontend_timing_ms": result.timing_ms,
        }
        for key, value in values.items():
            rows[key].append(value)
    arrays = {key: np.stack(values) for key, values in rows.items()}
    expected_length = record["steps"]
    if any(len(value) != expected_length for value in arrays.values()):
        raise ValueError(f"Derived episode alignment differs: {directory}")
    if not np.isfinite(arrays["state36_graph"]).all():
        raise ValueError(f"Graph State36 contains nonfinite values: {directory}")
    np.testing.assert_array_equal(
        arrays["state36_gt"][:, :15], arrays["state36_graph"][:, :15]
    )
    return arrays


def _pack_split(
    path: Path,
    records: list[dict[str, Any]],
    encoded: Path,
) -> dict[str, Any]:
    if not records:
        return {"file": None, "episodes": 0, "rows": 0, "sha256": None}
    episode_rows = []
    offsets = [0]
    for record in records:
        source = encoded / f"{record['seed']}.npz"
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        episode_rows.append(arrays)
        offsets.append(offsets[-1] + len(arrays["action_target"]))
    keys = episode_rows[0]
    packed = {key: np.concatenate([row[key] for row in episode_rows]) for key in keys}
    packed.update(
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        reset_seed=np.asarray([row["seed"] for row in records], dtype=np.int64),
        episode_dart=np.asarray([row["dart"] for row in records], dtype=bool),
    )
    _write_npz(path, **packed)
    return {
        "file": path.name,
        "episodes": len(records),
        "rows": offsets[-1],
        "sha256": file_sha256(path),
    }


def _validate_encoded_episode(path: Path, record: dict[str, Any]) -> None:
    """Validate a completed per-episode export before resume reuses it."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "state36_gt",
            "state36_graph",
            "action_target",
            "action_executed",
            "surface_xyz",
            "surface_valid",
            "observed",
            "node_valid",
        }
        if not required <= set(archive.files):
            raise ValueError(f"Incomplete encoded episode: {path}")
        if (
            archive["state36_gt"].shape != (record["steps"], 36)
            or archive["state36_graph"].shape != (record["steps"], 36)
            or archive["action_target"].shape != (record["steps"], 7)
            or archive["action_executed"].shape != (record["steps"], 7)
            or not np.isfinite(archive["state36_graph"]).all()
        ):
            raise ValueError(f"Encoded episode has invalid shapes or values: {path}")


def export(config: dict[str, Any]) -> None:
    """Run the frozen frontend once and publish aligned split-level State36 files."""
    raw_root = Path(config["raw_root"]).resolve()
    raw_manifest_path = raw_root / "dataset_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_format = raw_manifest.get("format")
    if raw_format not in (DATASET_FORMAT, SHARDED_DATASET_FORMAT):
        raise ValueError("Expected a completed Surface-Mamba raw dataset or shard")
    assignment = _validate_shard(raw_manifest.get("config", {}))
    if raw_manifest.get("shard") != assignment:
        raise ValueError("Raw dataset shard assignment differs from its effective config")
    expected_episode_count = sum(raw_manifest.get("counts", {}).values())
    if (
        raw_manifest.get("episode_count") != expected_episode_count
        or expected_episode_count < 1
    ):
        raise ValueError("Raw Surface-Mamba manifest episode count is incomplete")
    output = Path(config["output"]).resolve()
    effective = {k: v for k, v in config.items() if k != "trainer"}
    encoded = output / "episodes"
    if output.exists():
        if not config.get("resume") or (output / "dataset_manifest.json").exists():
            raise FileExistsError(f"Derived dataset output already exists: {output}")
        if json.loads((output / "effective_config.json").read_text()) != effective:
            raise ValueError("Resume requires an identical effective export config")
        for partial in encoded.glob("*.partial"):
            partial.rename(partial.with_suffix(f".abandoned-{int(time.time())}"))
    else:
        encoded.mkdir(parents=True)
        write_json(output / "effective_config.json", effective)
    frontend = SurfaceContextFrontend(config["frontend"])
    records = raw_manifest["episodes"]
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_state36_dataset",
        job_type="collection",
        seed=0,
        config=config,
    ) as logger:
        encoded_rows = []
        for index, record in enumerate(records, 1):
            destination = encoded / f"{record['seed']}.npz"
            if destination.exists():
                _validate_encoded_episode(destination, record)
            else:
                arrays = _encode_episode(raw_root, record, frontend)
                _write_npz(destination, **arrays)
            encoded_rows.append(
                {
                    "seed": record["seed"],
                    "split": record["split"],
                    "dart": record["dart"],
                    "steps": record["steps"],
                    "file": destination.name,
                    "sha256": file_sha256(destination),
                }
            )
            write_json(
                output / "progress.json",
                {"encoded": index, "total": len(records), "last_seed": record["seed"]},
            )
            print(f"Encoded Surface dataset {index}/{len(records)}", flush=True)
        splits = {}
        for split in ("train", "validation", "test"):
            selected = [row for row in records if row["split"] == split]
            splits[split] = _pack_split(output / f"{split}.npz", selected, encoded)
        expected = {
            split: sum(1 for row in records if row["split"] == split)
            for split in ("train", "validation", "test")
        }
        if {key: value["episodes"] for key, value in splits.items()} != expected:
            raise ValueError("Derived dataset split counts differ from the raw shard")
        manifest_path = output / "dataset_manifest.json"
        write_json(
            manifest_path,
            {
                "format": (
                    DERIVED_FORMAT
                    if raw_format == DATASET_FORMAT
                    else SHARDED_DERIVED_FORMAT
                ),
                "run_id": logger.run_id,
                "shard": assignment,
                "raw_manifest_sha256": file_sha256(raw_manifest_path),
                "raw_run_id": raw_manifest["run_id"],
                "frontend": config["frontend"],
                "splits": splits,
                "episodes": encoded_rows,
                "state36_layout": "Robot15 followed by Object21",
                "controller_input": "44D rotation6 plus Hole-frame tip/relative-rotation features",
            },
        )
        logger.link_output(manifest_path, kind="episode_metadata")
        for split in splits.values():
            if split["file"] is not None:
                logger.link_output(output / split["file"], kind="trajectory")
        logger.log(
            {
                "episodes": len(records),
                "rows": sum(split["rows"] for split in splits.values()),
            }
        )


def _validate_raw_shard_files(root: Path, manifest: dict[str, Any]) -> None:
    """Recheck every approved raw frame before packaging a shard."""
    for record in manifest["episodes"]:
        directory = root / record["path"]
        frames = sorted(directory.glob("frame_*.npz"))
        expected_hashes = record.get("frame_sha256", [])
        if len(frames) != record["observations"] or len(frames) != len(expected_hashes):
            raise ValueError(f"Raw shard frame inventory differs: {directory}")
        for frame, expected_hash in zip(frames, expected_hashes, strict=True):
            if file_sha256(frame) != expected_hash:
                raise ValueError(f"Raw shard frame checksum differs: {frame}")


def _validate_derived_shard_files(root: Path, manifest: dict[str, Any]) -> None:
    """Recheck encoded episodes and packed split files before packaging."""
    for record in manifest["episodes"]:
        path = root / "episodes" / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Derived shard episode checksum differs: {path}")
    for split in manifest["splits"].values():
        if split["file"] is None:
            continue
        path = root / split["file"]
        if file_sha256(path) != split["sha256"]:
            raise ValueError(f"Derived shard split checksum differs: {path}")


def _add_file(bundle: tarfile.TarFile, source: Path, archive_path: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Shard package input must be a regular file: {source}")
    bundle.add(source, arcname=archive_path, recursive=False)


def _write_raw_shard_archive(
    destination: Path,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    """Archive only approved raw episodes and their audit ledgers."""
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with tarfile.open(temporary, "w") as bundle:
        for name in ("effective_config.json", "source.json", "dataset_manifest.json"):
            _add_file(bundle, root / name, f"raw/{name}")
        for bucket_name in EXPECTED_BUCKETS:
            bucket_root = root / "buckets" / bucket_name
            for name in ("manifest.json", "attempts.jsonl"):
                _add_file(
                    bundle,
                    bucket_root / name,
                    f"raw/buckets/{bucket_name}/{name}",
                )
        for record in manifest["episodes"]:
            episode_root = root / record["path"]
            for path in sorted(episode_root.iterdir()):
                _add_file(bundle, path, f"raw/{record['path']}/{path.name}")
    temporary.replace(destination)


def _write_derived_shard_archive(
    destination: Path,
    root: Path,
    manifest: dict[str, Any],
) -> None:
    """Archive encoded episodes, packed splits and their completion manifest."""
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with tarfile.open(temporary, "w") as bundle:
        for name in ("effective_config.json", "progress.json", "dataset_manifest.json"):
            _add_file(bundle, root / name, f"state36/{name}")
        for record in manifest["episodes"]:
            _add_file(
                bundle,
                root / "episodes" / record["file"],
                f"state36/episodes/{record['file']}",
            )
        for split in manifest["splits"].values():
            if split["file"] is not None:
                _add_file(bundle, root / split["file"], f"state36/{split['file']}")
    temporary.replace(destination)


def _publish_shard_package(
    config: dict[str, Any],
    experiments_root: Path,
    run_id: str,
    output: Path,
) -> None:
    """Publish one finished package through the shared immutable publisher."""
    from utils.experiment_artifacts import (
        DEFAULT_REPOSITORIES,
        Repository,
        publish,
    )

    repositories = dict(DEFAULT_REPOSITORIES)
    repositories["rollouts"] = Repository(
        config.get("repository", "Graph-Recurrent-VLA/graph-recurrent-rollouts"),
        "dataset",
    )
    report = publish(
        experiments_root / run_id,
        ["trajectory"],
        repositories=repositories,
        retries=5,
    )
    write_json(output / "publication_report.json", report)
    if report.get("status") != "complete":
        raise RuntimeError("Hugging Face shard publication did not complete")


def package_shard(config: dict[str, Any]) -> None:
    """Verify, archive and optionally publish one completed raw/cache shard."""
    raw_root = Path(config["raw_root"]).resolve()
    derived_root = Path(config["derived_root"]).resolve()
    raw_manifest_path = raw_root / "dataset_manifest.json"
    derived_manifest_path = derived_root / "dataset_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    derived_manifest = json.loads(derived_manifest_path.read_text())
    if raw_manifest.get("format") != SHARDED_DATASET_FORMAT:
        raise ValueError("Shard packaging requires a distributed raw dataset shard")
    if derived_manifest.get("format") != SHARDED_DERIVED_FORMAT:
        raise ValueError("Shard packaging requires a distributed State36 shard")
    if raw_manifest.get("shard") != derived_manifest.get("shard"):
        raise ValueError("Raw and derived shard assignments differ")
    if derived_manifest.get("raw_manifest_sha256") != file_sha256(raw_manifest_path):
        raise ValueError("Derived shard was exported from a different raw manifest")
    raw_seeds = [row["seed"] for row in raw_manifest["episodes"]]
    derived_seeds = [row["seed"] for row in derived_manifest["episodes"]]
    if raw_seeds != derived_seeds:
        raise ValueError("Raw and derived shard episode order differs")

    _validate_raw_shard_files(raw_root, raw_manifest)
    _validate_derived_shard_files(derived_root, derived_manifest)

    output = Path(config["output"]).resolve()
    shard_index = raw_manifest["shard"]["index"]
    if output.exists():
        if not config.get("resume", False):
            raise FileExistsError(f"Shard package output already exists: {output}")
        index_path = output / "shard_package.json"
        request_path = output / "publication_request.json"
        if not index_path.is_file() or not request_path.is_file():
            raise ValueError("Existing shard package is incomplete; preserve it and use a new output")
        existing = json.loads(index_path.read_text())
        request = json.loads(request_path.read_text())
        if (
            existing.get("format") != SHARD_PACKAGE_FORMAT
            or existing.get("shard") != raw_manifest["shard"]
            or existing.get("raw_manifest_sha256") != file_sha256(raw_manifest_path)
            or existing.get("derived_manifest_sha256") != file_sha256(derived_manifest_path)
        ):
            raise ValueError("Existing shard package was built from different inputs")
        for archive in existing["archives"].values():
            path = output / archive["file"]
            if path.stat().st_size != archive["size"] or file_sha256(path) != archive["sha256"]:
                raise ValueError(f"Existing shard package archive differs: {path}")
        experiments_root = Path(request["experiments_dir"]).resolve()
        if config.get("publish", False):
            _publish_shard_package(
                config, experiments_root, request["run_id"], output
            )
        print(f"Reused Surface-Mamba shard package at {output}", flush=True)
        return

    output.mkdir(parents=True)
    raw_archive = output / f"raw_shard_{shard_index:03d}.tar"
    derived_archive = output / f"state36_shard_{shard_index:03d}.tar"
    experiments_root = Path(config.get("experiments_dir", output / "experiments")).resolve()

    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_dataset_shard",
        job_type="collection",
        seed=shard_index,
        config={key: value for key, value in config.items() if key != "trainer"},
        root=experiments_root,
    ) as logger:
        _write_raw_shard_archive(raw_archive, raw_root, raw_manifest)
        _write_derived_shard_archive(derived_archive, derived_root, derived_manifest)
        package_index = {
            "format": SHARD_PACKAGE_FORMAT,
            "run_id": logger.run_id,
            "shard": raw_manifest["shard"],
            "counts": raw_manifest["counts"],
            "global_counts": raw_manifest["global_counts"],
            "global_buckets": raw_manifest["global_buckets"],
            "bucket_assignments": raw_manifest["bucket_assignments"],
            "episodes": raw_manifest["episodes"],
            "raw_manifest_sha256": file_sha256(raw_manifest_path),
            "derived_manifest_sha256": file_sha256(derived_manifest_path),
            "archives": {
                "raw": {
                    "file": raw_archive.name,
                    "size": raw_archive.stat().st_size,
                    "sha256": file_sha256(raw_archive),
                },
                "state36": {
                    "file": derived_archive.name,
                    "size": derived_archive.stat().st_size,
                    "sha256": file_sha256(derived_archive),
                },
            },
        }
        index_path = output / "shard_package.json"
        write_json(index_path, package_index)
        logger.link_output(raw_archive, kind="trajectory")
        logger.link_output(derived_archive, kind="trajectory")
        logger.link_output(index_path, kind="episode_metadata")
        logger.log(
            {
                "episodes": len(raw_seeds),
                "raw_bytes": raw_archive.stat().st_size,
                "derived_bytes": derived_archive.stat().st_size,
            }
        )

    write_json(
        output / "publication_request.json",
        {
            "run_id": logger.run_id,
            "experiments_dir": str(experiments_root),
            "repository": config.get(
                "repository", "Graph-Recurrent-VLA/graph-recurrent-rollouts"
            ),
        },
    )
    if config.get("publish", False):
        _publish_shard_package(config, experiments_root, logger.run_id, output)


def merge_shard_packages(config: dict[str, Any]) -> None:
    """Validate every shard archive and produce one logical 900-episode index."""
    index_paths = [Path(path).resolve() for path in config["shard_package_indexes"]]
    if not index_paths:
        raise ValueError("shard_package_indexes must contain every shard package index")
    packages = [json.loads(path.read_text()) for path in index_paths]
    if any(package.get("format") != SHARD_PACKAGE_FORMAT for package in packages):
        raise ValueError("All inputs must be Surface-Mamba shard package indexes")
    shard_counts = {package["shard"]["count"] for package in packages}
    if len(shard_counts) != 1:
        raise ValueError("Shard packages disagree on the global shard count")
    shard_count = shard_counts.pop()
    by_index = {
        package["shard"]["index"]: (path, package)
        for path, package in zip(index_paths, packages, strict=True)
    }
    if len(by_index) != len(packages) or set(by_index) != set(range(shard_count)):
        raise ValueError("Shard package indexes must cover every index exactly once")

    episodes = []
    package_rows = []
    expected_global_counts = {
        name: expected[2] for name, expected in EXPECTED_BUCKETS.items()
    }
    global_buckets = packages[0].get("global_buckets")
    validated_global_buckets = _validate_buckets({"buckets": global_buckets})
    for shard_index in range(shard_count):
        index_path, package = by_index[shard_index]
        if (
            package.get("global_counts") != expected_global_counts
            or package.get("global_buckets") != global_buckets
        ):
            raise ValueError("Shard packages disagree on global bucket settings")
        expected_assignments = _partition_buckets(
            validated_global_buckets,
            {"index": shard_index, "count": shard_count},
        )
        if package.get("bucket_assignments") != expected_assignments:
            raise ValueError("Shard package bucket assignment is not deterministic")
        expected_local_counts = {
            bucket["name"]: bucket["count"] for bucket in expected_assignments
        }
        if package.get("counts") != expected_local_counts:
            raise ValueError("Shard package quota counts differ from its assignment")
        by_bucket = {bucket["name"]: bucket for bucket in expected_assignments}
        observed_local_counts = {name: 0 for name in EXPECTED_BUCKETS}
        for episode in package["episodes"]:
            matches = [
                name
                for name, (split, dart, _) in EXPECTED_BUCKETS.items()
                if episode["split"] == split and episode["dart"] == dart
            ]
            if len(matches) != 1:
                raise ValueError("Shard episode does not identify one fixed bucket")
            bucket_name = matches[0]
            bucket = by_bucket[bucket_name]
            if not bucket["start_seed"] <= episode["seed"] < (
                bucket["start_seed"] + bucket["max_attempts"]
            ):
                raise ValueError("Shard episode seed falls outside its assigned range")
            observed_local_counts[bucket_name] += 1
        if observed_local_counts != expected_local_counts:
            raise ValueError("Shard episode rows do not satisfy its local quotas")
        for archive in package["archives"].values():
            archive_path = index_path.parent / archive["file"]
            if (
                archive_path.stat().st_size != archive["size"]
                or file_sha256(archive_path) != archive["sha256"]
            ):
                raise ValueError(f"Shard archive checksum differs: {archive_path}")
        package_rows.append(
            {
                "shard": package["shard"],
                "run_id": package["run_id"],
                "index_sha256": file_sha256(index_path),
                "archives": package["archives"],
            }
        )
        episodes.extend(
            dict(row, shard_index=shard_index) for row in package["episodes"]
        )

    seeds = [row["seed"] for row in episodes]
    counts = {
        name: sum(
            1
            for row in episodes
            if row["split"] == expected[0] and row["dart"] == expected[1]
        )
        for name, expected in EXPECTED_BUCKETS.items()
    }
    expected_counts = expected_global_counts
    if len(episodes) != 900 or len(set(seeds)) != 900 or counts != expected_counts:
        raise ValueError("Merged shard packages do not form the exact 900-episode release")

    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "dataset_manifest.json",
        {
            "format": DATASET_FORMAT,
            "storage": "sharded_raw_and_state36_archives",
            "episode_count": len(episodes),
            "counts": counts,
            "episodes": episodes,
            "packages": package_rows,
        },
    )
