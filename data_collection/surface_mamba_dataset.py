"""Collect DART expert rollouts and export frozen Surface-Graph State36 pairs."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
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
DERIVED_FORMAT = "surface_mamba_state36_v1"
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
    buckets = _validate_buckets(collection)
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
            if completed.get("format") != DATASET_FORMAT or completed.get("episode_count") != 900:
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
        if len(episodes) != 900 or len(set(seeds)) != 900 or set(seeds) & excluded:
            raise ValueError("Final raw dataset count, uniqueness or exclusion check failed")
        manifest_path = output / "dataset_manifest.json"
        write_json(
            manifest_path,
            {
                "format": DATASET_FORMAT,
                "run_id": logger.run_id,
                "episodes": episodes,
                "episode_count": len(episodes),
                "counts": {name: expected[2] for name, expected in EXPECTED_BUCKETS.items()},
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
    if raw_manifest.get("format") != DATASET_FORMAT or raw_manifest.get("episode_count") != 900:
        raise ValueError("Expected the completed 900-episode Surface-Mamba raw dataset")
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
        expected = {"train": 720, "validation": 90, "test": 90}
        if {key: value["episodes"] for key, value in splits.items()} != expected:
            raise ValueError("Derived dataset split counts differ from 720/90/90")
        manifest_path = output / "dataset_manifest.json"
        write_json(
            manifest_path,
            {
                "format": DERIVED_FORMAT,
                "run_id": logger.run_id,
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
            logger.link_output(output / split["file"], kind="trajectory")
        logger.log(
            {
                "episodes": len(records),
                "rows": sum(split["rows"] for split in splits.values()),
            }
        )
