"""Validate and aggregate independently trained paired corner-tracking runs."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracks import FORMAT, read_episode
from evaluation.gnn.temporal.corner_tracking import paired_comparison
from utils.experiment_logger import ExperimentLogger, write


SCENARIOS = ("natural", "masked", "end_stress")
MATCHED_FIELDS = ("model", "gnn", "tracking", "masking", "calibration", "training", "loss",
                  "evaluation", "execution", "device", "implementation_sha256", "dataset_sha256")


def read_json(path):
    """Read required JSON files with an actionable missing-file message."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-run file is missing: {path}")
    return json.loads(path.read_text())


def verify_artifact_owner(path, metadata):
    """Reject missing files or sidecars belonging to a different experiment."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-run artifact is missing: {path}")
    sidecar = read_json(path.with_name(path.name + ".tracking.json"))
    if (sidecar.get("run_id") != metadata["run_id"]
            or Path(sidecar.get("artifact", "")).resolve() != path
            or str(path) not in metadata.get("outputs", [])):
        raise ValueError(f"Artifact owner/registration mismatch: {path}")


def validation_episodes(data_config):
    """Read validation episodes only, retaining the dataset's original order."""
    if data_config.get("kind") == "synthetic":
        from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
        settings = {key: value for key, value in data_config.items() if key != "kind"}
        groups, identity = synthetic_dataset(**settings)
        return groups["validation"], identity["dataset_sha256"]
    if data_config.get("kind") != "prepared" or not data_config.get("root"):
        raise ValueError("data must identify the original prepared dataset or synthetic fixture")
    root = Path(data_config["root"]).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    content = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if (manifest.get("format") != FORMAT or manifest.get("time_unit") != "saved_frame"
            or canonical_hash(content) != manifest.get("dataset_sha256")):
        raise ValueError("Prepared dataset manifest identity/format mismatch")
    seen, parent_splits, episodes = set(), {}, []
    for row in manifest["episodes"]:
        split = row["split"]
        if split not in ("train", "calibration", "validation", "test") or row["episode"] in seen:
            raise ValueError("Prepared dataset has duplicate episodes or unknown splits")
        seen.add(row["episode"])
        if parent_splits.setdefault(row["parent"], split) != split:
            raise ValueError("Prepared dataset parent overlaps multiple splits")
        if split != "validation":
            continue
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Validation episode checksum/path mismatch: {row['episode']}")
        episodes.append(read_episode(path, row))
    if not episodes:
        raise ValueError("Prepared dataset must contain validation episodes")
    return episodes, manifest["dataset_sha256"]


def inspect_run(directory):
    """Reject unfinished, failed, incomplete or internally inconsistent runs."""
    directory = Path(directory).expanduser().resolve()
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    sidecar = read_json(directory / "best.pt.tracking.json")
    metadata = read_json(Path(sidecar["experiment_directory"]) / "metadata.json")
    if metadata.get("status") != "finished" or metadata.get("run_id") != sidecar.get("run_id"):
        raise ValueError(f"Run is unfinished, failed or has inconsistent identity: {directory}")
    if Path(sidecar.get("artifact", "")).resolve() != best_path:
        raise ValueError(f"Checkpoint sidecar does not identify {best_path}")
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(f"Both best.pt and last.pt are required: {directory}")
    for path in (best_path, last_path, directory / "history.json", directory / "training_masks.json"):
        verify_artifact_owner(path, metadata)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    last = torch.load(last_path, map_location="cpu", weights_only=True)
    if best.get("kind") != "corner_tracking" or last.get("kind") != "corner_tracking_last":
        raise ValueError(f"Unexpected checkpoint kinds: {directory}")
    config = best["config"]
    for field in (*MATCHED_FIELDS, "architecture", "seed"):
        if config.get(field) != last["config"].get(field):
            raise ValueError(f"Best and last checkpoints disagree in {field}: {directory}")
    epochs = config["training"]["pretrain_epochs"] + config["training"]["joint_epochs"]
    history = read_json(directory / "history.json")
    ledger = read_json(directory / "training_masks.json")
    if (last["epoch"] != epochs or len(history) != epochs or len(ledger) != epochs
            or [row["epoch"] for row in history] != list(range(1, epochs + 1))
            or [row["epoch"] for row in ledger] != list(range(epochs))
            or canonical_hash(ledger) != canonical_hash(last["mask_ledger"])
            or canonical_hash(history) != canonical_hash(last["history"])):
        raise ValueError(f"Run did not complete its declared epoch/mask ledger: {directory}")
    if not config["training"]["pretrain_epochs"] < best["epoch"] <= epochs:
        raise ValueError(f"Best checkpoint is outside joint training: {directory}")
    joint_scores = [row["validation"]["object21_loss"] for row in history
                    if row["phase"] == "joint"]
    if (not joint_scores or not np.isfinite(joint_scores).all()
            or best["validation_score"] != min(joint_scores)
            or best["epoch"] != last["best_epoch"] or best["validation_score"] != last["best_score"]):
        raise ValueError(f"Best checkpoint disagrees with joint validation history: {directory}")
    for name in ("motion", "gnn"):
        if (best[name].keys() != last["best_state"][name].keys()
                or any(not torch.equal(value, last["best_state"][name][key])
                       for key, value in best[name].items())):
            raise ValueError(f"Best {name} weights disagree with last checkpoint: {directory}")
    initialization = best["shared_initialization_sha256"]
    if (initialization != last["shared_initialization_sha256"]
            or initialization != metadata.get("shared_initialization_sha256")
            or metadata.get("dataset_sha256") != config["dataset_sha256"]):
        raise ValueError(f"Logger and checkpoint initialization/dataset identity disagree: {directory}")
    data_identity = read_json(directory / "data_identity.json")
    if data_identity.get("dataset_sha256") != config["dataset_sha256"]:
        raise ValueError(f"Recorded dataset identity differs from checkpoint: {directory}")
    summaries, mask_hashes = {}, {}
    for scenario in SCENARIOS:
        folder = directory / scenario
        for name in ("summary.json", "mask_manifest.json", "tracking.json", "predictions.npz"):
            verify_artifact_owner(folder / name, metadata)
        summaries[scenario] = read_json(folder / "summary.json")
        masks = read_json(folder / "mask_manifest.json")
        mask_hashes[scenario] = canonical_hash(masks)
        read_json(folder / "tracking.json")
        if not (folder / "predictions.npz").is_file():
            raise FileNotFoundError(f"Missing completed {scenario} predictions: {folder}")
    if config["evaluation"]["ablations"]:
        for name in ("current_only", "hold", "constant_velocity"):
            folder = directory / f"ablation_{name}"
            for filename in ("summary.json", "mask_manifest.json", "tracking.json", "predictions.npz"):
                verify_artifact_owner(folder / filename, metadata)
            summaries[f"ablation_{name}"] = read_json(folder / "summary.json")
            read_json(folder / "mask_manifest.json")
            read_json(folder / "tracking.json")
            if not (folder / "predictions.npz").is_file():
                raise FileNotFoundError(f"Missing completed ablation predictions: {folder}")
    row = dict(architecture=config["architecture"], seed=config["seed"], run_id=metadata["run_id"],
               output=str(directory), checkpoint=str(best_path), checkpoint_sha256=file_sha256(best_path),
               best_epoch=best["epoch"], shared_initialization_sha256=initialization,
               training_mask_sha256=canonical_hash(ledger), validation_mask_sha256=mask_hashes["masked"],
               scenario_mask_sha256=mask_hashes, evaluation=summaries)
    return row, config


def check_paired_runs(rows, configs, seeds):
    """Enforce an exact architecture×seed matrix and shared protocol."""
    expected = {(architecture, seed) for seed in seeds for architecture in ("mamba", "transformer")}
    actual = [(row["architecture"], row["seed"]) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Expected exactly the Mamba/Transformer × {seeds} matrix; got {actual}")
    reference = configs[0]
    for config in configs[1:]:
        for key in MATCHED_FIELDS:
            if config.get(key) != reference.get(key):
                raise ValueError(f"Matched run configurations differ in {key}")
    for seed in seeds:
        pair = [row for row in rows if row["seed"] == seed]
        for key in ("shared_initialization_sha256", "training_mask_sha256", "validation_mask_sha256",
                    "scenario_mask_sha256"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Seed {seed} paired runs differ in {key}")


def verify_predictions(rows, episodes):
    """Check frame order and targets against the held-out dataset, not just peers."""
    expected_ids = np.asarray([frame.sample_id for episode in episodes for frame in episode.frames])
    expected_targets = np.stack([frame.target_object21 for episode in episodes for frame in episode.frames])
    for row in rows:
        for scenario in SCENARIOS:
            path = Path(row["output"]) / scenario / "predictions.npz"
            with np.load(path, allow_pickle=False) as arrays:
                if not np.array_equal(arrays["sample_ids"], expected_ids):
                    raise ValueError(f"{scenario} frame IDs/order disagree with validation dataset: {path}")
                if not np.array_equal(arrays["target"], expected_targets):
                    raise ValueError(f"{scenario} targets disagree with validation dataset: {path}")
                for name, width in (("prediction", 21), ("endpoint_geom", 3), ("endpoint_residual", 3)):
                    if arrays[name].shape != (len(expected_ids), width) or not np.isfinite(arrays[name]).all():
                        raise ValueError(f"{scenario} {name} is nonfinite or has invalid shape: {path}")


def aggregate(config):
    """Root-main entrypoint for a complete, independently launched seed matrix."""
    paths = config.get("runs")
    seeds = config.get("seeds", [0, 1, 2])
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("runs must be a nonempty list of distinct completed run directories")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct seed integers")
    inspected = [inspect_run(path) for path in paths]
    rows, configs = list(map(list, zip(*inspected)))
    check_paired_runs(rows, configs, seeds)
    data_config = dict(configs[0]["data"])
    data_config.update(config.get("data", {}))
    episodes, dataset_sha256 = validation_episodes(data_config)
    if dataset_sha256 != configs[0]["dataset_sha256"]:
        raise ValueError("Validation dataset identity differs from the trained runs")
    verify_predictions(rows, episodes)
    summary = paired_comparison(rows, episodes)
    with ExperimentLogger(team="graph", config=config, model="corner_tracking_comparison",
                          job_type="comparison") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        report = dict(dataset_sha256=dataset_sha256, runs=rows, aggregate=summary)
        write(output / "comparison.json", report)
        logger.record(dataset_sha256=dataset_sha256, source_run_ids=[row["run_id"] for row in rows])
        logger.link_output(output / "comparison.json")
        print(f"Matched corner tracking comparison: {output / 'comparison.json'}", flush=True)
        return str(output / "comparison.json")
