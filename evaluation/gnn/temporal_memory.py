"""Compare a frozen Object21 GNN with causal, GT-free hole-position memories."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from graph.build_graph import NodeType
from graph.utils.hole_memory import (
    HolePositionMemory, MemorySettings, ObservationQuality, replace_hole_position,
)
from training.gnn.cached_data import build_data
from training.gnn.trainer import build_model, resolve_config, seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def resolve_evaluation_config(config):
    """Fix the protocol before inference; no threshold search or test split access."""
    effective = deepcopy(config)
    effective.setdefault("device", "auto")
    settings = effective.setdefault("evaluation", {})
    for name, default in (("split", "validation"), ("batch_size", 64),
                          ("visibility_labels", None), ("bad_error_mm", 10.0),
                          ("stationary_tolerance_m", 0.0001)):
        settings.setdefault(name, default)
    if settings["split"] != "validation":
        raise ValueError("Temporal memory currently evaluates the published validation split only")
    if type(settings["batch_size"]) is not int or settings["batch_size"] < 1:
        raise ValueError("evaluation.batch_size must be an integer >= 1")
    for name in ("bad_error_mm", "stationary_tolerance_m"):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"evaluation.{name} must be finite and positive")
    source = Path(effective["checkpoint"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen best.pt: {source}")
    effective["checkpoint"] = str(source)
    cache = effective.get("data", {}).get("cache_dir")
    if not isinstance(cache, str) or not cache:
        raise ValueError("data.cache_dir must point to the extracted graph cache")
    effective["data"] = {"cache_dir": str(Path(cache).expanduser().resolve())}
    if settings["visibility_labels"] is not None:
        labels = Path(settings["visibility_labels"]).expanduser().resolve()
        if not labels.is_file():
            raise FileNotFoundError(f"Missing diagnostic visibility labels: {labels}; use null to omit this group")
        settings["visibility_labels"] = str(labels)
    variants = effective.get("memory")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("memory must contain named MemorySettings mappings")
    for name, parameters in variants.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "gnn":
            raise ValueError("Memory names must be lowercase snake_case and cannot be gnn")
        variants[name] = asdict(MemorySettings(**parameters))
    root = effective.setdefault("output", {}).setdefault("root", "outputs/temporal_memory")
    effective["output"]["root"] = str(Path(root).expanduser().resolve())
    return effective


def sample_order(sample_id):
    """For shard_000001/episode_000002/frame_000003.npz return episode and step 3."""
    path = PurePosixPath(str(sample_id))
    match = re.fullmatch(r"frame_(\d+)\.npz", path.name)
    if not match or len(path.parts) < 2:
        raise ValueError(f"Expected episode/frame_000000.npz sample ID, got {sample_id!r}")
    return str(path.parent), int(match[1])


def infer_validation(model, loader, device):
    """Read validated caches once and capture observation-only gate features."""
    collected = {name: [] for name in ("sample_ids", "prediction", "target", "tcp_world", "quality")}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            graph = batch.graph
            target_nodes = graph.node_valid & (
                (graph.object_type == NodeType.BOX) | (graph.object_type == NodeType.HOLE)
            )
            count = target_nodes.sum(1)
            known = target_nodes & graph.confidence_valid
            known_count = known.sum(1)
            mean_confidence = (graph.confidence * known).sum(1) / known_count.clamp_min(1)
            coverage = known_count / count.clamp_min(1)
            quality = torch.stack((count, mean_confidence, coverage), dim=1)
            collected["sample_ids"].extend(batch.sample_ids)
            collected["prediction"].append(model(graph.to(device))["object21"].cpu().numpy())
            collected["target"].append(batch.targets["object21"].numpy())
            collected["tcp_world"].append(graph.tcp_xyz_world.numpy())
            collected["quality"].append(quality.numpy())
            if batch_index % 80 == 0:
                print(f"Validation inference: {len(collected['sample_ids'])} frames", flush=True)
    if not collected["sample_ids"]:
        raise ValueError("Validation data is empty")
    if len(set(collected["sample_ids"])) != len(collected["sample_ids"]):
        raise ValueError("Duplicate validation sample IDs")
    order = sorted(range(len(collected["sample_ids"])),
                   key=lambda index: sample_order(collected["sample_ids"][index]))
    result = {name: np.concatenate(values)[order] for name, values in collected.items()
              if name != "sample_ids"}
    result["sample_ids"] = np.asarray(collected["sample_ids"])[order]
    result["episode"] = np.array([sample_order(value)[0] for value in result["sample_ids"]])
    result["frame"] = np.array([sample_order(value)[1] for value in result["sample_ids"]])
    if not all(np.isfinite(result[name]).all() for name in ("prediction", "target", "tcp_world", "quality")):
        raise ValueError("Validation predictions, targets and observation features must be finite")
    return result


def replay_memory(prediction, tcp_world, quality, episodes, frames, settings):
    """Replay chronologically without accepting any GT or diagnostic labels."""
    corrected = []
    flags = {name: [] for name in ("initialized", "updated", "held", "quality_accepted", "age_frames")}
    previous_episode = None
    for index, episode in enumerate(episodes):
        if episode != previous_episode:
            memory = HolePositionMemory(settings)
            previous_episode = episode
        count, confidence, coverage = quality[index]
        observation = ObservationQuality(int(count), float(confidence), float(coverage))
        estimate = memory.step(int(frames[index]), prediction[index, 10:13],
                               tcp_world[index], observation)
        corrected.append(replace_hole_position(prediction[index], estimate.hole_relative))
        for name in flags:
            flags[name].append(getattr(estimate, name))
    return {"object21": np.asarray(corrected), **{name: np.asarray(values) for name, values in flags.items()}}


def error_summary(values):
    """Return frame-weighted distances; absent groups have explicit null statistics."""
    values = np.asarray(values)
    if not len(values):
        return {"frames": 0, "mean_mm": None, "median_mm": None, "p90_mm": None}
    return {"frames": len(values), "mean_mm": float(values.mean()),
            "median_mm": float(np.median(values)), "p90_mm": float(np.percentile(values, 90))}


def scoring_groups(data, labels_path):
    """Read GT-derived occlusion diagnostics only after all replay outputs exist."""
    groups = {"all": np.ones(len(data["sample_ids"]), dtype=bool),
              "fewer_than_three_target_nodes": data["quality"][:, 0] < 3}
    if labels_path is None:
        return groups
    with np.load(labels_path, allow_pickle=False) as labels:
        identifiers = labels["sample_ids"].tolist()
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(data["sample_ids"]):
            raise ValueError("Visibility labels must list exactly the same validation sample IDs")
        mapping = {name: index for index, name in enumerate(identifiers)}
        order = [mapping[str(name)] for name in data["sample_ids"]]
        front = labels["front"][order]
        distances = labels["min_distance_m"][order]
        foreground = labels["foreground"][order]
        expected = (len(order), 8)
        if (front.shape != expected or distances.shape != expected or foreground.shape != expected
                or front.dtype != bool or foreground.dtype != bool
                or not np.all(front.sum(1) == 4) or np.isnan(distances).any()
                or np.any(distances < 0)):
            raise ValueError("Malformed visibility diagnostics: expected four front corners per frame")
        supported = (distances <= 0.006) & front
        occluded = foreground & front & ~supported
        groups["strong_occlusion_proxy"] = (occluded.sum(1) >= 3) & (supported.sum(1) < 2)
    return groups


def lifecycle_metrics(data, result, baseline_error, memory_error, threshold):
    """Audit cold starts, erroneous initializations, persistence, and reacquisition."""
    initial_errors = []
    initial_delays = []
    bad_initial_recovery = []
    bad_initial_unrecovered = 0
    uninitialized_episodes = 0
    reacquisition = []
    for episode in np.unique(data["episode"]):
        indices = np.flatnonzero(data["episode"] == episode)
        updates = indices[result["updated"][indices]]
        if not len(updates):
            uninitialized_episodes += 1
        else:
            first = updates[0]
            initial_errors.append(memory_error[first])
            initial_delays.append(int(data["frame"][first] - data["frame"][indices[0]]))
            if memory_error[first] > threshold:
                recovery = updates[memory_error[updates] <= threshold]
                if len(recovery):
                    bad_initial_recovery.append(int(data["frame"][recovery[0]] - data["frame"][first]))
                else:
                    bad_initial_unrecovered += 1
        quality = result["quality_accepted"][indices]
        starts = np.flatnonzero(quality[1:] & ~quality[:-1]) + 1
        for start in starts:
            before = indices[start - 1]
            if not result["initialized"][before]:
                continue
            stop = start
            while stop < len(indices) and quality[stop]:
                stop += 1
            run = indices[start:stop]
            run_updates = run[result["updated"][run]]
            first_update = int(run_updates[0]) if len(run_updates) else None
            reacquisition.append({
                "episode": str(episode), "return_frame": int(data["frame"][indices[start]]),
                "before_baseline_mm": float(baseline_error[before]),
                "before_memory_mm": float(memory_error[before]),
                "return_memory_mm": float(memory_error[indices[start]]),
                "latency_frames": int(data["frame"][first_update] - data["frame"][indices[start]])
                if first_update is not None else None,
                "after_memory_mm": float(memory_error[first_update]) if first_update is not None else None,
            })
    updated_events = [event for event in reacquisition if event["latency_frames"] is not None]
    initially_bad = [event for event in updated_events if event["before_memory_mm"] > threshold]
    held = result["held"]
    summary = {
        "initialized_episodes": len(initial_errors), "uninitialized_episodes": uninitialized_episodes,
        "initial_memory_error": error_summary(initial_errors),
        "initial_updates_over_threshold": int(np.sum(np.asarray(initial_errors) > threshold)),
        "initial_update_delay_mean_frames": float(np.mean(initial_delays)) if initial_delays else None,
        "initially_bad_recovered_episodes": len(bad_initial_recovery),
        "initially_bad_unrecovered_episodes": bad_initial_unrecovered,
        "bad_initial_recovery_mean_frames": float(np.mean(bad_initial_recovery)) if bad_initial_recovery else None,
        "held_error": error_summary(memory_error[held]),
        "held_frames_over_threshold": int(np.sum(held & (memory_error > threshold))),
        "cold_start_error": error_summary(memory_error[~result["initialized"]]),
        "reacquisition_events": len(reacquisition), "reacquisition_updated_events": len(updated_events),
        "reacquisition_without_update_events": len(reacquisition) - len(updated_events),
        "reacquisition_latency_mean_frames": float(np.mean([e["latency_frames"] for e in updated_events]))
        if updated_events else None,
        "reacquisition_before_memory_error": error_summary([e["before_memory_mm"] for e in reacquisition]),
        "reacquisition_after_memory_error": error_summary([e["after_memory_mm"] for e in updated_events]),
        "bad_before_reacquisition_events": len(initially_bad),
        "bad_before_reacquisition_fixed_events": sum(e["after_memory_mm"] <= threshold for e in initially_bad),
    }
    return summary, reacquisition


def evaluate(config):
    """Use the checkpoint's exact model/data settings, with no optimizer or cache writes."""
    effective = resolve_evaluation_config(config)
    checkpoint_path = Path(effective["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "best_weights":
        raise ValueError("checkpoint must be a best_weights snapshot (best.pt)")
    training_config = resolve_config(checkpoint["config"])
    if training_config["data"]["factory"] != "training.gnn.cached_data:build_data":
        raise ValueError("Temporal replay requires a checkpoint trained with the published cache-only factory")
    training_config["data"]["cache_dir"] = effective["data"]["cache_dir"]
    training_config["training"]["batch_size"] = effective["evaluation"]["batch_size"]
    if training_config["data"]["empty_inputs"] != "keep":
        raise ValueError("Temporal evaluation requires empty_inputs=keep to retain sequence frames")
    effective["source_training_config"] = training_config
    device = set_device(effective["device"])
    seed = training_config["seed"]
    seed_everything(seed, device)
    with ExperimentLogger(team="graph", config=effective, seed=seed,
                          model=training_config["model"]["name"], job_type="evaluation") as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(checkpoint_path, checkpoint=True)
        write_json(output / "effective_config.json", effective)
        bundle = build_data(training_config, torch.Generator().manual_seed(seed))
        if checkpoint["dataset_sha256"] != bundle.identity["dataset_sha256"]:
            raise ValueError("Checkpoint and validation cache dataset identities differ")
        model = build_model(training_config["model"], device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        model.requires_grad_(False)
        data = infer_validation(model, bundle.validation, device)
        variants = {
            name: replay_memory(data["prediction"], data["tcp_world"], data["quality"],
                                data["episode"], data["frame"], MemorySettings(**settings))
            for name, settings in effective["memory"].items()
        }
        # GT is used below for scoring only. It never enters replay_memory.
        settings = effective["evaluation"]
        max_drift = 0.0
        hole_world = data["target"][:, 10:13] + data["tcp_world"]
        for episode in np.unique(data["episode"]):
            positions = hole_world[data["episode"] == episode]
            max_drift = max(max_drift, float(np.linalg.norm(positions - positions[0], axis=1).max()))
        if max_drift > settings["stationary_tolerance_m"]:
            raise ValueError(f"Stationary-hole assumption failed: GT drift {max_drift:.6f} m")
        groups = scoring_groups(data, settings["visibility_labels"])
        predictions = {"gnn": data["prediction"], **{name: result["object21"] for name, result in variants.items()}}
        summary = {"frames": len(data["sample_ids"]), "episodes": len(set(data["episode"])),
                   "stationary_hole_max_gt_drift_m": max_drift, "split": "validation", "methods": {}}
        error = {name: np.linalg.norm(value[:, 10:13] - data["target"][:, 10:13], axis=1) * 1000
                 for name, value in predictions.items()}
        events = {}
        for name, prediction in predictions.items():
            method = {"groups": {group: error_summary(error[name][mask]) for group, mask in groups.items()}}
            endpoint = np.linalg.norm(prediction[:, 7:10] - data["target"][:, 7:10], axis=1) * 1000
            method["endpoint_error"] = error_summary(endpoint)
            episode_means = []
            improved_episodes = 0
            for episode in np.unique(data["episode"]):
                mask = data["episode"] == episode
                episode_means.append(float(error[name][mask].mean()))
                improved_episodes += error[name][mask].mean() < error["gnn"][mask].mean()
            method["episode_balanced_mean_mm"] = float(np.mean(episode_means))
            method["episodes_improved_over_gnn"] = int(improved_episodes)
            if name in variants:
                method["lifecycle"], events[name] = lifecycle_metrics(
                    data, variants[name], error["gnn"], error[name], settings["bad_error_mm"],
                )
                method["updated_frames"] = int(variants[name]["updated"].sum())
                method["held_frames"] = int(variants[name]["held"].sum())
            summary["methods"][name] = method
        arrays = dict(data)
        for name, result in variants.items():
            arrays.update({f"{name}_{key}": value for key, value in result.items()})
        np.savez_compressed(output / "predictions.npz", **arrays)
        write_json(output / "summary.json", summary)
        write_json(output / "reacquisition_events.json", events)
        source_files = [Path(__file__), Path(__file__).parents[2] / "graph/utils/hole_memory.py"]
        provenance = {"checkpoint_sha256": file_sha256(checkpoint_path),
                      "source_sha256": {str(path.relative_to(Path(__file__).parents[2])): file_sha256(path)
                                        for path in source_files},
                      "visibility_sha256": file_sha256(Path(settings["visibility_labels"]))
                      if settings["visibility_labels"] else None}
        write_json(output / "provenance.json", provenance)
        experiment.record(device=str(device), model_compilation=False,
                          dataset_sha256=bundle.identity["dataset_sha256"],
                          output_directory=str(output), checkpoint_sha256=provenance["checkpoint_sha256"],
                          memory_uses_gt=False, test_evaluated=False)
        for name, method in summary["methods"].items():
            experiment.log({"groups": method["groups"], "episode_balanced_mean_mm": method["episode_balanced_mean_mm"]}, prefix=name)
        for path in output.iterdir():
            experiment.link_output(path, kind="metrics")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Temporal memory output: {output}", flush=True)
        return output
