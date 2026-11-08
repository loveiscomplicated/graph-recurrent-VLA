"""Calibrate and evaluate simple detector-only online corner association."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import load_source_episode, read_source_manifest
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger, write


def representatives(source, frame, raw_indices):
    """Choose one certificate per ID for scoring only, never for tracking."""
    ids = source["physical_ids"][frame, raw_indices]
    result = {}
    for vertex_id in np.unique(ids[ids >= 0]):
        local = np.flatnonzero(ids == vertex_id)
        raw = raw_indices[local]
        errors = np.linalg.norm(source["xyz"][frame, raw] - source["reference_xyz"][frame, vertex_id], axis=-1)
        result[int(vertex_id)] = int(local[np.argmin(errors)])
    return result


def score_episode(model, source, device, threshold_m):
    """Track without IDs and score certified identities afterwards per frame."""
    tracker = DetectedCornerTracker(model, threshold_m, device)
    previous_id_track = {}
    previous_id_frame = {}
    track_label = {}
    vertex_tracks = defaultdict(set)
    counts = defaultdict(int)
    track_counts = []
    reappearance_by_gap = defaultdict(lambda: defaultdict(int))
    for frame in range(len(source["robot"])):
        raw_indices = np.flatnonzero(source["valid"][frame])
        xyz = source["xyz"][frame, raw_indices]
        kinds = source["kind"][frame, raw_indices]
        before = len(tracker.tracks)
        result = tracker.step(xyz, kinds, source["robot"][frame])
        ids = source["physical_ids"][frame, raw_indices]
        counts["frames"] += 1
        counts["detections"] += len(raw_indices)
        counts["uncertified_detections"] += int((ids < 0).sum())
        counts["new_tracks"] += result["new_tracks"]
        counts["matches"] += result["matches"]
        track_counts.append(len(tracker.tracks))
        current = representatives(source, frame, raw_indices)
        counts["duplicate_certificates"] += int((ids >= 0).sum()) - len(current)
        for vertex_id, detection_index in current.items():
            track_id = int(result["detection_tracks"][detection_index])
            counts["certified_observations"] += 1
            vertex_tracks[vertex_id].add(track_id)
            if track_id in track_label:
                counts["known_track_identity_checks"] += 1
                if track_label[track_id] != vertex_id:
                    counts["known_track_identity_conflicts"] += 1
            if vertex_id in previous_id_track:
                counts["certified_repeat_observations"] += 1
                if previous_id_track[vertex_id] != track_id:
                    counts["identity_switches"] += 1
                gap = frame - previous_id_frame[vertex_id] - 1
                if gap > 0:
                    counts["reappearances"] += 1
                    bucket = "1" if gap == 1 else "2-5" if gap <= 5 else "6-10" if gap <= 10 else "11-20" if gap <= 20 else "21+"
                    reappearance_by_gap[bucket]["events"] += 1
                    if previous_id_track[vertex_id] == track_id:
                        counts["reidentified"] += 1
                        reappearance_by_gap[bucket]["reidentified"] += 1
                    elif track_id >= before:
                        counts["reappearance_new_track"] += 1
                    else:
                        counts["reappearance_other_existing_track"] += 1
            previous_id_track[vertex_id] = track_id
            previous_id_frame[vertex_id] = frame
            track_label[track_id] = vertex_id
    counts["vertices_with_multiple_track_ids"] = sum(len(ids) > 1 for ids in vertex_tracks.values())
    counts["certified_vertices"] = len(vertex_tracks)
    counts["final_tracks"] = len(tracker.tracks)
    counts["max_tracks"] = max(track_counts, default=0)
    counts["track_frame_sum"] = sum(track_counts)
    return dict(counts=complete_counts(counts), reappearance_by_gap={key: dict(value) for key, value in reappearance_by_gap.items()})


COUNT_FIELDS = ("frames", "detections", "uncertified_detections", "new_tracks", "matches",
                "duplicate_certificates", "certified_observations", "known_track_identity_checks",
                "known_track_identity_conflicts", "certified_repeat_observations", "identity_switches",
                "reappearances", "reidentified", "reappearance_new_track", "reappearance_other_existing_track",
                "vertices_with_multiple_track_ids", "certified_vertices", "final_tracks", "max_tracks", "track_frame_sum")


def complete_counts(counts):
    return {key: int(counts.get(key, 0)) for key in COUNT_FIELDS}


def aggregate(episode_results):
    """Sum event counts and retain the worst per-episode track count."""
    sums = {key: sum(row["counts"][key] for row in episode_results) for key in COUNT_FIELDS}
    sums["max_tracks"] = max((row["counts"]["max_tracks"] for row in episode_results), default=0)
    gaps = defaultdict(lambda: defaultdict(int))
    for row in episode_results:
        for bucket, values in row["reappearance_by_gap"].items():
            for key, value in values.items():
                gaps[bucket][key] += value
    sums["reidentification_rate"] = sums["reidentified"] / sums["reappearances"] if sums["reappearances"] else None
    sums["known_track_conflict_rate"] = (sums["known_track_identity_conflicts"] / sums["known_track_identity_checks"]
                                          if sums["known_track_identity_checks"] else None)
    sums["id_switch_rate"] = sums["identity_switches"] / sums["certified_repeat_observations"] if sums["certified_repeat_observations"] else None
    sums["mean_tracks_per_frame"] = sums["track_frame_sum"] / sums["frames"] if sums["frames"] else None
    sums["reappearance_by_gap"] = {key: dict(value) for key, value in gaps.items()}
    return sums


def run_split(model, config, split, threshold, device):
    root = Path(config["data"]["source_dataset"]).resolve()
    manifest = read_source_manifest(root)
    if manifest["dataset_sha256"] != config["data"]["source_sha256"]:
        raise ValueError("Observed-corner source dataset hash mismatch")
    rows = [row for row in manifest["episodes"] if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} episodes")
    results = []
    for index, row in enumerate(rows):
        source = load_source_episode(root, row)
        result = score_episode(model, source, device, threshold / 1000)
        result["episode"] = row["episode"]
        results.append(result)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"{split} threshold={threshold}mm episodes={index + 1}/{len(rows)} "
                  f"reappearances={sum(r['counts']['reappearances'] for r in results)}", flush=True)
    return dict(threshold_mm=threshold, split=split, episodes=len(rows), aggregate=aggregate(results), per_episode=results)


def calibrate(config):
    """Score each candidate threshold on calibration episodes only."""
    device = torch.device(config["device"])
    model, checkpoint = load_temporal(config, device)
    with ExperimentLogger(team="graph", model="detected_corner_calibration", config=config,
                          seed=config["seed"], job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["temporal"]["checkpoint"], checkpoint=True)
        results = [run_split(model, config, "calibration", threshold, device)
                   for threshold in config["calibration"]["thresholds_mm"]]
        write(output / "calibration.json", dict(results=results, temporal_run_id=checkpoint["run_id"], config=config))
        logger.link_output(output / "calibration.json")
        logger.record(source_sha256=config["data"]["source_sha256"])
        print(f"Calibration completed: {output}", flush=True)
        return str(output)


def select_threshold(config):
    """Use the same threshold for all six runs, selecting on calibration only."""
    records = defaultdict(list)
    sources = []
    reference_data = None
    reference_thresholds = None
    reference_episodes = None
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            root = Path(config["calibration_root"]) / architecture / f"seed_{seed}"
            paths = list(root.glob("*/calibration.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one calibration result under {root}")
            data = json.loads(paths[0].read_text())
            if data["config"]["seed"] != seed or data["config"]["temporal"]["architecture"] != architecture:
                raise ValueError("Calibration architecture/seed mismatch")
            data_identity = data["config"]["data"]
            thresholds = data["config"]["calibration"]["thresholds_mm"]
            episodes = [[row["episode"] for row in result["per_episode"]] for result in data["results"]]
            if reference_data is None:
                reference_data, reference_thresholds, reference_episodes = data_identity, thresholds, episodes
            if data_identity != reference_data or thresholds != reference_thresholds or episodes != reference_episodes:
                raise ValueError("Calibration data, threshold candidates or episodes differ")
            sources.append(dict(path=str(paths[0]), sha256=file_sha256(paths[0])))
            for result in data["results"]:
                records[result["threshold_mm"]].append(result["aggregate"])
    candidates = []
    for threshold, values in sorted(records.items()):
        if len(values) != 6 or any(value["reidentification_rate"] is None or value["known_track_conflict_rate"] is None for value in values):
            raise ValueError("Each candidate requires six nonempty calibration results")
        candidates.append(dict(threshold_mm=threshold,
                               mean_reidentification=float(np.mean([value["reidentification_rate"] for value in values])),
                               max_known_track_conflict=float(max(value["known_track_conflict_rate"] for value in values))))
    allowed = [candidate for candidate in candidates if candidate["max_known_track_conflict"] <= config["max_conflict_rate"]]
    if not allowed:
        raise ValueError("No threshold meets the configured calibration identity-conflict bound")
    chosen = max(allowed, key=lambda item: (item["mean_reidentification"], -item["threshold_mm"]))
    result = dict(threshold_mm=chosen["threshold_mm"], candidates=candidates,
                  max_conflict_rate=config["max_conflict_rate"], split="calibration", source_results=sources)
    with ExperimentLogger(team="graph", model="detected_corner_threshold", config=config,
                          seed=0, job_type="evaluation") as logger:
        write(config["output"]["selection_file"], result)
        logger.link_output(config["output"]["selection_file"])
        logger.record(selected_threshold_mm=chosen["threshold_mm"])
    print(f"Selected common threshold: {chosen['threshold_mm']} mm", flush=True)
    return result


def evaluate(config):
    """Evaluate the calibration-selected threshold on untouched validation episodes."""
    selection = json.loads(Path(config["selection_file"]).read_text())
    if selection["split"] != "calibration":
        raise ValueError("Threshold must be selected on calibration")
    device = torch.device(config["device"])
    model, checkpoint = load_temporal(config, device)
    with ExperimentLogger(team="graph", model="detected_corner_tracking", config=config,
                          seed=config["seed"], job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["temporal"]["checkpoint"], checkpoint=True)
        result = run_split(model, config, "validation", selection["threshold_mm"], device)
        write(output / "validation.json", dict(result=result, selection=selection,
              temporal_run_id=checkpoint["run_id"], config=config, gt_used_in_tracking=False,
              known_id_metrics_are_gt_assisted=True, artificial_occlusion=False))
        logger.link_output(output / "validation.json")
        logger.record(source_sha256=config["data"]["source_sha256"], **result["aggregate"])
        print(f"Validation completed: {output}", flush=True)
        return str(output)


def compare(config):
    """Summarize the six completed validation runs on identical episodes."""
    rows = []
    reference_episodes = None
    reference_selection = None
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            root = Path(config["validation_root"]) / architecture / f"seed_{seed}"
            paths = list(root.glob("*/validation.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one validation result under {root}")
            value = json.loads(paths[0].read_text())
            if (value["config"]["seed"] != seed
                    or value["config"]["temporal"]["architecture"] != architecture):
                raise ValueError("Validation architecture/seed mismatch")
            episodes = [episode["episode"] for episode in value["result"]["per_episode"]]
            selection = value["selection"]
            if reference_episodes is None:
                reference_episodes, reference_selection = episodes, selection
            if episodes != reference_episodes or selection != reference_selection:
                raise ValueError("Validation episodes or calibration-selected threshold differ")
            rows.append(dict(architecture=architecture, seed=seed, path=str(paths[0]),
                             sha256=file_sha256(paths[0]), metrics=value["result"]["aggregate"]))
    architecture_means = {}
    for name in ("mamba", "transformer"):
        samples = [row["metrics"] for row in rows if row["architecture"] == name]
        keys = ("reidentification_rate", "known_track_conflict_rate", "id_switch_rate",
                "mean_tracks_per_frame", "vertices_with_multiple_track_ids", "final_tracks")
        architecture_means[name] = {key: float(np.mean([sample[key] for sample in samples])) for key in keys}
    with ExperimentLogger(team="graph", model="detected_corner_comparison", config=config,
                          seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        result = dict(threshold=reference_selection, validation_episodes=len(reference_episodes),
                      rows=rows, architecture_means=architecture_means,
                      gt_used_in_tracking=False, known_id_metrics_are_gt_assisted=True,
                      no_track_retirement=True)
        write(output / "comparison.json", result)
        logger.link_output(output / "comparison.json")
        logger.record(architecture_means=architecture_means)
        print(f"Detection-only comparison: {output}", flush=True)
        return str(output)
