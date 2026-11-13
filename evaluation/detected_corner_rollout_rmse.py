"""Permutation-invariant geometric RMSE for detector-only corner rollouts."""

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import load_source_episode, read_source_manifest
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger, write


def match_sets(positions, kinds, seen_ids, reference_xyz, reference_kind):
    """Pair point sets by minimum squared distance within detector object type.

    GT is used only here, after tracking. Every possible point is matched; no
    distance gate is applied. Extra or missing points are counted separately.
    """
    pairs = []
    for kind in (0, 1):
        predicted = np.flatnonzero(kinds == kind)
        targets = seen_ids[reference_kind[seen_ids] == kind]
        if not len(predicted) or not len(targets):
            continue
        differences = positions[predicted, None, :] - reference_xyz[targets][None, :, :]
        costs = np.square(differences).sum(axis=-1)
        local_prediction, local_target = linear_sum_assignment(costs)
        pairs.extend((int(predicted[p]), int(targets[t])) for p, t in zip(local_prediction, local_target))
    return pairs


def empty_totals():
    """Create additive totals needed for an exact frame-weighted 3D RMSE."""
    return dict(matched=0, squared_error_m2=0.0, predicted_nodes=0, seen_gt_nodes=0,
                unmatched_predicted_nodes=0, unmatched_seen_gt_nodes=0)


def summarize(totals):
    """Compute sqrt(mean(||prediction - GT||²)) in millimetres."""
    count = totals["matched"]
    return dict(**totals, rmse_3d_mm=(totals["squared_error_m2"] / count) ** .5 * 1000 if count else None)


def replay_episode(model, source, device, threshold_m):
    """Replay detector-only tracking and evaluate point sets after each step."""
    tracker = DetectedCornerTracker(model, threshold_m, device)
    seen = np.zeros(24, dtype=bool)
    totals = {name: empty_totals() for name in ("all", "detected", "predicted")}
    frames = []
    association = defaultdict(int)
    for frame in range(len(source["robot"])):
        valid = source["valid"][frame]
        xyz = source["xyz"][frame, valid]
        kinds = source["kind"][frame, valid]
        output = tracker.step(xyz, kinds, source["robot"][frame])
        association["matches"] += output["matches"]
        association["new_tracks"] += output["new_tracks"]
        # Oracle IDs affect only the evaluation target set, not tracking.
        certified = source["physical_ids"][frame, valid]
        seen[certified[certified >= 0]] = True
        seen_ids = np.flatnonzero(seen)
        track_kinds = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
        pairs = match_sets(output["positions"], track_kinds, seen_ids,
                           source["reference_xyz"][frame], source["reference_kind"])
        matched_tracks = {index for index, _ in pairs}
        frame_counts = dict(frame=frame, tracks=len(track_kinds), seen_gt=len(seen_ids),
                            matched=len(pairs), unmatched_tracks=len(track_kinds) - len(pairs),
                            unmatched_seen_gt=len(seen_ids) - len(pairs))
        frames.append(frame_counts)
        totals["all"]["predicted_nodes"] += len(track_kinds)
        totals["all"]["seen_gt_nodes"] += len(seen_ids)
        totals["all"]["unmatched_predicted_nodes"] += len(track_kinds) - len(pairs)
        totals["all"]["unmatched_seen_gt_nodes"] += len(seen_ids) - len(pairs)
        for group, observed in (("detected", True), ("predicted", False)):
            mask = output["observed"] == observed
            totals[group]["predicted_nodes"] += int(mask.sum())
            totals[group]["unmatched_predicted_nodes"] += sum(
                bool(mask[index]) and index not in matched_tracks for index in range(len(mask)))
        for track_index, vertex_id in pairs:
            difference = output["positions"][track_index] - source["reference_xyz"][frame, vertex_id]
            squared = float(np.dot(difference, difference))
            for group in ("all", "detected" if output["observed"][track_index] else "predicted"):
                totals[group]["matched"] += 1
                totals[group]["squared_error_m2"] += squared
    return dict(totals={name: summarize(value) for name, value in totals.items()},
                association=dict(matches=association["matches"], new_tracks=association["new_tracks"],
                                 final_tracks=len(tracker.tracks)), frame_records=frames)


def evaluate(config):
    """Evaluate a frozen seed on the recorded development-validation rollout."""
    selection = json.loads(Path(config["selection_file"]).read_text())
    if selection["split"] != "calibration":
        raise ValueError("The distance gate must be selected on calibration")
    root = Path(config["data"]["source_dataset"]).resolve()
    manifest = read_source_manifest(root)
    if manifest["dataset_sha256"] != config["data"]["source_sha256"]:
        raise ValueError("Observed-corner dataset hash mismatch")
    device = torch.device(config["device"])
    model, checkpoint = load_temporal(config, device)
    expected_path = Path(config["source_validation"]["path"])
    if file_sha256(expected_path) != config["source_validation"]["sha256"]:
        raise ValueError("Source validation result hash mismatch")
    expected_data = json.loads(expected_path.read_text())
    if (expected_data["config"]["temporal"]["sha256"] != config["temporal"]["sha256"]
            or expected_data["selection"]["threshold_mm"] != selection["threshold_mm"]):
        raise ValueError("Source validation checkpoint or distance gate differs")
    expected = {row["episode"]: row["counts"] for row in expected_data["result"]["per_episode"]}
    rows = [row for row in manifest["episodes"] if row["split"] == "validation"]
    if set(expected) != {row["episode"] for row in rows}:
        raise ValueError("Validation episode membership differs")
    with ExperimentLogger(team="graph", model="detected_corner_rollout_rmse", config=config,
                          seed=config["seed"], job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["temporal"]["checkpoint"], checkpoint=True)
        totals = {name: empty_totals() for name in ("all", "detected", "predicted")}
        episodes = []
        for index, row in enumerate(rows):
            source = load_source_episode(root, row)
            result = replay_episode(model, source, device, selection["threshold_mm"] / 1000)
            actual = result["association"]
            if any(actual[key] != expected[row["episode"]][key] for key in ("matches", "new_tracks", "final_tracks")):
                raise ValueError(f"Association replay differs from existing validation: {row['episode']}")
            for name in totals:
                for key in totals[name]:
                    totals[name][key] += result["totals"][name][key]
            episodes.append(dict(episode=row["episode"], **result))
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"RMSE {config['temporal']['architecture']} seed={config['seed']} episodes={index + 1}/{len(rows)}", flush=True)
        summary = {name: summarize(values) for name, values in totals.items()}
        result = dict(config=config, threshold_mm=selection["threshold_mm"], episodes=len(rows),
                      source_validation_sha256=config["source_validation"]["sha256"],
                      temporal_run_id=checkpoint["run_id"], summary=summary, episode_records=episodes,
                      gt_used_for_tracker=False, gt_used_for_set_matching_and_eligibility=True,
                      assignment="per-frame minimum sum of squared 3D distances by detector peg/box type",
                      gt_eligibility="GT physical IDs with a certified detection at or before this frame",
                      unmatched_nodes_excluded_from_rmse=True)
        write(output / "rollout_rmse.json", result)
        logger.link_output(output / "rollout_rmse.json")
        logger.record(source_sha256=manifest["dataset_sha256"], metrics=summary)
        print(f"Rollout RMSE: {output}", flush=True)
        return str(output)


def compare(config):
    """Compare three seeds per architecture after all rollout replays finish."""
    rows = []
    reference_episodes = None
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            root = Path(config["runs_root"]) / architecture / f"seed_{seed}"
            paths = list(root.glob("*/rollout_rmse.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one rollout result under {root}")
            result = json.loads(paths[0].read_text())
            if result["config"]["seed"] != seed or result["config"]["temporal"]["architecture"] != architecture:
                raise ValueError("Rollout architecture/seed mismatch")
            episodes = [row["episode"] for row in result["episode_records"]]
            if reference_episodes is None:
                reference_episodes = episodes
            elif episodes != reference_episodes:
                raise ValueError("Rollout validation episodes differ")
            rows.append(dict(architecture=architecture, seed=seed, sha256=file_sha256(paths[0]),
                             path=str(paths[0]), threshold_mm=result["threshold_mm"], summary=result["summary"]))
    if len({row["threshold_mm"] for row in rows}) != 1:
        raise ValueError("Rollout thresholds differ")
    with ExperimentLogger(team="graph", model="detected_corner_rollout_rmse_comparison",
                          config=config, seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        aggregate = {}
        for architecture in ("mamba", "transformer"):
            subset = [row for row in rows if row["architecture"] == architecture]
            aggregate[architecture] = {name: dict(
                mean_seed_rmse_3d_mm=float(np.mean([row["summary"][name]["rmse_3d_mm"] for row in subset])),
                seed_rmse_3d_mm=[row["summary"][name]["rmse_3d_mm"] for row in subset],
                seed_matched=[row["summary"][name]["matched"] for row in subset],
                seed_unmatched_tracks=[row["summary"][name]["unmatched_predicted_nodes"] for row in subset],
            ) for name in ("all", "detected", "predicted")}
        result = dict(episodes=len(reference_episodes), threshold_mm=rows[0]["threshold_mm"],
                      rows=rows, aggregate=aggregate,
                      metric="sqrt(mean(squared 3D Euclidean distance)) in mm",
                      gt_used_for_tracker=False, gt_used_for_set_matching_and_eligibility=True,
                      unmatched_nodes_excluded_from_rmse=True)
        write(output / "comparison.json", result)
        logger.link_output(output / "comparison.json")
        logger.record(metrics=aggregate)
        print(f"RMSE comparison: {output}", flush=True)
        return str(output)
