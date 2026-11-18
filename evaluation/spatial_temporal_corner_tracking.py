"""Evaluate teacher-bound forecasts and detector-only B2-3 association."""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import file_sha256
from graph.state_estimation.graph import batch_graphs
from graph.temporal.spatial_temporal_corner_tracker import (
    OnlineSpatialTemporalTracker,
    load_tracker_checkpoint,
)
from training.gnn.spatial_temporal_corner_data import (
    CornerTrackingDataset,
    collate_corner_tracking,
)
from training.spatial_temporal_corner_tracker import run_epoch
from utils.experiment_logger import ExperimentLogger, write


def evaluate_online(model, dataset, settings, device, *, frame_callback=None):
    """Replay detector-only association; use certificates only after each step."""
    totals = {
        "episodes": 0,
        "frames": 0,
        "matches": 0,
        "new_tracks": 0,
        "final_tracks": 0,
        "known_detections": 0,
        "association_targets": 0,
        "association_correct": 0,
        "identity_switches": 0,
        "known_track_conflicts": 0,
        "reappearances": 0,
        "reappearance_same_id": 0,
        "missing_squared": 0.0,
        "missing_targets": 0,
        "reappearance_squared": 0.0,
        "reappearance_targets": 0,
    }
    per_episode = []
    started = time.monotonic()
    for episode in dataset:
        tracker = OnlineSpatialTemporalTracker(
            model,
            distance_limit_m=settings["distance_limit_m"],
            embedding_weight_m=settings["embedding_weight_m"],
        )
        physical_track = {}
        track_identity = {}
        last_seen = {}
        counts = {name: 0 for name in (
            "matches",
            "new_tracks",
            "known_detections",
            "association_targets",
            "association_correct",
            "identity_switches",
            "known_track_conflicts",
            "reappearances",
            "reappearance_same_id",
            "missing_targets",
            "reappearance_targets",
        )}
        missing_squared = 0.0
        reappearance_squared = 0.0
        for frame_index, graph in enumerate(episode.graphs):
            # Canonical IDs remain available to the evaluator, but are removed
            # before the tracker sees the graph.
            evaluation_ids = graph.vertex_id.clone()
            model_graph = replace(graph, vertex_id=torch.full_like(graph.vertex_id, -1))
            result = tracker.step(batch_graphs([model_graph]).to(device))
            maximum = settings.get("max_tracks_per_frame")
            if maximum is not None and len(tracker.tracks) > maximum:
                raise ValueError(
                    f"Association produced {len(tracker.tracks)} tracks at "
                    f"{episode.episode_id}:{frame_index}; limit is {maximum}"
                )
            counts["matches"] += result["matches"]
            counts["new_tracks"] += result["new_tracks"]
            current_physical = set()
            for detection_index, row in enumerate(result["detection_rows"].cpu().tolist()):
                physical = int(evaluation_ids[row])
                if physical < 0:
                    continue
                track_id = int(result["detection_tracks"][detection_index])
                counts["known_detections"] += 1
                current_physical.add(physical)
                previous_identity = track_identity.get(track_id)
                if previous_identity is not None and previous_identity != physical:
                    counts["known_track_conflicts"] += 1
                prior_track = physical_track.get(physical)
                if prior_track is not None:
                    counts["association_targets"] += 1
                    counts["association_correct"] += int(prior_track == track_id)
                if prior_track is not None and prior_track != track_id:
                    counts["identity_switches"] += 1
                if physical in last_seen and frame_index - last_seen[physical] > 1:
                    counts["reappearances"] += 1
                    if prior_track == track_id:
                        counts["reappearance_same_id"] += 1
                    forecast = result["forecast_positions"][prior_track].cpu().numpy()
                    difference = forecast - episode.measurements[frame_index, physical]
                    reappearance_squared += float(difference @ difference)
                    counts["reappearance_targets"] += 1
                track_identity.setdefault(track_id, physical)
                physical_track[physical] = track_id
                last_seen[physical] = frame_index

            for physical, track_id in physical_track.items():
                if physical in current_physical or track_id >= len(result["positions"]):
                    continue
                difference = (
                    result["positions"][track_id].cpu().numpy()
                    - episode.targets[frame_index, physical]
                )
                missing_squared += float(difference @ difference)
                counts["missing_targets"] += 1
            if frame_callback is not None:
                frame_callback(episode, frame_index, result)

        row = {
            "episode": episode.episode_id,
            "frames": episode.length,
            "final_tracks": len(tracker.tracks),
            "missing_rmse_mm": (
                1000 * (missing_squared / counts["missing_targets"]) ** 0.5
                if counts["missing_targets"]
                else None
            ),
            **counts,
        }
        per_episode.append(row)
        totals["episodes"] += 1
        totals["frames"] += episode.length
        totals["final_tracks"] += len(tracker.tracks)
        totals["missing_squared"] += missing_squared
        totals["reappearance_squared"] += reappearance_squared
        for name, value in counts.items():
            totals[name] += value
        interval = settings.get("log_every_episodes", 25)
        if interval and (len(per_episode) == 1 or len(per_episode) % interval == 0):
            print(
                f"online episodes={len(per_episode)}/{len(dataset)} frames={totals['frames']} "
                f"max_final_tracks={max(row['final_tracks'] for row in per_episode)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    return {
        "episodes": totals["episodes"],
        "frames": totals["frames"],
        "matches": totals["matches"],
        "new_tracks": totals["new_tracks"],
        "mean_final_tracks": totals["final_tracks"] / totals["episodes"],
        "max_final_tracks": max(row["final_tracks"] for row in per_episode),
        "known_detections": totals["known_detections"],
        "association_targets": totals["association_targets"],
        "association_correct": totals["association_correct"],
        "association_accuracy": (
            totals["association_correct"] / totals["association_targets"]
            if totals["association_targets"] else None
        ),
        "identity_switches": totals["identity_switches"],
        "known_track_conflicts": totals["known_track_conflicts"],
        "reappearances": totals["reappearances"],
        "reappearance_same_id": totals["reappearance_same_id"],
        "reappearance_same_id_rate": (
            totals["reappearance_same_id"] / totals["reappearances"]
            if totals["reappearances"]
            else None
        ),
        "missing_targets": totals["missing_targets"],
        "reappearance_rmse_mm": (
            1000 * (totals["reappearance_squared"] / totals["reappearance_targets"]) ** 0.5
            if totals["reappearance_targets"] else None
        ),
        "missing_rmse_mm": (
            1000 * (totals["missing_squared"] / totals["missing_targets"]) ** 0.5
            if totals["missing_targets"]
            else None
        ),
        "per_episode": per_episode,
    }


def evaluate(config):
    """Evaluate one selected tracker checkpoint on development validation."""
    device = torch.device(config["device"])
    model, checkpoint = load_tracker_checkpoint(
        config["checkpoint"], config.get("checkpoint_sha256"), device
    )
    manifest = Path(config["data"]["manifest"]).resolve()
    if file_sha256(manifest) != checkpoint["identity"]["manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from tracker training")
    dataset = CornerTrackingDataset(
        manifest,
        "validation",
        max_edge_m=checkpoint["identity"]["max_edge_m"],
        graph_cache_root=checkpoint["config"]["data"].get("graph_cache_root"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["evaluation"].get("episode_batch_size", 1),
        shuffle=False,
        collate_fn=collate_corner_tracking,
    )
    teacher = run_epoch(
        model,
        loader,
        checkpoint["config"]["training"],
        device,
    )
    online = evaluate_online(model, dataset, config["association"], device)
    with ExperimentLogger(
        team="graph",
        model="spatial_temporal_corner_tracking_evaluation",
        seed=checkpoint["config"]["seed"],
        config=config,
        job_type="evaluation",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        result = {
            "checkpoint": str(Path(config["checkpoint"]).resolve()),
            "checkpoint_sha256": file_sha256(Path(config["checkpoint"])),
            "dataset_sha256": checkpoint["identity"]["dataset_sha256"],
            "teacher_bound": teacher,
            "detector_only_online": online,
            "gt_identity_in_forward_path": False,
            "checkpoint_selection": "development-validation missing-corner RMSE",
        }
        write(output / "results.json", result)
        logger.link_output(config["checkpoint"], checkpoint=True)
        logger.link_output(output / "results.json")
        logger.record(
            teacher_bound=teacher,
            detector_only_online={key: value for key, value in online.items() if key != "per_episode"},
        )
        return str(output)
