"""Select detector-only association settings on fixed validation episodes."""

from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from data_collection.episode_dataset import file_sha256
from evaluation.spatial_temporal_corner_tracking import evaluate_online
from graph.temporal.spatial_temporal_corner_tracker import load_tracker_checkpoint
from training.gnn.spatial_temporal_corner_data import CornerTrackingDataset
from utils.experiment_logger import ExperimentLogger, write


def calibration_indices(dataset, max_seeds):
    """Select complete five-mode seed groups without opening test episodes."""
    seeds = sorted({row["seed"] for row in dataset.records})
    if type(max_seeds) is not int or max_seeds < 1:
        raise ValueError("calibration.max_seeds must be a positive integer")
    positions = np.linspace(0, len(seeds) - 1, min(max_seeds, len(seeds)), dtype=int)
    selected_seeds = {seeds[position] for position in positions}
    return [index for index, row in enumerate(dataset.records) if row["seed"] in selected_seeds]


def selection_key(metrics):
    """Rank identity retention first, then missing error and track growth."""
    reappearance = metrics["reappearance_same_id_rate"]
    if reappearance is None:
        raise ValueError("Calibration needs certified reappearances")
    identity_error = (
        metrics["identity_switches"] + metrics["known_track_conflicts"]
    ) / max(metrics["known_detections"], 1)
    missing = metrics["missing_rmse_mm"]
    if missing is None:
        raise ValueError("Calibration needs missing-corner targets")
    return [identity_error + (1 - reappearance), missing, metrics["mean_final_tracks"]]


def calibrate(config):
    """Sweep fixed weights on validation and save all candidate results."""
    device = torch.device(config["device"])
    model, checkpoint = load_tracker_checkpoint(
        config["checkpoint"], config["checkpoint_sha256"], device
    )
    manifest = Path(config["data"]["manifest"]).resolve()
    if file_sha256(manifest) != checkpoint["identity"]["manifest_sha256"]:
        raise ValueError("Calibration dataset differs from tracker training")
    dataset = CornerTrackingDataset(
        manifest, "validation", max_edge_m=checkpoint["identity"]["max_edge_m"],
        graph_cache_root=checkpoint["config"]["data"].get("graph_cache_root"),
    )
    settings = config["calibration"]
    indices = calibration_indices(dataset, settings["max_seeds"])
    selected = Subset(dataset, indices)
    candidates = list(product(settings["distance_limits_m"], settings["embedding_weights_m"]))
    if not candidates:
        raise ValueError("Calibration requires nonempty gate and embedding-weight candidates")
    result = {
        "checkpoint_sha256": file_sha256(Path(config["checkpoint"])),
        "manifest_sha256": file_sha256(manifest),
        "split": "validation",
        "episodes": [dataset.records[index]["episode"] for index in indices],
        "selection": [
            "require no track-count guard violation",
            "minimize (switches + conflicts) / certified detections + reappearance failure rate",
            "then minimize missing RMSE",
            "then minimize mean final track count",
        ],
        "candidates": [],
    }
    with ExperimentLogger(
        team="graph", model="spatial_temporal_association_calibration",
        seed=checkpoint["config"]["seed"], config=config, job_type="calibration",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        logger.link_output(config["checkpoint"], checkpoint=True)
        for number, (gate, weight) in enumerate(candidates, 1):
            association = {"distance_limit_m": gate, "embedding_weight_m": weight}
            print(f"calibration candidate={number}/{len(candidates)} {association}", flush=True)
            row = {"association": association}
            try:
                metrics = evaluate_online(model, selected, {
                    **association,
                    "max_tracks_per_frame": settings["max_tracks_per_frame"],
                    "log_every_episodes": settings.get("log_every_episodes", 5),
                }, device)
                row.update(status="passed", metrics=metrics, selection_key=selection_key(metrics))
                scalar_metrics = {
                    key: value for key, value in metrics.items() if key != "per_episode"
                }
                logger.log({"candidate": scalar_metrics}, step=number)
            except ValueError as error:
                if "Association produced" not in str(error):
                    raise
                row.update(status="track_guard_failed", error=str(error))
            result["candidates"].append(row)
            write(output / "results.json", result)
        feasible = [row for row in result["candidates"] if row["status"] == "passed"]
        if not feasible:
            raise ValueError(
                f"All association candidates exceeded the track limit; inspect {output}"
            )
        best = min(feasible, key=lambda row: row["selection_key"])
        result["selected_association"] = best["association"]
        result["selected_metrics"] = best["metrics"]
        write(output / "results.json", result)
        logger.link_output(output / "effective_config.yaml")
        logger.link_output(output / "results.json")
        logger.record(selected_association=best["association"], selection_key=best["selection_key"])
    return str(output)
