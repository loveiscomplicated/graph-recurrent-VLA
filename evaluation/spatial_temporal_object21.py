"""Evaluate the selected B2-3 complete-graph Object21 checkpoint."""

from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss
from training.spatial_temporal_object21 import (
    collate_rollout_frames,
    complete_track_graph,
)
from utils.experiment_logger import ExperimentLogger, write


def evaluate_arrays(model, arrays, indices, model_config, device, *, batch_size):
    """Score explicit frozen frame indices without choosing a checkpoint or split."""
    if not len(indices) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Evaluation needs nonempty frame indices and a positive batch size")
    predictions = []
    errors = {}
    loss_total = 0.0
    with torch.inference_mode():
        for left in range(0, len(indices), batch_size):
            selected = indices[left : left + batch_size]
            values = collate_rollout_frames(arrays, selected, device)
            graph, extra = complete_track_graph(
                values["xyz"], values["kinds"], values["valid"], values["observed"],
                values["uncertainty"], values["age"], values["tcp"],
                model_config["downstream_features"],
            )
            output = model(graph, extra)
            loss = object21_loss(output, values["target"], **model_config["loss"])["total"]
            loss_total += float(loss) * len(selected)
            predictions.append(output["object21"].cpu().numpy())
            for name, value in per_frame_errors(output, values["target"]).items():
                errors.setdefault(name, []).append(value)
    prediction = np.concatenate(predictions)
    errors = {name: np.concatenate(values) for name, values in errors.items()}
    metrics = {name: statistics(value) for name, value in errors.items()}
    metrics["object21_loss"] = loss_total / len(indices)
    return prediction, errors, metrics


def save_predictions(path, arrays, indices, prediction, errors):
    """Save aligned predictions, labels and reliability counts for any explicit split."""
    observed_prefix = np.concatenate(([0], np.cumsum(arrays["observed"], dtype=np.int64)))
    offsets = arrays["offsets"]
    observed_counts = observed_prefix[offsets[indices + 1]] - observed_prefix[offsets[indices]]
    track_counts = offsets[indices + 1] - offsets[indices]
    np.savez_compressed(
        path,
        sample_ids=arrays["sample_ids"][indices], object21=prediction,
        target=arrays["target"][indices], observed_counts=observed_counts,
        track_counts=track_counts, **errors,
    )


def evaluate(config):
    """Replay the frozen validation rollout cache and save aligned predictions."""
    checkpoint_path = Path(config["checkpoint"]).expanduser().resolve()
    if config.get("checkpoint_sha256") and file_sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("B2-3 Object21 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "spatial_temporal_object21_v1":
        raise ValueError("Unsupported B2-3 Object21 checkpoint")
    cache_path = Path(
        config.get("rollout_cache") or checkpoint_path.parent / "frozen_tracker_rollouts.pt"
    ).expanduser().resolve()
    if file_sha256(cache_path) != checkpoint["rollout_cache_sha256"]:
        raise ValueError("Frozen B2-3 rollout cache checksum mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("format") != "spatial_temporal_rollouts_v2":
        raise ValueError("Unsupported B2-3 rollout cache")
    arrays = cache["arrays"]
    indices = np.flatnonzero(arrays["validation"])
    if not len(indices):
        raise ValueError("B2-3 rollout cache has no validation frames")

    device = torch.device(config["device"])
    model = VertexEdgeGatedMPNN(**checkpoint["config"]["gnn"]).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    batch_size = config["evaluation"]["batch_size"]
    prediction, errors, metrics = evaluate_arrays(
        model, arrays, indices, checkpoint["config"], device, batch_size=batch_size
    )
    with ExperimentLogger(
        team="graph",
        model="spatial_temporal_object21_evaluation",
        seed=checkpoint["config"]["seed"],
        config=config,
        job_type="evaluation",
    ) as logger:
        output_directory = Path(config["output"]["root"]) / logger.run_id
        output_directory.mkdir(parents=True, exist_ok=False)
        predictions_path = output_directory / "predictions.npz"
        save_predictions(predictions_path, arrays, indices, prediction, errors)
        result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "rollout_cache_sha256": file_sha256(cache_path),
            "validation_frames": len(indices),
            "metrics": metrics,
            "tracker_frozen": True,
            "downstream_topology": "complete_directed_without_self_edges",
        }
        write(output_directory / "results.json", result)
        logger.link_output(checkpoint_path, checkpoint=True)
        logger.link_output(predictions_path)
        logger.link_output(output_directory / "results.json")
        logger.record(validation_frames=len(indices), metrics=metrics)
        return str(output_directory)
