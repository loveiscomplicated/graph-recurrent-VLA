"""Measure downstream accuracy and deformation on identical validation frames."""

from pathlib import Path

import numpy as np
import torch

from data_collection.detected_corner_visibility import CONDITIONS
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from evaluation.gnn.five_mode_corner_oracle import METRICS
from evaluation.gnn.five_mode_routed import pinned_path, summarize_mm
from graph.temporal.persistent_corner_tracking import predict_object21, shape_errors, shape_means
from training.gnn.loss import object21_loss
from training.gnn.tasks import geometry_errors
from utils.experiment_logger import ExperimentLogger


def batch_rows(dataset, rows, device):
    """Trim padding to this batch's maximum track count."""
    batch = {name: value[rows] for name, value in dataset.items()}
    width = max(1, int((batch["kinds"] >= 0).sum(dim=1).max()))
    for name in ("anchors", "kinds", "age", "observed", "features", "motion_targets"):
        batch[name] = batch[name][:, :width]
    return {name: value.to(device) for name, value in batch.items()}


@torch.inference_mode()
def predict(gnn, tracker, dataset, rows, device, batch_size):
    """Return frame-aligned position errors and within-object distance errors."""
    if tracker is not None:
        tracker.eval()
    parts = []
    for ids in rows.split(batch_size):
        batch = batch_rows(dataset, ids, device)
        positions = batch["anchors"] if tracker is None else tracker(batch)
        output = predict_object21(gnn, batch, positions, use_forecasts=tracker is not None)
        errors = geometry_errors(output["object21"], batch["target"])
        if tracker is None:
            parts.append(torch.stack([1000 * errors[name] for name in METRICS], dim=1).cpu().numpy())
            continue
        deformation, masks = shape_errors(positions, batch["motion_targets"], batch["kinds"], batch["observed"])
        shape_mm, pairs = shape_means(deformation.abs() * 1000, masks)
        values = [*(1000 * errors[name] for name in METRICS), *shape_mm.unbind(1), *pairs.unbind(1)]
        parts.append(torch.stack(values, dim=1).cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite persistent-corner validation error")
    return result


@torch.inference_mode()
def validation_loss(gnn, tracker, dataset, device, batch_size):
    """Give Condition 2 and 3 equal weight for checkpoint selection."""
    tracker.eval()
    losses = {}
    for label in (2, 3):
        rows = (dataset["condition"] == label).nonzero(as_tuple=True)[0]
        total = 0.0
        for ids in rows.split(batch_size):
            batch = batch_rows(dataset, ids, device)
            output = predict_object21(gnn, batch, tracker(batch))
            total += float(object21_loss(output, batch["target"])["total"]) * len(ids)
        losses[CONDITIONS[label - 1]] = total / len(rows)
    return losses


def rim_recency(dataset):
    """Separate current, recent, old and never-seen detector Hole rims."""
    absent = torch.iinfo(dataset["age"].dtype).max
    age = dataset["age"].masked_fill(dataset["kinds"] != 2, absent).min(dim=1).values.numpy()
    return {
        "current_rim": age == 0,
        "rim_age_1_to_8": (age >= 1) & (age <= 8),
        "rim_age_9_or_more": (age >= 9) & (age < absent),
        "rim_never_detected": age == absent,
    }


def summarize(dataset, traces):
    """Report paired geometry, deformation and how far rim memory reaches."""
    conditions = dataset["condition"].numpy()
    masks = {name: conditions == index for index, name in enumerate(CONDITIONS, start=1)}
    masks.update({f"condition_3/{name}": (conditions == 3) & mask
                  for name, mask in rim_recency(dataset).items()})
    report = {}
    for name, mask in masks.items():
        report[name] = groups = {"frames": int(mask.sum())}
        if not mask.any():
            continue
        for route, trace in traces.items():
            selected = trace[mask]
            groups[route] = {metric: summarize_mm(selected[:, col]) for col, metric in enumerate(METRICS)}
            if selected.shape[1] > len(METRICS):
                for index, body in enumerate(("peg", "box_with_rim")):
                    valid = selected[:, 5 + index] > 0
                    if valid.any():
                        groups[route][f"{body}_shape_error"] = summarize_mm(selected[valid, 3 + index])
    return report


def compare(config, gnn, fitted, validation, frame_ids, device):
    """Evaluate both selected checkpoints and the previous model on the same frames."""
    settings = config["evaluation"]
    rows = torch.arange(len(frame_ids))
    traces = {"detector_only": predict(gnn, None, validation, rows, device, settings["batch_size"])}
    for name, (tracker, _) in fitted.items():
        traces[name] = predict(gnn, tracker, validation, rows, device, settings["batch_size"])
    with np.load(pinned_path(settings["recent_validation_trace"]), allow_pickle=False) as previous:
        if not (np.array_equal(previous["frame_ids"], frame_ids)
                and np.array_equal(previous["condition"], validation["condition"].numpy())
                and previous["columns"].tolist()[1:4] == list(METRICS)):
            raise ValueError("Recent-model validation trace uses different frames or metrics")
        errors = previous["learned"][:, 1:4].copy()
        direct = previous["detector_only"][:, 1:4]
        first = validation["condition"].numpy() == 1
        if not np.allclose(traces["detector_only"], direct, atol=0.002, rtol=0):
            raise ValueError("Current detections changed relative to the recent-model validation")
        errors[first] = direct[first]
        traces["recent_eight"] = errors
    for name in fitted:
        if not np.allclose(traces[name][first, :3], traces["detector_only"][first], atol=0.002, rtol=0):
            raise ValueError("Condition 1 did not preserve detector-only routing")

    results = summarize(validation, traces)
    with ExperimentLogger(team="graph", model="persistent_corner_tracking", config=config,
                          job_type="evaluate") as logger:
        root = Path(config["output_root"]) / logger.run_id
        root.mkdir(parents=True, exist_ok=False)
        checkpoints = {}
        for name, (_, checkpoint) in fitted.items():
            logger.link_output(checkpoint, checkpoint=True)
            checkpoints[name] = dict(path=str(checkpoint), sha256=file_sha256(checkpoint))
        logger.record(split="validation", device=str(device), checkpoints=checkpoints,
                      frame_ids_sha256=canonical_hash(frame_ids))
        trace_path = root / "validation_errors.npz"
        np.savez_compressed(trace_path, frame_ids=frame_ids, condition=validation["condition"].numpy(),
                            columns=np.asarray((*METRICS, "peg_shape_mm", "box_shape_mm", "peg_pairs", "box_pairs")),
                            track_counts=(validation["kinds"] >= 0).sum(dim=1).numpy(), **traces)
        logger.link_output(trace_path)
        result_path = root / "results.json"
        write_json(result_path, {"checkpoints": checkpoints, "condition_results": results})
        logger.link_output(result_path, kind="metrics")
        print(f"Completed paired validation: {result_path}", flush=True)
