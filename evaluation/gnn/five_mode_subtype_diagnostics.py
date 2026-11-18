"""Diagnose one frozen C2 expert on validation, without reading held-out samples."""

from contextlib import ExitStack
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from data_collection.detected_corner_visibility import classify_episode, coplanarity_tolerance
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from evaluation.gnn.five_mode_endpoint_diagnostics import corner_measurements, predict_object21
from evaluation.gnn.five_mode_routed import load_frames, pinned_path
from graph.five_mode_hard_routing import load_surface_model
from training.gnn.packed_five_mode_data import indexed_frames, read_archive_member
from training.gnn.tasks import geometry_errors
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


SUBTYPES = ("C2-A", "C2-B", "C2-C", "C2-D", "C2-E")
POLICIES = (
    "Fixed_Hole_expert", "Fixed_Hole_expert_random", "Fixed_Hole_expert_random_expert",
    "Fixed_Hole_random", "Moving_Hole_expert",
)


def subtype_masks(measurements, tolerance_m):
    """Partition detector-selected C2 frames into the five disjoint support types."""
    tolerance = coplanarity_tolerance(tolerance_m)
    values = np.asarray(measurements, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Detector measurements must have shape [N,3]: Peg count, rim count, plane RMS")
    peg_count, rim_count, plane_rms = values.T
    counts = values[:, :2]
    enough_peg = peg_count >= 4
    if (not np.isfinite(counts).all() or np.any(counts != np.floor(counts))
            or np.any(peg_count < 2) or np.any(rim_count < 1)
            or not np.isfinite(plane_rms[enough_peg]).all()
            or np.any(plane_rms[enough_peg] < 0)):
        raise ValueError("Invalid condition_2 detector support or Peg plane RMS")
    few_peg = ~enough_peg
    coplanar = enough_peg & (plane_rms <= tolerance)
    noncoplanar = enough_peg & (plane_rms > tolerance)
    few_rim = rim_count < 3
    groups = {
        "C2-A": few_peg & few_rim,
        "C2-B": few_peg & ~few_rim,
        "C2-C": coplanar & few_rim,
        "C2-D": coplanar & ~few_rim,
        "C2-E": noncoplanar & few_rim,
    }
    if not np.all(np.stack(list(groups.values())).sum(axis=0) == 1):
        raise ValueError("Detector measurements include C1 frames or disagree with the C2 index")
    return groups


def validate_config(config):
    """Require an explicitly pinned validation cohort before any data or model read."""
    if config.get("split") != "validation":
        raise ValueError("C2 subtype diagnostics accepts split: validation only")
    batch_size = config.get("batch_size", {}).get("validation")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size.validation must be an integer >= 1")
    threads = config.get("torch_num_threads")
    if type(threads) is not int or threads < 1:
        raise ValueError("torch_num_threads must be an integer >= 1")
    cohort = config.get("cohorts", {}).get("validation", {}).get("condition_2", {})
    if type(cohort.get("frames")) is not int or cohort["frames"] < 1:
        raise ValueError("The condition_2 validation cohort needs a positive frame count")
    if not isinstance(cohort.get("frame_ids_sha256"), str):
        raise ValueError("The condition_2 validation cohort needs frame_ids_sha256")


def validation_metadata(manifest_path, manifest, index, frame_ids, tolerance_m):
    """Read only validation detector members and align evidence to packed frame order."""
    records = manifest["records"]
    assignments = {(row["seed"], row["policy_mode"]): row["split"] for row in records}
    seed_splits = {}
    for row in records:
        if seed_splits.setdefault(row["seed"], row["split"]) != row["split"]:
            raise ValueError(f"Reset seed crosses dataset splits: {row['seed']}")
    selected = indexed_frames(index, "condition_2", ("validation",), assignments)
    archive_names = {row["file"] for row in manifest["archives"]}
    detector_ids, measurements = [], []
    with ExitStack() as stack:
        archives = {}
        for episode in records:
            if episode["split"] != "validation":
                continue
            key = episode["seed"], episode["policy_mode"]
            frames = selected[key]
            if not frames:
                continue
            receipt = episode["files"]["corners"]
            name = receipt["archive"]
            path = (manifest_path.parent / name).resolve()
            if name not in archive_names or not path.is_relative_to(manifest_path.parent):
                raise ValueError(f"Unexpected Five-mode detector archive: {path}")
            if path not in archives:
                archives[path] = stack.enter_context(tarfile.open(path, "r:"))
            corners = read_archive_member(archives[path], receipt, "corners")
            classified = classify_episode(corners, episode["frames"], tolerance_m=tolerance_m)
            if classified["condition_2"] != frames:
                raise ValueError(f"Current detector conditions differ from the C2 index: {key}")
            measurements.extend(corner_measurements(corners, frames))
            detector_ids.extend(f"{key[0]}:{key[1]}:{frame}" for frame in frames)
    if detector_ids != frame_ids:
        raise ValueError("Detector evidence order differs from validation prediction frame IDs")
    return np.asarray(measurements, dtype=np.float64)


def physical_errors(prediction, target):
    """Reuse training's per-frame geometry metrics, with distance errors in millimetres."""
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError("Expected finite predictions aligned to validation Object21 targets")
    result = {}
    for name, values in geometry_errors(prediction, target).items():
        if name.endswith("_m"):
            name = name.removesuffix("_m") + "_mm"
            values = values * 1000
        result[name] = values.cpu().numpy()
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Every validation geometry error must be finite")
    return result


def summarize_group(mask, errors, frame_ids):
    """Report frame-weighted errors and independent episode/seed support for one group."""
    rows = np.flatnonzero(mask)
    episodes = {frame_ids[row].rsplit(":", 1)[0] for row in rows}
    seeds = {episode.split(":", 1)[0] for episode in episodes}
    metrics = {}
    for name, values in errors.items():
        selected = values[mask]
        metrics[name] = None if not len(rows) else {
            "mean": float(np.mean(selected, dtype=np.float64)),
            "median": float(np.median(selected)),
            "p95": float(np.quantile(selected, 0.95, method="linear")),
        }
    return {"frames": len(rows), "episodes": len(episodes), "seeds": len(seeds), "metrics": metrics}


def build_report(groups, errors, frame_ids):
    """Separate error severity from prevalence, including policy and Hole-motion strata."""
    policies = np.asarray([frame_id.split(":")[1] for frame_id in frame_ids])
    if not np.isin(policies, POLICIES).all():
        raise ValueError("Validation frame IDs include an unknown Five-mode policy")
    motion = np.where(policies == "Moving_Hole_expert", "Moving_Hole", "Fixed_Hole")
    all_frames = np.ones(len(frame_ids), dtype=bool)

    def stratify(mask):
        report = summarize_group(mask, errors, frame_ids)
        report["by_policy"] = {
            policy: summarize_group(mask & (policies == policy), errors, frame_ids)
            for policy in POLICIES
        }
        report["by_motion"] = {
            kind: summarize_group(mask & (motion == kind), errors, frame_ids)
            for kind in ("Fixed_Hole", "Moving_Hole")
        }
        return report

    overall = stratify(all_frames)
    subtypes = {name: stratify(mask) for name, mask in groups.items()}
    endpoint = errors["endpoint_error_mm"]
    total_error = float(np.sum(endpoint, dtype=np.float64))
    for name, report in subtypes.items():
        group_error = float(np.sum(endpoint[groups[name]], dtype=np.float64))
        report["frame_fraction"] = report["frames"] / len(frame_ids)
        report["endpoint_error_fraction"] = group_error / total_error if total_error else 0.0
        report["endpoint_mean_contribution_mm"] = group_error / len(frame_ids)
    populated = [name for name in groups if subtypes[name]["frames"]]
    worst = {
        "highest_mean_endpoint": max(
            populated, key=lambda name: subtypes[name]["metrics"]["endpoint_error_mm"]["mean"],
        ),
        "largest_endpoint_error_contribution": max(
            populated, key=lambda name: subtypes[name]["endpoint_error_fraction"],
        ),
        "highest_mean_by_metric": {
            metric: max(populated, key=lambda name: subtypes[name]["metrics"][metric]["mean"])
            for metric in errors
        },
    }
    return {"overall": overall, "subtypes": subtypes, "worst_groups": worst}, policies, motion


def evaluate(config):
    """Run the pinned baseline once on validation; do not select or update any model."""
    validate_config(config)
    tolerance = coplanarity_tolerance(config["coplanarity_tolerance_m"])
    torch.set_num_threads(config["torch_num_threads"])
    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config["matmul_precision"])
    manifest_path = pinned_path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(pinned_path(config["condition_index"]).read_text())
    model = load_surface_model(config["checkpoint"], device, "condition_2")

    with ExperimentLogger(
        team="graph", model="vertex_edge_gated_mpnn", config=config, job_type="evaluate",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            dataset_sha256=config["manifest"]["sha256"],
            condition_index_sha256=config["condition_index"]["sha256"],
            checkpoint_sha256=config["checkpoint"]["sha256"],
            split="validation", test_samples_read=False, model_selection_performed=False,
            device=str(device), torch_version=str(torch.__version__),
            torch_num_threads=torch.get_num_threads(),
            evaluator_sha256=file_sha256(Path(__file__)),
            output_directory=str(output.resolve()),
        )
        logger.link_output(config["checkpoint"]["path"], checkpoint=True)
        loader = load_frames(
            manifest_path, manifest, index, "validation", "condition_2", config, device,
        )
        print(f"Loaded {len(loader.ids):,} pinned C2 validation frames", flush=True)
        measurements = validation_metadata(manifest_path, manifest, index, loader.ids, tolerance)
        groups = subtype_masks(measurements, tolerance)
        subtype_counts = {name: int(mask.sum()) for name, mask in groups.items()}
        print(f"Validated detector subtypes: {subtype_counts}", flush=True)
        prediction, residual = predict_object21(loader, model)
        target = loader.target.cpu()
        errors = physical_errors(prediction, target)
        report, policies, motion = build_report(groups, errors, loader.ids)
        labels = np.empty(len(loader.ids), dtype="U4")
        for name, mask in groups.items():
            labels[mask] = name

        trace_path = output / "validation_predictions.npz"
        np.savez_compressed(
            trace_path, frame_ids=np.asarray(loader.ids), subtype=labels, policy_mode=policies,
            hole_motion=motion, peg_corner_count=measurements[:, 0],
            hole_rim_count=measurements[:, 1], peg_plane_rms_m=measurements[:, 2],
            prediction_object21=prediction.numpy(), target_object21=target.numpy(),
            endpoint_residual_m=residual.numpy(), **errors,
        )
        logger.link_output(trace_path)
        result_path = output / "results.json"
        write_json(result_path, {
            "run_id": logger.run_id, "config": config, "split": "validation",
            "frame_ids_sha256": canonical_hash(loader.ids),
            "test_samples_read": False, "model_selection_performed": False,
            "metric_definition": "frame-weighted means; endpoint is Object21[7:10] error in world axes",
            **report,
        })
        logger.link_output(result_path, kind="metrics")
        logger.log(report, step=1, prefix="validation/condition_2")
        print(f"Completed {result_path}: {report['worst_groups']}", flush=True)
        return output
