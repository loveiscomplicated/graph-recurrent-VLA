"""Measure endpoint errors by detector visibility and individual GT substitutions."""

from contextlib import ExitStack
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from data_collection.detected_corner_visibility import coplanarity_tolerance
from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from evaluation.gnn.five_mode_routed import (
    ENDPOINT, endpoint_errors_mm, load_frames, load_model, pinned_path, summarize_mm,
)
from graph.gated_mpnn import rotate_local_x
from training.gnn.packed_five_mode_data import indexed_frames, read_archive_member
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


PEG_CENTER = slice(0, 3)
PEG_QUATERNION = slice(3, 7)
HOLE_CENTER = slice(10, 13)
PEG_HALF_LENGTH = slice(17, 18)


def corner_measurements(corners, frames):
    """Count detector corners and measure the Peg's best-fit plane RMS."""
    offsets = corners["corner_offsets"]
    kinds = corners["corner_object_id"]
    xyz = corners["corner_world_xyz"]
    rim = corners["corner_contour_hole"]
    for frame in frames:
        start, stop = offsets[frame:frame + 2]
        frame_kinds = kinds[start:stop]
        peg = xyz[start:stop][frame_kinds == 0].astype(np.float64)
        peg_count = len(peg)
        rim_count = int(np.count_nonzero((frame_kinds == 1) & rim[start:stop]))
        plane_rms = np.nan
        if peg_count >= 4:
            singular_values = np.linalg.svd(peg - peg.mean(axis=0), compute_uv=False)
            plane_rms = singular_values[-1] / np.sqrt(peg_count)
        yield peg_count, rim_count, plane_rms


def load_corner_measurements(manifest_path, manifest, index, frame_ids):
    """Align detector metadata to the test loader using the original frame IDs."""
    records = manifest["records"]
    splits = {(row["seed"], row["policy_mode"]): row["split"] for row in records}
    frame_lists = indexed_frames(index, "condition_2", ("test",), splits)
    archive_names = {row["file"] for row in manifest["archives"]}
    detector_ids, measurements = [], []
    with ExitStack() as stack:
        archives = {}
        for episode in records:
            key = episode["seed"], episode["policy_mode"]
            frames = frame_lists.get(key)
            if not frames:
                continue
            receipt = episode["files"]["corners"]
            name = receipt["archive"]
            path = (manifest_path.parent / name).resolve()
            if name not in archive_names or not path.is_relative_to(manifest_path.parent):
                raise ValueError(f"Unexpected Five-mode corner archive: {path}")
            if path not in archives:
                archives[path] = stack.enter_context(tarfile.open(path, "r:"))
            corners = read_archive_member(archives[path], receipt, "corners")
            measurements.extend(corner_measurements(corners, frames))
            detector_ids.extend(f"{key[0]}:{key[1]}:{frame}" for frame in frames)
    if detector_ids != frame_ids:
        raise ValueError("Detector metadata order differs from the evaluation frames")
    return np.asarray(measurements)


def visibility_groups(measurements, tolerance_m):
    """Partition condition_2 with the same Peg and rim rules as the release."""
    peg_count, rim_count, plane_rms = measurements.T
    peg_sparse = peg_count < 4
    peg_coplanar = plane_rms <= tolerance_m
    peg_limited = peg_sparse | peg_coplanar
    rim_limited = rim_count < 3
    if np.any((peg_count < 2) | (rim_count < 1) | ~(peg_limited | rim_limited)):
        raise ValueError("Detector measurements disagree with the condition_2 index")
    return {
        "peg_only": peg_limited & ~rim_limited,
        "rim_only": ~peg_limited & rim_limited,
        "both": peg_limited & rim_limited,
    }, {
        "peg_only_few_corners": peg_sparse & ~rim_limited,
        "peg_only_coplanar": peg_coplanar & ~rim_limited,
        "both_few_peg_corners": peg_sparse & rim_limited,
        "both_coplanar_peg": peg_coplanar & rim_limited,
    }


@torch.inference_mode()
def predict_object21(loader, model):
    """Collect predicted Object21 and the residual actually added by the GNN."""
    predictions, residuals = [], []
    for batch in loader:
        output = model(batch.graph)
        predictions.append(output["object21"].cpu())
        residuals.append(output["endpoint_residual"].cpu())
    return torch.cat(predictions), torch.cat(residuals)


def endpoint_with_gt(prediction, target, residual):
    """Replace one geometric term at a time while retaining the learned residual."""
    peg, gt_peg = prediction[:, PEG_CENTER], target[:, PEG_CENTER]
    hole, gt_hole = prediction[:, HOLE_CENTER], target[:, HOLE_CENTER]
    axis = rotate_local_x(prediction[:, PEG_QUATERNION])
    gt_axis = rotate_local_x(target[:, PEG_QUATERNION])
    length, gt_length = prediction[:, PEG_HALF_LENGTH], target[:, PEG_HALF_LENGTH]

    geometric = peg + axis * length - hole
    torch.testing.assert_close(geometric + residual, prediction[:, ENDPOINT], rtol=0, atol=1e-7)
    gt_endpoint = gt_peg + gt_axis * gt_length - gt_hole
    torch.testing.assert_close(gt_endpoint, target[:, ENDPOINT], rtol=0, atol=1e-7)

    return {
        "original": prediction[:, ENDPOINT],
        "gt_peg_center": gt_peg + axis * length - hole + residual,
        "gt_peg_rotation": peg + gt_axis * length - hole + residual,
        "gt_peg_half_length": peg + axis * gt_length - hole + residual,
        "gt_hole_center": peg + axis * length - gt_hole + residual,
    }


def summarize_groups(groups, errors):
    """Aggregate every substitution over the same frames as its baseline."""
    results = {}
    for name, group in groups.items():
        if not group.any():
            results[name] = {"count": 0}
            continue
        stats = {variant: summarize_mm(values[group]) for variant, values in errors.items()}
        baseline = stats["original"]
        for summary in stats.values():
            summary["mean_reduction_mm"] = baseline["mean_mm"] - summary["mean_mm"]
            summary["p95_reduction_mm"] = baseline["p95_mm"] - summary["p95_mm"]
        results[name] = {"count": baseline["count"], "variants": stats}
    return results


def evaluate(config):
    """Run the selected checkpoint on the pinned condition_2 test cohort."""
    manifest_path = pinned_path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(pinned_path(config["condition_index"]).read_text())
    tolerance_m = coplanarity_tolerance(config["coplanarity_tolerance_m"])
    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config["matmul_precision"])
    model = load_model(config["checkpoint"], device)

    with ExperimentLogger(
        team="graph", model="vertex_edge_gated_mpnn", config=config, job_type="evaluate",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            dataset_sha256=config["manifest"]["sha256"],
            condition_index_sha256=config["condition_index"]["sha256"],
            device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            torch_version=str(torch.__version__),
            evaluator_sha256=file_sha256(Path(__file__)),
            output_directory=str(output.resolve()),
        )
        logger.link_output(config["checkpoint"]["path"], checkpoint=True)
        loader = load_frames(manifest_path, manifest, index, "test", "condition_2", config, device)
        measurements = load_corner_measurements(manifest_path, manifest, index, loader.ids)
        groups, subgroups = visibility_groups(measurements, tolerance_m)
        groups = {"all": np.ones(len(loader.ids), dtype=bool), **groups}
        predicted, residual = predict_object21(loader, model)
        target = loader.target.cpu()
        variants = endpoint_with_gt(predicted, target, residual)
        errors = {
            name: endpoint_errors_mm(endpoint.numpy(), target[:, ENDPOINT].numpy())
            for name, endpoint in variants.items()
        }
        if not all(np.isfinite(values).all() for values in errors.values()):
            raise ValueError("Endpoint errors must be finite for every substitution")
        results = summarize_groups(groups, errors)
        subgroup_results = summarize_groups(subgroups, errors)

        trace_path = output / "test_endpoint_diagnostics.npz"
        np.savez_compressed(
            trace_path, frame_ids=np.asarray(loader.ids), peg_corner_count=measurements[:, 0],
            rim_corner_count=measurements[:, 1], peg_plane_rms_m=measurements[:, 2],
            prediction_object21=predicted.numpy(), target_object21=target.numpy(),
            endpoint_residual_m=residual.numpy(),
            **{f"{name}_error_mm": values for name, values in errors.items()},
        )
        logger.link_output(trace_path)
        result_path = output / "results.json"
        write_json(result_path, {
            "run_id": logger.run_id, "config": config,
            "frame_ids_sha256": canonical_hash(loader.ids),
            "groups": results, "peg_subgroups": subgroup_results,
        })
        logger.link_output(result_path, kind="metrics")
        logger.log(results, step=1, prefix="test/endpoint_by_visibility")
        logger.log(subgroup_results, step=2, prefix="test/endpoint_by_peg_degeneracy")
        print(f"Completed {result_path}", flush=True)
