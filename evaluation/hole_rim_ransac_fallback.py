"""Replay a preserved grazing-view failure through the Hole RANSAC fallback.

The original CUDA detector candidates remain byte-for-byte unchanged.  This
evaluation appends only candidates explicitly emitted by the opt-in RANSAC
fallback, then reruns the causal matcher, frozen tracker, and frozen GNN.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_surface_context import surface_sequence_arrays
from evaluation.five_mode_corner_tracker_comparison import summarize
from evaluation.frozen_surface_tracker_gnn import position_errors_mm
from graph.corner_association import CornerMatcher
from graph.frozen_tracker_graph import build_tracker_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker
from training.frozen_surface_tracker_gnn import state_digest
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector


def _verified_path(path, expected_sha256, label):
    """Resolve a file and reject any bytes that differ from the receipt."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA256 differs: expected {expected_sha256}, got {actual}"
        )
    return path


def _load_npz(path):
    """Materialize an NPZ so no file handle survives beyond this call."""
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _extract_frames(archive, destination, expected_frames):
    """Extract one verified case archive and return its ordered RGB-D frames."""
    subprocess.run(
        ["tar", "--zstd", "-xf", str(archive), "-C", str(destination)],
        check=True,
    )
    frames = sorted(destination.rglob("frame_*.npz"))
    if len(frames) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} raw frames, found {len(frames)} in {archive}"
        )
    expected_names = [f"frame_{index:06d}.npz" for index in range(expected_frames)]
    if [path.name for path in frames] != expected_names:
        raise ValueError("Raw frame names are not contiguous or ordered")
    return frames


def _fallback_points(detector, frame_path):
    """Return only the world points newly proposed by the RANSAC fallback."""
    with np.load(frame_path, allow_pickle=False) as frame:
        result = detector(
            frame["observation/sensor_data/base_camera/rgb"][0],
            frame["observation/sensor_data/base_camera/depth"][0],
            frame["observation/sensor_param/base_camera/intrinsic_cv"][0],
            frame["observation/sensor_param/base_camera/extrinsic_cv"][0],
        )
    selected = np.asarray(
        [
            metadata.get("plane_fit_method") == "ransac_fallback"
            for metadata in result.metadata
        ],
        dtype=bool,
    )
    if selected.any() and not np.all(result.object_id[selected].cpu().numpy() == 1):
        raise ValueError("Hole fallback emitted a non-Box candidate")
    return result.xyz[selected].cpu().numpy().astype(np.float32, copy=False)


def _associate_with_fallback(raw_corners, fallback_by_frame, tcp):
    """Append fallback hints after original candidates and run causal IDs once."""
    frame_count = len(tcp)
    matcher = CornerMatcher()
    detected = np.zeros((frame_count, 20, 3), dtype=np.float32)
    observed = np.zeros((frame_count, 20), dtype=bool)

    offsets = raw_corners["corner_offsets"]
    for frame_index, fallback in enumerate(fallback_by_frame):
        start, stop = offsets[frame_index : frame_index + 2]
        original_points = raw_corners["corner_world_xyz"][start:stop]
        original_objects = raw_corners["corner_object_id"][start:stop]
        original_hints = raw_corners["corner_contour_hole"][start:stop] | (
            raw_corners["corner_method"][start:stop] == "box_opening"
        )

        points = np.concatenate((original_points, fallback), axis=0)
        objects = np.concatenate(
            (original_objects, np.ones(len(fallback), dtype=original_objects.dtype))
        )
        hints = np.concatenate((original_hints, np.ones(len(fallback), dtype=bool)))
        family = np.where(objects == 0, 0, np.where(hints, 2, 1))
        assignment = matcher.step(points, family, tcp[frame_index])
        present = assignment >= 0
        observed[frame_index] = present
        detected[frame_index, present] = points[assignment[present]]
    return detected, observed


def _detector_metrics(fallback_by_frame, hole_truth):
    """Score raw fallback proposals against the four GT Hole rim vertices."""
    errors = []
    matched_within_6mm = []
    for points, truth in zip(fallback_by_frame, hole_truth, strict=True):
        if not len(points):
            matched_within_6mm.append(0)
            continue
        distance_mm = np.linalg.norm(points[:, None] - truth[None], axis=-1) * 1000
        rows, columns = linear_sum_assignment(distance_mm)
        frame_errors = distance_mm[rows, columns]
        errors.extend(frame_errors.tolist())
        matched_within_6mm.append(int((frame_errors <= 6).sum()))
    counts = np.asarray([len(points) for points in fallback_by_frame])
    matched_within_6mm = np.asarray(matched_within_6mm)
    return {
        "frames_with_any_fallback": int((counts > 0).sum()),
        "frames_with_at_least_3_fallback_points": int((counts >= 3).sum()),
        "frames_with_at_least_3_gt_matches_within_6mm": int(
            (matched_within_6mm >= 3).sum()
        ),
        "candidate_count": int(counts.sum()),
        "assignment_error_mm": summarize(np.asarray(errors)) if errors else None,
    }


def _load_models(tracker_path, gnn_path, device):
    """Load the exact frozen tracker/GNN pair and its training fallback."""
    tracker_payload = torch.load(tracker_path, map_location="cpu", weights_only=False)
    gnn_payload = torch.load(gnn_path, map_location="cpu", weights_only=False)
    if tracker_payload["config"]["training"]["supervision"] != "after_first_observation":
        raise ValueError("Expected the observed-only surface tracker")

    tracker = SurfaceSpatialCornerTracker(**tracker_payload["config"]["model"])
    tracker.load_state_dict(tracker_payload["model"], strict=True)
    settings = dict(gnn_payload["config"]["model"])
    settings.pop("name", None)
    settings.pop("factory", None)
    gnn = VertexEdgeGatedMPNN(**settings)
    gnn.load_state_dict(gnn_payload["model"], strict=True)
    for model in (tracker, gnn):
        model.to(device).eval().requires_grad_(False)
    return tracker, gnn, tracker_payload["fallback"].cpu().numpy()


@torch.inference_mode()
def _infer(episode, tracker, gnn, fallback, device):
    """Run the unchanged surface tracker and two-flag graph construction."""
    arrays, _, _ = surface_sequence_arrays([episode], fallback)
    input_names = (
        "features",
        "anchor",
        "detected",
        "observed",
        "valid",
        "surface_features",
        "surface_valid",
    )
    inputs = [torch.as_tensor(arrays[name], device=device) for name in input_names]
    corners = tracker(*inputs)[0]
    observed = torch.as_tensor(episode["observed"], device=device)
    torch.testing.assert_close(corners[observed], inputs[2][0][observed], rtol=0, atol=0)

    seen = np.logical_or.accumulate(episode["observed"], axis=0)
    graph, flags = build_tracker_graph(
        corners,
        observed,
        torch.as_tensor(seen, device=device),
        torch.as_tensor(episode["tcp"], device=device),
    )
    output = gnn(graph, flags)
    return {
        "corners": corners.cpu().numpy(),
        "object21": output["object21"].cpu().numpy(),
        "endpoint_residual": output["endpoint_residual"].cpu().numpy(),
        "ever_seen": seen,
    }


def _metric_summary(prediction, target):
    return {
        name: summarize(values)
        for name, values in position_errors_mm(prediction, target).items()
    }


def _first_hole_observation(observed):
    frames = np.flatnonzero(observed[:, 16:20].any(axis=1))
    return int(frames[0]) if len(frames) else None


def evaluate(config):
    """Replay one checksum-pinned failure and record detector-to-GNN effects."""
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation was requested but CUDA is unavailable")

    case_root = Path(config["case_root"]).expanduser().resolve()
    receipt_path = case_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["case_name"] != config["expected_case"]:
        raise ValueError("Diagnostic receipt is for a different case")
    for name, expected in receipt["files"].items():
        _verified_path(case_root / name, expected, name)
    archive = _verified_path(
        config["raw_archive"], receipt["raw_archive"]["sha256"], "raw archive"
    )
    tracker_path = _verified_path(
        config["tracker"]["path"], config["tracker"]["sha256"], "tracker checkpoint"
    )
    gnn_path = _verified_path(
        config["gnn"]["path"], config["gnn"]["sha256"], "GNN checkpoint"
    )

    features = _load_npz(case_root / "features.npz")
    raw_corners = _load_npz(case_root / "raw_corners.npz")
    published = _load_npz(case_root / "predictions.npz")
    frame_count = receipt["frames"]
    if len(features["target"]) != frame_count:
        raise ValueError("Feature count differs from the case receipt")

    detector = TorchCornerDetector.main(
        device=config.get("detector_device", config["device"]),
        plane_fit_method="grid",
        hole_rim_ransac_fallback=True,
    )
    with tempfile.TemporaryDirectory(prefix="hole_rim_fallback_") as temporary:
        frame_paths = _extract_frames(archive, Path(temporary), frame_count)
        fallback_by_frame = []
        for index, frame_path in enumerate(frame_paths):
            fallback_by_frame.append(_fallback_points(detector, frame_path))
            if (index + 1) % 25 == 0 or index + 1 == frame_count:
                print(f"Replayed fallback {index + 1}/{frame_count} frames", flush=True)

    detected, observed = _associate_with_fallback(
        raw_corners, fallback_by_frame, features["tcp"]
    )
    if not np.array_equal(observed[:, :16], features["observed"][:, :16]):
        raise ValueError("Fallback changed a Peg or Box exterior observation mask")
    existing = features["observed"][:, :16]
    if not np.array_equal(detected[:, :16][existing], features["detected"][:, :16][existing]):
        raise ValueError("Fallback changed an existing Peg or Box exterior coordinate")

    tracker, gnn, fallback = _load_models(tracker_path, gnn_path, device)
    tracker_before, gnn_before = state_digest(tracker), state_digest(gnn)
    common = {
        name: features[name]
        for name in ("tcp", "gt", "depth_mm", "surface_xyz", "surface_valid")
    }
    common.update(seed=receipt["seed"], mode=receipt["policy_mode"])
    baseline_episode = dict(common, detected=features["detected"], observed=features["observed"])
    fallback_episode = dict(common, detected=detected, observed=observed)
    baseline = _infer(baseline_episode, tracker, gnn, fallback, device)
    changed = _infer(fallback_episode, tracker, gnn, fallback, device)
    if state_digest(tracker) != tracker_before or state_digest(gnn) != gnn_before:
        raise RuntimeError("A frozen model changed during evaluation")

    output = Path(config["output"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(
        team="graph",
        model="hole_rim_ransac_fallback",
        job_type="evaluate",
        config=config,
        root=output / "experiments",
    ) as logger:
        predictions_path = output / "predictions.npz"
        np.savez_compressed(
            predictions_path,
            baseline_corners=baseline["corners"],
            fallback_corners=changed["corners"],
            baseline_object21=baseline["object21"],
            fallback_object21=changed["object21"],
            target=features["target"],
            detected=detected,
            observed=observed,
            ever_seen=changed["ever_seen"],
            fallback_counts=np.asarray([len(points) for points in fallback_by_frame]),
        )
        result = {
            "run_id": logger.run_id,
            "case": receipt["case_name"],
            "frames": frame_count,
            "isolation": (
                "original CUDA candidates preserved; only grid->RANSAC fallback "
                "candidates appended; baseline and fallback rerun on the same device"
            ),
            "detector": _detector_metrics(fallback_by_frame, features["gt"][:, 16:20]),
            "association": {
                "baseline_first_hole_observation_frame": _first_hole_observation(
                    features["observed"]
                ),
                "fallback_first_hole_observation_frame": _first_hole_observation(observed),
                "baseline_hole_observed_frames": int(
                    features["observed"][:, 16:20].any(axis=1).sum()
                ),
                "fallback_hole_observed_frames": int(
                    observed[:, 16:20].any(axis=1).sum()
                ),
            },
            "metrics_mm": {
                "published_original": _metric_summary(
                    published["object21"], features["target"]
                ),
                "controlled_device_baseline": _metric_summary(
                    baseline["object21"], features["target"]
                ),
                "fallback": _metric_summary(changed["object21"], features["target"]),
            },
            "published_vs_controlled_baseline_max_abs_object21": float(
                np.max(np.abs(published["object21"] - baseline["object21"]))
            ),
            "checkpoints": {
                "tracker_sha256": config["tracker"]["sha256"],
                "gnn_sha256": config["gnn"]["sha256"],
            },
            "raw_archive_sha256": receipt["raw_archive"]["sha256"],
            "predictions_sha256": file_sha256(predictions_path),
            "limitations": (
                "Only the one locally preserved 251-frame never-observed episode was "
                f"replayed on {device}."
            ),
            "config": config,
        }
        write(output / "results.json", result)
        logger.link_output(predictions_path)
        logger.link_output(output / "results.json")
        logger.record(
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            cuda_executed=device.type == "cuda",
            frames=frame_count,
            tracker_frozen=True,
            gnn_frozen=True,
        )
        logger.log(result["metrics_mm"], step=0)
    print(json.dumps(result, indent=2), flush=True)
