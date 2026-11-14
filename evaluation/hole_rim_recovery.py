"""Compare Hole opening recovery on identical reset and pilot observations."""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import load_cohort, verify_camera
from evaluation.pilot_corner_detection import sample_frames
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, score, visible_reference


class PreviousOpeningDetector(TorchCornerDetector):
    """Replay the original opening fit and append-only merge for comparison."""

    _refine_opening = False


def initialize(trace_points=False):
    """Keep detector workspace and thread pools local to each process."""
    global detectors, record_points
    record_points = trace_points
    torch.set_num_threads(1)
    detectors = {
        "baseline": TorchCornerDetector("cpu"),
        "opening": PreviousOpeningDetector("cpu", recover_hole_rim=True),
        "refined": TorchCornerDetector("cpu", recover_hole_rim=True),
    }


def coverage(targets, points):
    """Count reference vertices with a same-object candidate within 3 mm."""
    points = np.asarray(points).reshape(-1, 3)
    return int(
        (np.linalg.norm(targets[:, None] - points[None], axis=2) <= 0.003).any(1).sum()
    )


def opening_misses(result, box_score, references, all_references):
    """Classify unmatched opening predictions using the existing assignment."""
    box_indices = (result.object_id == 1).nonzero().flatten().tolist()
    matched = {match["prediction"] for match in box_score["matches"]}
    visible = np.array(
        [ref["xyz"] for ref in references if ref["object_id"] == 1]
    ).reshape(-1, 3)
    physical = np.array([ref["xyz"] for ref in all_references if ref["object_id"] == 1])
    counts = Counter()
    for index, tensor_index in enumerate(box_indices):
        if result.metadata[tensor_index]["method"] != "box_opening":
            continue
        point = result.xyz[tensor_index].numpy()
        distance = np.linalg.norm(physical - point, axis=1).min()
        if index in matched:
            reason = "matched"
        elif (np.linalg.norm(visible - point, axis=1) <= 0.003).any():
            reason = "duplicate_supported"
        elif distance <= 0.003:
            reason = "unsupported_physical"
        elif distance <= 0.005:
            reason = "offset_3_to_5_mm"
        elif distance <= 0.010:
            reason = "offset_5_to_10_mm"
        else:
            reason = "offset_over_10_mm"
        counts[f"opening_{reason}"] += 1
    return counts


def evaluate_pair(sensor, gt, ids, *, front_only=False):
    """Measure cumulative candidate coverage and final one-to-one recall."""
    references = visible_reference(gt, sensor, ids)
    all_references = corners(gt)
    if front_only:
        for ref in references + all_references:
            if ref["kind"] == "hole_rim" and ref["corner_id"] >= 4:
                ref["kind"] = "hole_rear"
    world = np.array(
        [ref["xyz"] for ref in references if ref["kind"] == "hole_rim"]
    ).reshape(-1, 3)
    ext = sensor["extrinsic_cv"]
    targets = world @ ext[:, :3].T + ext[:, 3]
    outputs = {}
    for name, detector in detectors.items():
        result = detector(sensor["rgb"], sensor["depth"], sensor["intrinsic_cv"], ext)
        proposed, verified = [], []
        opening_proposed, opening_rectangular, opening_verified = [], [], []
        for row in detector.rows:
            points, _, oid, flags, _ = detector.blocks[row["block"]]
            if oid != 1:
                continue
            index = row["row"]
            point = points[index].numpy()
            accepted = bool(flags[index])
            proposed.append(point)
            if accepted:
                verified.append(point)
            meta = row["meta"]
            if meta["method"] == "box_opening":
                opening_proposed.append(point)
                if meta["rectangular"]:
                    opening_rectangular.append(point)
                if accepted:
                    opening_verified.append(point)
        objects = score(result.as_dict()["corners"], references, all_references)
        box = objects[1]
        outputs[name] = {
            "rim_reference": len(targets),
            "proposed": coverage(targets, proposed),
            "verified": coverage(targets, verified),
            "after_merge": coverage(
                targets, result.camera_xyz[result.object_id == 1].numpy()
            ),
            "opening_proposed": coverage(targets, opening_proposed),
            "opening_rectangular": coverage(targets, opening_rectangular),
            "opening_depth_verified": coverage(targets, opening_verified),
            "rim_matched": box["by_kind"].get("hole_rim", {}).get("matched", 0),
            "outer_matched": box["by_kind"].get("box_outer", {}).get("matched", 0),
            "outer_reference": box["by_kind"].get("box_outer", {}).get("visible", 0),
            "peg_matched": objects[0]["matched_3mm"],
            "box_predictions": box["predictions"],
            "box_unmatched": box["predictions"] - box["matched_3mm"],
        }
        if name != "baseline":
            outputs[name].update(
                opening_misses(result, box, references, all_references)
            )
        if name == "baseline":
            baseline = result
        elif name == "opening":
            previous_opening = result
            count = len(baseline.xyz)
            for field in ("camera_xyz", "uv", "object_id"):
                torch.testing.assert_close(
                    getattr(result, field)[:count],
                    getattr(baseline, field),
                    atol=0,
                    rtol=0,
                )
            # The world transform can pick a different matmul kernel as N grows.
            torch.testing.assert_close(
                result.xyz[:count], baseline.xyz, atol=2e-7, rtol=0
            )
    if record_points:
        result = previous_opening
        box = result.object_id == 1
        outputs["trace"] = {
            "xyz": result.xyz[box].tolist(),
            "uv": result.uv[box].tolist(),
            "method": [m["method"] for m, keep in zip(result.metadata, box) if keep],
            "concave": [
                bool(m.get("contour_hole"))
                for m, keep in zip(result.metadata, box)
                if keep
            ],
            "references": [
                [ref["kind"], *ref["xyz"].tolist()]
                for ref in references
                if ref["object_id"] == 1
            ],
            "all_references": [
                [ref["kind"], *ref["xyz"].tolist()]
                for ref in all_references
                if ref["object_id"] == 1
            ],
        }
    return outputs


def reset_results(config):
    """Render the full original reset cohort and verify calibration and references."""
    peg, hole = load_cohort(config)
    env = make_peg_insertion_env(device="cpu", obs_mode="rgb+depth+segmentation")
    try:
        for seed in sorted(peg):
            obs, _ = env.reset(seed=seed)
            sensor = {
                key: value[0].cpu().numpy()
                for key, value in obs["sensor_data"]["base_camera"].items()
            }
            calibration = {
                key: value[0].cpu().numpy()
                for key, value in obs["sensor_param"]["base_camera"].items()
            }
            verify_camera(sensor, calibration, config)
            sensor.update(calibration)
            gt = {
                key: value[0].cpu().numpy()
                for key, value in vars(get_gt_state(env)).items()
            }
            np.testing.assert_allclose(
                gt["peg_pose"], peg[seed]["gt_peg_pose"], atol=1e-6, rtol=0
            )
            np.testing.assert_allclose(
                gt["peg_half_sizes"], peg[seed]["peg_half_sizes"], atol=1e-6, rtol=0
            )
            front = [
                ref["xyz"]
                for ref in corners(gt)
                if ref["kind"] == "hole_rim" and ref["corner_id"] < 4
            ]
            ext = calibration["extrinsic_cv"]
            camera = np.asarray(front) @ ext[:, :3].T + ext[:, 3]
            uv = camera @ calibration["intrinsic_cv"].T
            np.testing.assert_allclose(
                uv[:, :2] / uv[:, 2:],
                [r["uv"] for r in hole[seed]["corners"]],
                atol=1e-3,
                rtol=0,
            )
            np.testing.assert_allclose(
                camera[:, 2],
                [r["camera_depth_m"] for r in hole[seed]["corners"]],
                atol=1e-5,
                rtol=0,
            )
            task = env.unwrapped
            ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
            yield {
                "seed": seed,
                "scores": evaluate_pair(sensor, gt, ids, front_only=True),
            }
    finally:
        env.close()


def pilot_frame(job):
    """Replay one checksum-verified RGB-D frame from the pilot manifest."""
    path, checksum, ids, labels = job
    if file_sha256(path) != checksum:
        raise ValueError(f"Pilot frame checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as frame:
        sensor = {
            name: frame[f"observation/sensor_data/base_camera/{name}"][0]
            for name in ("rgb", "depth", "segmentation")
        }
        sensor.update(
            {
                name: frame[f"observation/sensor_param/base_camera/{name}"][0]
                for name in ("intrinsic_cv", "extrinsic_cv")
            }
        )
        gt = {
            name: frame[f"ground_truth/{name}"][0]
            for name in (
                "peg_pose",
                "box_pose",
                "hole_pose",
                "peg_half_sizes",
                "hole_radius",
            )
        }
    return dict(**labels, scores=evaluate_pair(sensor, gt, ids))


def pilot_jobs(root, evidence):
    """Use the original sample rule and the previously verified actor IDs."""
    manifest_path = root / "manifest.json"
    if file_sha256(manifest_path) != evidence["source_manifest_sha256"]:
        raise ValueError("Pilot manifest differs from the baseline evaluation")
    manifest = json.loads(manifest_path.read_text())
    if (
        evidence["actor_id_reset_checks"] != 100
        or len(evidence["actor_id_values"]) != 1
    ):
        raise ValueError(
            "Expected one pair of actor IDs verified across all pilot resets"
        )
    peg, box = evidence["actor_id_values"][0]
    for row in manifest["episodes"]:
        for frame in sample_frames(row):
            yield (
                root / row["path"] / f"frame_{frame:06d}.npz",
                row["frame_sha256"][frame],
                {0: peg, 1: box},
                {"seed": row["seed"], "mode": row["mode"], "frame": frame},
            )


def collect_scores(results, total):
    """Keep a progress count for long reset and replay evaluations."""
    rows = []
    for row in results:
        rows.append(row)
        if len(rows) % 100 == 0 or len(rows) == total:
            print(f"Paired corner evaluation {len(rows)}/{total}", flush=True)
    return rows


def summarize(rows):
    """Sum common and recovery-specific stages over the same reference set."""
    totals = {name: Counter() for name in detectors}
    for row in rows:
        for name, counts in totals.items():
            counts.update(row["scores"][name])
    return totals


def evaluate(config):
    """Replay both cohorts, verify baseline counts per frame and save paired results."""
    trace_points = config.get("trace_points", False)
    initialize(trace_points)
    cohort = config["cohort"]
    evidence_path = Path(config["baseline"])
    evidence = json.loads(evidence_path.read_text())
    with ExperimentLogger(
        team="graph",
        model="hole_rim_recovery",
        seed=200,
        config=config,
        job_type="evaluation",
    ) as logger:
        if cohort == "reset":
            original = {
                (row["seed"],): row["scores"]["3"]["families"]
                for row in evidence["rows"]
            }
            labels = ("seed",)
            families = ("hole_front", "box_outer", "peg")
            rows = collect_scores(reset_results(config["reset"]), len(original))
        elif cohort == "pilot":
            original = {
                (row["seed"], row["mode"], row["frame"]): {
                    key: value
                    for obj in row["scores"]["0.003"]
                    for key, value in obj["by_kind"].items()
                }
                for row in evidence["rows"]
            }
            labels = ("seed", "mode", "frame")
            families = ("hole_rim", "box_outer", "peg")
            with ProcessPoolExecutor(
                max_workers=config["workers"],
                initializer=initialize,
                initargs=(trace_points,),
            ) as pool:
                results = pool.map(
                    pilot_frame,
                    pilot_jobs(Path(config["dataset"]), evidence),
                    chunksize=4,
                )
                rows = collect_scores(results, len(original))
        else:
            raise ValueError("cohort must be reset or pilot")
        if len(rows) != len(original):
            raise ValueError("Paired evaluation frame count differs from baseline")
        for row in rows:
            key = tuple(row[label] for label in labels)
            ref = original.pop(key)
            actual = row["scores"]["baseline"]
            for family, prefix in zip(families, ("rim", "outer", "peg"), strict=True):
                if actual[f"{prefix}_matched"] != ref.get(family, {}).get("matched", 0):
                    raise ValueError(
                        f"Baseline {family} match count differs at frame {key}"
                    )
        totals = summarize(rows)
        output = Path(config["output_root"]) / logger.run_id / "results.json"
        output.parent.mkdir(parents=True, exist_ok=False)
        write_json(
            output,
            {
                "config": config,
                "baseline_sha256": file_sha256(evidence_path),
                "frames": len(rows),
                "totals": totals,
                "rows": rows,
            },
        )
        logger.record(frames=len(rows), totals=totals)
        logger.link_output(output)
        print(output, json.dumps(totals), flush=True)
