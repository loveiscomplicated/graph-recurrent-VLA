"""Paired replay of fixed Box exterior and Peg position corrections."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.trajectory_rim_recovery import (
    LongSpanDetector,
    ShortSpanDetector,
    depth_ranked_rims,
    load_frame,
)
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, score, visible_reference
from visual.corner_detection.hole_rim import verified_long_rim
from visual.corner_detection.rim_memory import StaticRimMemory, replace_rim_square


def initialize(variant="exterior", *, recover_peg_rear_end=False):
    """Prepare paired detector paths with the existing Peg and rim settings."""
    global baseline, corrected_detector, rim_alternatives
    torch.set_num_threads(1)
    options = dict(
        device="cpu",
        recover_hole_rim=True,
        recover_small_geometry=True,
        recover_fragmented_rim=True,
    )
    compare_peg = variant != "exterior"
    compare_rim = variant in ("hole_depth_fallback", "hole_rim_replacement")
    if recover_peg_rear_end:
        if variant != "hole_rim_replacement":
            raise ValueError("The main preset requires the hole_rim_replacement comparison")
        baseline = TorchCornerDetector.main(device="cpu")
    else:
        baseline = TorchCornerDetector(
            **options,
            correct_box_exterior_pixel_center=compare_peg,
            correct_peg_position=variant == "peg_plane_center" or compare_rim,
            center_peg_plane_bounds=compare_rim,
            verify_rejected_rim_squares=variant == "hole_rim_replacement",
        )
    corrected_detector = TorchCornerDetector(
        **options,
        correct_box_exterior_pixel_center=True,
        correct_peg_position=compare_peg,
        center_peg_plane_bounds=variant == "peg_plane_center" or compare_rim,
        verify_rejected_rim_squares=compare_rim,
    )
    rim_alternatives = [LongSpanDetector(**options), ShortSpanDetector(**options)]


def stage_counts(references, baseline_result, ext):
    """Locate initially missed exterior vertices before and after validation."""
    target = np.asarray(
        [p["xyz"] for p in references if p["kind"] == "box_outer"]
    ).reshape(-1, 3)
    target = target @ ext[:, :3].T + ext[:, 3]
    distance = np.full((3, len(target)), np.inf)
    for points, _, oid, flags, _ in baseline.blocks:
        if oid != 1:
            continue
        error = np.linalg.norm(points.numpy()[:, None] - target, axis=-1)
        distance[0] = np.minimum(distance[0], error.min(0))
        if flags.any():
            distance[1] = np.minimum(distance[1], error[flags.numpy()].min(0))
    points = baseline_result.camera_xyz[baseline_result.object_id == 1].numpy()
    if len(points):
        distance[2] = np.linalg.norm(points[:, None] - target, axis=-1).min(0)
    return {
        "raw_within_3mm": int((distance[0] <= 0.003).sum()),
        "validated_within_3mm": int((distance[1] <= 0.003).sum()),
        "merged_within_3mm": int((distance[2] <= 0.003).sum()),
        "raw_within_6mm": int((distance[0] <= 0.006).sum()),
    }


def metrics(predictions, references, physical):
    """Return one-to-one 3 mm counts for all corner families."""
    peg, box = score(predictions, references, physical)
    return {
        "outer_matches": box["by_kind"].get("box_outer", {}).get("matched", 0),
        "outer_reference": box["by_kind"].get("box_outer", {}).get("visible", 0),
        "rim_matches": box["by_kind"].get("hole_rim", {}).get("matched", 0),
        "rim_reference": box["by_kind"].get("hole_rim", {}).get("visible", 0),
        "peg_matches": peg["matched_3mm"],
        "peg_reference": peg["visible_reference"],
        "peg_unmatched": peg["predictions"] - peg["matched_3mm"],
        "box_unmatched": box["predictions"] - box["matched_3mm"],
        "corrected_exterior_predictions": sum(
            p.get("box_exterior_pixel_center_corrected", False) for p in predictions
        ),
    }


def rim_miss_stages(detector, predictions, references, physical, extrinsic):
    """Classify each final rim miss against the primary detector's candidates."""
    box_refs = [ref for ref in references if ref["object_id"] == 1]
    box_score = score(predictions, references, physical)[1]
    matched = {match["reference"] for match in box_score["matches"]}
    missing = np.asarray(
        [
            ref["xyz"]
            for i, ref in enumerate(box_refs)
            if ref["kind"] == "hole_rim" and i not in matched
        ]
    ).reshape(-1, 3)
    target = missing @ extrinsic[:, :3].T + extrinsic[:, 3]
    generated = np.zeros(len(target), bool)
    verified = np.zeros_like(generated)
    for points, _, oid, flags, _ in detector.blocks:
        if oid != 1:
            continue
        nearby = np.linalg.norm(points.numpy()[:, None] - target, axis=-1) <= 0.003
        generated |= nearby.any(axis=0)
        verified |= nearby[flags.numpy()].any(axis=0)
    return {
        "generation": int((~generated).sum()),
        "verification": int((generated & ~verified).sum()),
        "merge_and_recovery": int(verified.sum()),
    }


def score_episode(job):
    """Compare both paths on identical frames and identical supplemental rims."""
    root, cohort, rows, variant = job
    compare_peg = variant != "exterior"
    result_rows = []
    memory = StaticRimMemory()
    improved_memory = StaticRimMemory()
    for row in sorted(rows, key=lambda item: item["frame"]):
        sensor, gt, grasp = load_frame(
            root, row, require_grasp=compare_peg and cohort == "trajectory"
        )
        args = tuple(
            sensor[key] for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")
        )
        base_result = baseline(*args)
        base = base_result.as_dict()["corners"]
        fitted_result = base_result if variant == "hole_rim_replacement" else corrected_detector(*args)
        fitted = fitted_result.as_dict()["corners"]
        if cohort == "trajectory":
            supplemental = [
                point
                for detector in rim_alternatives
                for point in detector(*args).as_dict()["corners"]
                if point["method"] == "box_opening"
            ]
            recovered = memory.update(base_result, *args[1:], supplemental)
            ranked = depth_ranked_rims(base, recovered, base_result, *args[1:])
            if variant == "hole_depth_fallback":
                improved = improved_memory.update(fitted_result, *args[1:], supplemental)
                fitted = depth_ranked_rims(fitted, improved, fitted_result, *args[1:])
            else:
                fitted.extend(ranked[len(base) :])
            base = ranked
            if variant == "hole_rim_replacement":
                box = baseline.prepare(1, base_result.masks[1])
                square = verified_long_rim(
                    baseline.depth.numpy(), box.support_cpu, box.planes_cpu,
                    box.eligible_cpu, baseline.k_cpu,
                )
                fitted = replace_rim_square(base, square, *args[2:])
        ids = {int(key): value for key, value in row["actor_ids"].items()}
        references = visible_reference(gt, sensor, ids)
        physical = corners(gt)
        if cohort == "reset":
            for ref in references + physical:
                if ref["kind"] == "hole_rim" and ref["corner_id"] >= 4:
                    ref["kind"] = "hole_rear"
        if variant == "hole_rim_replacement":
            stage_metrics = rim_miss_stages(baseline, base, references, physical, args[-1])
        else:
            stage_metrics = stage_counts(references, base_result, args[-1])
        result_rows.append(
            {
                "seed": row["seed"],
                "frame": row["frame"],
                "mode": row.get("mode"),
                "grasp": grasp,
                "stages": stage_metrics,
                "baseline": metrics(base, references, physical),
                variant: metrics(fitted, references, physical),
            }
        )
    return result_rows


def prepare_holdout(config, output):
    """Re-render saved reset seeds and recover actor IDs for saved rollouts."""
    from data_collection.mixed_trajectory_collection import validate_camera
    from data_collection.storage import save_frame
    from envs.maniskill import get_gt_state
    from evaluation.hole_rim_holdout import actor_ids, make_environment

    protocol_path = Path(config["protocol"])
    protocol = json.loads(protocol_path.read_text())
    root = Path(config["dataset"])
    tuning = Path(config["tuning_dataset"])
    cohorts = ("reset", "trajectory")
    manifests = {cohort: root / f"{cohort}.json" for cohort in cohorts}
    rows = {
        cohort: json.loads(path.read_text())["rows"] for cohort, path in manifests.items()
    }
    tuning_sources = [tuning / f"{cohort}.json" for cohort in cohorts]
    tuning_seeds = {
        row["seed"]
        for path in tuning_sources
        for row in json.loads(path.read_text())["rows"]
    }
    holdout_seeds = {row["seed"] for cohort in cohorts for row in rows[cohort]}
    if holdout_seeds & tuning_seeds:
        raise ValueError("Box exterior holdout overlaps the tuning seeds")

    seeds = sorted(holdout_seeds)
    reset_rows = {row["seed"]: row for row in rows["reset"]}
    env = make_environment({"seed": seeds[0]}, protocol)
    ids = {}
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            validate_camera(observation, protocol["camera"])
            ids[seed] = actor_ids(env)
            if seed not in reset_rows:
                continue
            path = output / f"reset_{seed}.npz"
            save_frame(path, observation, ground_truth=vars(get_gt_state(env)))
            reset_rows[seed].update(
                path=str(path.resolve()), sha256=file_sha256(path)
            )
    finally:
        env.close()

    for row in rows["trajectory"]:
        row["path"] = str((root / row["path"]).resolve())
    sources = [protocol_path, *manifests.values(), *tuning_sources]
    selection = {
        "source_sha256": {str(path): file_sha256(path) for path in sources},
        "cohorts": {},
    }
    for cohort in cohorts:
        selection["cohorts"][cohort] = [
            {
                **{key: value for key, value in row.items() if key != "scores"},
                "actor_ids": ids[row["seed"]],
            }
            for row in rows[cohort]
        ]
    path = output / "selection.json"
    write_json(path, selection)
    return path, selection["cohorts"]


def evaluate(config):
    """Replay frozen reset and trajectory manifests and record paired counts."""
    root = Path(config["dataset"])
    variant = config.get("comparison", "exterior")
    if config.get("compare_peg_position", False):
        variant = "peg_position"
    if variant not in ("exterior", "peg_position", "peg_plane_center", "hole_depth_fallback", "hole_rim_replacement"):
        raise ValueError(f"Unsupported corner correction comparison: {variant}")
    compare_peg = variant != "exterior"
    with ExperimentLogger(
        team="graph",
        model=variant if compare_peg else "box_exterior_recovery",
        config=config,
        seed=20261114,
        job_type="evaluation",
    ) as logger, ProcessPoolExecutor(
        max_workers=config["workers"],
        initializer=partial(initialize, variant=variant),
    ) as pool:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        selection = None
        if "tuning_dataset" in config:
            selection, holdout = prepare_holdout(config, output)
            logger.link_output(selection)
        if "selection_manifest" in config:
            selection = Path(config["selection_manifest"])
            holdout = json.loads(selection.read_text())["cohorts"]
        cohorts = ("trajectory",) if compare_peg else ("reset", "trajectory")
        for cohort in cohorts:
            if selection is None:
                source = root / config.get(f"{cohort}_manifest", f"{cohort}.json")
                inputs = json.loads(source.read_text())["rows"]
            else:
                source = selection
                inputs = holdout[cohort]
            groups = defaultdict(list)
            for row in inputs:
                groups[(row["seed"], row.get("mode"))].append(row)
            jobs = ((root, cohort, frames, variant) for frames in groups.values())
            rows = [row for episode in pool.map(score_episode, jobs) for row in episode]
            totals = {name: Counter() for name in ("stages", "baseline", variant)}
            for row in rows:
                for name, counts in totals.items():
                    counts.update(row[name])
            path = output / f"{cohort}.json"
            source_hash = file_sha256(source)
            write_json(
                path,
                {
                    "frames": len(rows),
                    "source_sha256": source_hash,
                    "totals": totals,
                    "rows": rows,
                },
            )
            logger.link_output(path)
            logger.record(**{cohort: totals, f"{cohort}_source_sha256": source_hash})
            print(path, json.dumps(totals), flush=True)
