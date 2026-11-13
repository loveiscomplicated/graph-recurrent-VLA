"""Causal H1 feature replay, fixed development selection, and paired diagnostics."""

import numpy as np

from evaluation.gnn.temporal.hybrid_diagnostics import (
    paired_bootstrap,
    vector_relationship,
)
from evaluation.gnn.temporal.insertion import distribution


def selection_score(method, baseline, settings):
    """Apply preregistered pose/axial/motion guards before a fixed geometry score.

    Baseline refers to H1 on the same fine-tuning development parents. The old
    126 episodes and final test never enter this function during selection.
    """
    current, original = method["groups"]["all"], baseline["groups"]["all"]
    failures = []
    for metric, unit, key in (
        ("peg_position", "mm", "peg_position_tolerance_mm"),
        ("peg_rotation", "deg", "peg_rotation_tolerance_deg"),
        ("axial", "mm", "axial_tolerance_mm"),
    ):
        if current[metric]["mean_" + unit] > original[metric]["mean_" + unit] + settings[key]:
            failures.append(metric)
    moving = method["groups"]["peg_moving"]["peg_position"]["mean_mm"]
    moving_original = baseline["groups"]["peg_moving"]["peg_position"]["mean_mm"]
    if moving is not None and moving > moving_original + settings["moving_peg_tolerance_mm"]:
        failures.append("moving_peg_position")
    weights = settings["score_weights"]
    score = sum(
        weights[name] * current[metric][statistic + "_mm"]
        for name, metric, statistic in (
            ("relative_mean", "relative_3d", "mean"),
            ("lateral_mean", "lateral", "mean"),
            ("relative_p95", "relative_3d", "p95"),
            ("lateral_p95", "lateral", "p95"),
        )
    )
    return float(score), failures


def analyze_finetuning(data, predictions, errors, manifest, *, seed, resamples):
    """Compare endpoint/Hole vectors, tails, motion, and parent-paired improvements."""
    target = data["target"].astype(np.float64)
    results = {}
    for name, prediction in predictions.items():
        prediction = prediction.astype(np.float64)
        hole_error = prediction[:, 10:13] - target[:, 10:13]
        endpoint_error = (
            prediction[:, 7:10] + prediction[:, 10:13] - target[:, 7:10] - target[:, 10:13]
        )
        relative_error = prediction[:, 7:10] - target[:, 7:10]
        np.testing.assert_allclose(relative_error, endpoint_error - hole_error, atol=1e-12)
        cross = -2 * np.sum(endpoint_error * hole_error, axis=1) * 1e6
        identity = np.sum(endpoint_error**2 + hole_error**2, axis=1) * 1e6 + cross
        np.testing.assert_allclose(identity, np.sum(relative_error**2, axis=1) * 1e6, atol=1e-8)
        result = {
            "endpoint_position": distribution(np.linalg.norm(endpoint_error, axis=1) * 1000, "mm"),
            "endpoint_bias_xyz_mm": (endpoint_error.mean(0) * 1000).tolist(),
            "peg_bias_xyz_mm": ((prediction[:, :3] - target[:, :3]).mean(0) * 1000).tolist(),
            "error_cross_term_mean_mm2": float(cross.mean()),
            "endpoint_hole_relationship": vector_relationship(endpoint_error, hole_error),
        }
        if name != "original_h1":
            baseline = predictions["original_h1"]
            np.testing.assert_array_equal(prediction[:, 10:17], baseline[:, 10:17])
            result["hole_pose_bitwise_unchanged"] = True
            result["paired"] = {}
            for metric in ("relative_3d", "lateral", "axial", "peg_position"):
                delta = errors[name][metric] - errors["original_h1"][metric]
                result["paired"][metric] = {
                    "improved_frame_percent": float(np.mean(delta < -1e-6) * 100),
                    "worsened_frame_percent": float(np.mean(delta > 1e-6) * 100),
                    "parent_bootstrap": paired_bootstrap(data, delta, manifest, seed, resamples),
                }
        failure_episode = "shard_000022/episode_000016"
        mask = data["episode"] == failure_episode
        result["historical_failure_episode"] = {
            "episode": failure_episode,
            "present": bool(mask.any()),
            "metrics": {
                metric: distribution(values[mask], "deg" if "rotation" in metric else "mm")
                for metric, values in errors[name].items()
            }
            if mask.any()
            else None,
        }
        results[name] = result
    return results
