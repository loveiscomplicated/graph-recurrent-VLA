"""Paired size diagnostics and relative-error interventions on saved predictions.

No GT values enter a student model. Oracle substitutions operate on saved
outputs and hold the learned endpoint residual fixed unless explicitly removed.
"""

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import (
    ABLATION_MODELS, cluster_interval, load_aligned_predictions,
    observation_groups, statistics, correlation,
)
from graph.pose_geometry import peg_axis, hole_frame_vector
from utils.experiment_logger import ExperimentLogger


MODELS = ABLATION_MODELS


def group_masks(counts):
    """Keep original count thresholds and add total-count and published bins."""
    groups = observation_groups(counts, 4, 4)
    peg, box, hole = counts.T
    groups.update({
        "target_present": box + hole > 0,
        "peg_le4": peg <= 4, "peg_5_6": (peg >= 5) & (peg <= 6),
        "peg_ge7": peg >= 7,
    })
    for count in np.unique(counts.sum(1)):
        groups[f"total_vertices_{count}"] = counts.sum(1) == count
    return groups


def size_diagnostics(predictions, parents, repetitions, groups=None):
    """Summarize count strata, signed bias, size tracking and paired uncertainty."""
    baseline = predictions["baseline"]
    identifiers = baseline["sample_ids"]
    episodes = np.asarray([str(Path(value).parent) for value in identifiers])
    parent_ids = np.asarray([parents[episode] for episode in episodes]) if parents else None
    gt = baseline["target"][:, 17].astype(np.float64) * 1000
    result = {}
    if groups is None:
        groups = group_masks(baseline["counts"])
    for group, mask in groups.items():
        rows = {
            "frames": int(mask.sum()), "episodes": len(set(episodes[mask])),
            "parents": len(set(parent_ids[mask])) if parents else None,
            "gt_size_x_mm": statistics(gt[mask]), "models": {}, "paired": {},
        }
        for name, data in predictions.items():
            predicted = data["object21"][:, 17].astype(np.float64) * 1000
            signed = predicted - gt
            error = np.abs(signed)
            variance = np.var(gt[mask]) if mask.any() else 0
            slope = (np.cov(gt[mask], predicted[mask], ddof=0)[0, 1] / variance
                     if variance > 1e-12 else None)
            rows["models"][name] = {
                **statistics(error[mask]), "signed_bias_mm": float(signed[mask].mean()) if mask.any() else None,
                "overestimate_fraction": float((signed[mask] > 0).mean()) if mask.any() else None,
                "prediction_std_mm": float(predicted[mask].std()) if mask.any() else None,
                "gt_std_mm": float(gt[mask].std()) if mask.any() else None,
                "prediction_on_gt_slope": slope,
                "prediction_gt_pearson": correlation(predicted[mask], gt[mask]),
            }
            if name == "baseline":
                baseline_error = error
                continue
            delta = (error - baseline_error)[mask]
            rows["paired"][name] = {
                "candidate_minus_baseline_mm": statistics(delta),
                "episode_ci95": cluster_interval(delta, episodes[mask], repetitions),
                "parent_ci95": cluster_interval(delta, parent_ids[mask], repetitions) if parents else None,
            }
        result[group] = rows
    return result


def write_count_table(path, size):
    """Export complete count strata as a flat, reviewable CSV table."""
    columns = ["group", "frames", "episodes", "parents", "model", "mean_mm", "median_mm",
               "p90_mm", "p95_mm", "paired_mean_mm", "parent_ci_low", "parent_ci_high"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group, row in size.items():
            for model, values in row["models"].items():
                paired = row["paired"].get(model, {})
                interval = paired.get("parent_ci95") or {}
                writer.writerow(dict(
                    group=group, frames=row["frames"], episodes=row["episodes"], parents=row["parents"],
                    model=model, mean_mm=values["mean"], median_mm=values["median"],
                    p90_mm=values["p90"], p95_mm=values["p95"],
                    paired_mean_mm=paired.get("candidate_minus_baseline_mm", {}).get("mean"),
                    parent_ci_low=interval.get("low"), parent_ci_high=interval.get("high"),
                ))


def paired_metric_diagnostics(predictions, parents, repetitions):
    """Report paired pose/endpoint uncertainty on original observation strata."""
    baseline = predictions["baseline"]
    episodes = [str(Path(identifier).parent) for identifier in baseline["sample_ids"]]
    clusters = np.asarray([parents[episode] for episode in episodes] if parents else episodes)
    groups = group_masks(baseline["counts"])
    result = {}
    for group in ("all", "high_support_proxy", "low_support_proxy", "missing_target", "target_present"):
        mask = groups[group]
        result[group] = {}
        for name in MODELS[1:]:
            result[group][name] = {
                metric: cluster_interval((values - baseline[metric])[mask], clusters[mask], repetitions)
                for metric, values in predictions[name].items()
                if metric.endswith(("_error_mm", "_error_deg"))
            }
    return result


def endpoint_components(data):
    """Decompose geometric and corrected relative errors in world axes, in m."""
    prediction = data["object21"].astype(np.float64)
    target = data["target"].astype(np.float64)
    predicted_tip = prediction[:, :3] + peg_axis(torch.tensor(prediction[:, 3:7])).numpy() * prediction[:, 17:18]
    target_tip = target[:, :3] + peg_axis(torch.tensor(target[:, 3:7])).numpy() * target[:, 17:18]
    np.testing.assert_allclose(target_tip - target[:, 10:13], target[:, 7:10], atol=1e-6)
    geometric = predicted_tip - prediction[:, 10:13]
    np.testing.assert_allclose(geometric, data["endpoint_geom"], atol=1e-6)
    residual = data["endpoint_residual"].astype(np.float64)
    np.testing.assert_allclose(geometric + residual, prediction[:, 7:10], atol=1e-6)
    tip_error = predicted_tip - target_tip
    hole_error = prediction[:, 10:13] - target[:, 10:13]
    relative_error = tip_error - hole_error
    return dict(tip_error=tip_error, hole_error=hole_error,
                geometric_relative_error=relative_error, residual=residual,
                final_relative_error=relative_error + residual)


def relative_score(error, target):
    """Score existing relative, lateral and axial metrics without subtracting twice."""
    local = hole_frame_vector(torch.tensor(error), torch.tensor(target[:, 13:17], dtype=torch.float64)).numpy()
    return {
        "relative_3d_mm": statistics(np.linalg.norm(error, axis=1) * 1000),
        "lateral_mm": statistics(np.linalg.norm(local[:, 1:], axis=1) * 1000),
        "axial_mm": statistics(np.abs(local[:, 0]) * 1000),
    }


def oracle_replacements(data):
    """Replace one output component and keep the original residual fixed.

    These are output-space diagnostics; the residual head is never re-evaluated
    with oracle values. The separate no_residual intervention sets it to zero.
    """
    prediction = data["object21"].astype(np.float64)
    target = data["target"].astype(np.float64)
    target_tip = target[:, :3] + peg_axis(torch.tensor(target[:, 3:7])).numpy() * target[:, 17:18]
    result = {}
    for condition in ("original", "gt_peg_position", "gt_peg_rotation", "gt_peg_size",
                      "gt_hole_position", "gt_peg_endpoint", "no_residual"):
        replaced = prediction.copy()
        residual = data["endpoint_residual"].astype(np.float64).copy()
        slices = {"gt_peg_position": slice(0, 3), "gt_peg_rotation": slice(3, 7),
                  "gt_peg_size": slice(17, 20), "gt_hole_position": slice(10, 13)}
        if condition in slices:
            selected = slices[condition]
            replaced[:, selected] = target[:, selected]
        if condition == "no_residual":
            residual[:] = 0
        endpoint = replaced[:, :3] + peg_axis(torch.tensor(replaced[:, 3:7])).numpy() * replaced[:, 17:18]
        if condition == "gt_peg_endpoint":
            endpoint = target_tip
        error = endpoint - replaced[:, 10:13] + residual - target[:, 7:10]
        result[condition] = relative_score(error, target)
    return result


def relative_diagnostics(predictions):
    """Check the squared-error identity and separate direction/residual terms."""
    result = {}
    vectors = {}
    for name, data in predictions.items():
        components = endpoint_components(data)
        vectors[name] = components
        endpoint = components["tip_error"]
        hole = components["hole_error"]
        relative = components["geometric_relative_error"]
        residual = components["residual"]
        final = components["final_relative_error"]
        endpoint_squared = np.sum(endpoint ** 2, axis=1)
        hole_squared = np.sum(hole ** 2, axis=1)
        dot = np.sum(endpoint * hole, axis=1)
        relative_squared = np.sum(relative ** 2, axis=1)
        np.testing.assert_allclose(relative_squared, endpoint_squared + hole_squared - 2 * dot, atol=1e-12)
        denom = np.linalg.norm(endpoint, axis=1) * np.linalg.norm(hole, axis=1)
        supported = denom > 1e-12
        cosine = np.clip(dot[supported] / denom[supported], -1, 1)
        squared = {
            "endpoint": endpoint_squared, "hole": hole_squared,
            "minus_2_endpoint_dot_hole": -2 * dot,
            "geometric_relative": relative_squared,
            "residual": np.sum(residual ** 2, axis=1),
            "plus_2_relative_dot_residual": 2 * np.sum(relative * residual, axis=1),
            "final_relative": np.sum(final ** 2, axis=1),
        }
        np.testing.assert_allclose(squared["final_relative"], relative_squared + squared["residual"] +
                                   squared["plus_2_relative_dot_residual"], atol=1e-12)
        result[name] = {
            "mean_squared_mm2": {key: float(value.mean() * 1e6) for key, value in squared.items()},
            "endpoint_hole_dot_mm2": statistics(dot * 1e6),
            "endpoint_hole_cosine": statistics(cosine),
            "endpoint_hole_aligned_fraction": float((cosine > 0).mean()),
            "endpoint_hole_angle_deg": statistics(np.rad2deg(np.arccos(cosine))),
            "geometry": relative_score(relative, data["target"]),
            "final": relative_score(final, data["target"]),
            "oracle_fixed_residual": oracle_replacements(data),
        }
        missing = data["counts"][:, 1:].sum(axis=1) == 0
        if missing.any():
            predicted_hole = data["object21"][missing, 10:13].astype(np.float64)
            target_hole = data["target"][missing, 10:13].astype(np.float64)
            result[name]["missing_target_hole"] = {
                "frames": int(missing.sum()),
                "unique_predicted_positions": len(np.unique(predicted_hole, axis=0)),
                "prediction_xyz_mean_mm": (predicted_hole.mean(axis=0) * 1000).tolist(),
                "prediction_xyz_range_mm": (np.ptp(predicted_hole, axis=0) * 1000).tolist(),
                "target_xyz_std_mm": (target_hole.std(axis=0) * 1000).tolist(),
                "fraction_of_total_hole_squared_error": float(hole_squared[missing].sum() / hole_squared.sum()),
                "fraction_of_total_hole_absolute_error": float(
                    np.sqrt(hole_squared[missing]).sum() / np.sqrt(hole_squared).sum()),
            }
    return result, vectors


def selected_cases(predictions, per_direction=5):
    """Select extreme high-support changes, retaining one case per episode."""
    baseline = predictions["baseline"]
    mask = group_masks(baseline["counts"])["high_support_proxy"]
    delta = predictions["combined"]["peg_half_size_x_error_mm"] - baseline["peg_half_size_x_error_mm"]
    rows = []
    for direction, sign in (("worse", -1), ("better", 1)):
        order = np.argsort(sign * delta)
        seen = set()
        for index in order:
            identifier = str(baseline["sample_ids"][index])
            episode = str(Path(identifier).parent)
            if not mask[index] or episode in seen:
                continue
            seen.add(episode)
            rows.append({
                "direction": direction, "sample_id": identifier, "index": int(index),
                "counts": baseline["counts"][index].tolist(),
                "gt_size_x_mm": float(baseline["target"][index, 17] * 1000),
                "predicted_size_x_mm": {name: float(data["object21"][index, 17] * 1000)
                                        for name, data in predictions.items()},
                "ab_minus_baseline_error_mm": float(delta[index]),
                "gt_peg_tcp_relative_pose": baseline["target"][index, :7].tolist(),
                "gt_hole_tcp_relative_pose": baseline["target"][index, 10:17].tolist(),
                "gt_peg_half_size_m": baseline["target"][index, 17:20].tolist(),
                "predicted_peg_tcp_relative_pose": {
                    name: data["object21"][index, :7].tolist() for name, data in predictions.items()
                },
                "predicted_hole_tcp_relative_pose": {
                    name: data["object21"][index, 10:17].tolist() for name, data in predictions.items()
                },
            })
            if len(seen) == per_direction:
                break
    return rows


def analyze(config):
    """Run saved-output diagnostics with a separate local experiment identity."""
    effective = deepcopy(config)
    predictions, hashes = load_aligned_predictions(config["predictions"])
    parents = None
    if config.get("parent_manifest"):
        manifest = json.loads(Path(config["parent_manifest"]).read_text())
        parents = manifest["parent_by_episode"]
    effective["prediction_sha256"] = hashes
    with ExperimentLogger(team="graph", config=effective, seed=config.get("seed", 0),
                          model="vertex_gnn_diagnostics", job_type="evaluate") as experiment:
        output = Path(config["output"]["root"]).resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        size = size_diagnostics(predictions, parents, config.get("bootstrap_repetitions", 2000))
        relative, vectors = relative_diagnostics(predictions)
        write_json(output / "size_diagnostics.json", size)
        write_count_table(output / "count_metrics.csv", size)
        write_json(output / "relative_diagnostics.json", relative)
        write_json(output / "paired_metrics.json", paired_metric_diagnostics(
            predictions, parents, config.get("bootstrap_repetitions", 2000)))
        write_json(output / "selected_cases.json", selected_cases(predictions))
        write_json(output / "effective_config.json", effective)
        np.savez_compressed(output / "error_vectors.npz", sample_ids=predictions["baseline"]["sample_ids"],
                            **{f"{model}_{key}": value for model, values in vectors.items()
                               for key, value in values.items()})
        for path in output.iterdir():
            experiment.link_output(path, kind="metrics")
        experiment.record(prediction_sha256=hashes, frames=len(predictions["baseline"]["sample_ids"]))
        print(f"Vertex diagnostics: {output}", flush=True)
        return output
