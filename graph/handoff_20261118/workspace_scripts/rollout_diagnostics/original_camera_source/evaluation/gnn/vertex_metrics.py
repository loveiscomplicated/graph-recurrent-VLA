"""Physical error and paired visibility diagnostics for vertex GNN ablations."""

from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.pose_geometry import (
    angular_error_rad, hole_frame_vector,
)
from training.gnn.tasks import geometry_errors


def per_frame_errors(output, target):
    """Score world-axis Object21; use GT hole rotation only for error projection.

    Object21[7:10] already is endpoint minus hole. Physical endpoint position is
    that vector plus hole position, so the two errors must be distinguished.
    """
    prediction = output["object21"]
    errors = {}
    for name, values in geometry_errors(prediction, target).items():
        if name == "endpoint_error_m":
            continue
        key = name.removesuffix("_m") + "_mm" if name.endswith("_m") else name
        errors[key] = values * 1000 if name.endswith("_m") else values
    for name, start in (("peg_rotation_error_deg", 3), ("hole_rotation_error_deg", 13)):
        errors[name] = torch.rad2deg(angular_error_rad(
            prediction[:, start:start + 4], target[:, start:start + 4],
        ))
    relative_error = prediction[:, 7:10] - target[:, 7:10]
    endpoint_error = relative_error + prediction[:, 10:13] - target[:, 10:13]
    geometry_error = output["endpoint_geom"] - target[:, 7:10]
    geometry_position_error = geometry_error + prediction[:, 10:13] - target[:, 10:13]
    local_error = hole_frame_vector(relative_error, target[:, 13:17])
    vectors = {
        "endpoint_position_error_mm": endpoint_error,
        "relative_3d_error_mm": relative_error,
        "endpoint_geom_relative_error_mm": geometry_error,
        "endpoint_geom_position_error_mm": geometry_position_error,
        "endpoint_residual_magnitude_mm": output["endpoint_residual"],
    }
    errors.update({name: torch.linalg.vector_norm(vector, dim=-1) * 1000
                   for name, vector in vectors.items()})
    errors["lateral_error_mm"] = torch.linalg.vector_norm(local_error[:, 1:], dim=-1) * 1000
    errors["axial_error_mm"] = local_error[:, 0].abs() * 1000
    # Positive values mean the existing residual reduced relative endpoint error.
    errors["endpoint_residual_gain_mm"] = (
        errors["endpoint_geom_relative_error_mm"] - errors["relative_3d_error_mm"]
    )
    return {name: values.cpu().numpy() for name, values in errors.items()}


def statistics(values):
    """Summarize frame-weighted values, representing absent groups with nulls."""
    if not len(values):
        return dict(mean=None, median=None, p90=None, p95=None)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
    }


def correlation(left, right):
    """Return descriptive Pearson r, or null for too few/constant observations."""
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def observation_groups(counts, sufficient_peg, sufficient_target):
    """Use observed counts, without assuming missing or occluded shape is known."""
    peg, box, hole = counts.T
    target = box + hole
    sufficient = (peg >= sufficient_peg) & (target >= sufficient_target)
    groups = {
        "all": np.ones(len(counts), dtype=bool),
        "high_support_proxy": sufficient,
        "low_support_proxy": ~sufficient,
        "empty_graph": (peg + target) == 0,
        "missing_peg": peg == 0,
        "missing_target": target == 0,
    }
    for name, values in (("peg", peg), ("box", box), ("hole", hole), ("target", target)):
        for count in np.unique(values):
            groups[f"{name}_vertices_{count}"] = values == count
    return groups


def summarize_errors(errors, groups):
    """Report every metric and the size/endpoint association within each group."""
    summaries = {}
    for name, mask in groups.items():
        size = errors["peg_half_size_x_error_mm"][mask]
        summaries[name] = {
            "frames": int(mask.sum()),
            "metrics": {metric: statistics(values[mask]) for metric, values in errors.items()},
            "size_x_endpoint_pearson": {
                metric: correlation(size, errors[metric][mask])
                for metric in ("endpoint_position_error_mm", "relative_3d_error_mm",
                               "endpoint_geom_relative_error_mm")
            },
        }
    return summaries


def paired_improvements(baseline, candidate, groups):
    """Compare the same frames; positive error reductions favor the candidate."""
    result = {}
    for name, mask in groups.items():
        deltas = {metric: (values - candidate[metric])[mask]
                  for metric, values in baseline.items() if "_error_" in metric}
        size_gain = deltas["peg_half_size_x_error_mm"]
        result[name] = {
            "frames": int(mask.sum()),
            "baseline_minus_candidate": {metric: statistics(values)
                                         for metric, values in deltas.items()},
            "size_endpoint_change": {},
        }
        if "endpoint_residual_gain_mm" in baseline:
            result[name]["candidate_minus_baseline_residual_gain_mm"] = statistics(
                (candidate["endpoint_residual_gain_mm"] - baseline["endpoint_residual_gain_mm"])[mask]
            )
        for metric in ("endpoint_position_error_mm", "relative_3d_error_mm",
                       "endpoint_geom_relative_error_mm"):
            endpoint_gain = deltas[metric]
            size_improved = size_gain > 0
            size_count = int(size_improved.sum())
            result[name]["size_endpoint_change"][metric] = {
                "pearson": correlation(size_gain, endpoint_gain),
                "size_x_improved_frames": size_count,
                "both_improved_frames": int((size_improved & (endpoint_gain > 0)).sum()),
                "endpoint_improved_given_size_improved": (
                    float((endpoint_gain[size_improved] > 0).mean()) if size_count else None
                ),
            }
    return result


def comparison_markdown(summary):
    """Produce a concise mean-error table; full quantiles remain in summary.json."""
    names = ("baseline", "geometry", "multiseed", "combined")
    lines = ["| Metric (mean) | Baseline | A | B | AB |",
             "| --- | ---: | ---: | ---: | ---: |"]
    metrics = summary["models"]["baseline"]["groups"]["all"]["metrics"]
    for metric in metrics:
        row = [f"{summary['models'][name]['groups']['all']['metrics'][metric]['mean']:.6f}"
               for name in names]
        lines.append("| " + " | ".join([metric, *row]) + " |")
    lines.extend(["", "Frame-weighted validation statistics; this split also selects best checkpoints.",
                  "Correlations are descriptive; parameter differences do not establish causality."])
    return "\n".join(lines) + "\n"


ABLATION_MODELS = ("baseline", "geometry", "multiseed", "combined")


def load_aligned_predictions(directory):
    """Join unique episode/frame IDs to baseline order and check targets/counts."""
    directory = Path(directory)
    predictions = {}
    hashes = {}
    reference_ids = None
    for name in ABLATION_MODELS:
        path = directory / f"{name}_predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        identifiers = data["sample_ids"]
        if not len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name}: sample_ids must be nonempty and unique")
        if reference_ids is None:
            reference_ids = identifiers
        if set(identifiers) != set(reference_ids):
            raise ValueError(f"{name}: validation frame membership differs")
        lookup = {value: index for index, value in enumerate(identifiers)}
        order = [lookup[value] for value in reference_ids]
        data = {key: value[order] for key, value in data.items()}
        for key, shape in (("target", (len(order), 21)),
                           ("object21", (len(order), 21)),
                           ("counts", (len(order), 3))):
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"{name}: {key} must be finite with shape {shape}")
        if predictions:
            for key in ("target", "counts"):
                if not np.array_equal(data[key], predictions["baseline"][key]):
                    raise ValueError(f"{name}: {key} differs on aligned frames")
        expected_size_error = np.abs(data["object21"][:, 17] - data["target"][:, 17]) * 1000
        np.testing.assert_allclose(data["peg_half_size_x_error_mm"], expected_size_error,
                                   atol=1e-5, rtol=1e-5)
        predictions[name] = data
        hashes[name] = file_sha256(path)
    return predictions, hashes


def cluster_interval(values, clusters, repetitions=2000, seed=0):
    """Bootstrap whole clusters for a frame-weighted paired mean difference.

    The cluster set is sampled with replacement. Every selected cluster retains
    all its frames; the ratio of total sums to total counts matches the original
    frame-weighted estimand. This does not treat frames as independent samples.
    """
    if not len(values):
        return None
    unique, indices = np.unique(clusters, return_inverse=True)
    sums = np.bincount(indices, weights=values)
    counts = np.bincount(indices)
    random = np.random.default_rng(seed)
    draws = random.integers(len(unique), size=(repetitions, len(unique)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "mean": float(np.mean(values)), "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)), "clusters": len(unique),
        "repetitions": repetitions, "estimand": "frame-weighted mean",
    }
