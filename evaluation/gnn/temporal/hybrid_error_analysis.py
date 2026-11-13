"""Replay-free, identity-checked Hybrid-H1 geometry attribution on saved validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.temporal.hybrid_diagnostics import (
    paired_bootstrap,
    vector_relationship,
)
from evaluation.gnn.temporal.hybrid_error_geometry import (
    endpoint_factor_vectors,
    endpoint_interventions,
    endpoint_position,
    factor_squared_terms,
    geometry_groups,
    local_vectors,
    ordered_oracle_reductions,
    squared_decomposition,
)
from evaluation.gnn.temporal.hybrid_inputs import (
    DATA_FIELDS,
    load_reference,
    restore_branches,
)
from evaluation.gnn.temporal.insertion import (
    distribution,
    insertion_errors,
    precision_percentages,
)
from graph.pose_geometry import geometric_endpoint
from graph.temporal.hybrid import assemble_hybrid
from utils.experiment_logger import ExperimentLogger

METHODS = ("mamba", "hybrid_h1_a")
BOOTSTRAP_METRICS = ("relative_3d", "lateral", "axial")


def load_diagnostic_inputs(config):
    """Validate saved data by complete ordered identities before combining runs."""
    root = Path(config["hybrid_evaluation"])
    if file_sha256(root / "predictions.npz") != config["hybrid_predictions_sha256"]:
        raise ValueError("Hybrid predictions checksum differs from the selected frozen result")
    source_config = json.loads((root / "effective_config.json").read_text())
    data, baselines, reference, _, provenance, identity, frames = load_reference(source_config)
    # Reuse existing checkpoint/source/provenance checks, without running either model.
    mamba, gate, audit = restore_branches(
        source_config, reference, provenance, identity, torch.device("cpu")
    )
    del mamba, gate
    if audit["checkpoints"]["gate"]["epoch"] != 15:
        raise ValueError("Expected the selected epoch-15 Learned Gate checkpoint")
    saved_frames = json.loads((root / "frame_manifest.json").read_text())
    if saved_frames != frames:
        raise ValueError(
            "Hybrid and reference frame manifests differ in identity, order or raw checksum"
        )
    old_audit = json.loads((root / "audit.json").read_text())
    frame_hash = hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest()
    if old_audit["frame_identity_sha256"] != frame_hash:
        raise ValueError("Hybrid frame manifest differs from its audited checksum")
    predictions = {}
    with np.load(root / "predictions.npz", allow_pickle=False) as saved:
        for name in DATA_FIELDS:
            if name in data:
                np.testing.assert_array_equal(
                    saved[name], data[name], err_msg=f"Cross-run mismatch: {name}"
                )
        for name in ("gnn", "learned_gate", "mamba", "hybrid_h1_a"):
            value = saved[name + "_object21"].copy()
            if value.shape != data["target"].shape or not np.isfinite(value).all():
                raise ValueError(f"{name}: expected finite frame-aligned Object21 predictions")
            predictions[name] = value
    for name in baselines:
        np.testing.assert_allclose(predictions[name], baselines[name], atol=2e-5, rtol=1e-4)
    expected = assemble_hybrid(
        *(torch.from_numpy(predictions[name]) for name in ("mamba", "learned_gate", "gnn"))
    )
    np.testing.assert_allclose(predictions["hybrid_h1_a"], expected.numpy(), atol=1e-7, rtol=0)
    for region in (slice(0, 3), slice(17, 21)):
        np.testing.assert_array_equal(
            predictions["mamba"][:, region], predictions["hybrid_h1_a"][:, region]
        )
    # Hybrid assembly renormalizes float32 quaternions (up to one ULP here).
    for region in (slice(3, 7), slice(13, 17)):
        np.testing.assert_allclose(
            predictions["mamba"][:, region],
            predictions["hybrid_h1_a"][:, region],
            atol=1e-7,
            rtol=0,
        )
    np.testing.assert_array_equal(
        predictions["hybrid_h1_a"][:, 10:13], predictions["learned_gate"][:, 10:13]
    )
    np.testing.assert_array_equal(predictions["mamba"][:, 17:21], predictions["gnn"][:, 17:21])
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary["split"] != "validation"
        or summary["independent_test"]
        or summary["excluded_frames"]
    ):
        raise ValueError("Expected complete reused validation with zero excluded frames")
    audit.update(
        frame_identity_sha256=frame_hash,
        reused_evaluation=str(root.resolve()),
        inference_executed=False,
        current_device="cpu",
        source_inference_context=old_audit["inference_context"],
        source_runtime_backend=old_audit["mamba_runtime_backend"],
        source_reproduction=old_audit["reproduction"],
        phase_label_audit={
            "configured_labels": reference["evaluation"]["phase_labels"],
            "snapshot_fields": list(data),
            "unavailable": ["before_grasp", "grasp_lift", "transport", "physical_contact"],
            "scope": "No phase/contact labels in the identified snapshot; raw /workspace episodes unavailable here.",
        },
    )
    inputs = [
        root / name
        for name in (
            "predictions.npz",
            "frame_manifest.json",
            "audit.json",
            "summary.json",
            "effective_config.json",
        )
    ]
    inputs += [Path(source_config[name + "_checkpoint"]) for name in ("gnn", "mamba", "gate")]
    audit["diagnostic_inputs"] = {str(path): file_sha256(path) for path in inputs}
    return data, predictions, reference["temporal_split_manifest"], frames, summary, audit


def prepare_geometry(data, predictions, source_summary):
    """Canonicalize only saved vector roundoff through float64 Method A, then score."""
    target = data["target"].astype(np.float64)
    original_target_vector = target[:, 7:10].copy()
    target[:, 7:10] = geometric_endpoint(torch.from_numpy(target)).numpy()
    canonical = {}
    errors = {}
    residuals = {
        "target_vector_max_difference_m": float(
            np.max(np.abs(target[:, 7:10] - original_target_vector))
        )
    }
    for name in METHODS:
        value = predictions[name].astype(np.float64)
        saved_vector = value[:, 7:10].copy()
        if name == "hybrid_h1_a":
            residuals["shared_quaternion_max_component_difference"] = float(
                max(
                    np.max(np.abs(value[:, 3:7] - canonical["mamba"][:, 3:7])),
                    np.max(np.abs(value[:, 13:17] - canonical["mamba"][:, 13:17])),
                )
            )
            value[:, 3:7] = canonical["mamba"][:, 3:7]
            value[:, 13:17] = canonical["mamba"][:, 13:17]
        value[:, 7:10] = geometric_endpoint(torch.from_numpy(value)).numpy()
        residuals[name + "_vector_max_difference_m"] = float(
            np.max(np.abs(value[:, 7:10] - saved_vector))
        )
        np.testing.assert_allclose(value[:, 7:10], saved_vector, atol=2e-7, rtol=0)
        errors[name] = insertion_errors(value, target)
        errors[name]["endpoint_position"] = (
            np.linalg.norm(endpoint_position(value) - endpoint_position(target), axis=1) * 1000
        )
        canonical[name] = value
        for metric, values in errors[name].items():
            if metric == "endpoint_position":
                continue
            unit = "deg" if "rotation" in metric else "mm"
            previous = source_summary["methods"][name]["groups"]["all"][metric]["mean_" + unit]
            if abs(float(values.mean()) - previous) > 0.001:
                raise ValueError(
                    f"{name}/{metric}: source metric does not reproduce within .001 {unit}"
                )
    np.testing.assert_allclose(target[:, 7:10], original_target_vector, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(
        endpoint_position(canonical["mamba"]), endpoint_position(canonical["hybrid_h1_a"])
    )
    return target, canonical, errors, residuals


def decomposition_summary(terms, mask):
    """Average squared terms in a common group, retaining cancellation signs."""
    if not mask.any():
        return {"frames": 0}
    result = {
        name: float(values[mask].mean())
        for name, values in terms.items()
        if name not in ("cosine", "identity_residual_mm2")
    }
    cosine = terms["cosine"][mask]
    result.update(
        frames=int(mask.sum()),
        cosine=distribution(cosine[np.isfinite(cosine)], "unitless"),
        cancellation_percent=float(np.mean(terms["dot_mm2"][mask] > 0) * 100),
        amplification_percent=float(np.mean(terms["dot_mm2"][mask] < 0) * 100),
        orthogonal_or_zero_percent=float(np.mean(terms["dot_mm2"][mask] == 0) * 100),
        identity_max_residual_mm2=float(np.max(np.abs(terms["identity_residual_mm2"][mask]))),
    )
    return result


def summarize_group(data, errors, mask):
    """Report frame distributions and per-episode worst frames for every metric."""
    episodes = np.unique(data["episode"][mask])
    metrics = {
        name: distribution(values[mask], "deg" if "rotation" in name else "mm")
        for name, values in errors.items()
    }
    worst = {}
    for episode in episodes:
        rows = np.flatnonzero(mask & (data["episode"] == episode))
        worst[str(episode)] = {}
        for name, values in errors.items():
            row = rows[np.argmax(values[rows])]
            worst[str(episode)][name] = {
                "value": float(values[row]),
                "sample_id": str(data["sample_ids"][row]),
            }
    return dict(
        frames=int(mask.sum()),
        episodes=len(episodes),
        metrics=metrics,
        precision=precision_percentages(errors, mask),
        episode_worst=worst,
    )


def summarize_interventions(interventions, target, hole_error, mask):
    """Measure oracle endpoint and downstream relative/lateral/axial precision."""
    result = {}
    for replacement, endpoint_error in interventions.items():
        relative = endpoint_error - hole_error
        local = local_vectors(relative, target)
        result[str(replacement)] = {
            "gt_replacements": [
                factor
                for bit, factor in enumerate(("translation", "rotation", "size"))
                if replacement & (1 << bit)
            ],
            "endpoint_mm": distribution(np.linalg.norm(endpoint_error[mask], axis=1) * 1000, "mm"),
            "endpoint_mse_mm2": float(np.mean(np.sum(endpoint_error[mask] ** 2, axis=1)) * 1e6)
            if mask.any()
            else None,
            "relative_mm": distribution(np.linalg.norm(relative[mask], axis=1) * 1000, "mm"),
            "lateral_mm": distribution(np.linalg.norm(local[mask, 1:], axis=1) * 1000, "mm"),
            "axial_mm": distribution(np.abs(local[mask, 0]) * 1000, "mm"),
        }
    return {"conditions": result, "order_effects": ordered_oracle_reductions(interventions, mask)}


def cancellation_changes(data, endpoint_error, hole_errors, target, errors, groups):
    """Explain relative MSE change via Hole gain and lost cancellation per axis."""
    old_hole, new_hole = hole_errors["mamba"], hole_errors["hybrid_h1_a"]
    hole_gain = np.sum(old_hole**2 - new_hole**2, axis=1) * 1e6
    lost_cancellation = 2 * np.sum(endpoint_error * (old_hole - new_hole), axis=1) * 1e6
    delta_squared = (
        np.sum((endpoint_error - new_hole) ** 2, axis=1)
        - np.sum((endpoint_error - old_hole) ** 2, axis=1)
    ) * 1e6
    np.testing.assert_allclose(delta_squared, lost_cancellation - hole_gain, atol=1e-8, rtol=0)
    hole_delta = errors["hybrid_h1_a"]["hole_position"] - errors["mamba"]["hole_position"]
    relative_delta = errors["hybrid_h1_a"]["relative_3d"] - errors["mamba"]["relative_3d"]
    tolerance = 1e-6
    hole_improves, hole_worsens = hole_delta < -tolerance, hole_delta > tolerance
    rel_improves, rel_worsens = relative_delta < -tolerance, relative_delta > tolerance
    masks = {
        "hole_improves_relative_improves": hole_improves & rel_improves,
        "hole_improves_relative_worsens": hole_improves & rel_worsens,
        "hole_worsens_relative_improves": hole_worsens & rel_improves,
        "hole_worsens_relative_worsens": hole_worsens & rel_worsens,
    }
    masks["tie_in_either_metric"] = ~np.logical_or.reduce(list(masks.values()))
    result = {"tie_tolerance_mm": tolerance, "quadrants": {}, "axis_change_by_group": {}}
    for name, mask in masks.items():
        result["quadrants"][name] = {
            "frames": int(mask.sum()),
            "percent_of_all": float(mask.mean() * 100),
            "relative_delta_mm": distribution(relative_delta[mask], "mm"),
            "mean_hole_squared_gain_mm2": float(hole_gain[mask].mean()) if mask.any() else None,
            "mean_lost_cancellation_mm2": float(lost_cancellation[mask].mean())
            if mask.any()
            else None,
            "mean_relative_squared_delta_mm2": float(delta_squared[mask].mean())
            if mask.any()
            else None,
        }
    local_endpoint = local_vectors(endpoint_error, target)
    local_old = local_vectors(old_hole, target)
    local_new = local_vectors(new_hole, target)
    gain_axes = (local_old**2 - local_new**2) * 1e6
    lost_axes = 2 * local_endpoint * (local_old - local_new) * 1e6
    for name, mask in groups.items():
        result["axis_change_by_group"][name] = (
            {
                axis: dict(
                    hole_squared_gain_mm2=float(gain_axes[mask, index].mean()),
                    lost_cancellation_mm2=float(lost_axes[mask, index].mean()),
                    relative_squared_delta_mm2=float((lost_axes - gain_axes)[mask, index].mean()),
                )
                for index, axis in enumerate(("x_axial", "y_lateral", "z_lateral"))
            }
            if mask.any()
            else {}
        )
    arrays = dict(
        hole_squared_gain_mm2=hole_gain,
        lost_cancellation_mm2=lost_cancellation,
        relative_squared_delta_mm2=delta_squared,
        quadrant=np.select(list(masks.values()), list(masks), default="unclassified"),
    )
    return result, arrays


def analyze(data, target, predictions, errors, manifest, config):
    """Run deterministic GT-only diagnostics on all frames, never learning parameters."""
    groups, geometry = geometry_groups(target, config["geometry_proxies"])
    endpoint_error = endpoint_position(predictions["mamba"]) - endpoint_position(target)
    hole_errors = {name: value[:, 10:13] - target[:, 10:13] for name, value in predictions.items()}
    decompositions = {
        name: squared_decomposition(endpoint_error, value) for name, value in hole_errors.items()
    }
    local_endpoint = local_vectors(endpoint_error, target)
    local_holes = {name: local_vectors(value, target) for name, value in hole_errors.items()}
    interventions = endpoint_interventions(predictions["mamba"], target)
    factors = endpoint_factor_vectors(predictions["mamba"], target)
    factor_terms = factor_squared_terms(factors)
    np.testing.assert_allclose(sum(factors.values()), endpoint_error, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        sum(factor_terms.values()), np.sum(endpoint_error**2, axis=1) * 1e6, atol=1e-8, rtol=0
    )
    summary = {
        "scope": "repeatedly reused development validation; GT diagnostics only; no insertion rollouts",
        "frames": len(target),
        "episodes": len(np.unique(data["episode"])),
        "groups": {},
        "episodes_detail": {},
        "paired": {},
        "endpoint_oracles": {},
        "decomposition": {},
        "factor_terms_mm2": {},
        "vector_relationships": {},
        "axis_decomposition": {},
    }
    for name in METHODS:
        for index, axis in enumerate(("x", "y", "z")):
            errors[name]["signed_gt_hole_" + axis] = (
                local_endpoint[:, index] - local_holes[name][:, index]
            ) * 1000
            errors[name]["absolute_gt_hole_" + axis] = np.abs(
                errors[name]["signed_gt_hole_" + axis]
            )
    for group, mask in groups.items():
        summary["groups"][group] = {
            name: summarize_group(data, values, mask) for name, values in errors.items()
        }
        summary["decomposition"][group] = {
            name: decomposition_summary(terms, mask) for name, terms in decompositions.items()
        }
        summary["axis_decomposition"][group] = {
            name: {
                axis: decomposition_summary(
                    squared_decomposition(
                        local_endpoint[:, index : index + 1],
                        local_holes[name][:, index : index + 1],
                    ),
                    mask,
                )
                for index, axis in enumerate(("x_axial", "y_lateral", "z_lateral"))
            }
            for name in METHODS
        }
        summary["factor_terms_mm2"][group] = {
            name: float(values[mask].mean()) if mask.any() else None
            for name, values in factor_terms.items()
        }
        summary["endpoint_oracles"][group] = summarize_interventions(
            interventions, target, hole_errors["hybrid_h1_a"], mask
        )
        summary["paired"][group] = {}
        if mask.any():
            subset = {name: value[mask] for name, value in data.items()}
            for metric in BOOTSTRAP_METRICS:
                delta = errors["hybrid_h1_a"][metric][mask] - errors["mamba"][metric][mask]
                summary["paired"][group][metric] = paired_bootstrap(
                    subset, delta, manifest, config["seed"], config["bootstrap_resamples"]
                )
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        summary["episodes_detail"][str(episode)] = {
            "frames": int(mask.sum()),
            "decomposition": {
                name: decomposition_summary(terms, mask) for name, terms in decompositions.items()
            },
            "oracle": summarize_interventions(
                interventions, target, hole_errors["hybrid_h1_a"], mask
            ),
            "metrics": {
                name: {
                    metric: distribution(values[mask], "deg" if "rotation" in metric else "mm")
                    for metric, values in method.items()
                }
                for name, method in errors.items()
            },
        }
    for name in METHODS:
        summary["vector_relationships"][name] = {
            "world": vector_relationship(endpoint_error, hole_errors[name]),
            "gt_hole_frame": vector_relationship(local_endpoint, local_holes[name]),
        }
    cancellation, changes = cancellation_changes(
        data, endpoint_error, hole_errors, target, errors, groups
    )
    summary["cancellation"] = cancellation
    differences = summary["paired"]["all"]["relative_3d"]["episode_mean_deltas_mm"]
    ordered = sorted(differences, key=differences.get)
    summary["representatives"] = {
        "most_improved": ordered[:5],
        "most_degraded": ordered[-5:][::-1],
        "required": config["required_episode"],
    }
    if config["required_episode"] not in differences:
        raise ValueError("Required failure episode is absent from the verified validation")
    # Expose threshold sensitivity rather than treating one geometric cut as phase GT.
    local = geometry["gt_local_relative_m"]
    sensitivity = {}
    for lateral_mm in (7.5, 10.0, 15.0):
        for angle_deg in (5.0, 10.0, 15.0):
            mask = (
                (local[:, 0] >= config["geometry_proxies"]["approach_min_x_m"])
                & (local[:, 0] < config["geometry_proxies"]["head_min_x_m"])
                & (np.linalg.norm(local[:, 1:], axis=1) <= lateral_mm / 1000)
                & (geometry["gt_axis_alignment_deg"] <= angle_deg)
            )
            sensitivity[f"lateral_{lateral_mm:g}mm_axis_{angle_deg:g}deg"] = {
                "frames": int(mask.sum()),
                "episodes": len(np.unique(data["episode"][mask])),
                **{
                    name: {
                        metric: distribution(values[mask], "mm")
                        for metric, values in errors[name].items()
                        if metric in BOOTSTRAP_METRICS
                    }
                    for name in METHODS
                },
            }
    summary["approach_threshold_sensitivity"] = sensitivity
    size_error = np.abs(predictions["mamba"][:, 17] - target[:, 17]) * 1000
    tail_masks = {
        "hybrid_relative_top_5_percent": errors["hybrid_h1_a"]["relative_3d"]
        >= np.percentile(errors["hybrid_h1_a"]["relative_3d"], 95),
        "peg_rotation_at_least_5deg": errors["mamba"]["peg_rotation"] >= 5,
        "half_length_error_at_least_3mm": size_error >= 3,
        "half_length_error_below_3mm": size_error < 3,
    }
    summary["tail_diagnostics"] = {
        name: {
            "frames": int(mask.sum()),
            "half_length_error_mm": distribution(size_error[mask], "mm"),
            "oracle": summarize_interventions(
                interventions, target, hole_errors["hybrid_h1_a"], mask
            ),
            "metrics": {
                method: {
                    metric: distribution(values[mask], "deg" if "rotation" in metric else "mm")
                    for metric, values in errors[method].items()
                }
                for method in METHODS
            },
        }
        for name, mask in tail_masks.items()
    }
    arrays = dict(
        sample_ids=data["sample_ids"],
        episode=data["episode"],
        frame=data["frame"],
        endpoint_error_m=endpoint_error,
        **geometry,
        **changes,
    )
    for name in METHODS:
        arrays[name + "__hole_error_m"] = hole_errors[name]
        arrays.update({name + "__" + metric: values for metric, values in errors[name].items()})
        arrays.update({name + "__" + term: values for term, values in decompositions[name].items()})
    arrays.update(
        {"oracle_" + str(key) + "__endpoint_error_m": value for key, value in interventions.items()}
    )
    arrays.update({"factor__" + name + "_m": value for name, value in factors.items()})
    arrays.update({"group__" + name: mask for name, mask in groups.items()})
    return summary, arrays, groups


def evaluate(config):
    """Check existing artifacts, analyze all validation frames and write a fresh run."""
    if type(config["bootstrap_resamples"]) is not int or config["bootstrap_resamples"] < 100:
        raise ValueError("bootstrap_resamples must be an integer >=100")
    if config.get("device", "cpu") != "cpu":
        raise ValueError(
            "Offline saved-array diagnostics support device: cpu; no accelerator inference runs"
        )
    torch.set_num_threads(1)
    data, saved_predictions, manifest, frames, previous, audit = load_diagnostic_inputs(config)
    target, predictions, errors, residuals = prepare_geometry(data, saved_predictions, previous)
    summary, arrays, groups = analyze(data, target, predictions, errors, manifest, config)
    audit["float64_method_a_roundoff"] = residuals
    audit["diagnostic_source_sha256"] = {
        str(path): file_sha256(path)
        for path in (
            Path(__file__),
            Path("evaluation/gnn/temporal/hybrid_error_geometry.py"),
            Path("evaluation/gnn/temporal/hybrid_error_plots.py"),
        )
    }
    with ExperimentLogger(
        team="graph",
        model="hybrid_error_diagnostics",
        config=config,
        seed=config["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        write_json(output / "summary.json", summary)
        write_json(output / "audit.json", audit)
        write_json(output / "frame_manifest.json", frames)
        np.savez_compressed(output / "frame_diagnostics.npz", **arrays)
        from evaluation.gnn.temporal.hybrid_error_plots import save_plots

        save_plots(output, data, target, predictions, errors, summary, arrays, groups)
        for name in METHODS:
            experiment.log(summary["groups"]["all"][name]["metrics"], prefix=name)
        experiment.record(
            frame_identity_sha256=audit["frame_identity_sha256"],
            split="validation",
            independent_test=False,
            inference_executed=False,
            gt_usage="post-inference scoring, geometry proxies and oracle interventions",
        )
        for path in output.iterdir():
            experiment.link_output(path)
        # Neither checkpoint linking nor training occurs. Check every input remains byte-identical.
        for path, digest in audit["diagnostic_inputs"].items():
            if file_sha256(path) != digest:
                raise ValueError(f"Input changed during diagnostics: {path}")
        print(f"Hybrid error diagnostics saved: {output}", flush=True)
        return output
