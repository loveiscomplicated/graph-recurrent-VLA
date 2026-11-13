"""Offline cancellation analysis, parent-cluster comparisons and diagnostic plots."""

import numpy as np

from evaluation.gnn.temporal.hybrid_inputs import BASELINE_FIELDS
from evaluation.gnn.temporal.insertion import distribution, score_insertion_methods

MAIN_METHODS = ("gnn", "learned_gate", "mamba", "hybrid_h1_a", "hybrid_h2_a")


def vector_relationship(left, right):
    """Describe correlated signed vector errors, with zero-norm directions excluded."""
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    product = left_norm * right_norm
    valid = product > 1e-12
    cosine = np.sum(left[valid] * right[valid], axis=1) / product[valid]
    correlations = []
    for axis in range(3):
        if left[:, axis].std() > 1e-12 and right[:, axis].std() > 1e-12:
            correlations.append(float(np.corrcoef(left[:, axis], right[:, axis])[0, 1]))
        else:
            correlations.append(None)
    return {
        "direction_cosine": distribution(cosine, "unitless"),
        "positive_dot_percent": float(np.mean(cosine > 0) * 100) if len(cosine) else None,
        "valid_direction_frames": int(valid.sum()),
        "pearson_signed_xyz": correlations,
        "mean_dot_mm2": float(np.sum(left * right, axis=1).mean() * 1e6),
    }


def paired_bootstrap(data, delta, manifest, seed, resamples):
    """Resample parent clusters; never treat individual frames as independent draws."""
    episodes = np.unique(data["episode"])
    parents = manifest["parent_by_episode"]
    clusters = sorted({parents[str(episode)] for episode in episodes})
    episode_means = {
        str(episode): float(delta[data["episode"] == episode].mean()) for episode in episodes
    }
    episode_sums, episode_counts, frame_sums, frame_counts = [], [], [], []
    for parent in clusters:
        members = [str(episode) for episode in episodes if parents[str(episode)] == parent]
        mask = np.isin(data["episode"], members)
        episode_sums.append(sum(episode_means[episode] for episode in members))
        episode_counts.append(len(members))
        frame_sums.append(float(delta[mask].sum()))
        frame_counts.append(int(mask.sum()))
    draws = np.random.default_rng(seed).integers(0, len(clusters), (resamples, len(clusters)))
    episode_draws = np.asarray(episode_sums)[draws].sum(1) / np.asarray(episode_counts)[draws].sum(
        1
    )
    frame_draws = np.asarray(frame_sums)[draws].sum(1) / np.asarray(frame_counts)[draws].sum(1)
    return {
        "unit": "parent cluster",
        "parents": len(clusters),
        "episodes": len(episodes),
        "resamples": resamples,
        "seed": seed,
        "confidence": 0.95,
        "episode_equal_mean_delta_mm": float(np.mean(list(episode_means.values()))),
        "episode_equal_percentile_ci_mm": np.percentile(episode_draws, [2.5, 97.5]).tolist(),
        "frame_weighted_mean_delta_mm": float(delta.mean()),
        "frame_weighted_percentile_ci_mm": np.percentile(frame_draws, [2.5, 97.5]).tolist(),
        "episodes_improved": sum(value < 0 for value in episode_means.values()),
        "episodes_worsened": sum(value > 0 for value in episode_means.values()),
        "episode_mean_deltas_mm": episode_means,
    }


def frame_changes(data, baseline, candidate):
    """Retain signed deltas, conditional improvement magnitudes and worst identities."""
    delta = candidate - baseline
    # A tiny tolerance prevents floating-point roundoff from being called an effect.
    improved, worsened = delta < -1e-6, delta > 1e-6
    worst = np.argsort(delta)[::-1][:10]
    return {
        "delta_definition": "hybrid minus Mamba, mm; negative is improvement",
        "tie_tolerance_mm": 1e-6,
        "mean_delta_mm": float(delta.mean()),
        "improved_percent": float(improved.mean() * 100),
        "worsened_percent": float(worsened.mean() * 100),
        "tied_percent": float((~(improved | worsened)).mean() * 100),
        "mean_improvement_on_improved_mm": float(-delta[improved].mean())
        if improved.any()
        else None,
        "mean_degradation_on_worsened_mm": float(delta[worsened].mean())
        if worsened.any()
        else None,
        "delta_distribution": distribution(delta, "mm"),
        "worst_deltas": [
            dict(
                sample_id=str(data["sample_ids"][row]),
                episode=str(data["episode"][row]),
                frame=int(data["frame"][row]),
                delta_mm=float(delta[row]),
                mamba_mm=float(baseline[row]),
                hybrid_mm=float(candidate[row]),
            )
            for row in worst
        ],
    }


def analyze_hybrid(data, predictions, errors, manifest, *, seed=0, resamples=10000):
    """Use GT strictly after prediction to test whether Hole replacement breaks cancellation."""
    if type(resamples) is not int or resamples < 100:
        raise ValueError("bootstrap_resamples must be an integer >= 100")
    target, mamba, gate = (
        np.asarray(value, dtype=np.float64)
        for value in (data["target"], predictions["mamba"], predictions["learned_gate"])
    )
    peg_error = mamba[:, :3] - target[:, :3]
    hole_error = mamba[:, 10:13] - target[:, 10:13]
    gate_error = gate[:, 10:13] - target[:, 10:13]
    endpoint_error = mamba[:, 7:10] + mamba[:, 10:13] - target[:, 7:10] - target[:, 10:13]
    hole_gain = np.sum(hole_error**2 - gate_error**2, axis=1) * 1e6
    lost_cancellation = 2 * np.sum(endpoint_error * (hole_error - gate_error), axis=1) * 1e6
    expected_change = lost_cancellation - hole_gain
    relative_before = endpoint_error - hole_error
    relative_after = endpoint_error - gate_error
    actual_change = np.sum(relative_after**2 - relative_before**2, axis=1) * 1e6
    np.testing.assert_allclose(actual_change, expected_change, atol=1e-8, rtol=1e-8)
    cancellation = {
        "peg_hole_mamba": vector_relationship(peg_error, hole_error),
        "endpoint_hole_mamba": vector_relationship(endpoint_error, hole_error),
        "endpoint_hole_gate": vector_relationship(endpoint_error, gate_error),
        "squared_relative_change_identity": "lost_cancellation - hole_squared_error_reduction (mm^2)",
        "mean_hole_squared_error_reduction_mm2": float(hole_gain.mean()),
        "mean_lost_cancellation_mm2": float(lost_cancellation.mean()),
        "mean_squared_relative_change_mm2": float(actual_change.mean()),
        "squared_identity_max_residual_mm2": float(np.max(np.abs(actual_change - expected_change))),
        "hole_replacement_improves_hole_percent": float(np.mean(hole_gain > 0) * 100),
        "hole_improves_but_relative_worsens_percent": float(
            np.mean((hole_gain > 0) & (actual_change > 0)) * 100
        ),
    }
    for label, mask in (
        ("relative_improved", actual_change < 0),
        ("relative_worsened", actual_change > 0),
    ):
        cancellation[label] = {
            "frames": int(mask.sum()),
            "mean_hole_squared_error_reduction_mm2": float(hole_gain[mask].mean())
            if mask.any()
            else None,
            "mean_lost_cancellation_mm2": float(lost_cancellation[mask].mean())
            if mask.any()
            else None,
            "mean_squared_relative_change_mm2": float(actual_change[mask].mean())
            if mask.any()
            else None,
        }
    endpoint_errors = {}
    for name, prediction in predictions.items():
        error = prediction[:, 7:10] + prediction[:, 10:13] - target[:, 7:10] - target[:, 10:13]
        endpoint_errors[name] = distribution(np.linalg.norm(error, axis=1) * 1000, "mm")
    results = {
        "error_cancellation": cancellation,
        "endpoint_position_error": endpoint_errors,
        "scope": "reused validation, offline GT diagnostic; no action controller or insertion rollouts",
        "conditions": "GT motion/alignment proxies; observed-support absence is not physical visibility",
        "comparisons": {},
    }
    for name in ("hybrid_h1_a", "hybrid_h2_a"):
        comparisons = {}
        for metric in ("relative_3d", "lateral"):
            baseline, candidate = errors["mamba"][metric], errors[name][metric]
            comparisons[metric] = frame_changes(data, baseline, candidate)
            comparisons[metric]["paired_bootstrap"] = paired_bootstrap(
                data, candidate - baseline, manifest, seed, resamples
            )
        results["comparisons"][name] = comparisons
    differences = results["comparisons"]["hybrid_h1_a"]["relative_3d"]["paired_bootstrap"][
        "episode_mean_deltas_mm"
    ]
    ordered = sorted(differences, key=differences.get)
    results["representative_episodes"] = {
        "largest_improvement": ordered[0],
        "largest_degradation": ordered[-1],
        "delta_mm": {episode: differences[episode] for episode in (ordered[0], ordered[-1])},
    }
    results["representative_episode_metrics"] = {}
    for episode in (ordered[0], ordered[-1]):
        mask = data["episode"] == episode
        results["representative_episode_metrics"][episode] = {
            name: {metric: float(values[mask].mean()) for metric, values in errors[name].items()}
            for name in MAIN_METHODS
        }
    support = (data["pose_quality"][:, 0] > 0) & (data["quality"][:, 0] > 0)
    results["observed_support_slices"] = {}
    for label, mask in (
        ("both_object_support_present", support),
        ("episode_cold_start", data["frame"] == 0),
    ):
        results["observed_support_slices"][label] = {
            name: {
                metric: distribution(values[mask], "deg" if "rotation" in metric else "mm")
                for metric, values in errors[name].items()
            }
            for name in MAIN_METHODS
        }
    return results


def save_hybrid_plots(output, data, predictions, errors, diagnostics):
    """Save ECDFs, paired frames/episodes and full pose traces for both extremes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(MAIN_METHODS, ("#777777", "#d99722", "#2576b9", "#198754", "#aa4d9c")))
    for metric in ("relative_3d", "lateral"):
        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
        for name in MAIN_METHODS:
            values = np.sort(errors[name][metric])
            axis.plot(
                values, np.arange(1, len(values) + 1) / len(values), label=name, color=colors[name]
            )
        axis.set(
            xlabel=f"{metric} estimation error (mm)",
            ylabel="Frame cumulative fraction",
            title="Same validation frames; historical baseline assemblies",
        )
        axis.legend()
        figure.savefig(output / f"{metric}_distribution.png", dpi=160)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, metric in zip(axes, ("relative_3d", "lateral")):
        left, right = errors["mamba"][metric], errors["hybrid_h1_a"][metric]
        maximum = max(left.max(), right.max())
        axis.hexbin(left, right, gridsize=65, mincnt=1, bins="log", cmap="viridis")
        axis.plot([0, maximum], [0, maximum], "--", color="red", linewidth=1)
        axis.set(
            xlabel=f"Mamba {metric} (mm)",
            ylabel="Hybrid-H1 (mm)",
            title="Below diagonal: Hybrid improves",
        )
    figure.savefig(output / "frame_scatter.png", dpi=160)
    plt.close(figure)
    episodes = np.unique(data["episode"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, metric in zip(axes, ("relative_3d", "lateral")):
        baseline = [
            errors["mamba"][metric][data["episode"] == episode].mean() for episode in episodes
        ]
        hybrid = [
            errors["hybrid_h1_a"][metric][data["episode"] == episode].mean() for episode in episodes
        ]
        axis.plot(baseline, label="Mamba", linewidth=1)
        axis.plot(hybrid, label="Hybrid-H1", linewidth=1)
        axis.set(xlabel="Validation episode index (same order)", ylabel=f"Mean {metric} (mm)")
        axis.legend()
    figure.savefig(output / "episode_comparison.png", dpi=160)
    plt.close(figure)
    for role in ("largest_improvement", "largest_degradation"):
        episode = diagnostics["representative_episodes"][role]
        save_episode_plot(
            output / f"{role}_episode.png", episode, data, predictions, errors, colors
        )


def save_episode_plot(path, episode, data, predictions, errors, colors):
    """Plot every world pose component, estimation errors and predicted-frame geometry."""
    import matplotlib.pyplot as plt
    import torch

    from graph.pose_geometry import hole_frame_vector

    rows = np.flatnonzero(data["episode"] == episode)
    time = data["frame"][rows]
    figure = plt.figure(figsize=(18, 21), layout="constrained")
    grid = figure.add_gridspec(6, 4)
    values = {
        name: prediction[rows].copy()
        for name, prediction in predictions.items()
        if name in MAIN_METHODS
    }
    values["GT"] = data["target"][rows].copy()
    for value in values.values():
        value[:, :3] += data["tcp_world"][rows]
        value[:, 10:13] += data["tcp_world"][rows]
        # Canonicalize only the visualization, so q/-q does not create false jumps.
        for start in (3, 13):
            quaternion = value[:, start : start + 4]
            value[:, start : start + 4] = np.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    handles = {}
    for block, start, label in (
        (0, 0, "Peg world position"),
        (1, 10, "Hole world position"),
        (2, 3, "Peg wxyz"),
        (3, 13, "Hole wxyz"),
    ):
        width = 3 if block < 2 else 4
        for component in range(width):
            axis = figure.add_subplot(grid[block, component])
            for name, value in values.items():
                (line,) = axis.plot(
                    time,
                    value[:, start + component],
                    label=name,
                    color=colors.get(name, "black"),
                    linestyle="--" if name == "GT" else "-",
                    linewidth=1.1,
                    alpha=0.85,
                )
                handles[name] = line
            axis.set(
                title=f"{label} {'xyz'[component] if block < 2 else 'wxyz'[component]}",
                xlabel="Frame index",
                ylabel="metres" if block < 2 else "unit quaternion",
            )
    for column, metric in enumerate(
        ("peg_position", "hole_position", "peg_rotation", "hole_rotation")
    ):
        axis = figure.add_subplot(grid[4, column])
        for name in MAIN_METHODS:
            axis.plot(time, errors[name][metric][rows], color=colors[name], linewidth=1)
        axis.set(
            title=metric, xlabel="Frame index", ylabel="degrees" if "rotation" in metric else "mm"
        )
    for column, metric in enumerate(("relative_3d", "lateral", "axial", "relative_rotation")):
        axis = figure.add_subplot(grid[5, column])
        for name in MAIN_METHODS:
            axis.plot(time, errors[name][metric][rows], color=colors[name], linewidth=1)
        axis.set(
            title=metric, xlabel="Frame index", ylabel="degrees" if "rotation" in metric else "mm"
        )
    axis = figure.add_subplot(grid[:2, 3])
    for name in MAIN_METHODS:
        local = hole_frame_vector(
            torch.tensor(values[name][:, 7:10]), torch.tensor(values[name][:, 13:17])
        ).numpy()
        axis.plot(time, np.linalg.norm(local[:, 1:], axis=1) * 1000, label=name, color=colors[name])
    gt_local = hole_frame_vector(
        torch.tensor(values["GT"][:, 7:10]), torch.tensor(values["GT"][:, 13:17])
    ).numpy()
    axis.plot(time, np.linalg.norm(gt_local[:, 1:], axis=1) * 1000, "--", color="black")
    axis.set(
        title="Predicted Hole-frame lateral distance\n(inference geometry, not estimation error)",
        xlabel="Frame index",
        ylabel="mm",
    )
    figure.legend(handles.values(), handles.keys(), loc="outside lower center", ncols=6)
    figure.suptitle(f"{path.stem}: {episode}\nFull episode; no warm-up discarded", fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def check_reproduction(predictions, baselines, summary, settings, data):
    """Require original assemblies to reproduce before interpreting hybrid differences."""
    differences = {}
    for name in BASELINE_FIELDS:
        difference = float(np.max(np.abs(predictions[name] - baselines[name])))
        np.testing.assert_allclose(predictions[name], baselines[name], atol=2e-5, rtol=1e-4)
        differences[name] = difference
    reproduced, _ = score_insertion_methods(
        data, {name: predictions[name] for name in BASELINE_FIELDS}, settings
    )
    mean_differences = {}
    for name, source in (("gnn", "gnn"), ("learned_gate", "learned_gate"), ("mamba", "primary")):
        current = reproduced["methods"][name]["groups"]["all"]
        previous = summary["methods"][source]["groups"]["all"]
        mean_differences[name] = {}
        for metric in (
            "peg_position",
            "peg_rotation",
            "hole_position",
            "hole_rotation",
            "relative_3d",
            "lateral",
            "axial",
            "relative_rotation",
        ):
            unit = "deg" if "rotation" in metric else "mm"
            delta = current[metric]["mean_" + unit] - previous[metric]["mean_" + unit]
            if abs(delta) > 0.01:
                raise ValueError(f"{name}/{metric} differs from original mean by {delta}")
            mean_differences[name][metric] = delta
    return {
        "maximum_object21_component_difference": differences,
        "mean_metric_difference_mm_or_deg": mean_differences,
        "scope": "fresh streaming replay vs original CUDA full-episode snapshot, all frames",
    }
