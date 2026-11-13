"""Static export figures for offline Hybrid error attribution; no model execution."""

import numpy as np

from evaluation.gnn.temporal.hybrid_error_geometry import endpoint_position

COLORS = {"mamba": "#2673b8", "hybrid_h1_a": "#168451"}


def save_episode_trace(path, episode, data, target, predictions, errors, arrays):
    """Trace world endpoints, Hole estimates, signed geometry and cancellation."""
    import matplotlib.pyplot as plt

    rows = np.flatnonzero(data["episode"] == episode)
    time = data["frame"][rows]
    tcp = data["tcp_world"][rows]
    figure, axes = plt.subplots(6, 3, figsize=(17, 20), layout="constrained")
    endpoint_gt = endpoint_position(target)[rows] + tcp
    endpoint_pred = endpoint_position(predictions["mamba"])[rows] + tcp
    for component, axis_name in enumerate("xyz"):
        axis = axes[0, component]
        axis.plot(time, endpoint_gt[:, component], "--", color="black", label="GT")
        axis.plot(time, endpoint_pred[:, component], color=COLORS["mamba"], label="Mamba = Hybrid")
        axis.set(title=f"Endpoint world {axis_name}", ylabel="m")
        axis = axes[1, component]
        axis.plot(
            time, target[rows, 10 + component] + tcp[:, component], "--", color="black", label="GT"
        )
        for name, value in predictions.items():
            axis.plot(
                time,
                value[rows, 10 + component] + tcp[:, component],
                color=COLORS[name],
                label="Gate / H1" if name == "hybrid_h1_a" else "Mamba",
            )
        axis.set(title=f"Hole world {axis_name}", ylabel="m")
    for column, metric in enumerate(("relative_3d", "lateral", "axial")):
        axis = axes[2, column]
        for name in COLORS:
            axis.plot(time, errors[name][metric][rows], color=COLORS[name], label=name)
        axis.set(title=metric + " estimation error", ylabel="mm")
    axis = axes[3, 0]
    axis.plot(time, errors["mamba"]["endpoint_position"][rows], color="#8a42ac", label="Endpoint")
    for name in COLORS:
        axis.plot(
            time, errors[name]["hole_position"][rows], color=COLORS[name], label=name + " Hole"
        )
    axis.set(title="Absolute point errors", ylabel="mm")
    for column, term in ((1, "cosine"), (2, "dot_mm2")):
        axis = axes[3, column]
        for name in COLORS:
            axis.plot(time, arrays[name + "__" + term][rows], color=COLORS[name], label=name)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set(
            title="Endpoint-Hole error "
            + ("direction cosine" if term == "cosine" else "dot product"),
            ylabel="unitless" if term == "cosine" else "mm²",
        )
    for column, axis_name in enumerate("xyz"):
        axis = axes[4, column]
        for name in COLORS:
            axis.plot(
                time,
                arrays[name + "__signed_gt_hole_" + axis_name][rows],
                color=COLORS[name],
                label=name,
            )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set(title=f"Signed relative error: GT Hole {axis_name}", ylabel="mm")
    axis = axes[5, 0]
    for replacement, label in (
        (0, "Predicted"),
        (1, "GT position"),
        (2, "GT rotation"),
        (4, "GT size"),
    ):
        values = arrays[f"oracle_{replacement}__endpoint_error_m"][rows]
        axis.plot(time, np.linalg.norm(values, axis=1) * 1000, label=label)
    axis.set(title="Oracle Endpoint interventions", ylabel="mm")
    axis = axes[5, 1]
    for field, label in (
        ("hole_squared_gain_mm2", "Hole squared gain"),
        ("lost_cancellation_mm2", "Lost cancellation"),
        ("relative_squared_delta_mm2", "Relative squared change"),
    ):
        axis.plot(time, arrays[field][rows], label=label)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set(title="H1 minus Mamba MSE identity", ylabel="mm²")
    axis = axes[5, 2]
    local = arrays["gt_local_relative_m"][rows] * 1000
    axis.plot(time, local[:, 0], label="GT head X")
    axis.plot(time, np.linalg.norm(local[:, 1:], axis=1), label="GT lateral distance")
    axis.plot(time, arrays["gt_axis_alignment_deg"][rows], label="GT axis angle (deg)")
    axis.set(title="Offline geometry (not phase/contact GT)", ylabel="mm / deg")
    for axis in axes.flat:
        axis.set_xlabel("Saved frame index")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.15)
    figure.suptitle(
        episode + "\nFrozen predictions; GT used only for offline diagnosis", fontsize=16
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_plots(output, data, target, predictions, errors, summary, arrays, groups):
    """Export ten extreme episodes plus overview and phase-specific oracle charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representatives = summary["representatives"]
    episodes = list(
        dict.fromkeys(
            representatives["most_improved"]
            + representatives["most_degraded"]
            + [representatives["required"]]
        )
    )
    for episode in episodes:
        filename = "episode_" + episode.replace("/", "_") + ".png"
        save_episode_trace(output / filename, episode, data, target, predictions, errors, arrays)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for column, metric in enumerate(("relative_3d", "lateral", "axial")):
        axis = axes[0, column]
        for name in COLORS:
            values = np.sort(errors[name][metric])
            axis.plot(
                values, np.arange(1, len(values) + 1) / len(values), label=name, color=COLORS[name]
            )
        axis.set(xlabel="Estimation error (mm)", ylabel="ECDF", title=metric)
        axis.legend()
    axis = axes[1, 0]
    for name in COLORS:
        terms = summary["decomposition"]["all"][name]
        keys = (
            "endpoint_squared_mm2",
            "hole_squared_mm2",
            "cross_term_mm2",
            "relative_squared_mm2",
        )
        offset = -0.18 if name == "mamba" else 0.18
        axis.bar(
            np.arange(4) + offset,
            [terms[key] for key in keys],
            width=0.36,
            color=COLORS[name],
            label=name,
        )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(
        xticks=range(4),
        xticklabels=["Endpoint²", "Hole²", "-2 E·H", "Relative²"],
        ylabel="Mean mm²",
        title="Exact squared decomposition",
    )
    axis.legend()
    labels = [
        "all",
        "alignment_geometry_proxy",
        "insertion_approach_geometry_proxy",
        "head_in_region_geometry_proxy",
    ]
    axis = axes[1, 1]
    for replacement, label in (
        (0, "Predicted"),
        (1, "GT position"),
        (2, "GT rotation"),
        (4, "GT size"),
    ):
        means = [
            summary["endpoint_oracles"][group]["conditions"][str(replacement)]["endpoint_mm"][
                "mean_mm"
            ]
            for group in labels
        ]
        axis.plot(range(len(labels)), means, marker="o", label=label)
    axis.set(
        xticks=range(4),
        xticklabels=["All", "Alignment", "Approach", "Head region"],
        ylabel="Endpoint error mm",
        title="Diagnostic-only oracle replacements",
    )
    axis.legend(fontsize=8)
    axis = axes[1, 2]
    terms = summary["cancellation"]["axis_change_by_group"]["all"]
    for index, (key, label) in enumerate(
        (
            ("hole_squared_gain_mm2", "Hole gain"),
            ("lost_cancellation_mm2", "Lost cancellation"),
            ("relative_squared_delta_mm2", "Relative change"),
        )
    ):
        axis.bar(
            np.arange(3) + (index - 1) * 0.25,
            [terms[name][key] for name in terms],
            width=0.25,
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(
        xticks=range(3),
        xticklabels=["X axial", "Y lateral", "Z lateral"],
        ylabel="Mean mm²",
        title="Why Hole replacement helps or harms",
    )
    axis.legend(fontsize=8)
    figure.suptitle(
        f"Hybrid-H1 error attribution: reused validation, {summary['episodes']} episodes / {summary['frames']:,} frames"
    )
    figure.savefig(output / "error_attribution_overview.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    deltas = summary["paired"]["all"]["relative_3d"]["episode_mean_deltas_mm"]
    ordered = sorted(deltas, key=deltas.get)
    axes[0].bar(range(len(ordered)), [deltas[episode] for episode in ordered], color="#168451")
    axes[0].set(
        title="Paired episode mean relative change",
        xlabel="Episodes ordered by change",
        ylabel="H1 minus Mamba (mm)",
    )
    for column, metric in ((1, "lateral"), (2, "axial")):
        axis = axes[column]
        for name in COLORS:
            values = [
                summary["groups"][group][name]["metrics"][metric]["mean_mm"] for group in labels
            ]
            axis.plot(range(4), values, marker="o", color=COLORS[name], label=name)
        axis.set(
            xticks=range(4),
            xticklabels=["All", "Alignment", "Approach", "Head region"],
            ylabel="Mean estimation error mm",
            title=metric + " across geometry proxies",
        )
        axis.legend()
    figure.savefig(output / "episode_and_geometry_comparison.png", dpi=160)
    plt.close(figure)
