"""Static, exportable figures for the BOX/Hole diagnostic report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_collection.storage import write_json
from evaluation.gnn.box_hole_geometry import physical_vertices


def draw_wireframe(axis, points, color, alpha=.5):
    """Connect sign-labelled cuboid corners along its twelve physical edges."""
    for left in range(8):
        for right in range(left + 1, 8):
            if (left ^ right) in (1, 2, 4):
                line = points[[left, right]] * 1000
                axis.plot(*line.T, color=color, alpha=alpha, linewidth=.8)


def make_plots(output, geometry, states, coverage, groups, inference, fitting):
    """Plot paired Oracles, geometric support, signed bias and selected 3D cases."""
    import json
    summary = json.loads((output / "summary.json").read_text())
    original = inference["original"]["combined"]
    errors = original["errors"]["hole_position_error_mm"]
    counts = (geometry["object_type"] == 1).sum(1)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, values, title in zip(axes, (counts, coverage["unique_rim_candidates_3mm"], coverage["rank_1mm"]),
                                    ("Observed BOX node count", "Unique rim proximity candidates (3 mm)", "Observed rank (1 mm singular threshold)")):
        support = values > 0 if title.startswith("Observed BOX") else groups["target_present"]
        bins = sorted(set(values[support].astype(int)))
        data = [errors[(values == value) & support] for value in bins]
        axis.boxplot(data, positions=bins, showfliers=False, widths=.55)
        axis.set_xticks(bins, [str(value) for value in bins], fontsize=8)
        axis.set(xlabel=title, ylabel="AB Hole error (mm)")
        axis.xaxis.label.set_fontsize(9)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Validation observation strata; proximity does not certify identity")
    figure.tight_layout()
    figure.savefig(output / "observation_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    condition_labels = {"original": "Original", "gt_outer_original_peg": "GT outer",
                        "gt_rim_original_peg": "GT rim", "gt_target_original_peg": "GT target",
                        "full_gt": "Full GT"}
    for axis, group in zip(axes, ("target_present", "high_support_proxy")):
        labels, means, intervals = [], [], []
        for condition in inference:
            for model in ("baseline", "combined"):
                row = summary["groups"][group]["conditions"][condition][model]["hole_position_error_mm"]
                labels.append(f"{condition_labels[condition]}\n{'Base' if model == 'baseline' else 'AB'}")
                means.append(row["mean"])
                intervals.append([row["mean"] - row["parent_mean_ci95"]["low"], row["parent_mean_ci95"]["high"] - row["mean"]])
        axis.bar(range(len(means)), means, color=["#6b91b8", "#d87946"] * len(inference))
        axis.errorbar(range(len(means)), means, yerr=np.asarray(intervals).T, fmt="none", color="black", capsize=2)
        axis.set_xticks(range(len(means)), labels, rotation=35, ha="right", fontsize=8)
        axis.set(title=group, ylabel="Mean Hole position error (mm)")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Frozen checkpoints, same frames; parent bootstrap 95% CI\nGT conditions change visibility and information")
    figure.tight_layout()
    figure.savefig(output / "oracle_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    signed = (original["arrays"]["object21"][:, 10:13] - geometry["target"][:, 10:13]) * 1000
    specifications = (("half_length_mm", [85, 95, 105, 115, 125.001], "Fixture half-length (mm)"),
                      ("yaw_deg", [67.49, 78.75, 90, 101.25, 112.51], "Fixture yaw (degrees)"),
                      ("unique_rim_candidates_3mm", [0, 1, 3, 5, 9], "Rim proximity candidate count"))
    for axis, (key, edges, label) in zip(axes, specifications):
        centers = (np.asarray(edges[1:]) + np.asarray(edges[:-1])) / 2
        for coordinate, name in enumerate("XYZ"):
            means = []
            for lower, upper in zip(edges[:-1], edges[1:]):
                mask = groups["target_present"] & (coverage[key] >= lower) & (coverage[key] < upper)
                means.append(signed[mask, coordinate].mean() if mask.any() else np.nan)
            axis.plot(centers, means, marker="o", label=name)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel=label, ylabel="Signed predicted - GT error (mm)")
        axis.legend()
        axis.grid(alpha=.2)
    figure.suptitle("AB conditional signed bias; descriptive validation associations")
    figure.tight_layout()
    figure.savefig(output / "signed_bias.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for family in ("outer", "rim"):
        for subset in ("triangle", "one_face", "full"):
            means = [np.mean([row["hole_error_mm"] for row in fitting
                              if row["family"] == family and row["subset"] == subset and row["sigma_mm"] == sigma])
                     for sigma in (0, .5, 1)]
            axis.plot((0, .5, 1), means, marker="o", label=f"{family}/{subset}")
    axis.set(xlabel="Independent isotropic coordinate noise sigma (mm)", ylabel="GT fitting Hole error (mm)",
             title="Known identity + shape; outer fitting additionally uses GT Hole offset")
    axis.legend(fontsize=7)
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "geometric_noise.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    cases, selected = [], set()
    for group, ranking in (("proximity_rim_ge3", "worst"), ("proximity_rim_ge3", "best"),
                           ("proximity_outer_ge3_no_rim", "worst"), ("high_support_proxy", "worst")):
        candidates = [index for index in np.flatnonzero(groups[group]) if index not in selected]
        if not candidates:
            continue
        index = max(candidates, key=lambda i: errors[i]) if ranking == "worst" else min(candidates, key=lambda i: errors[i])
        selected.add(index)
        cases.append(dict(index=int(index), sample_id=str(geometry["sample_ids"][index]), group=group,
                          selection=ranking, ab_hole_error_mm=float(errors[index])))
    figure = plt.figure(figsize=(12, 10))
    for number, case in enumerate(cases):
        index = case["index"]
        state = states[str(Path(case["sample_id"]).parent)]
        outer, rim = physical_vertices(state["box_pose"], state["hole_pose"], state["peg_half_sizes"][0], state["hole_radius"])
        origin = np.asarray(state["hole_pose"])[:3]
        points = geometry["xyz_world"][index][geometry["object_type"][index] == 1]
        prediction = original["arrays"]["object21"][index, 10:13] + geometry["tcp_world"][index]
        anchor = original["arrays"]["hole_anchor"][index] + geometry["tcp_world"][index]
        axis = figure.add_subplot(2, 2, number + 1, projection="3d")
        draw_wireframe(axis, outer - origin, "#999999")
        draw_wireframe(axis, rim - origin, "#469d8c")
        axis.scatter(*((outer - origin) * 1000).T, s=12, color="gray", label="GT outer")
        axis.scatter(*((rim - origin) * 1000).T, s=15, color="#469d8c", label="GT rim")
        axis.scatter(*((points - origin) * 1000).T, s=25, color="black", label="Observed BOX")
        axis.scatter(0, 0, 0, color="green", marker="*", s=100, label="GT Hole")
        axis.scatter(*((prediction - origin) * 1000), color="red", marker="x", s=60, label="AB Hole")
        axis.scatter(*((anchor - origin) * 1000), color="blue", marker="+", s=60, label="Target anchor")
        extent = max(130, float(np.abs(np.concatenate((outer - origin, points - origin, (prediction - origin)[None]))).max() * 1050))
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent),
                 xlabel="World X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                 title=f"{case['group']}/{case['selection']}\nAB error {errors[index]:.2f} mm; {len(points)} nodes")
        axis.set_box_aspect((1, 1, 1))
        if number == 0:
            axis.legend(fontsize=6, loc="upper left")
        case.update(observed_box_world_m=points.tolist(), gt_outer_world_m=outer.tolist(), gt_rim_world_m=rim.tolist(),
                    gt_hole_world_m=origin.tolist(), predicted_hole_world_m=prediction.tolist(), anchor_world_m=anchor.tolist())
    figure.suptitle("Selected cases in world axes relative to GT Hole\nEqual XYZ units within each panel; no inferred correspondences")
    figure.tight_layout()
    figure.savefig(output / "geometry_cases.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_json(output / "cases.json", cases)
