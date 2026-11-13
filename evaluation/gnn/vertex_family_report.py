"""Export scientific figures and residual diagnostics from completed family runs."""

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.storage import write_json
from data_collection.vertex_family_labels import physical_references, pose_rotation
from evaluation.gnn.vertex_family import MODES, annotation_groups, summarize_metric
from evaluation.gnn.box_hole_plots import draw_wireframe
from evaluation.gnn.vertex_metrics import (
    correlation, statistics,
)
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


COLORS = ("#637a91", "#e09442", "#479887", "#a078ad")
NAMES = dict(baseline="M0: XYZ/type", detector="M1: detector", oracle="M2: Oracle", type_control="Type control")


def local_path(path, config):
    """Resolve an artifact after copying its preserved remote directory tree."""
    path = Path(path)
    if path.exists():
        return path
    remote = Path(config["remote_prefix"])
    if path.is_relative_to(remote):
        candidate = Path(config["local_prefix"]) / path.relative_to(remote)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required report artifact is unavailable: {path}")


def save_figure(figure, output, name):
    """Save readable static scientific figures without clipping labels."""
    figure.tight_layout()
    figure.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(summary, output):
    """Compare fixed-seed mean errors with whole-Parent bootstrap intervals."""
    selected = ("all", "target_present", "fully_certified_target", "oracle_unique_rim_ge3")
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for axis, group in zip(axes, selected):
        row = summary["groups"][group]
        for index, mode in enumerate(MODES):
            result = row["modes"][mode]["hole_position_error_mm"]
            interval = result["parent_mean_ci95"]
            axis.bar(index, result["mean"], color=COLORS[index])
            axis.errorbar(index, result["mean"], yerr=[[result["mean"] - interval["low"]],
                                                      [interval["high"] - result["mean"]]], color="black", capsize=3)
        axis.set_xticks(range(4), ["M0", "M1", "M2", "Control"])
        axis.set(title=f"{group}\nn={row['frames']}, parents={row['parents']}", ylabel="Hole error (mm)")
        axis.title.set_fontsize(9)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Same coordinates and visibility; mean error over seeds 0,1,2\nParent bootstrap 95% CI is conditional on these training seeds")
    save_figure(figure, output, "hole_comparison.png")


def plot_observation_groups(summary, output):
    """Keep detector-defined and GT-certified subgroup interpretations separate."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for axis, prefix in zip(axes, ("detector", "oracle")):
        groups = [f"{prefix}_{suffix}" for suffix in ("exterior_only", "mixed", "rim_1_2", "rim_ge3")]
        x = np.arange(len(groups))
        for index, mode in enumerate(MODES):
            means = [summary["groups"][group]["modes"][mode]["hole_position_error_mm"]["mean"] for group in groups]
            axis.bar(x + (index - 1.5) * .18, means, width=.18, color=COLORS[index], label=NAMES[mode])
        axis.set_xticks(x, [f"{group.removeprefix(prefix + '_')}\nn={summary['groups'][group]['frames']}" for group in groups], fontsize=8)
        axis.set(title=f"{prefix} annotations (groups overlap)", ylabel="Hole error (mm)")
        axis.legend(fontsize=7)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("GT-certified subgroups are used only for analysis; uncertified nodes may remain")
    save_figure(figure, output, "observation_groups.png")


def plot_labels_and_interventions(summary, interventions, output):
    """Show the measured label confusion and fixed-weight label sensitivity."""
    confusion = np.asarray(summary["label_quality"]["confusion_gt_rows_exterior_rim_detector_columns_exterior_rim_unknown"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(3):
            axes[0].text(column, row, str(confusion[row, column]), ha="center", va="center")
    axes[0].set(xticks=range(3), xticklabels=["Exterior", "Rim", "Unknown"],
                yticks=range(2), yticklabels=["Exterior", "Rim"], xlabel="Detector-derived mapping", ylabel="Certified GT family",
                title="Certified validation nodes only")
    for index, mode in enumerate(("detector", "oracle")):
        values = [summary["groups"]["target_present"]["modes"][mode]["hole_position_error_mm"]["mean"]]
        values.extend(interventions[mode][name]["target_present"]["hole_position"]["mean"] for name in ("erase", "swap"))
        axes[1].plot(range(3), values, marker="o", color=COLORS[index + 1], label=NAMES[mode])
    axes[1].set(xticks=range(3), xticklabels=["Natural", "Erase", "Swap"], ylabel="Hole error (mm)",
                title="Fixed weights, same target-present frames")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    save_figure(figure, output, "labels_and_interventions.png")


def plot_training_curves(summary, config, output):
    """Show every seed's original history; do not select a new best checkpoint."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for index, mode in enumerate(MODES):
        histories = []
        for run in summary["runs"]:
            if run["mode"] == mode:
                histories.append(json.loads(local_path(Path(run["output"]) / "history.json", config).read_text()))
        for axis, metric in zip(axes, ("loss", "hole_position_error_m")):
            values = np.asarray([[row["validation"][metric] for row in history] for history in histories])
            if metric.endswith("_m"):
                values *= 1000
            epochs = np.arange(1, values.shape[1] + 1)
            axis.plot(epochs, values.mean(0), color=COLORS[index], label=NAMES[mode])
            axis.fill_between(epochs, values.min(0), values.max(0), color=COLORS[index], alpha=.12)
            axis.set(xlabel="Epoch", ylabel="Validation total loss" if metric == "loss" else "Validation Hole error (mm)")
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Mean and min-max across three training seeds; checkpoint selection uses total loss")
    save_figure(figure, output, "training_curves.png")


def residual_diagnostics(evaluation, features, groups, parents, seeds):
    """Measure remaining bias and head-offset errors with GT confined to scoring."""
    result = {}
    for mode in MODES:
        signed, required, predicted = [], [], []
        for seed in seeds:
            with np.load(evaluation / f"{mode}_seed_{seed}_head_diagnostics.npz", allow_pickle=False) as archive:
                signed.append(archive["signed_error_mm"])
                required.append(archive["desired_offset"])
                predicted.append(archive["predicted_offset"])
        signed_stack = np.asarray(signed)
        mean_error = np.linalg.norm(signed_stack, axis=2).mean(axis=0)
        required_norm = np.linalg.norm(np.asarray(required), axis=2).mean(axis=0) * 1000
        predicted_norm = np.linalg.norm(np.asarray(predicted), axis=2).mean(axis=0) * 1000
        signed = signed_stack.mean(axis=0)
        result[mode] = {}
        for group, mask in groups.items():
            values = dict(required_offset_norm_mm=required_norm,
                          predicted_offset_norm_mm=predicted_norm, hole_error_mm=mean_error)
            values.update({f"signed_{name}_mm": signed[:, axis] for axis, name in enumerate("xyz")})
            result[mode][group] = {name: summarize_metric(value, parents, mask) for name, value in values.items()}
        target = groups["target_present"]
        result[mode]["descriptive_correlations"] = {
            name: correlation(mean_error[target], features[name][target])
            for name in ("half_length_mm", "hole_radius_mm", "oracle_unique_rim")}
    return result


def observed_rim_geometry(root, rows, groups):
    """Describe certified physical coverage and remaining coordinate noise for scoring."""
    values = {name: [] for name in ("rim_s2_mm", "rim_y_span_fraction", "rim_z_span_fraction",
                                    "both_mouths", "rim_coordinate_error_mm")}
    for row in rows:
        with np.load(root / row["labels_path"], allow_pickle=False) as archive:
            mask = archive["oracle_family"] == 1
            points = archive["xyz_world"][mask].astype(np.float64)
            physical_ids = archive["physical_id"][mask]
            pose = archive["hole_pose"].reshape(7)
            local = (points - pose[:3]) @ pose_rotation(pose)
            radius = float(archive["hole_radius"].reshape(-1)[0])
            if len(points) > 1:
                singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
                second = singular[1] * 1000
                spans = np.ptp(local[:, 1:], axis=0) / (2 * radius)
            else:
                second, spans = 0., np.zeros(2)
            both = bool((physical_ids < 12).any() and (physical_ids >= 12).any())
            references = archive["physical_xyz_world"][physical_ids]
            error = float(np.linalg.norm(points - references, axis=1).mean() * 1000) if len(points) else np.nan
            values["rim_s2_mm"].append(second)
            values["rim_y_span_fraction"].append(spans[0])
            values["rim_z_span_fraction"].append(spans[1])
            values["both_mouths"].append(float(both))
            values["rim_coordinate_error_mm"].append(error)
    arrays = {name: np.asarray(value) for name, value in values.items()}
    result = {}
    for group in ("target_present", "oracle_unique_rim_ge3", "fully_certified_target"):
        result[group] = {}
        for name, array in arrays.items():
            selected = array[groups[group]]
            result[group][name] = statistics(selected[np.isfinite(selected)])
    return result


def plot_size_residuals(summary, output):
    """Show the predeclared fixture-size strata without implying a causal size effect."""
    groups = [f"half_length_{lower}_{upper}" for lower, upper in
              ((85, 95), (95, 105), (105, 115), (115, "125.001"))]
    figure, axis = plt.subplots(figsize=(7, 4))
    for index, mode in enumerate(MODES):
        means = [summary["groups"][group]["modes"][mode]["hole_position_error_mm"]["mean"] for group in groups]
        axis.plot(range(4), means, marker="o", color=COLORS[index], label=NAMES[mode])
    axis.set(xticks=range(4), xticklabels=["85-95", "95-105", "105-115", "115-125"],
             xlabel="Fixture half-length (mm)", ylabel="Hole error (mm)",
             title="Target-present size strata; visibility and parents are not matched")
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    save_figure(figure, output, "size_residuals.png")


def plot_geometry_cases(root, rows, features, seeds, evaluation, output):
    """Inspect raw RGB-D evidence and 3D corner roles on predeclared case types."""
    selected = []
    candidates = np.flatnonzero(features["wrong_certified"] > 0)
    if len(candidates):
        selected.append(int(candidates[np.argmax(features["wrong_certified"][candidates])]))
    candidates = np.flatnonzero(features["oracle_unique_rim"] >= 3)
    if len(candidates):
        errors_by_seed = []
        for seed in seeds:
            with np.load(evaluation / f"oracle_seed_{seed}_natural.npz", allow_pickle=False) as archive:
                errors_by_seed.append(archive["hole_position_error_mm"])
        mean_errors = np.mean(errors_by_seed, axis=0)
        selected.append(int(candidates[np.argmax(mean_errors[candidates])]))
    selected = list(dict.fromkeys(selected))
    figure = plt.figure(figsize=(12, 5.5 * len(selected)))
    case_rows = []
    for number, index in enumerate(selected):
        row = rows[index]
        observation, gt = read_frame(root / row["raw_path"], "base_camera")
        with np.load(root / row["labels_path"], allow_pickle=False) as archive:
            uv, xyz = archive["uv"], archive["xyz_world"]
            detected, oracle = archive["detector_family"], archive["oracle_family"]
            target = archive["object_type"] == 1
        image_axis = figure.add_subplot(len(selected), 2, number * 2 + 1)
        image_axis.imshow(observation["rgb"])
        for family, color, name in ((0, "orange", "Exterior"), (1, "cyan", "Rim"), (-1, "magenta", "Unknown")):
            mask = target & (oracle == family)
            image_axis.scatter(*uv[mask].T, facecolors="none", edgecolors=color, s=65, label=name)
        for point, family, prediction, is_target in zip(uv, oracle, detected, target):
            if is_target and family >= 0 and prediction != family:
                image_axis.text(*point, "x", color="red", fontsize=14)
        image_axis.set(title="Certified roles; red x = detector disagreement", xlim=(0, 512), ylim=(512, 0))
        image_axis.legend(fontsize=7)
        axis = figure.add_subplot(len(selected), 2, number * 2 + 2, projection="3d")
        origin = np.asarray(gt["hole_pose"]).reshape(7)[:3]
        references = physical_references(gt)
        for family, color, name in ((0, "orange", "Exterior"), (1, "#2b9ca3", "Rim"), (-1, "magenta", "Unknown")):
            mask = target & (oracle == family)
            axis.scatter(*((xyz[mask] - origin) * 1000).T, color=color, label=name)
        reference_xyz = np.asarray([reference["world"] for reference in references])
        draw_wireframe(axis, reference_xyz[:8] - origin, "gray", alpha=.3)
        draw_wireframe(axis, reference_xyz[8:] - origin, "#2b9ca3", alpha=.3)
        axis.scatter(*((reference_xyz - origin) * 1000).T, color="gray", alpha=.3, s=12, label="GT references")
        axis.scatter(0, 0, 0, color="green", marker="*", s=100, label="GT Hole")
        predictions = {}
        for mode, color in zip(("baseline", "detector", "oracle"), COLORS):
            with np.load(evaluation / f"{mode}_seed_0_natural.npz", allow_pickle=False) as archive:
                relative = archive["object21"][index, 10:13]
            world = relative + np.asarray(gt["ee_pose"]).reshape(7)[:3]
            predictions[mode] = world.tolist()
            axis.scatter(*((world - origin) * 1000), color=color, marker="x", s=55, label=NAMES[mode])
        axis.set(xlabel="World X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title="GT Hole-relative geometry; seed-zero predictions")
        all_points = np.concatenate((reference_xyz - origin, xyz[target] - origin,
                                     np.asarray(list(predictions.values())) - origin))
        extent = max(130, float(np.abs(all_points).max() * 1100))
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
        axis.set_box_aspect((1, 1, 1))
        axis.legend(fontsize=6)
        case_rows.append(dict(sample_id=row["sample_id"], parent_id=row["parent_id"],
                              detected_family=detected.tolist(), certified_family=oracle.tolist(),
                              uv=uv.tolist(), observed_xyz_world=xyz.tolist(), predictions_world=predictions))
    figure.suptitle("Selected diagnostic cases, not representative performance estimates")
    save_figure(figure, output, "geometry_cases.png")
    write_json(output / "cases.json", case_rows)


def export(config):
    """Produce report figures, residual tables and a final source snapshot locally."""
    root = Path(config["family_dataset"])
    evaluation = Path(config["evaluation"])
    manifest = read_manifest(root)
    summary = json.loads((evaluation / "summary.json").read_text())
    if summary["core_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Report dataset and evaluation identity differ")
    rows, features, groups, quality = annotation_groups(root, manifest)
    parents = np.asarray([row["parent_id"] for row in rows])
    with ExperimentLogger(team="graph", config=config, seed=0, model="vertex_family_report", job_type="analysis") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        interventions = json.loads((evaluation / "interventions.json").read_text())
        plot_comparison(summary, output)
        plot_observation_groups(summary, output)
        plot_labels_and_interventions(summary, interventions, output)
        plot_training_curves(summary, config, output)
        plot_size_residuals(summary, output)
        residual = residual_diagnostics(evaluation, features, groups, parents, summary["seeds"])
        write_json(output / "residual_diagnostics.json", residual)
        write_json(output / "observed_rim_geometry.json", observed_rim_geometry(root, rows, groups))
        plot_geometry_cases(root, rows, features, summary["seeds"], evaluation, output)
        sources = [Path("main.py"), Path("tests/test_vertex_family.py")]
        for directory in ("data_collection", "graph", "graph/build_graph", "graph/utils", "training/gnn",
                          "evaluation/gnn", "utils", "visual/corner_detection", "visual/corner_detection/backends"):
            sources.extend(Path(directory).glob("*.py"))
        hashes = {}
        for source in sources:
            destination = output / "sources" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[str(source)] = file_sha256(source)
        write_json(output / "source_sha256.json", hashes)
        write_json(output / "effective_config.json", config)
        logger.record(core_dataset_sha256=manifest["dataset_sha256"], source_evaluation=str(evaluation))
        for path in output.rglob("*"):
            if path.is_file():
                logger.link_output(path)
        print(f"Vertex family report exports: {output}", flush=True)
    return output
