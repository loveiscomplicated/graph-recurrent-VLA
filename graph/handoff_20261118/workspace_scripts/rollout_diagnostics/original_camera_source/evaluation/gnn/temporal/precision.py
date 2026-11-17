"""Shared GRU/Mamba estimation precision; explicitly separate task-head predicates."""

import numpy as np
import torch

from graph.pose_geometry import (
    angular_error_rad,
    geometric_endpoint,
    hole_frame_vector,
    peg_axis,
    relative_rotation,
)

PRECISION_CONTRACT = "peg_head_plus_x_hole_x_axial_yz_lateral_v1"


def precision_errors(prediction, target, endpoint_from_pose=True):
    """Return per-frame geometry errors after inference; GT never enters the model.

    Prediction/target positions may both be world or TCP-relative. Relative
    vectors use world axes. In the GT Hole frame X is axial and YZ lateral.
    A second local-vector error includes uncertainty in predicted Hole rotation.
    """
    relative = geometric_endpoint(prediction) if endpoint_from_pose else prediction[..., 7:10]
    target_relative = target[..., 7:10]
    error = relative - target_relative
    local_error = hole_frame_vector(error, target[..., 13:17])
    predicted_local = hole_frame_vector(relative, prediction[..., 13:17])
    target_local = hole_frame_vector(target_relative, target[..., 13:17])
    predicted_rotation = relative_rotation(prediction[..., 3:7], prediction[..., 13:17])
    target_rotation = relative_rotation(target[..., 3:7], target[..., 13:17])
    predicted_axis, target_axis = peg_axis(predicted_rotation), peg_axis(target_rotation)
    axis_error = torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(predicted_axis, target_axis), dim=-1),
        (predicted_axis * target_axis).sum(-1),
    )
    return {
        "relative_3d_mm": torch.linalg.vector_norm(error, dim=-1) * 1000,
        "lateral_mm": torch.linalg.vector_norm(local_error[..., 1:], dim=-1) * 1000,
        "axial_mm": local_error[..., 0].abs() * 1000,
        "relative_in_predicted_hole_frame_mm": torch.linalg.vector_norm(
            predicted_local - target_local, dim=-1
        )
        * 1000,
        "relative_orientation_deg": torch.rad2deg(
            angular_error_rad(predicted_rotation, target_rotation)
        ),
        "relative_axis_direction_deg": torch.rad2deg(axis_error),
        "signed_gt_hole_x_mm": local_error[..., 0] * 1000,
        "signed_gt_hole_y_mm": local_error[..., 1] * 1000,
        "signed_gt_hole_z_mm": local_error[..., 2] * 1000,
    }


def distribution(values):
    """Summarize nonnegative estimation errors, with explicit empty groups."""
    values = np.asarray(values)
    if not len(values):
        return dict(frames=0, mean=None, median=None, p90=None, p95=None, maximum=None)
    return dict(
        frames=len(values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        maximum=float(values.max()),
    )


def precision_summary(errors, mask):
    """Fractions are state-estimation accuracy, never insertion success rates."""
    count = int(mask.sum())
    metrics = {
        name: distribution(values[mask])
        for name, values in errors.items()
        if not name.startswith("signed_")
    }
    thresholds = {
        f"lateral_lt_{bound}_mm": float(np.mean(errors["lateral_mm"][mask] < bound))
        if count
        else None
        for bound in (1, 2, 3)
    }
    thresholds["relative_3d_lt_3_mm"] = (
        float(np.mean(errors["relative_3d_mm"][mask] < 3)) if count else None
    )
    return {"errors": metrics, "estimation_accuracy_fractions": thresholds, "frames": count}


def precision_diagnostics(data, prediction, groups, maximum_failures=10):
    """Save frame-level arrays, episode worst cases and labeled failures for audit."""
    predicted, target = torch.as_tensor(prediction), torch.as_tensor(data["target"])
    # Score the actual State/Object21 output, including the historical GNN's
    # learned endpoint residual. Pose-reconstructed geometry is a separate ablation.
    errors = {
        name: value.numpy()
        for name, value in precision_errors(predicted, target, endpoint_from_pose=False).items()
    }
    errors["pose_reconstructed_relative_3d_mm"] = (
        torch.linalg.vector_norm(geometric_endpoint(predicted) - target[:, 7:10], dim=-1).numpy()
        * 1000
    )
    # Absolute pose distributions also expose tails/degradation alongside geometry.
    for name, start in (("peg", 0), ("hole", 10)):
        errors[name + "_position_mm"] = (
            np.linalg.norm(
                prediction[:, start : start + 3] - data["target"][:, start : start + 3], axis=1
            )
            * 1000
        )
        errors[name + "_rotation_deg"] = torch.rad2deg(
            angular_error_rad(predicted[:, start + 3 : start + 7], target[:, start + 3 : start + 7])
        ).numpy()
    summaries = {
        "contract": PRECISION_CONTRACT,
        "groups": {name: precision_summary(errors, mask) for name, mask in groups.items()},
        "episodes": {
            str(episode): precision_summary(errors, data["episode"] == episode)
            for episode in np.unique(data["episode"])
        },
    }
    worst = np.argsort(errors["relative_3d_mm"])[::-1][:maximum_failures]
    summaries["worst_relative_frames"] = [
        {
            "sample_id": str(data["sample_ids"][index]),
            "episode": str(data["episode"][index]),
            "frame": int(data["frame"][index]),
            **{name: float(values[index]) for name, values in errors.items()},
        }
        for index in worst
    ]
    failure = (errors["relative_3d_mm"] >= 3.0) | (errors["lateral_mm"] >= 3.0)
    summaries["precision_failure_frames"] = int(failure.sum())
    summaries["worst_precision_failures"] = [
        row
        for row in summaries["worst_relative_frames"]
        if row["relative_3d_mm"] >= 3.0 or row["lateral_mm"] >= 3.0
    ]
    actual_local = hole_frame_vector(target[:, 7:10], target[:, 13:17]).numpy()
    summaries["gt_physical_alignment"] = {
        "meaning": "GT head-centre geometry, not estimation error or a rollout success rate",
        "lateral_distance_mm": distribution(np.linalg.norm(actual_local[:, 1:], axis=1) * 1000),
        "head_position_predicate_frames": int(
            np.sum(
                (actual_local[:, 0] >= -0.015)
                & (np.abs(actual_local[:, 1]) <= data["target"][:, 20])
                & (np.abs(actual_local[:, 2]) <= data["target"][:, 20])
            )
        ),
        "predicate": "hole-frame head x>=-0.015m and abs(y),abs(z)<=GT hole_radius; no orientation flag",
    }
    return summaries, errors


def motion_precision_groups(data, settings):
    """Offline GT/proprioceptive slices; no grasp state is inferred from geometry."""
    count = len(data["frame"])
    reversal = np.zeros(count, dtype=bool)
    together = np.zeros(count, dtype=bool)
    alignment = np.zeros(count, dtype=bool)
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        world = data["target"][rows, :3] + data["tcp_world"][rows]
        peg_motion = np.diff(world, axis=0)
        tcp_motion = np.diff(data["tcp_world"][rows], axis=0)
        speed = np.linalg.norm(peg_motion, axis=1)
        tcp_speed = np.linalg.norm(tcp_motion, axis=1)
        difference = np.linalg.norm(peg_motion - tcp_motion, axis=1)
        together[rows[1:]] = (
            (speed > settings["motion_threshold_m"])
            & (tcp_speed > settings["motion_threshold_m"])
            & (difference <= settings["co_motion_tolerance_m"])
        )
        if len(rows) > 2:
            dot = (peg_motion[1:] * peg_motion[:-1]).sum(-1)
            cosine = dot / np.maximum(speed[1:] * speed[:-1], 1e-12)
            reversal[rows[2:]] = (
                (speed[1:] > settings["motion_threshold_m"])
                & (speed[:-1] > settings["motion_threshold_m"])
                & (cosine < settings["direction_change_cosine"])
            )
        local = hole_frame_vector(
            torch.tensor(data["target"][rows, 7:10]), torch.tensor(data["target"][rows, 13:17])
        ).numpy()
        alignment[rows] = (
            (local[:, 0] >= settings["alignment_axial_min_m"])
            & (local[:, 0] <= settings["alignment_axial_max_m"])
            & (np.linalg.norm(local[:, 1:], axis=1) <= settings["alignment_lateral_m"])
        )
    return {
        "peg_tcp_co_motion_proxy": together,
        "fast_direction_change": reversal,
        "insertion_alignment_geometry_proxy": alignment,
    }


def validation_precision(prediction, target, mask, settings):
    """Compute selection arrays from validation labels only; labels stay outside input."""
    prediction, target = prediction.detach(), target.detach()
    result = {
        name: values.cpu().numpy()
        for name, values in precision_errors(prediction[mask], target[mask]).items()
    }
    pair_valid = mask[:, 1:] & mask[:, :-1]
    moving = torch.zeros_like(mask)
    pair_rotation = torch.zeros_like(target[:, 1:, 0])
    if pair_valid.any():
        pair_rotation[pair_valid] = angular_error_rad(
            target[:, 1:, 3:7][pair_valid], target[:, :-1, 3:7][pair_valid]
        )
    motion = torch.linalg.vector_norm(target[:, 1:, :3] - target[:, :-1, :3], dim=-1)
    moving[:, 1:] = pair_valid & (
        (motion > settings["motion_threshold_m"])
        | (pair_rotation > settings["motion_threshold_rad"])
    )
    for name, start in (("peg", 0), ("hole", 10)):
        position = (
            torch.linalg.vector_norm(
                prediction[mask][:, start : start + 3] - target[mask][:, start : start + 3], dim=-1
            )
            * 1000
        )
        rotation = torch.rad2deg(
            angular_error_rad(
                prediction[mask][:, start + 3 : start + 7], target[mask][:, start + 3 : start + 7]
            )
        )
        result[name + "_position_mm"] = position.cpu().numpy()
        result[name + "_rotation_deg"] = rotation.cpu().numpy()
        if name == "peg":
            result["peg_moving_position_mm"] = position[moving[mask]].cpu().numpy()
            result["peg_moving_rotation_deg"] = rotation[moving[mask]].cpu().numpy()
    return result


def selection_score(metrics, settings):
    """Weighted validation geometry/motion/absolute score; never use final-test metrics."""
    components = {}
    for name, specification in settings["components"].items():
        value = metrics[name]
        if value is None and name.startswith("peg_moving_"):
            value = metrics[name.replace("peg_moving_", "peg_")]
        if value is None:
            raise ValueError(f"Checkpoint selection metric is unavailable: {name}")
        components[name] = float(value) / specification["scale"] * specification["weight"]
    return sum(components.values()), components


def save_precision_plots(output, data, errors, maximum):
    """Plot relative/lateral/axial estimation errors and directed relative rotation."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = [
        ("relative_3d_mm", "Relative 3D estimation error (mm)"),
        ("lateral_mm", "Lateral estimation error (mm)"),
        ("axial_mm", "Axial estimation error (mm)"),
        ("relative_orientation_deg", "Relative orientation error (deg)"),
    ]
    for number, episode in enumerate(np.unique(data["episode"])[:maximum]):
        rows = data["episode"] == episode
        frames = data["frame"][rows]
        figure, axes = plt.subplots(2, 2, sharex=True, figsize=(13, 7))
        for axis, (field, title) in zip(axes.flat, panels):
            for name, values in errors.items():
                axis.plot(frames, values[field][rows], label=name)
            if field in ("relative_3d_mm", "lateral_mm"):
                axis.axhline(
                    3.0,
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                    label="3 mm estimation reference",
                )
            axis.set_ylabel(title)
            axis.set_xlabel("Frame")
        axes[0, 0].legend(fontsize=6)
        figure.suptitle(f"{episode}: estimation precision (not insertion success)")
        figure.tight_layout()
        figure.savefig(output / f"precision_trace_{number:03d}.png", dpi=130)
        plt.close(figure)
