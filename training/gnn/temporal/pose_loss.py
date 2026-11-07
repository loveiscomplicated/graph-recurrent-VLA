"""Dimensionless pose supervision with padding masks and q/-q equivalence."""

import torch
from torch.nn import functional as F

from graph.pose_geometry import unit_quaternion
from graph.temporal.pose_gru import MODES

INSERTION_ABLATIONS = {
    "pose_only": (),
    "pose_relative": ("relative_position",),
    "pose_relative_lateral": ("relative_position", "lateral_position"),
    "pose_relative_full": ("relative_position", "lateral_position", "relative_orientation"),
}


def pose_loss(prediction, target, mask, mode, settings):
    """Normalize position by metres and rotation by radians; mask before geometry.

    Rotation uses 4*(1-dot(q,q_gt)^2), a sign-invariant smooth squared-angle
    surrogate (approximately angle² near zero), divided by rotation_unit_rad².
    This avoids acos gradient singularities at perfect agreement.
    """
    if (
        prediction.shape != target.shape
        or mask.shape != target.shape[:2]
        or mask.dtype != torch.bool
    ):
        raise ValueError("Expected matching pose[B,T,21] tensors and boolean mask[B,T]")
    if not mask.any():
        raise ValueError("Pose supervision needs at least one valid frame")
    prediction, target = prediction[mask], target[mask]
    terms = {}
    for name in MODES[mode]:
        start = 0 if name == "peg" else 10
        position_error = (
            prediction[:, start : start + 3] - target[:, start : start + 3]
        ) / settings["position_unit_m"]
        terms[name + "_position"] = F.huber_loss(
            position_error, torch.zeros_like(position_error), delta=1.0
        )
        predicted_q = unit_quaternion(prediction[:, start + 3 : start + 7])
        target_q = unit_quaternion(target[:, start + 3 : start + 7])
        # Algebraically 4*(1-dot²), evaluated through sign-invariant chord length
        # so identical quaternions give exactly zero despite float32 dot rounding.
        chord = torch.minimum(
            (predicted_q - target_q).square().sum(-1), (predicted_q + target_q).square().sum(-1)
        )
        terms[name + "_rotation"] = (
            4 * chord * (1 - 0.25 * chord).clamp_min(0) / settings["rotation_unit_rad"] ** 2
        ).mean()
    loss = sum(settings["weights"][name] * value for name, value in terms.items())
    if not torch.isfinite(loss):
        raise ValueError("Pose loss must be finite")
    return loss, terms


def insertion_loss(prediction, target, mask, mode, settings):
    """Dimensionless robust pose/relative supervision in the GT Hole frame.

    The lateral term measures prediction error at the actual GT arrangement;
    it never forces the peg onto the Hole axis. Relative terms backpropagate
    through the reconstructed endpoint into both active pose heads. Full directed
    SO(3) orientation is used: square cross-section symmetry is not quotiented
    out because the existing labeled pose contract retains orientation and +X.
    """
    from graph.pose_geometry import (
        angular_error_rad,
        geometric_endpoint,
        hole_frame_vector,
        relative_rotation,
    )

    if (
        prediction.shape != target.shape
        or prediction.shape[-1] != 21
        or mask.shape != target.shape[:2]
        or mask.dtype != torch.bool
        or not mask.any()
    ):
        raise ValueError("Expected matching Object21[B,T,21] and nonempty boolean mask[B,T]")
    prediction, target = prediction[mask], target[mask]
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Valid insertion supervision must be finite")

    def robust(error, unit):
        normalized = error / unit
        return F.huber_loss(normalized, torch.zeros_like(normalized), delta=1.0)

    terms = {}
    # Preserve the earlier pose objective exactly for the pose_only ablation.
    _, pose_terms = pose_loss(
        prediction[None],
        target[None],
        torch.ones((1, len(target)), dtype=torch.bool, device=target.device),
        mode,
        settings,
    )
    terms.update(pose_terms)
    # The insertion Mamba/GRU recipe explicitly requests robust geodesic angles.
    # Older pose recipes retain their original squared-angle surrogate by default.
    if settings.get("rotation_objective") == "geodesic_huber":
        for name, start in (("peg", 0), ("hole", 10)):
            terms[name + "_position"] = robust(
                prediction[:, start : start + 3] - target[:, start : start + 3],
                settings["position_unit_m"],
            )
            angle = angular_error_rad(
                prediction[:, start + 3 : start + 7], target[:, start + 3 : start + 7]
            )
            terms[name + "_rotation"] = robust(angle, settings["rotation_unit_rad"])
    # Do not trust a cached/copied relative-vector slot: recomputation guarantees
    # that relative supervision reaches the refined poses and predicted size.
    relative_error = geometric_endpoint(prediction) - target[:, 7:10]
    terms["relative_position"] = robust(relative_error, settings["position_unit_m"])
    lateral_error = hole_frame_vector(relative_error, target[:, 13:17])[:, 1:]
    terms["lateral_position"] = robust(lateral_error, settings["position_unit_m"])
    relative_angle = angular_error_rad(
        relative_rotation(prediction[:, 3:7], prediction[:, 13:17]),
        relative_rotation(target[:, 3:7], target[:, 13:17]),
    )
    terms["relative_orientation"] = robust(relative_angle, settings["rotation_unit_rad"])
    active = [name + "_" + term for name in MODES[mode] for term in ("position", "rotation")]
    active.extend(INSERTION_ABLATIONS[settings["ablation"]])
    loss = sum(settings["weights"][name] * terms[name] for name in active)
    if not torch.isfinite(loss):
        raise ValueError("Insertion loss must be finite")
    return loss, terms
