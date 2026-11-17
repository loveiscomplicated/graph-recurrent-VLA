"""Dimensionless Peg supervision and differentiable Method A relative geometry."""

import torch
from torch.nn import functional as F

from graph.pose_geometry import angular_error_rad, geometric_endpoint, hole_frame_vector

TERMS = ("peg_position", "peg_rotation", "relative_position", "lateral_position")


def geometry_terms(prediction, target, settings):
    """Supervise the GT arrangement, using GT Hole +X only as a loss reference.

    Positions retain the pretrained 0.01 m scale, rotations its 0.1 rad scale.
    The lateral Huber uses Hole-frame YZ, matching the existing insertion scorer
    and objective. GT sizes never enter predicted endpoint reconstruction.
    """
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[-1] != 21:
        raise ValueError("Expected aligned nonempty Object21[N,21] prediction and target")
    if (
        len(prediction) == 0
        or not torch.isfinite(target).all()
        or not torch.isfinite(prediction).all()
    ):
        raise ValueError("Geometry supervision requires nonempty finite valid frames")

    def robust(error, unit):
        normalized = error / unit
        return F.huber_loss(normalized, torch.zeros_like(normalized), delta=1.0)

    relative_error = geometric_endpoint(prediction) - target[:, 7:10]
    angle = angular_error_rad(prediction[:, 3:7], target[:, 3:7])
    return {
        "peg_position": robust(prediction[:, :3] - target[:, :3], settings["position_unit_m"]),
        "peg_rotation": robust(angle, settings["rotation_unit_rad"]),
        "relative_position": robust(relative_error, settings["position_unit_m"]),
        "lateral_position": robust(
            hole_frame_vector(relative_error, target[:, 13:17])[:, 1:], settings["position_unit_m"]
        ),
    }


def gradient_vector(loss, parameters):
    """Measure a term's head gradient without accumulating it in parameter.grad."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=False)
    return torch.cat([value.flatten() for value in gradients])


def measure_gradients(terms, parameters, weights):
    """Record raw/weighted norms and relative/lateral overlap on the same frames."""
    vectors = {name: gradient_vector(value, parameters) for name, value in terms.items()}
    result = {
        name: {
            "loss": float(terms[name].detach()),
            "gradient_norm": float(vector.norm()),
            "weighted_gradient_norm": float(vector.norm()) * weights[name],
        }
        for name, vector in vectors.items()
    }
    relative, lateral = vectors["relative_position"], vectors["lateral_position"]
    denominator = relative.norm() * lateral.norm()
    result["relative_lateral_cosine"] = (
        float((relative @ lateral) / denominator) if denominator > 0 else None
    )
    pose = (
        weights["peg_position"] * vectors["peg_position"]
        + weights["peg_rotation"] * vectors["peg_rotation"]
    )
    result["pose_gradient_norm"] = float(pose.norm())
    return result
