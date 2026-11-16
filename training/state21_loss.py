"""Shared Object21D objective with optional static size supervision."""

import math

import torch
from torch.nn import functional as F

from graph.state_estimation.codec import SIZE_SCALES_M
from training.gnn.loss import object21_loss


def state21_loss(
    output,
    target,
    *,
    shape_weight=0.0,
    pos_unit_m=0.01,
    endpoint_transition_m=None,
    endpoint_weight=1.0,
):
    """Use sign-invariant rotation and supervise auxiliary Peg/Hole dimensions."""
    if not math.isfinite(shape_weight) or shape_weight < 0:
        raise ValueError("shape_weight must be nonnegative and finite")
    terms = object21_loss(
        output, target, pos_unit_m=pos_unit_m, size_units_m=SIZE_SCALES_M
    )
    shape = target.new_zeros(())
    if "shape" in output:
        scale = target.new_tensor(SIZE_SCALES_M)
        shape = F.smooth_l1_loss(output["shape"] / scale, target[:, 17:] / scale)
    terms["shape"] = shape
    terms["state21"] = terms["total"]
    if endpoint_transition_m is not None:
        if not math.isfinite(endpoint_transition_m) or endpoint_transition_m <= 0:
            raise ValueError("endpoint_transition_m must be positive and finite")
        if not math.isfinite(endpoint_weight) or endpoint_weight <= 0:
            raise ValueError("endpoint_weight must be positive and finite")
        delta = output["object21"][:, 7:10] - target[:, 7:10]
        distance = delta.float().norm(dim=-1)
        endpoint = distance_loss(distance, pos_unit_m, endpoint_transition_m)
        weighted = endpoint_weight * endpoint
        terms["total"] = terms["total"] - terms["weighted/endpoint"] + weighted
        terms["endpoint_radial"] = endpoint
        terms["weighted/endpoint_radial"] = weighted
    if shape_weight:
        terms["total"] = terms["total"] + shape_weight * shape
    return terms


def distance_loss(distance, scale_m, transition_m):
    """Apply Smooth L1 to Euclidean distances with a metre-valued transition."""
    normalized = distance / scale_m
    return F.smooth_l1_loss(
        normalized, torch.zeros_like(normalized), beta=transition_m / scale_m
    )


def spatial_vertex_errors(prediction, current, windows):
    """Gather only GT-certified, observed vertex rows in the current frame."""
    batch_rows, node_rows, references = [], [], []
    for batch, (window, observed) in enumerate(
        zip(windows, current.observed_vertex_rows, strict=True)
    ):
        labels = window.sample.vertex_targets
        if labels is None:
            raise ValueError(
                "Spatial vertex supervision requires offline vertex annotations"
            )
        for key, position in labels.items():
            batch_rows.append(batch)
            node_rows.append(observed[key])
            references.append(position)
    if not references:
        return prediction.new_empty(0), prediction.new_empty(0)
    reference = prediction.new_tensor(references)
    predicted_xyz = prediction[batch_rows, node_rows]
    observed_xyz = current.position[batch_rows, node_rows]
    return (
        (predicted_xyz - reference).norm(dim=-1),
        (observed_xyz - reference).norm(dim=-1),
    )
