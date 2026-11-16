"""Shared Object21D objective with optional static size supervision."""

import math

from torch.nn import functional as F

from graph.state_estimation.codec import SIZE_SCALES_M
from training.gnn.loss import object21_loss


def state21_loss(output, target, *, shape_weight=0.0, pos_unit_m=0.01):
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
    if shape_weight:
        terms["total"] = terms["total"] + shape_weight * shape
    return terms
