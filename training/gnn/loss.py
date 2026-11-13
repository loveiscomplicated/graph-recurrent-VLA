"""Training objectives for corner-based object geometry prediction."""

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


# -------------------------------------------
# GT-supervised Object21D Loss
# -------------------------------------------


def object21_loss(
    output: Dict[str, Tensor],
    gt_object21: Tensor,
    pos_unit_m: float = 0.01,
    consistency_weight: float = 0.05,
    size_units_m: Tuple[float, float, float, float] = (0.04, 0.01, 0.01, 0.02),
) -> Dict[str, Tensor]:
    """Compare Object21D predictions with GT and regularize endpoint corrections."""
    pred = output["object21"]

    if gt_object21.shape != pred.shape:
        raise ValueError("gt_object21 must be (B,21)")

    if pos_unit_m <= 0 or any(s <= 0 for s in size_units_m):
        raise ValueError("Loss scales must be positive")

    def huber(a, b, unit):
        return F.smooth_l1_loss(
            a / unit,
            b / unit,
            beta=1.0,
        )

    def quat_loss(a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        return (1.0 - torch.sum(a * b, dim=-1).square().clamp(max=1)).mean()

    terms = {
        "peg_position": huber(
            pred[:, 0:3],
            gt_object21[:, 0:3],
            pos_unit_m,
        ),
        "peg_rotation": quat_loss(
            pred[:, 3:7],
            gt_object21[:, 3:7],
        ),
        "endpoint": huber(
            pred[:, 7:10],
            gt_object21[:, 7:10],
            pos_unit_m,
        ),
        "hole_position": huber(
            pred[:, 10:13],
            gt_object21[:, 10:13],
            pos_unit_m,
        ),
        "hole_rotation": quat_loss(
            pred[:, 13:17],
            gt_object21[:, 13:17],
        ),
        "size": F.smooth_l1_loss(
            pred[:, 17:21] / pred.new_tensor(size_units_m),
            gt_object21[:, 17:21] / pred.new_tensor(size_units_m),
            beta=1.0,
        ),
        "endpoint_consistency": (output["endpoint_residual"] / pos_unit_m)
        .square()
        .mean(),
    }

    total = (
        terms["peg_position"]
        + terms["hole_position"]
        + 2.0 * terms["endpoint"]
        + 0.5 * (terms["peg_rotation"] + terms["hole_rotation"])
        + 0.5 * terms["size"]
        + consistency_weight * terms["endpoint_consistency"]
    )

    contributions = {
        "weighted/peg_position": terms["peg_position"],
        "weighted/hole_position": terms["hole_position"],
        "weighted/endpoint": 2.0 * terms["endpoint"],
        "weighted/peg_rotation": 0.5 * terms["peg_rotation"],
        "weighted/hole_rotation": 0.5 * terms["hole_rotation"],
        "weighted/size": 0.5 * terms["size"],
        "weighted/endpoint_consistency": consistency_weight * terms["endpoint_consistency"],
    }
    return {"total": total, **terms, **contributions}


# -------------------------------------------
# Optional: Paired Delta Loss
# -------------------------------------------


def paired_delta_loss(
    pred_a: Tensor,
    pred_b: Tensor,
    gt_a: Tensor,
    gt_b: Tensor,
    delta_unit_m: float = 0.005,
) -> Tensor:
    """Match paired position changes for the peg, endpoint and hole."""
    if not (pred_a.shape == pred_b.shape == gt_a.shape == gt_b.shape):
        raise ValueError("Paired inputs must match")

    if pred_a.ndim != 2 or pred_a.shape[1] != 21 or delta_unit_m <= 0:
        raise ValueError("Expected (B,21)")

    losses = []

    for sl in (
        slice(0, 3),
        slice(7, 10),
        slice(10, 13),
    ):
        dp = (pred_b[:, sl] - pred_a[:, sl]) / delta_unit_m

        dg = (gt_b[:, sl] - gt_a[:, sl]) / delta_unit_m

        losses.append(F.smooth_l1_loss(dp, dg, beta=1.0))

    return sum(losses) / len(losses)
