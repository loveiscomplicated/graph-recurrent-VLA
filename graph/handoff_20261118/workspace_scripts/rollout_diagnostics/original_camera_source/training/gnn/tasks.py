"""Task contracts keep prediction layouts and losses out of the epoch loop."""

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn

from .loss import object21_loss


def geometry_errors(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Return detached per-sample errors in physical units, before averaging."""
    prediction = prediction.detach()
    target = target.detach()
    errors = {}
    for name, start in (("peg_position_error_m", 0), ("endpoint_error_m", 7),
                        ("hole_position_error_m", 10)):
        difference = prediction[:, start:start + 3] - target[:, start:start + 3]
        errors[name] = torch.linalg.vector_norm(difference, dim=-1)
    for name, start in (("peg_rotation_error_deg", 3), ("hole_rotation_error_deg", 13)):
        predicted_quaternion = torch.nn.functional.normalize(prediction[:, start:start + 4], dim=-1)
        target_quaternion = torch.nn.functional.normalize(target[:, start:start + 4], dim=-1)
        # q and -q describe the same rotation. Clamp roundoff before acos.
        cosine = (predicted_quaternion * target_quaternion).sum(dim=-1).abs().clamp(max=1)
        errors[name] = torch.rad2deg(2 * torch.acos(cosine))
    size_names = ("peg_half_size_x_error_mm", "peg_half_size_y_error_mm",
                  "peg_half_size_z_error_mm", "hole_radius_error_mm")
    for index, name in enumerate(size_names, start=17):
        errors[name] = (prediction[:, index] - target[:, index]).abs() * 1000
    return errors


@dataclass
class StepResult:
    """A differentiable scalar loss and scalar means with explicit denominators."""

    loss: Tensor
    metrics: dict[str, Tensor | float | int]
    metric_weights: dict[str, int]


class Task(Protocol):
    """A task owns the relationship between batch targets and model outputs."""

    def compute(self, model: nn.Module, batch) -> StepResult: ...


class Object21Task:
    """Supervise pose/size geometry with the existing Object21 objective."""

    def __init__(self, pos_unit_m=0.01, consistency_weight=0.05,
                 size_units_m=(0.04, 0.01, 0.01, 0.02)):
        for name, value in (("pos_unit_m", pos_unit_m),
                            ("consistency_weight", consistency_weight)):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"task.{name} must be finite")
        if pos_unit_m <= 0 or consistency_weight < 0:
            raise ValueError("Task position unit must be positive and consistency weight nonnegative")
        if not isinstance(size_units_m, (list, tuple)) or len(size_units_m) != 4:
            raise ValueError("task.size_units_m must contain four positive finite scales")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
               for value in size_units_m):
            raise ValueError("task.size_units_m must contain four positive finite scales")
        self.loss_settings = dict(pos_unit_m=pos_unit_m, consistency_weight=consistency_weight,
                                  size_units_m=tuple(size_units_m))

    def compute(self, model: nn.Module, batch) -> StepResult:
        """Compute per-sample means; empty detections still have valid GT labels."""
        output = model(batch.graph)
        if not isinstance(output, dict) or not {"object21", "endpoint_residual"} <= output.keys():
            raise ValueError("Object21Task requires dict outputs object21 and endpoint_residual")
        terms = object21_loss(output, batch.targets["object21"], **self.loss_settings)
        count = len(batch.sample_ids)
        metrics = {name: value.detach() for name, value in terms.items()}
        metrics["loss"] = metrics.pop("total")
        errors = geometry_errors(output["object21"], batch.targets["object21"])
        metrics.update({name: error.mean() for name, error in errors.items()})
        empty_inputs = ~batch.graph.node_valid.any(dim=1)
        metrics["empty_input_fraction"] = empty_inputs.float().mean()
        weights = {name: count for name in metrics}
        if not model.training:
            # Transfer only group counts. Absent groups are omitted instead of
            # reporting zero errors; each group has its own epoch denominator.
            empty_count = int(empty_inputs.sum().item())
            for group, mask, group_count in (
                ("empty_input", empty_inputs, empty_count),
                ("nonempty_input", ~empty_inputs, count - empty_count),
            ):
                if not group_count:
                    continue
                for name, error in errors.items():
                    key = f"{group}/{name}"
                    metrics[key] = error.masked_fill(~mask, 0).sum() / group_count
                    weights[key] = group_count
        return StepResult(terms["total"], metrics, weights)
