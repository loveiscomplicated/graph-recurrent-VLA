"""Task contracts keep prediction layouts and losses out of the epoch loop."""

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn

from .loss import object21_loss


@dataclass
class StepResult:
    """A differentiable scalar loss and scalar means with explicit denominators."""

    loss: Tensor
    metrics: dict[str, float]
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
        metrics = {name: float(value.detach()) for name, value in terms.items()}
        metrics["loss"] = metrics.pop("total")
        # Error units are metres, independent of normalized loss scales.
        for name, start in (("peg_position_error_m", 0), ("endpoint_error_m", 7),
                            ("hole_position_error_m", 10)):
            error = output["object21"][:, start:start+3] - batch.targets["object21"][:, start:start+3]
            metrics[name] = float(torch.linalg.vector_norm(error.detach(), dim=-1).mean())
        metrics["empty_input_fraction"] = float((~batch.graph.node_valid.any(dim=1)).float().mean())
        return StepResult(terms["total"], metrics, {name: count for name in metrics})
