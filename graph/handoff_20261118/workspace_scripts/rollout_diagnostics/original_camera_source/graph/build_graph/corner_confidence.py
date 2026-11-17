"""Configurable evidence quality scores; these are not calibrated probabilities."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConfidenceSettings:
    """Provisional scales and weights to validate against saved-frame errors."""

    residual_scale_px: float = 2.0
    plane_reference: float = 3.0
    source_reference: float = 3.0
    residual_weight: float = 0.6
    plane_weight: float = 0.3
    source_weight: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"confidence.{name} must be finite and nonnegative")
            if not name.endswith("weight") and value == 0:
                raise ValueError(f"confidence.{name} must be positive")
        if self.residual_weight + self.plane_weight + self.source_weight == 0:
            raise ValueError("At least one confidence weight must be positive")


def compute_confidence(
    metadata: Sequence[Mapping],
    settings: ConfidenceSettings | None = None,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Average available evidence only; absent evidence returns zero and False."""
    settings = settings or ConfidenceSettings()
    scores, valid = [], []
    for index, evidence in enumerate(metadata):
        if not isinstance(evidence, Mapping):
            raise TypeError(f"metadata[{index}] must be a mapping")
        weighted_score = 0.0
        available_weight = 0.0
        for name, weight, scale in (
            ("line_residual_px", settings.residual_weight, settings.residual_scale_px),
            ("supporting_planes", settings.plane_weight, settings.plane_reference),
            ("sources", settings.source_weight, settings.source_reference),
        ):
            value = evidence.get(name)
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"metadata[{index}].{name} must be finite and nonnegative")
            if name == "line_residual_px":
                ratio = value / scale
                # Avoid overflowing Python's square for extreme finite inputs.
                score = math.exp(-ratio * ratio)
            else:
                score = min(value / scale, 1.0)
            weighted_score += weight * score
            available_weight += weight
        scores.append(weighted_score / available_weight if available_weight else 0.0)
        valid.append(available_weight > 0)
    return (
        torch.tensor(scores, dtype=torch.float32, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )
