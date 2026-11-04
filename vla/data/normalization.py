"""Feature-independent statistics used by the current robot-state normalizer."""
from dataclasses import dataclass
import torch


@dataclass
class StateStatistics:
    mean: torch.Tensor
    std: torch.Tensor

    def as_dict(self):
        return dict(mean=self.mean, std=self.std)
