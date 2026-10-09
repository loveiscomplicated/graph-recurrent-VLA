"""Train-only statistics shared by A, B and C; A/C feature equivalence retained."""
from dataclasses import dataclass

import torch

from .graph_builder import node_features, GraphBatch


@dataclass
class FeatureNormalizer:
    node_mean: torch.Tensor
    node_std: torch.Tensor
    context_mean: torch.Tensor
    context_std: torch.Tensor

    @classmethod
    def fit(cls, train_geometry):
        x, _, _, context = node_features(train_geometry)
        mean = x.mean((0, 1))
        std = x.std((0, 1), correction=0).clamp_min(.001)
        # Categorical indicators are not standardized.
        mean[:5] = 0
        std[:5] = 1
        return cls(mean, std, context.mean(0), context.std(0, correction=0).clamp_min(.001))

    def nodes(self, x):
        return (x-self.node_mean)/self.node_std

    def context(self, x):
        return (x-self.context_mean)/self.context_std

    def as_dict(self):
        return {k: getattr(self, k) for k in ("node_mean", "node_std", "context_mean", "context_std")}

    def apply(self, name, geometry, builder):
        if name == "flat":
            x, _, _, context = node_features(geometry)
            return torch.cat((self.nodes(x).flatten(1), self.context(context)), -1)
        graph = builder(geometry)
        graph.x = self.nodes(graph.x)
        graph.context = self.context(graph.context)
        return graph
