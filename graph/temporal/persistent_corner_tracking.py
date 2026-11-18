"""Forecast every active detector track and regularize its rigid geometry."""

import torch
from torch import nn
from torch.nn import functional as F

from graph.temporal.recent_corner_completion import mlp
from training.detected_corner_gnn import graph_batch


class PersistentCornerTracking(nn.Module):
    """Combine track history with the set of current detections."""

    def __init__(self, hidden, max_offset_m):
        super().__init__()
        if hidden < 1 or max_offset_m <= 0:
            raise ValueError("Expected a positive hidden width and displacement scale")
        self.max_offset_m = max_offset_m
        self.observation = mlp(6, hidden, hidden)
        self.context = mlp(3 * hidden + 3, hidden, hidden)
        self.update = mlp(hidden + 18, hidden, 4)
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, batch):
        anchors, features = batch["anchors"], batch["features"]
        tags = F.one_hot(batch["kinds"].long().clamp_min(0), 3).to(anchors.dtype)
        observed = batch["observed"]
        observed_tags = tags * observed.unsqueeze(-1)
        encoded = self.observation(torch.cat((features[..., :3], tags), dim=-1))
        counts = observed_tags.sum(dim=1)
        pooled = observed_tags.transpose(1, 2) @ encoded / counts.clamp_min(1).unsqueeze(-1)
        context = self.context(torch.cat((pooled.flatten(1), counts.log1p()), dim=-1))
        inputs = torch.cat((features, tags, context[:, None].expand(-1, anchors.shape[1], -1)), dim=-1)
        update = self.update(inputs)
        correction = self.max_offset_m * torch.tanh(update[..., :3])
        tool_delta = features[..., 3:6] * 0.15
        predicted = anchors + correction + update[..., 3:].sigmoid() * tool_delta
        return torch.where(observed.unsqueeze(-1), anchors, predicted)


def predict_object21(gnn, batch, positions, *, use_forecasts=True):
    """Use original detections for Condition 1 and keep forecast nodes separate."""
    observed = batch["observed"]
    kinds = batch["kinds"].long().clamp_max(1)
    valid = observed
    if use_forecasts:
        valid = observed | ((batch["condition"][:, None] != 1) & (kinds >= 0))
    graph, _ = graph_batch(positions, kinds, valid, observed, batch["tcp"])
    return gnn(graph)


def shape_errors(positions, targets, kinds, observed):
    """Return distance residuals for Peg and Box pairs involving a forecast."""
    valid = kinds >= 0
    upper = torch.ones((kinds.shape[1], kinds.shape[1]), dtype=torch.bool, device=kinds.device).triu(1)
    forecast = valid & ~observed
    eligible = upper & valid[:, :, None] & valid[:, None, :]
    eligible &= forecast[:, :, None] | forecast[:, None, :]
    peg = kinds == 0
    peg_pairs = eligible & peg[:, :, None] & peg[:, None, :]
    box_pairs = eligible & ~peg[:, :, None] & ~peg[:, None, :]
    actual = torch.linalg.vector_norm(positions[:, :, None] - positions[:, None, :], dim=-1)
    reference = torch.linalg.vector_norm(targets[:, :, None] - targets[:, None, :], dim=-1)
    return actual - reference, (peg_pairs, box_pairs)


def shape_means(values, masks):
    """Average valid pairs per frame, avoiding a bias toward larger point sets."""
    pair_counts = torch.stack([mask.sum(dim=(1, 2)) for mask in masks], dim=1)
    sums = torch.stack([(values * mask).sum(dim=(1, 2)) for mask in masks], dim=1)
    return sums / pair_counts.clamp_min(1), pair_counts
