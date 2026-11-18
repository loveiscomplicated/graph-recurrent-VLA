"""Recover recent corners and score the empty, single, and paired subsets."""

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from training.detected_corner_gnn import graph_batch


def mlp(input_size, hidden, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.SiLU(), nn.Linear(hidden, output_size))


class RecentCornerCompletion(nn.Module):
    """Predict XYZ corrections and score every feasible subset for the frozen GNN."""

    def __init__(self, max_candidates, hidden, max_offset_m):
        super().__init__()
        if max_candidates < 2 or max_offset_m <= 0:
            raise ValueError("Corner completion needs at least two slots and a positive offset scale")
        self.max_offset_m = max_offset_m
        self.observation = mlp(6, hidden, hidden)
        self.context = mlp(hidden * 3 + 3, hidden, hidden)
        self.candidate = mlp(hidden + 17, hidden, hidden)
        self.recovery = nn.Linear(hidden, 3)
        self.selector = mlp(hidden * 2 + 5, hidden, 1)
        nn.init.zeros_(self.recovery.weight)
        nn.init.zeros_(self.recovery.bias)
        actions = [(-1, -1)] + [(slot, -1) for slot in range(max_candidates)]
        actions += list(combinations(range(max_candidates), 2))
        self.register_buffer("actions", torch.tensor(actions))

    def forward(self, batch):
        observed_kinds = batch["observed_families"].long()
        observed_tags = F.one_hot(observed_kinds.clamp_min(0), 3).float()
        observed_tags = observed_tags * (observed_kinds >= 0).unsqueeze(-1)
        observations = (batch["xyz"] - batch["tcp"][:, None]) / 0.15
        encoded = self.observation(torch.cat((observations, observed_tags), dim=-1))
        counts = observed_tags.sum(dim=1)
        pooled = observed_tags.transpose(1, 2) @ encoded / counts.clamp_min(1).unsqueeze(-1)
        context = self.context(torch.cat((pooled.flatten(1), counts.log1p()), dim=-1))

        candidate_kinds = batch["candidate_kinds"].long()
        tags = F.one_hot(candidate_kinds.clamp_min(0), 3).float()
        expanded_context = context[:, None].expand(-1, tags.shape[1], -1)
        inputs = torch.cat((batch["features"], tags, expanded_context), dim=-1)
        encoded_corners = self.candidate(inputs)
        offset = self.max_offset_m * torch.tanh(self.recovery(encoded_corners))
        recovered = batch["anchors"] + offset

        slots = self.actions.clamp_min(0)
        active = self.actions >= 0
        available = (candidate_kinds[:, slots] >= 0) | ~active
        valid_actions = available.all(dim=-1)
        encoded_sets = (encoded_corners[:, slots] * active[..., None]).sum(dim=-2)
        family_counts = (tags[:, slots] * active[..., None]).sum(dim=-2)
        pair_xyz = recovered[:, slots]
        spacing = torch.linalg.vector_norm(pair_xyz[:, :, 0] - pair_xyz[:, :, 1], dim=-1) / 0.15
        spacing = spacing * active.all(dim=-1)
        sizes = active.sum(dim=-1).expand_as(spacing)
        context = context[:, None].expand(-1, len(slots), -1)
        selection_inputs = torch.cat((context, encoded_sets, family_counts,
                                      sizes.unsqueeze(-1), spacing.unsqueeze(-1)), dim=-1)
        logits = self.selector(selection_inputs).squeeze(-1).masked_fill(~valid_actions, -torch.inf)
        return recovered, logits, valid_actions


def run_actions(gnn, batch, recovered, options, rows):
    """Append real-valued recovery nodes for the requested row/subset pairs."""
    xyz = batch["xyz"][rows]
    kinds = batch["node_kinds"][rows].long()
    chosen = options.clamp_min(0)
    points = recovered[rows[:, None], chosen]
    candidate_kinds = batch["candidate_kinds"][rows[:, None], chosen].long().clamp_max(1)
    candidate_kinds = candidate_kinds.masked_fill(options < 0, -1)
    xyz = torch.cat((xyz, points), dim=1)
    kinds = torch.cat((kinds, candidate_kinds), dim=1)
    valid = kinds >= 0
    graph, _ = graph_batch(xyz, kinds, valid, valid, batch["tcp"][rows])
    return gnn(graph)
