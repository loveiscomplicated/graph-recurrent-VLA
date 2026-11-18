"""Twenty-slot spatial/temporal Transformer from the graph2 feasibility model."""

import torch
from torch import nn

from graph.temporal.causal_transformer import (
    CausalTransformerBlock,
    sinusoidal_positions,
)


class SpatialTemporalCornerTracker(nn.Module):
    """Preserve detections and complete missing slots using only causal inputs."""

    def __init__(self, width=48, heads=4):
        super().__init__()
        self.input = nn.Linear(10, width)
        self.ids = nn.Embedding(20, width)
        self.temporal = nn.ModuleList(
            CausalTransformerBlock(width, heads, width * 2, 0.0) for _ in range(2)
        )
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.zeros_(self.output[1].weight)
        nn.init.zeros_(self.output[1].bias)
        body = torch.tensor([0] * 8 + [1] * 12)
        self.register_buffer(
            "spatial_mask", body[:, None] != body[None, :], persistent=False
        )

    def forward(self, features, anchor, detected, observed, valid):
        batch_size, length, slots, _ = features.shape
        padding = (~valid[:, None, :]).expand(batch_size, slots, length)
        padding = padding.reshape(batch_size * slots, length)
        hidden = self.input(features) + self.ids.weight[None, None]
        positions = sinusoidal_positions(
            length, hidden.shape[-1], offset=0, device=hidden.device, dtype=hidden.dtype
        )
        hidden = hidden + positions[None, :, None]

        def attend_over_time(values, block):
            sequences = values.transpose(1, 2).reshape(batch_size * slots, length, -1)
            attended = block(sequences, padding)
            return attended.reshape(batch_size, slots, length, -1).transpose(1, 2)

        hidden = attend_over_time(hidden, self.temporal[0])
        per_frame = self.spatial_norm(hidden).reshape(batch_size * length, slots, -1)
        update, _ = self.spatial(
            per_frame,
            per_frame,
            per_frame,
            attn_mask=self.spatial_mask,
            need_weights=False,
        )
        hidden = hidden + update.reshape_as(hidden)
        hidden = attend_over_time(hidden, self.temporal[1])
        prediction = anchor + self.output(hidden).float() * 0.1
        return torch.where(observed[..., None], detected, prediction)
