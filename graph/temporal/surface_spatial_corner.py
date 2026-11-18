"""Add current Peg surface tokens only to the corner tracker's spatial attention."""

import torch
from torch import nn

from graph.temporal.causal_transformer import sinusoidal_positions
from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker


class SurfaceSpatialCornerTracker(SpatialTemporalCornerTracker):
    """Query corner states against corners and masked, same-frame Peg surfaces."""

    def __init__(self, width=48, heads=4):
        # Initialize the shared parameters before the new encoder, preserving
        # the baseline initialization for the same random seed.
        super().__init__(width=width, heads=heads)
        self.surface_input = nn.Sequential(nn.Linear(6, width), nn.LayerNorm(width))
        self.heads = heads

    def forward(
        self,
        features,
        anchor,
        detected,
        observed,
        valid,
        surface_features,
        surface_valid,
    ):
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
        queries = self.spatial_norm(hidden).reshape(batch_size * length, slots, -1)
        surface_count = surface_features.shape[2]
        surfaces = self.surface_input(surface_features).reshape(
            batch_size * length, surface_count, -1
        )
        keys = torch.cat((queries, surfaces), dim=1)
        surface_allowed = surface_valid.reshape(batch_size * length, surface_count)
        peg_queries = torch.arange(slots, device=hidden.device) < 8
        surface_blocked = ~(peg_queries[None, :, None] & surface_allowed[:, None, :])
        corner_blocked = self.spatial_mask[None].expand(batch_size * length, -1, -1)
        blocked = torch.cat((corner_blocked, surface_blocked), dim=-1)
        blocked = blocked.repeat_interleave(self.heads, dim=0)
        update, _ = self.spatial(
            queries, keys, keys, attn_mask=blocked, need_weights=False
        )
        hidden = hidden + update.reshape_as(hidden)
        hidden = attend_over_time(hidden, self.temporal[1])
        prediction = anchor + self.output(hidden).float() * 0.1
        return torch.where(observed[..., None], detected, prediction)
