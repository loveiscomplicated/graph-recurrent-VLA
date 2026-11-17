"""Fixed physical-ID corner models for the spatial-relation ablation."""
import torch
from torch import nn

from graph.temporal.causal_transformer import CausalTransformerBlock, sinusoidal_positions

# Peg has its own rigid body. Hole and exterior points share the Box body.
BODY = (0,) * 8 + (1,) * 12


class CornerTransformer(nn.Module):
    def __init__(self, spatial, width=48, heads=4):
        super().__init__()
        self.input = nn.Linear(10, width)
        self.ids = nn.Embedding(20, width)
        self.temporal = nn.ModuleList(CausalTransformerBlock(width, heads, width * 2, 0.0)
                                      for _ in range(2))
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.zeros_(self.output[1].weight)
        nn.init.zeros_(self.output[1].bias)
        group = torch.tensor(BODY) if spatial else torch.arange(20)
        self.register_buffer('spatial_mask', group[:, None] != group[None, :], persistent=False)

    def forward(self, features, anchor, detected, observed, valid):
        b, t, s, _ = features.shape
        pad = (~valid[:, None, :]).expand(b, s, t).reshape(b * s, t)
        hidden = self.input(features) + self.ids.weight[None, None]
        pos = sinusoidal_positions(t, hidden.shape[-1], offset=0,
                                   device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + pos[None, :, None]

        def across_time(x, block):
            y = x.transpose(1, 2).reshape(b * s, t, -1)
            return block(y, pad).reshape(b, s, t, -1).transpose(1, 2)

        hidden = across_time(hidden, self.temporal[0])
        per_frame = self.spatial_norm(hidden).reshape(b * t, s, -1)
        update, _ = self.spatial(per_frame, per_frame, per_frame, attn_mask=self.spatial_mask,
                                 need_weights=False)
        hidden = hidden + update.reshape_as(hidden)
        hidden = across_time(hidden, self.temporal[1])
        prediction = anchor + self.output(hidden).float() * .1
        return torch.where(observed[..., None], detected, prediction)
