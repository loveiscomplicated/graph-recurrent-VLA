"""B2-2 spatial-first recovery of missing fixed-ID corner positions."""

import torch
from torch import nn

from graph.state_estimation.codec import COORDINATE_SCALE_M
from graph.state_estimation.spatial import SpatialBackbone
from graph.state_estimation.temporal import CausalAttention


class SpatialCornerTracker(nn.Module):
    """Encode each frame spatially, then predict corner displacements causally."""

    def __init__(self, hidden=64, layers=2, heads=4):
        super().__init__()
        self.spatial = SpatialBackbone(hidden, layers, 0.0)
        self.temporal = CausalAttention(hidden, heads, 0.0)
        self.displacement = nn.Linear(hidden, 3)
        nn.init.zeros_(self.displacement.weight)
        nn.init.zeros_(self.displacement.bias)

    def forward(self, graph, rows, visible, anchors):
        """Return measured locations when visible and anchor plus delta otherwise."""
        batch, steps, slots = rows.shape
        hidden = self.spatial.propagate(self.spatial.encode(graph), graph)
        frame_rows = hidden[torch.arange(batch * steps, device=rows.device)[:, None],
                            rows.reshape(batch * steps, slots).clamp_min(0)]
        sequence = frame_rows.reshape(batch, steps, slots, -1).transpose(1, 2).flatten(0, 1)
        available = visible.transpose(1, 2).flatten(0, 1)
        times = torch.arange(steps, device=rows.device).expand(batch * slots, -1)
        starts = torch.zeros(batch * slots, dtype=torch.long, device=rows.device)
        predictions = self.temporal(sequence, available, times, starts)
        delta = self.displacement(predictions).reshape(batch, slots, steps, 3).transpose(1, 2)
        return torch.where(visible[..., None], anchors, anchors + delta * COORDINATE_SCALE_M)
