"""Replay existing Object21 readouts/heads on frozen baseline node features.

This diagnostic wrapper reuses existing modules and preserves their decoding
rules. It adds no message-passing architecture or production model interface.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from graph.gated_mpnn import normalize_quat_wxyz, rotate_local_x


COMPONENTS = ("peg_pool", "hole_pool", "peg_head", "hole_head", "endpoint_residual")


class ReadoutProbe(nn.Module):
    """Run existing readouts/heads on detached node features and original anchors."""

    def __init__(self, source):
        super().__init__()
        for name in COMPONENTS:
            setattr(self, name, deepcopy(getattr(source, name)))
        self.register_buffer("coordinate_scale", source.coord_scale_m.detach().clone())
        self.register_buffer("size_scales", source.size_scales_m.detach().clone())

    def forward(self, features):
        """Preserve the pretrained model's Object21 order, units and correction."""
        h = features["h"]
        peg_pool, peg_summary = self.peg_pool(h, features["confidence"], features["peg_mask"])
        hole_pool, hole_summary = self.hole_pool(h, features["confidence"], features["target_mask"])
        peg_features = torch.cat((peg_pool, peg_summary), dim=-1)
        hole_features = torch.cat((hole_pool, hole_summary), dim=-1)
        raw_peg = self.peg_head(peg_features)
        raw_hole = self.hole_head(hole_features)
        peg_position = features["peg_anchor"] + raw_peg[:, :3] * self.coordinate_scale
        hole_position = features["hole_anchor"] + raw_hole[:, :3] * self.coordinate_scale
        peg_rotation = normalize_quat_wxyz(raw_peg[:, 3:7])
        hole_rotation = normalize_quat_wxyz(raw_hole[:, 3:7])
        peg_size = F.softplus(raw_peg[:, 7:10]) * self.size_scales[:3]
        hole_radius = F.softplus(raw_hole[:, 7:8]) * self.size_scales[3:4]
        geometric = peg_position + rotate_local_x(peg_rotation) * peg_size[:, :1] - hole_position
        residual = self.endpoint_residual(torch.cat((peg_features, hole_features), dim=-1)) * self.coordinate_scale
        object21 = torch.cat((peg_position, peg_rotation, geometric + residual,
                              hole_position, hole_rotation, peg_size, hole_radius), dim=-1)
        return {"object21": object21, "endpoint_geom": geometric, "endpoint_residual": residual}
