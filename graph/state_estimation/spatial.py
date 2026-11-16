"""Adapt the leaderboard GNN's message passing and readout to observed graphs."""

import torch
from torch import nn
from torch.nn import functional as F

from graph.gated_mpnn import (
    ConfidenceAttentionPool,
    EdgeGatedMPNNLayer,
    mlp,
    normalize_quat_wxyz,
    rotate_local_x,
)
from graph.state_estimation.codec import COORDINATE_SCALE_M, SIZE_SCALES_M
from graph.state_estimation.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM


class SpatialBackbone(nn.Module):
    """Reuse edge-conditioned sigmoid gates, messages and per-object pools."""

    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.node_encoder = mlp(NODE_FEATURE_DIM, hidden, hidden)
        self.peg_layers = nn.ModuleList(
            EdgeGatedMPNNLayer(hidden, EDGE_FEATURE_DIM, dropout) for _ in range(layers)
        )
        self.hole_layers = nn.ModuleList(
            EdgeGatedMPNNLayer(hidden, EDGE_FEATURE_DIM, dropout) for _ in range(layers)
        )
        self.peg_pool = ConfidenceAttentionPool(hidden)
        self.hole_pool = ConfidenceAttentionPool(hidden)

    def encode(self, graph):
        """Encode geometry and provenance without making padding a candidate."""
        return self.node_encoder(graph.features) * graph.valid.unsqueeze(-1)

    def propagate(self, hidden, graph):
        """Pass messages on the Delaunay edges of each object independently."""
        edges = graph.edge_index
        peg = graph.object_id.flatten() == 0
        peg_edges = peg[edges[0]]
        features = torch.cat(
            (graph.edge_attr[:, :4] / COORDINATE_SCALE_M, graph.edge_attr[:, 4:]),
            dim=-1,
        )
        quality = graph.quality.flatten()
        # The existing scatter layer expects all operands to share a dtype.
        with torch.autocast(hidden.device.type, enabled=False):
            flat = hidden.flatten(0, 1).float()
            for peg_layer, hole_layer in zip(self.peg_layers, self.hole_layers):
                peg_updated = peg_layer(
                    flat, edges[:, peg_edges], features[peg_edges], quality
                )
                hole_updated = hole_layer(
                    flat, edges[:, ~peg_edges], features[~peg_edges], quality
                )
                flat = torch.where(peg[:, None], peg_updated, hole_updated)
        return flat.reshape_as(hidden)

    def pool(self, hidden, kinds, quality, supported):
        """Return Peg/Hole geometry latents and presence/confidence summaries."""
        peg, peg_summary = self.peg_pool(hidden, quality, supported & (kinds == 0))
        hole, hole_summary = self.hole_pool(hidden, quality, supported & (kinds == 1))
        return torch.stack(
            (torch.cat((peg, peg_summary), -1), torch.cat((hole, hole_summary), -1)),
            dim=1,
        )


class Object21Head(nn.Module):
    """Read two object latents with the leaderboard's geometric endpoint contract."""

    def __init__(self, width, hidden):
        super().__init__()
        self.peg = mlp(width, hidden, 10)
        self.hole = mlp(width, hidden, 8)
        self.endpoint = mlp(width * 2, hidden, 3)
        nn.init.zeros_(self.endpoint[-1].weight)
        nn.init.zeros_(self.endpoint[-1].bias)
        self.register_buffer("size_scales", torch.tensor(SIZE_SCALES_M))

    def forward(self, objects, anchors):
        """Return TCP-relative positions, world wxyz, shape and endpoint residual."""
        peg = self.peg(objects[:, 0]).float()
        hole = self.hole(objects[:, 1]).float()
        peg_rel = anchors[:, 0] + peg[:, :3] * COORDINATE_SCALE_M
        hole_rel = anchors[:, 1] + hole[:, :3] * COORDINATE_SCALE_M
        peg_quat = normalize_quat_wxyz(peg[:, 3:7])
        hole_quat = normalize_quat_wxyz(hole[:, 3:7])
        size = F.softplus(peg[:, 7:10]) * self.size_scales[:3]
        radius = F.softplus(hole[:, 7:8]) * self.size_scales[3:]
        endpoint_residual = (
            self.endpoint(objects.flatten(1)).float() * COORDINATE_SCALE_M
        )
        endpoint = (
            peg_rel
            + rotate_local_x(peg_quat) * size[:, :1]
            - hole_rel
            + endpoint_residual
        )
        return {
            "object21": torch.cat(
                (peg_rel, peg_quat, endpoint, hole_rel, hole_quat, size, radius), dim=-1
            ),
            "endpoint_residual": endpoint_residual,
        }


def position_anchors(graph):
    """Compute current observed TCP-relative centroids in metres per object."""
    relative = graph.features[..., :3] * COORDINATE_SCALE_M
    anchors = []
    for kind in (0, 1):
        support = graph.geometry & (graph.object_id == kind)
        weighted = relative * support.unsqueeze(-1)
        anchors.append(
            weighted.sum(dim=1) / support.sum(dim=1, keepdim=True).clamp_min(1)
        )
    return torch.stack(anchors, dim=1)
