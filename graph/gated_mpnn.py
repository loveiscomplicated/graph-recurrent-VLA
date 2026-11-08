"""Observed PEG/BOX/HOLE graphs to object geometry predictions."""

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from graph.build_graph import EdgeType, GraphBatch, NodeType
from graph.vertex_aggregation import GlobalGeometryAggregation, MultiSeedAttentionPool


def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


# -------------------------------------------
# Edge-aware Gated Message Passing
# -------------------------------------------


class EdgeGatedMPNNLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        in_dim = 2 * hidden + edge_dim

        self.message = mlp(in_dim, hidden, hidden)
        self.gate = mlp(in_dim, hidden, hidden)
        self.update = mlp(2 * hidden, hidden, hidden)

        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_feature: Tensor,
        sender_confidence: Tensor,
    ) -> Tensor:

        if edge_index.shape[1] == 0:
            return h

        src, dst = edge_index

        z = torch.cat(
            (h[src], h[dst], edge_feature),
            dim=-1,
        )

        # Confidence-aware gating
        weight = sender_confidence[src].unsqueeze(-1)

        messages = torch.sigmoid(self.gate(z)) * self.message(z) * weight

        # Aggregate messages at receivers
        aggregated = torch.zeros_like(h)
        aggregated = aggregated.index_add(0, dst, messages)

        denom = torch.zeros(
            (h.shape[0], 1),
            dtype=h.dtype,
            device=h.device,
        )
        denom = denom.index_add(0, dst, weight)

        aggregated = aggregated / denom.clamp_min(1.0)

        # Only nodes receiving supported messages keep candidate updates.
        receives_message = denom[:, 0] > 0
        if self.training and self.dropout.p > 0:
            # Keep the original dropout draw count/order for stochastic recipes.
            update_inputs = torch.cat(
                (h[receives_message], aggregated[receives_message]), dim=-1,
            )
            updated = self.update(update_inputs)
            result = h.clone()
            result[receives_message] = self.norm(
                h[receives_message] + self.dropout(updated)
            )
            return result

        # Dense candidates avoid data-dependent node counts; unsupported nodes
        # retain their previous states rather than receiving an update.
        update_inputs = torch.cat((h, aggregated), dim=-1)
        updated = self.norm(h + self.update(update_inputs))
        return torch.where(receives_message.unsqueeze(-1), updated, h)


# -------------------------------------------
# Confidence-aware Attention Pooling
# -------------------------------------------


class ConfidenceAttentionPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attention = mlp(hidden, hidden, 1)

    def forward(
        self,
        h: Tensor,
        confidence: Tensor,
        valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        attention = self.attention(h).squeeze(-1)
        logits = attention + torch.log(confidence.clamp_min(1e-6))
        logits = logits.masked_fill(~valid, -torch.inf)
        # Empty object groups get a finite softmax input, then zero weight.
        logits = torch.where(
            valid.any(dim=1, keepdim=True), logits, torch.zeros_like(logits),
        )
        weights = torch.softmax(logits, dim=1) * valid.to(h.dtype)

        weights = weights / (weights.sum(dim=1, keepdim=True).clamp_min(1e-8))

        pooled = torch.sum(
            weights.unsqueeze(-1) * h,
            dim=1,
        )

        present = valid.any(dim=1).to(h.dtype).unsqueeze(-1)

        mean_conf = (confidence * valid).sum(1, keepdim=True) / valid.sum(
            1, keepdim=True
        ).clamp_min(1)

        summary = torch.cat(
            (present, mean_conf),
            dim=-1,
        )

        return pooled, summary


# -------------------------------------------
# Quaternion Utilities
# -------------------------------------------


def normalize_quat_wxyz(raw: Tensor) -> Tensor:
    """Normalize an identity-offset prediction, falling back to identity near zero."""
    identity = torch.zeros_like(raw)
    identity[..., 0] = 1.0

    shifted = raw + identity
    norm = torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
    epsilon = 1e-8
    normalized = shifted / norm.clamp_min(epsilon)
    quat = torch.where(norm > epsilon, normalized, identity)

    # q and -q represent the same rotation
    return torch.where(
        quat[..., :1] < 0,
        -quat,
        quat,
    )


def rotate_local_x(quat_wxyz: Tensor) -> Tensor:
    """R(q) @ [1, 0, 0]"""
    w, x, y, z = quat_wxyz.unbind(dim=-1)

    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ),
        dim=-1,
    )


# -------------------------------------------
# Vertex Graph -> Object21D
# -------------------------------------------


class VertexEdgeGatedMPNN(nn.Module):
    def __init__(
        self,
        num_semantic_ids: int,
        hidden: int = 128,
        layers: int = 3,
        edge_embedding_dim: int = 16,
        dropout: float = 0.0,
        coord_scale_m: float = 0.05,
        size_scales_m: Tuple[float, float, float, float] = (0.04, 0.01, 0.01, 0.02),
        use_semantic_ids: bool = False,
        use_confidence: bool = False,
        global_geometry: bool = False,
        geometry_dim: int = 16,
        readout: str = "attention",
        readout_seeds: int = 4,
        readout_dim: int = 16,
        readout_heads: int = 4,
    ):
        super().__init__()

        if num_semantic_ids < 1 or hidden < 8 or layers < 1:
            raise ValueError("Invalid model configuration")

        if any(s <= 0 for s in size_scales_m) or coord_scale_m <= 0:
            raise ValueError("Scales must be positive")

        if type(use_semantic_ids) is not bool or type(use_confidence) is not bool:
            raise ValueError("use_semantic_ids and use_confidence must be bool")
        if type(global_geometry) is not bool:
            raise ValueError("global_geometry must be bool")
        for name, value in (("geometry_dim", geometry_dim), ("readout_seeds", readout_seeds),
                            ("readout_dim", readout_dim), ("readout_heads", readout_heads)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if readout not in ("attention", "multiseed"):
            raise ValueError("readout must be attention or multiseed")
        if readout_dim % readout_heads:
            raise ValueError("readout_dim must be divisible by readout_heads")
        self.hidden = hidden
        self.num_semantic_ids = num_semantic_ids
        self.use_semantic_ids = use_semantic_ids
        self.use_confidence = use_confidence

        self.semantic_embed = nn.Embedding(num_semantic_ids, 32)
        self.object_embed = nn.Embedding(3, 8)

        self.node_encoder = mlp(
            32 + 8 + 3 + 1 + 1,
            hidden,
            hidden,
        )

        self.edge_embed = nn.Embedding(4, edge_embedding_dim)

        edge_dim = edge_embedding_dim + 3 + 1 + 1

        # Separate peg and target-side (box/hole) message passing
        self.peg_layers = nn.ModuleList(
            [EdgeGatedMPNNLayer(hidden, edge_dim, dropout) for _ in range(layers)]
        )

        self.hole_layers = nn.ModuleList(
            [EdgeGatedMPNNLayer(hidden, edge_dim, dropout) for _ in range(layers)]
        )

        # Cross-object message passing
        self.cross_layer = EdgeGatedMPNNLayer(hidden, edge_dim, dropout)

        self.peg_pool = ConfidenceAttentionPool(hidden)
        self.hole_pool = ConfidenceAttentionPool(hidden)

        # Peg: position3 + quaternion4 + halfsize3
        self.peg_head = mlp(hidden + 2, hidden, 10)

        # Hole: position3 + quaternion4 + radius1
        self.hole_head = mlp(hidden + 2, hidden, 8)

        self.endpoint_residual = mlp(
            2 * (hidden + 2),
            hidden,
            3,
        )

        # Initially endpoint = geometric endpoint
        nn.init.zeros_(self.endpoint_residual[-1].weight)
        nn.init.zeros_(self.endpoint_residual[-1].bias)

        self.register_buffer(
            "size_scales_m",
            torch.tensor(
                size_scales_m,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "coord_scale_m",
            torch.tensor(
                coord_scale_m,
                dtype=torch.float32,
            ),
        )

        # Initialize optional modules last: legacy keys, initialization draws and
        # all existing heads remain identical when the flags are disabled.
        # Separate RNG scopes also give A/AB identical geometry initialization,
        # B/AB identical readout initialization and the same later training RNG.
        with torch.random.fork_rng(devices=[]):
            self.global_geometry = (
                GlobalGeometryAggregation(hidden, geometry_dim) if global_geometry else None
            )
        with torch.random.fork_rng(devices=[]):
            if readout == "multiseed":
                self.peg_pool = MultiSeedAttentionPool(hidden, readout_dim, readout_seeds, readout_heads)
                self.hole_pool = MultiSeedAttentionPool(hidden, readout_dim, readout_seeds, readout_heads)

    def forward(
        self,
        batch: GraphBatch,
    ) -> Dict[str, Tensor]:
        """Predict hole geometry using box and/or hole observations.

        BOX and HOLE retain different embeddings and edge types. Their features
        share the target-side message layers and readout, whose output is the
        hole pose/radius. hole_observed reports actual HOLE nodes only.

        Validate the graph before calling this method. The training data loader
        validates during collation, avoiding repeated checks on the model device.
        """
        xyz = batch.xyz_world
        b, n, _ = xyz.shape

        node_is_present = batch.node_valid
        observed = node_is_present
        # Unknown quality is neutral; it must not erase a detected node.
        conf = torch.ones_like(batch.confidence)
        if self.use_confidence:
            conf = torch.where(batch.confidence_valid, batch.confidence, conf)
        conf = conf * observed.to(conf.dtype)

        # World-frame relative coordinates
        xyz_relative = (xyz - batch.tcp_xyz_world[:, None, :]) * observed.unsqueeze(-1)

        observed_flat = observed.reshape(-1)
        conf_flat = conf.reshape(-1)

        peg_obs = observed & (batch.object_type == NodeType.PEG)
        target_nodes = (
            (batch.object_type == NodeType.BOX) | (batch.object_type == NodeType.HOLE)
        )
        hole_obs = observed & target_nodes

        # Confidence-weighted object centroid
        def centroid(mask: Tensor) -> Tensor:
            w = conf * mask.to(conf.dtype)

            return (xyz_relative * w.unsqueeze(-1)).sum(dim=1) / w.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        peg_anchor = centroid(peg_obs)
        hole_anchor = centroid(hole_obs)

        node_anchor = torch.where(
            (batch.object_type == NodeType.PEG).unsqueeze(-1),
            peg_anchor[:, None, :],
            hole_anchor[:, None, :],
        )

        xyz_centered = (xyz_relative - node_anchor) * observed.unsqueeze(-1)

        coordinate_scale = self.coord_scale_m.to(dtype=xyz.dtype)

        # Unknown or disabled semantics do not contribute a learned identity.
        semantic_features = torch.zeros((b, n, 32), dtype=xyz.dtype, device=xyz.device)
        if self.use_semantic_ids:
            semantic_features = self.semantic_embed(batch.semantic_id.clamp_min(0))
            semantic_features = torch.where(
                batch.semantic_valid.unsqueeze(-1), semantic_features, 0,
            )

        # PAD/unknown IDs are -1. Use a safe embedding index, then mask its output.
        object_features = self.object_embed(batch.object_type.clamp_min(0))
        object_features = torch.where(node_is_present.unsqueeze(-1), object_features, 0)

        # Node encoding
        node_features = torch.cat(
            (
                xyz_centered / coordinate_scale,
                conf.unsqueeze(-1),
                observed.to(xyz.dtype).unsqueeze(-1),
                semantic_features,
                object_features,
            ),
            dim=-1,
        )

        h = self.node_encoder(node_features)
        h = torch.where(node_is_present.unsqueeze(-1), h, 0)
        h = h.reshape(b * n, -1)

        h = h * (node_is_present.reshape(-1, 1).to(h.dtype))

        # Edge geometric features
        src, dst = batch.edge_index

        edge_observed = observed_flat[src] & observed_flat[dst]

        rel = batch.relative_xyz

        edge_feature = torch.cat(
            (
                rel / coordinate_scale,
                batch.distance / coordinate_scale,
                edge_observed.to(xyz.dtype).unsqueeze(-1),
                self.edge_embed(batch.edge_type),
            ),
            dim=-1,
        )

        peg_edges = batch.edge_type == EdgeType.PEG
        hole_edges = (
            (batch.edge_type == EdgeType.BOX) | (batch.edge_type == EdgeType.HOLE)
        )
        cross_edges = batch.edge_type == EdgeType.CROSS

        peg_nodes = (batch.object_type.reshape(-1) == NodeType.PEG).unsqueeze(-1)

        hole_nodes = target_nodes.reshape(-1, 1)

        # Intra-object message passing
        for peg_layer, hole_layer in zip(
            self.peg_layers,
            self.hole_layers,
        ):
            peg_h = peg_layer(
                h,
                batch.edge_index[:, peg_edges],
                edge_feature[peg_edges],
                conf_flat,
            )

            hole_h = hole_layer(
                h,
                batch.edge_index[:, hole_edges],
                edge_feature[hole_edges],
                conf_flat,
            )

            h = torch.where(
                peg_nodes,
                peg_h,
                torch.where(
                    hole_nodes,
                    hole_h,
                    torch.zeros_like(h),
                ),
            )

        # Cross-object interaction
        h = self.cross_layer(
            h,
            batch.edge_index[:, cross_edges],
            edge_feature[cross_edges],
            conf_flat,
        )

        h = h.reshape(b, n, -1)

        if self.global_geometry is not None:
            centered = xyz_centered / coordinate_scale
            peg_geometry = self.global_geometry(h, centered, peg_obs)
            target_geometry = self.global_geometry(h, centered, hole_obs)
            h = peg_geometry + target_geometry

        # Object-level readout
        peg_pool, peg_summary = self.peg_pool(
            h,
            conf,
            observed & (batch.object_type == NodeType.PEG),
        )

        hole_pool, hole_summary = self.hole_pool(
            h,
            conf,
            observed & target_nodes,
        )

        peg_features = torch.cat(
            (peg_pool, peg_summary),
            dim=-1,
        )

        hole_features = torch.cat(
            (hole_pool, hole_summary),
            dim=-1,
        )

        raw_peg = self.peg_head(peg_features)
        raw_hole = self.hole_head(hole_features)

        # Peg pose
        peg_rel = peg_anchor + raw_peg[:, :3] * coordinate_scale

        peg_quat = normalize_quat_wxyz(raw_peg[:, 3:7])

        # Hole pose
        hole_rel = hole_anchor + raw_hole[:, :3] * coordinate_scale

        hole_quat = normalize_quat_wxyz(raw_hole[:, 3:7])

        # Positive physical dimensions
        scales = self.size_scales_m.to(dtype=xyz.dtype)

        peg_half_size = F.softplus(raw_peg[:, 7:10]) * scales[:3]

        hole_radius = F.softplus(raw_hole[:, 7:8]) * scales[3:4]

        # Geometric Peg tip relative to Hole
        endpoint_geom = (
            peg_rel + rotate_local_x(peg_quat) * peg_half_size[:, :1] - hole_rel
        )

        # Learned endpoint correction
        endpoint_delta = (
            self.endpoint_residual(
                torch.cat(
                    (peg_features, hole_features),
                    dim=-1,
                )
            )
            * coordinate_scale
        )

        endpoint = endpoint_geom + endpoint_delta

        # Object21D contract
        object21 = torch.cat(
            (
                peg_rel,  # 0:3
                peg_quat,  # 3:7
                endpoint,  # 7:10
                hole_rel,  # 10:13
                hole_quat,  # 13:17
                peg_half_size,  # 17:20
                hole_radius,  # 20:21
            ),
            dim=-1,
        )

        assert object21.shape == (b, 21)

        return {
            "object21": object21,
            "endpoint_geom": endpoint_geom,
            "endpoint_residual": endpoint_delta,
            "peg_anchor": peg_anchor,
            "hole_anchor": hole_anchor,
            "peg_observed": peg_summary[:, 0] > 0,
            "hole_observed": (observed & (batch.object_type == NodeType.HOLE)).any(dim=1),
            "box_observed": (observed & (batch.object_type == NodeType.BOX)).any(dim=1),
            "target_support_observed": hole_summary[:, 0] > 0,
            "peg_mean_confidence": peg_summary[:, 1],
            "hole_mean_confidence": hole_summary[:, 1],
        }


# -------------------------------------------
# Object21D -> State38D
# -------------------------------------------


def assemble_state38(
    robot15: Tensor,
    object21: Tensor,
    freshness2: Tensor,
) -> Tensor:

    if robot15.ndim != 2 or robot15.shape[-1] != 15:
        raise ValueError("robot15 must be (B,15)")

    if object21.ndim != 2 or object21.shape[-1] != 21:
        raise ValueError("object21 must be (B,21)")

    if freshness2.ndim != 2 or freshness2.shape[-1] != 2:
        raise ValueError("freshness2 must be (B,2)")

    if not (robot15.shape[0] == object21.shape[0] == freshness2.shape[0]):
        raise ValueError("Batch sizes must match")

    return torch.cat(
        (robot15, object21, freshness2),
        dim=-1,
    )
