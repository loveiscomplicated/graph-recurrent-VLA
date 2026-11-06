from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PEG_EDGE, HOLE_EDGE, CROSS_EDGE = 0, 1, 2
PEG, HOLE, PAD = 0, 1, -1


@dataclass
class VertexBatch:
    vertex_xyz_world: Tensor  # (B,N,3)
    tcp_xyz_world: Tensor  # (B,3)
    semantic_id: Tensor  # (B,N)
    object_id: Tensor  # (B,N)
    visible: Tensor  # (B,N)
    confidence: Tensor  # (B,N)
    edge_index: Tensor  # (2,E)
    edge_type: Tensor  # (E,)

    def to(self, device: torch.device | str) -> "VertexBatch":
        return VertexBatch(
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
            }
        )

    def validate(self, num_semantic_ids: int) -> None:
        xyz = self.vertex_xyz_world
        tcp = self.tcp_xyz_world
        ids = self.semantic_id
        obj = self.object_id

        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("vertex_xyz_world must be (B,N,3)")

        b, n, _ = xyz.shape

        for name, value in (
            ("semantic_id", ids),
            ("object_id", obj),
            ("visible", self.visible),
            ("confidence", self.confidence),
        ):
            if value.shape != (b, n):
                raise ValueError(f"{name} must be (B,N)")

        if tcp.shape != (b, 3):
            raise ValueError("tcp_xyz_world must be (B,3)")

        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be (2,E)")

        if self.edge_type.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_type must be (E,)")

        if not (
            ids.dtype == torch.long
            and obj.dtype == torch.long
            and self.edge_index.dtype == torch.long
            and self.edge_type.dtype == torch.long
        ):
            raise TypeError("IDs and edge tensors must be torch.long")

        if torch.any((ids < 0) | (ids >= num_semantic_ids)):
            raise ValueError("Invalid semantic IDs")

        if torch.any((obj < -1) | (obj > 1)):
            raise ValueError("object_id must be -1/0/1")

        if self.visible.dtype != torch.bool:
            raise TypeError("visible must be bool")

        if not (
            torch.isfinite(xyz).all()
            and torch.isfinite(tcp).all()
            and torch.isfinite(self.confidence).all()
        ):
            raise ValueError("Non-finite inputs")

        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence must be in [0,1]")

        if torch.any(self.visible & (obj == PAD)):
            raise ValueError("Padding cannot be visible")

        if torch.any((~self.visible) & (self.confidence != 0)):
            raise ValueError("Missing vertices must have confidence=0")

        if self.edge_index.numel() == 0:
            return

        src, dst = self.edge_index

        if torch.any((src < 0) | (dst < 0) | (src >= b * n) | (dst >= b * n)):
            raise ValueError("Out-of-range edge_index")

        if torch.any(src // n != dst // n):
            raise ValueError("Edges cannot cross batch samples")

        if torch.any((self.edge_type < 0) | (self.edge_type > 2)):
            raise ValueError("Invalid edge_type")

        flat_obj = obj.reshape(-1)
        osrc = flat_obj[src]
        odst = flat_obj[dst]

        if torch.any((osrc == PAD) | (odst == PAD)):
            raise ValueError("Edges cannot touch padding")

        expected = torch.where(
            osrc != odst,
            CROSS_EDGE,
            osrc,
        )

        if torch.any(self.edge_type != expected):
            raise ValueError("Edge type disagrees with object types")


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

        updated = self.update(torch.cat((h, aggregated), dim=-1))

        candidate = self.norm(h + self.dropout(updated))

        return torch.where(
            denom > 0,
            candidate,
            h,
        )


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

        weights = self.attention(h).squeeze(-1) + torch.log(confidence.clamp_min(1e-6))

        weights = weights.masked_fill(~valid, -1e4)

        weights = torch.softmax(weights, dim=1) * valid.to(h.dtype)

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
    ):
        super().__init__()

        if num_semantic_ids < 1 or hidden < 8 or layers < 1:
            raise ValueError("Invalid model configuration")

        if any(s <= 0 for s in size_scales_m) or coord_scale_m <= 0:
            raise ValueError("Scales must be positive")

        self.num_semantic_ids = num_semantic_ids

        self.semantic_embed = nn.Embedding(num_semantic_ids, 32)
        self.object_embed = nn.Embedding(2, 8)

        self.node_encoder = mlp(
            32 + 8 + 3 + 1 + 1,
            hidden,
            hidden,
        )

        self.edge_embed = nn.Embedding(3, edge_embedding_dim)

        edge_dim = edge_embedding_dim + 3 + 1 + 1

        # Separate Peg and Hole message passing
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

    def forward(
        self,
        batch: VertexBatch,
    ) -> Dict[str, Tensor]:
        """Validate the vertex graph and predict object geometry and confidence."""
        batch.validate(self.num_semantic_ids)

        xyz = batch.vertex_xyz_world
        b, n, _ = xyz.shape

        node_is_present = batch.object_id != PAD

        observed = node_is_present & batch.visible & (batch.confidence > 0)

        conf = batch.confidence * observed.to(batch.confidence.dtype)

        # World-frame relative coordinates
        xyz_relative = (xyz - batch.tcp_xyz_world[:, None, :]) * observed.unsqueeze(-1)

        xyz_flat = xyz_relative.reshape(b * n, 3)
        observed_flat = observed.reshape(-1)
        conf_flat = conf.reshape(-1)

        peg_obs = observed & (batch.object_id == PEG)
        hole_obs = observed & (batch.object_id == HOLE)

        # Confidence-weighted object centroid
        def centroid(mask: Tensor) -> Tensor:
            w = conf * mask.to(conf.dtype)

            return (xyz_relative * w.unsqueeze(-1)).sum(dim=1) / w.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

        peg_anchor = centroid(peg_obs)
        hole_anchor = centroid(hole_obs)

        node_anchor = torch.where(
            (batch.object_id == PEG).unsqueeze(-1),
            peg_anchor[:, None, :],
            hole_anchor[:, None, :],
        )

        xyz_centered = (xyz_relative - node_anchor) * observed.unsqueeze(-1)

        coordinate_scale = self.coord_scale_m.to(dtype=xyz.dtype)

        # Node encoding
        node_features = torch.cat(
            (
                xyz_centered / coordinate_scale,
                conf.unsqueeze(-1),
                observed.to(xyz.dtype).unsqueeze(-1),
                self.semantic_embed(batch.semantic_id),
                self.object_embed(batch.object_id.clamp_min(0)),
            ),
            dim=-1,
        )

        h = self.node_encoder(node_features)
        h = h.reshape(b * n, -1)

        h = h * (node_is_present.reshape(-1, 1).to(h.dtype))

        # Edge geometric features
        src, dst = batch.edge_index

        edge_observed = observed_flat[src] & observed_flat[dst]

        rel = (xyz_flat[src] - xyz_flat[dst]) * edge_observed.unsqueeze(-1)

        edge_feature = torch.cat(
            (
                rel / coordinate_scale,
                torch.linalg.vector_norm(rel, dim=-1, keepdim=True) / coordinate_scale,
                edge_observed.to(xyz.dtype).unsqueeze(-1),
                self.edge_embed(batch.edge_type),
            ),
            dim=-1,
        )

        peg_edges = batch.edge_type == PEG_EDGE
        hole_edges = batch.edge_type == HOLE_EDGE
        cross_edges = batch.edge_type == CROSS_EDGE

        peg_nodes = (batch.object_id.reshape(-1) == PEG).unsqueeze(-1)

        hole_nodes = (batch.object_id.reshape(-1) == HOLE).unsqueeze(-1)

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

        # Object-level readout
        peg_pool, peg_summary = self.peg_pool(
            h,
            conf,
            observed & (batch.object_id == PEG),
        )

        hole_pool, hole_summary = self.hole_pool(
            h,
            conf,
            observed & (batch.object_id == HOLE),
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
            "hole_observed": hole_summary[:, 0] > 0,
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
