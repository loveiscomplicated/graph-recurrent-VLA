"""Optional observed-vertex aggregation for the Object21 gated MPNN.

These are small AG-Pose-inspired and GMT-inspired components, not reproductions
of either complete network. See docs/gnn_training.md for sources and deviations.
"""

import math

import torch
from torch import Tensor, nn


AGGREGATION_DEFAULTS = dict(
    global_geometry=False, geometry_dim=16, readout="attention",
    readout_seeds=4, readout_dim=16, readout_heads=4,
)


class GlobalGeometryAggregation(nn.Module):
    """Fuse masked pairwise positional means and a global object feature.

    One shared module processes PEG and target-side groups separately. Coordinates
    are object-centered world axes divided by the existing coordinate scale.
    Self pairs are included, as in AG-Pose's global positional aggregation.
    """

    def __init__(self, hidden: int, geometry_dim: int):
        super().__init__()
        self.relative_position = nn.Sequential(
            nn.Linear(3, geometry_dim), nn.SiLU(), nn.Linear(geometry_dim, geometry_dim),
        )
        self.centered_position = nn.Sequential(
            nn.Linear(3, geometry_dim), nn.SiLU(), nn.Linear(geometry_dim, geometry_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden + geometry_dim, geometry_dim),
            nn.SiLU(),
            nn.Linear(geometry_dim, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, xyz: Tensor, valid: Tensor) -> Tensor:
        """Return equivariant [B,N,H] features, zero outside this group."""
        h = torch.where(valid.unsqueeze(-1), h, 0)
        xyz = torch.where(valid.unsqueeze(-1), xyz, 0)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        relative = xyz.unsqueeze(2) - xyz.unsqueeze(1)
        pair_features = self.relative_position(relative)
        pair_features = torch.where(pair_valid.unsqueeze(-1), pair_features, 0)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        geometry = pair_features.sum(dim=2) / count
        geometry = geometry + self.centered_position(xyz)
        global_feature = h.sum(dim=1, keepdim=True) / count
        global_feature = global_feature.expand_as(h)
        update = self.fuse(torch.cat((h, global_feature, geometry), dim=-1))
        result = self.norm(h + update)
        return torch.where(valid.unsqueeze(-1), result, 0)


class MultiSeedAttentionPool(nn.Module):
    """GMT-inspired seed-to-node attention, projected back to the existing head.

    Keys and values use the existing MPNN features. This omits GMT's additional
    graph convolutions and hierarchical/self-attention pooling stages. A small
    attention width limits parameter growth while preserving four distinct seeds.
    """

    def __init__(self, hidden: int, attention_dim: int, seeds: int, heads: int):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("readout_dim must be divisible by readout_heads")
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.seeds = nn.Parameter(torch.empty(seeds, attention_dim))
        nn.init.xavier_uniform_(self.seeds)
        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(hidden, attention_dim)
        self.value = nn.Linear(hidden, attention_dim)
        self.norm = nn.LayerNorm(attention_dim)
        self.feedforward = nn.Linear(attention_dim, attention_dim)
        self.projection = nn.Linear(seeds * attention_dim, hidden)

    def forward(self, h: Tensor, confidence: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Pool a node multiset; empty groups return exactly zero and [0,0]."""
        batch_size, node_count, _ = h.shape
        h = torch.where(valid.unsqueeze(-1), h, 0)
        confidence = torch.where(valid, confidence, 0)
        projected_seeds = self.query(self.seeds)
        query = projected_seeds.reshape(-1, self.heads, self.head_dim)
        key = self.key(h).reshape(batch_size, node_count, self.heads, self.head_dim)
        value = self.value(h).reshape(batch_size, node_count, self.heads, self.head_dim)
        logits = torch.einsum("shd,bnhd->bhsn", query, key) / math.sqrt(self.head_dim)
        logits = logits + confidence.clamp_min(1e-6).log()[:, None, None, :]
        logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
        present = valid.any(dim=1)
        logits = torch.where(present[:, None, None, None], logits, 0)
        weights = logits.softmax(dim=-1) * valid[:, None, None, :].to(h.dtype)
        attended = torch.einsum("bhsn,bnhd->bshd", weights, value)
        attended = attended.reshape(batch_size, self.seeds.shape[0], -1)
        tokens = self.norm(attended + projected_seeds.unsqueeze(0))
        tokens = tokens + torch.relu(self.feedforward(tokens))
        pooled = self.projection(tokens.flatten(start_dim=1))
        pooled = torch.where(present.unsqueeze(-1), pooled, 0)
        mean_confidence = confidence.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        summary = torch.stack((present.to(h.dtype), mean_confidence), dim=-1)
        return pooled, summary
