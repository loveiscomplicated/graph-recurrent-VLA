"""Model-independent observed-corner graphs, in world axes and metres."""

from dataclasses import dataclass, fields
from enum import IntEnum

import torch
from torch import Tensor


class NodeType(IntEnum):
    PAD = -1
    PEG = 0
    BOX = 1
    HOLE = 2


class EdgeType(IntEnum):
    PEG = 0
    BOX = 1
    HOLE = 2
    CROSS = 3


@dataclass
class GraphBatch:
    """Padded observed nodes and directed, flattened edges.

    Unknown semantic IDs are -1. Invalid confidence is zero, independently of
    node presence. An all-empty batch still has one padding slot per sample.
    relative_xyz is sender minus receiver, with world-aligned axes.
    """

    xyz_world: Tensor
    object_type: Tensor
    node_valid: Tensor
    semantic_id: Tensor
    semantic_valid: Tensor
    confidence: Tensor
    confidence_valid: Tensor
    tcp_xyz_world: Tensor
    edge_index: Tensor
    edge_type: Tensor
    relative_xyz: Tensor
    distance: Tensor

    def to(self, device: torch.device | str) -> "GraphBatch":
        """Move all tensor fields together without changing their dtypes."""
        moved_fields = {
            field.name: getattr(self, field.name).to(device)
            for field in fields(self)
        }
        return GraphBatch(**moved_fields)

    def validate(self, num_semantic_ids: int | None = None) -> None:
        """Reject inconsistent shapes, devices, masks, types and edge geometry."""
        xyz = self.xyz_world
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz_world must have shape [B,N,3]")
        batch_size, node_count, _ = xyz.shape
        if batch_size < 1 or node_count < 1:
            raise ValueError("GraphBatch requires B >= 1 and N >= 1 (padding allowed)")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        edge_count = self.edge_index.shape[1]
        shapes = {
            "tcp_xyz_world": (batch_size, 3),
            "edge_type": (edge_count,),
            "relative_xyz": (edge_count, 3),
            "distance": (edge_count, 1),
        }
        bool_names = {"node_valid", "semantic_valid", "confidence_valid"}
        int_names = {"object_type", "semantic_id", "edge_index", "edge_type"}
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            expected = shapes.get(name, (batch_size, node_count))
            if name not in {"xyz_world", "edge_index"} and value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if value.device != xyz.device:
                raise ValueError(f"{name} must be on {xyz.device}")
            if name in bool_names:
                if value.dtype != torch.bool:
                    raise TypeError(f"{name} must be torch.bool")
            elif name in int_names:
                if value.dtype != torch.long:
                    raise TypeError(f"{name} must be torch.long")
            elif value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite torch.float32")

        present = self.node_valid
        if torch.any(present & ((self.object_type < 0) | (self.object_type > 2))):
            raise ValueError("Valid nodes must have PEG, BOX or HOLE object_type")
        if torch.any(~present & (self.object_type != NodeType.PAD)):
            raise ValueError("Padding must have object_type=PAD")
        if torch.any((self.semantic_valid | self.confidence_valid) & ~present):
            raise ValueError("Padding cannot have valid semantic IDs or confidence")
        if torch.any(~present.unsqueeze(-1) & (xyz != 0)):
            raise ValueError("Padding xyz_world must be zero")
        if torch.any(~self.semantic_valid & (self.semantic_id != -1)):
            raise ValueError("Unknown semantic IDs must be -1")
        known_ids = self.semantic_id[self.semantic_valid]
        if torch.any(known_ids < 0):
            raise ValueError("Invalid semantic IDs: known IDs must be nonnegative")
        if num_semantic_ids is not None and torch.any(known_ids >= num_semantic_ids):
            raise ValueError("Invalid semantic IDs: ID exceeds model vocabulary")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confidence must be in [0,1]")
        if torch.any(~self.confidence_valid & (self.confidence != 0)):
            raise ValueError("Invalid confidence and padding must have confidence=0")

        src, dst = self.edge_index
        if torch.any((self.edge_index < 0) | (self.edge_index >= batch_size * node_count)):
            raise ValueError("Out-of-range edge_index")
        if torch.any(src // node_count != dst // node_count):
            raise ValueError("Edges cannot cross batch samples")
        if torch.any(src == dst):
            raise ValueError("Self edges are not part of the corner graph contract")
        if torch.any(~present.flatten()[self.edge_index]):
            raise ValueError("Edges cannot touch padding")
        if torch.unique(src * batch_size * node_count + dst).numel() != edge_count:
            raise ValueError("Duplicate directed edges are not allowed")
        objects = self.object_type.flatten()
        expected_types = torch.where(
            objects[src] == objects[dst], objects[src], EdgeType.CROSS,
        )
        if not torch.equal(self.edge_type, expected_types):
            raise ValueError("Edge type disagrees with object types")
        expected_relative = xyz.reshape(-1, 3)[src] - xyz.reshape(-1, 3)[dst]
        if not torch.allclose(self.relative_xyz, expected_relative, atol=1e-6, rtol=1e-5):
            raise ValueError("relative_xyz must be sender minus receiver")
        expected_distance = torch.linalg.vector_norm(expected_relative, dim=-1, keepdim=True)
        if not torch.allclose(self.distance, expected_distance, atol=1e-6, rtol=1e-5):
            raise ValueError("distance must be the Euclidean edge length in metres")
