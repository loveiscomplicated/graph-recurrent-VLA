"""Convert detector results into GT-free, padded corner interaction graphs."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .corner_confidence import ConfidenceSettings, compute_confidence
from .graph_batch import EdgeType, GraphBatch, NodeType


class CornerResult(Protocol):
    """TensorResult subset consumed here, without importing the visual backend."""

    xyz: Tensor
    object_id: Tensor
    metadata: list


@dataclass(frozen=True)
class GraphSettings:
    """None selects complete directed edges; k selects nearest senders per receiver.

    Intra-object k applies within each type, cross-object k across all other
    types. kNN need not be reciprocal; equal-distance ties use input order.
    """

    intra_k: int | None = None
    cross_k: int | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be None or an integer >= 1")


class GraphBuilder:
    """Build graphs from single-camera observations and separately supplied TCP.

    Pass explicit object_types only when an upstream classifier has established
    PEG/BOX/HOLE labels. Detector object_id=1 always means BOX by default.
    semantic_ids uses -1 for unknown IDs; detection order is never an identity.
    """

    def __init__(
        self,
        settings: GraphSettings | None = None,
        confidence_settings: ConfidenceSettings | None = None,
    ):
        self.settings = settings or GraphSettings()
        self.confidence_settings = confidence_settings or ConfidenceSettings()

    def _edges(self, xyz: Tensor, objects: Tensor) -> Tensor:
        """Select directed edges locally before adding batch offsets."""
        node_count = len(xyz)
        if node_count < 2:
            return torch.empty((2, 0), dtype=torch.long, device=xyz.device)
        distances = torch.cdist(xyz, xyz)
        pairs = []
        indices = torch.arange(node_count, device=xyz.device)
        for receiver in range(node_count):
            same_object = objects == objects[receiver]
            for candidates, limit in (
                (same_object & (indices != receiver), self.settings.intra_k),
                (~same_object, self.settings.cross_k),
            ):
                senders = indices[candidates]
                if limit is not None:
                    order = torch.argsort(distances[receiver, senders], stable=True)
                    senders = senders[order[:limit]]
                receivers = torch.full_like(senders, receiver)
                pairs.append(torch.stack((senders, receivers)))
        return torch.cat(pairs, dim=1)

    def __call__(
        self,
        results: Sequence[CornerResult],
        tcp_xyz_world: Tensor,
        *,
        semantic_ids: Sequence[Tensor] | None = None,
        object_types: Sequence[Tensor] | None = None,
        pad_to: int | None = None,
    ) -> GraphBatch:
        """Clone inference tensors into ordinary training inputs on the TCP device."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Build training graphs outside torch.inference_mode()")
        batch_size = len(results)
        if batch_size < 1:
            raise ValueError("results must contain at least one detector result")
        if tcp_xyz_world.shape != (batch_size, 3):
            raise ValueError("tcp_xyz_world must have shape [len(results),3]")
        if not tcp_xyz_world.is_floating_point() or not torch.isfinite(tcp_xyz_world).all():
            raise ValueError("tcp_xyz_world must contain finite floating point metres")
        optional_labels = (("semantic_ids", semantic_ids), ("object_types", object_types))
        for name, values in optional_labels:
            if values is not None and len(values) != batch_size:
                raise ValueError(f"{name} must contain one tensor per result")
        device = tcp_xyz_world.device
        node_count = max(1, max(len(result.xyz) for result in results))
        if pad_to is not None:
            if type(pad_to) is not int or pad_to < node_count:
                raise ValueError(f"pad_to must be an integer >= {node_count}")
            node_count = pad_to

        xyz = torch.zeros(batch_size, node_count, 3, dtype=torch.float32, device=device)
        objects = torch.full((batch_size, node_count), NodeType.PAD, dtype=torch.long, device=device)
        valid = torch.zeros(batch_size, node_count, dtype=torch.bool, device=device)
        semantics = torch.full_like(objects, -1)
        confidence = torch.zeros(batch_size, node_count, dtype=torch.float32, device=device)
        confidence_valid = torch.zeros_like(valid)
        edges = []
        for sample_index, result in enumerate(results):
            count = len(result.xyz)
            if result.xyz.shape != (count, 3) or not result.xyz.is_floating_point():
                raise ValueError(f"results[{sample_index}].xyz must be floating point [N,3]")
            if not torch.isfinite(result.xyz).all():
                raise ValueError(f"results[{sample_index}].xyz must be finite")
            if result.object_id.shape != (count,) or result.object_id.dtype != torch.long:
                raise ValueError(f"results[{sample_index}].object_id must be int64 [N]")
            if torch.any((result.object_id < 0) | (result.object_id > 1)):
                raise ValueError("Detector object_id must be 0=PEG or 1=BOX")
            if len(result.metadata) != count:
                raise ValueError(f"results[{sample_index}].metadata must have N entries")
            # detach().clone() outside inference mode permits autograd to save
            # these inputs for parameter gradients; no detector gradient is used.
            points = result.xyz.detach().to(device=device, dtype=torch.float32).clone()
            labels = result.object_id.detach().to(device).clone()
            if object_types is not None:
                supplied = object_types[sample_index]
                if supplied.shape != (count,) or supplied.dtype != torch.long:
                    raise ValueError("object_types entries must be int64 [N]")
                if torch.any((supplied < NodeType.PEG) | (supplied > NodeType.HOLE)):
                    raise ValueError("object_types entries must be PEG, BOX or HOLE")
                labels = supplied.detach().to(device).clone()
            xyz[sample_index, :count] = points
            objects[sample_index, :count] = labels
            valid[sample_index, :count] = True
            if semantic_ids is not None:
                supplied = semantic_ids[sample_index]
                if supplied.shape != (count,) or supplied.dtype != torch.long:
                    raise ValueError("semantic_ids entries must be int64 [N], -1 for unknown")
                if torch.any(supplied < -1):
                    raise ValueError("semantic_ids entries must be >= -1")
                semantics[sample_index, :count] = supplied.detach().to(device).clone()
            scores, score_valid = compute_confidence(
                result.metadata, self.confidence_settings, device=device,
            )
            confidence[sample_index, :count] = scores
            confidence_valid[sample_index, :count] = score_valid
            edges.append(self._edges(points, labels) + sample_index * node_count)

        edge_index = torch.cat(edges, dim=1)
        src, dst = edge_index
        flat_objects = objects.flatten()
        edge_types = torch.where(
            flat_objects[src] == flat_objects[dst], flat_objects[src], EdgeType.CROSS,
        )
        flat_xyz = xyz.reshape(-1, 3)
        relative = flat_xyz[src] - flat_xyz[dst]
        graph = GraphBatch(
            xyz_world=xyz, object_type=objects, node_valid=valid,
            semantic_id=semantics, semantic_valid=semantics >= 0,
            confidence=confidence, confidence_valid=confidence_valid,
            tcp_xyz_world=tcp_xyz_world.detach().to(dtype=torch.float32).clone(),
            edge_index=edge_index, edge_type=edge_types, relative_xyz=relative,
            distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
        )
        graph.validate()
        return graph
