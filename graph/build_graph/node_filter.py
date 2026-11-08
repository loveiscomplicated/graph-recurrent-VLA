"""Filter observed graph nodes without changing retained node identities."""

from dataclasses import replace

import torch

from .graph_batch import GraphBatch


def filter_nodes(graph: GraphBatch, keep: torch.Tensor) -> GraphBatch:
    """Keep selected present nodes and their induced edges, leaving padding in place.

    For k-nearest-neighbor graphs this keeps existing edges between retained
    nodes; it does not construct replacement neighbors.
    """
    if not isinstance(keep, torch.Tensor) or keep.dtype != torch.bool:
        raise ValueError("keep must be a boolean tensor with shape [B,N]")
    if keep.shape != graph.node_valid.shape or keep.device != graph.node_valid.device:
        raise ValueError("keep must have the graph's [B,N] shape and device")
    present = graph.node_valid & keep
    edge_mask = present.flatten()[graph.edge_index].all(dim=0)
    filtered = replace(
        graph,
        xyz_world=graph.xyz_world.masked_fill(~present.unsqueeze(-1), 0),
        object_type=graph.object_type.masked_fill(~present, -1),
        node_valid=present,
        semantic_id=graph.semantic_id.masked_fill(~present, -1),
        semantic_valid=graph.semantic_valid & present,
        confidence=graph.confidence.masked_fill(~present, 0),
        confidence_valid=graph.confidence_valid & present,
        edge_index=graph.edge_index[:, edge_mask],
        edge_type=graph.edge_type[edge_mask],
        relative_xyz=graph.relative_xyz[edge_mask],
        distance=graph.distance[edge_mask],
    )
    filtered.validate()
    return filtered
