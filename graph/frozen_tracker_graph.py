"""Construct masked fixed-slot graphs from frozen tracker coordinates."""

import torch

from graph.build_graph import EdgeType, GraphBatch, NodeType


def build_tracker_graph(xyz, observed, valid, tcp):
    """Exclude never-observed slots and distinguish measurements from predictions."""
    batch, count, _ = xyz.shape
    if count != 20 or valid.shape != (batch, count) or observed.shape != valid.shape:
        raise ValueError("Expected 20 tracker slots with [B,20] boolean masks")
    if torch.any(observed & ~valid):
        raise ValueError("Observed tracker slots must be valid")
    xyz = torch.where(valid[..., None], xyz, 0)
    kinds = torch.tensor([NodeType.PEG] * 8 + [NodeType.BOX] * 8 + [NodeType.HOLE] * 4,
                         device=xyz.device).expand(batch, -1)
    kinds = torch.where(valid, kinds, NodeType.PAD)
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~torch.eye(count, dtype=torch.bool, device=xyz.device)[None]
    sample, source, target = allowed.nonzero(as_tuple=True)
    relative = xyz[sample, source] - xyz[sample, target]
    graph = GraphBatch(
        xyz_world=xyz, object_type=kinds, node_valid=valid,
        semantic_id=torch.full_like(kinds, -1), semantic_valid=torch.zeros_like(valid),
        confidence=xyz.new_zeros(valid.shape), confidence_valid=torch.zeros_like(valid),
        tcp_xyz_world=tcp, edge_index=torch.stack((sample * count + source, sample * count + target)),
        edge_type=torch.where(kinds[sample, source] == kinds[sample, target],
                              kinds[sample, source], EdgeType.CROSS),
        relative_xyz=relative, distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    flags = torch.stack((observed & valid, valid & ~observed), dim=-1).to(xyz.dtype)
    return graph, flags
