"""Build frame graphs from frozen twenty-slot causal tracker outputs."""

import torch

from graph.build_graph import EdgeType, GraphBatch, NodeType


def completed_graph(xyz, seen, observed, tcp):
    """Omit never-seen slots and encode detection versus prediction provenance.

    As in the original graph2 GNN, Peg has type 0 and Box/Hole share type 1.
    Slot numbers are not semantic-ID features. No GT enters this function.
    """
    batch_size, count, _ = xyz.shape
    if count != 20 or seen.shape != xyz.shape[:2] or observed.shape != seen.shape:
        raise ValueError("Expected xyz [B,20,3] and seen/observed [B,20]")
    kinds = torch.tensor([0] * 8 + [1] * 12, device=xyz.device)
    types = torch.where(seen, kinds[None], NodeType.PAD)
    xyz = torch.where(seen[..., None], xyz, 0)
    allowed = seen[:, :, None] & seen[:, None, :]
    allowed &= ~torch.eye(count, device=xyz.device, dtype=torch.bool)[None]
    sample, source, target = allowed.nonzero(as_tuple=True)
    edges = torch.stack((sample * count + source, sample * count + target))
    edge_types = torch.where(
        types[sample, source] == types[sample, target],
        types[sample, source], EdgeType.CROSS,
    )
    relative = xyz[sample, source] - xyz[sample, target]
    graph = GraphBatch(
        xyz_world=xyz, object_type=types, node_valid=seen,
        semantic_id=torch.full_like(types, -1), semantic_valid=torch.zeros_like(seen),
        confidence=xyz.new_zeros((batch_size, count)),
        confidence_valid=torch.zeros_like(seen), tcp_xyz_world=tcp,
        edge_index=edges, edge_type=edge_types, relative_xyz=relative,
        distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    return graph, observed.to(xyz.dtype)[..., None]
