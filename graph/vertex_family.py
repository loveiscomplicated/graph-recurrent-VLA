"""Experimental family-token views of an unchanged observed vertex graph."""

from dataclasses import replace

import torch

from graph.build_graph import GraphBatch, NodeType


FAMILY_MODES = ("baseline", "detector", "oracle", "type_control")


def with_family_tokens(graph: GraphBatch, families, mode: str) -> GraphBatch:
    """Change semantic fields only; UNKNOWN and padding remain invalid -1 IDs.

    Family IDs are experiment-specific 0=exterior, 1=rim. They do not identify
    individual physical corners. type_control uses redundant object-type tokens
    (BOX=0, PEG=1), activating the same embedding rows without adding information.
    """
    if mode not in FAMILY_MODES:
        raise ValueError(f"family mode must be one of {FAMILY_MODES}; got {mode!r}")
    semantics = torch.full_like(graph.semantic_id, -1)
    if mode in ("detector", "oracle"):
        families = torch.as_tensor(families, dtype=torch.long, device=semantics.device)
        if families.ndim == 1 and semantics.shape[0] == 1:
            families = families[None]
        if families.shape != semantics.shape or not torch.all((families >= -1) & (families <= 1)):
            raise ValueError("Family labels must have graph [B,N] shape and values -1,0,1")
        target = graph.node_valid & (graph.object_type == NodeType.BOX)
        semantics[target] = families[target]
    elif mode == "type_control":
        semantics[graph.node_valid & (graph.object_type == NodeType.BOX)] = 0
        semantics[graph.node_valid & (graph.object_type == NodeType.PEG)] = 1
    return replace(graph, semantic_id=semantics, semantic_valid=semantics >= 0)
