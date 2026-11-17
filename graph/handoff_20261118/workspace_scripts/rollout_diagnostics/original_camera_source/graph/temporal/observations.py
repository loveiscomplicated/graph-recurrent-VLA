"""Shared observation-only contract for temporal state estimators."""

import torch

from graph.build_graph import NodeType

QUALITY_NAMES = tuple(
    f"{kind}_{stat}"
    for kind in ("box", "hole")
    for stat in ("count", "present", "missing", "confidence_fraction",
                 "confidence_present", "confidence_mean", "confidence_min", "confidence_max")
) + ("all_vertices_missing",)

OBSERVATION_NAMES = ("gnn_hole_world_x", "gnn_hole_world_y", "gnn_hole_world_z",
                     "tcp_world_x", "tcp_world_y", "tcp_world_z") + QUALITY_NAMES
OBSERVATION_CONTRACT = "hole_world_tcp_quality_v1"


def temporal_inputs(prediction, tcp_world, quality):
    """Build [...,23] raw observations; only the model scales metre coordinates.

    The learned gate uses these same external observations. Its previous estimate,
    innovation and initialized flag belong to its own recurrence, not the data.
    No target, GT visibility or previous GT position is accepted here.
    """
    if prediction.shape[-1] != 21:
        raise ValueError('prediction must have shape [...,21]')
    leading = prediction.shape[:-1]
    if tcp_world.shape != leading + (3,) or quality.shape != leading + (17,):
        raise ValueError('Expected aligned tcp_world[...,3] and quality[...,17]')
    inputs = torch.cat((prediction[..., 10:13] + tcp_world, tcp_world, quality), dim=-1)
    if not torch.isfinite(inputs).all():
        raise ValueError('Temporal observations must be finite')
    return inputs


def observation_features(graph, include_peg=False):
    """Summarize actual valid vertices; unknown confidence is masked, never imputed."""
    features = []
    kinds = (NodeType.PEG, NodeType.BOX, NodeType.HOLE) if include_peg else (NodeType.BOX, NodeType.HOLE)
    for kind in kinds:
        present = graph.node_valid & (graph.object_type == kind)
        count = present.sum(1)
        known = present & graph.confidence_valid
        known_count = known.sum(1)
        confidence = graph.confidence
        mean = (confidence * known).sum(1) / known_count.clamp_min(1)
        minimum = confidence.masked_fill(~known, float('inf')).min(1).values
        minimum = torch.where(known_count > 0, minimum, 0)
        maximum = confidence.masked_fill(~known, 0).max(1).values
        features.extend((count, count > 0, count == 0, known_count / count.clamp_min(1),
                         known_count > 0, mean, minimum, maximum))
    features.append(~graph.node_valid.any(1))
    return torch.stack(features, dim=1).float()
