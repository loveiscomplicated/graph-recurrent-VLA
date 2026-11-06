"""Observation-only scalar temporal gate; positions use world axes and metres."""

import torch
from torch import nn

from graph.build_graph import NodeType
from graph.gated_mpnn import assemble_state38

QUALITY_NAMES = tuple(
    f"{kind}_{stat}"
    for kind in ("box", "hole")
    for stat in ("count", "present", "missing", "confidence_fraction",
                 "confidence_present", "confidence_mean", "confidence_min", "confidence_max")
) + ("all_vertices_missing",)


def observation_features(graph):
    """Summarize actual valid vertices; unknown confidence is masked, never imputed."""
    features = []
    for kind in (NodeType.BOX, NodeType.HOLE):
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


class LearnedTemporalGate(nn.Module):
    """Predict one alpha without count/confidence rules or GT state inputs.

    Cold start seeds memory with the current prediction (effective alpha=1).
    Thereafter the MLP sees its own previous output, including during training.
    """

    def __init__(self, hidden=32, position_scale_m=0.1):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError('hidden must be a positive integer')
        if not 0 < position_scale_m < float('inf'):
            raise ValueError('position_scale_m must be finite and positive')
        self.position_scale_m = position_scale_m
        self.network = nn.Sequential(nn.Linear(30, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1), nn.Sigmoid())

    def step(self, hole_relative, tcp_world, quality, previous_world=None):
        """Advance one frame; callers pass None on every episode reset."""
        if hole_relative.shape != (3,) or tcp_world.shape != (3,) or quality.shape != (17,):
            raise ValueError('Expected hole_relative[3], tcp_world[3], quality[17]')
        current = hole_relative + tcp_world
        initialized = previous_world is not None
        previous = previous_world if initialized else torch.zeros_like(current)
        if previous.shape != (3,):
            raise ValueError('previous_world must have shape [3] or be None')
        inputs = torch.cat((current / self.position_scale_m,
                            previous / self.position_scale_m,
                            (current - previous) / self.position_scale_m,
                            tcp_world / self.position_scale_m, quality,
                            current.new_tensor([float(initialized)])))
        if not torch.isfinite(inputs).all():
            raise ValueError('Gate observations and previous memory must be finite')
        proposed_alpha = self.network(inputs).squeeze(-1)
        alpha = proposed_alpha if initialized else (proposed_alpha * 0 + 1)
        memory = (1 - alpha) * previous + alpha * current
        return memory, alpha


def correct_object21(object21, hole_relative, preserve_endpoint=True):
    """Replace only hole XYZ, optionally preserving estimated peg-tip geometry."""
    if object21.shape[-1] != 21 or hole_relative.shape != object21.shape[:-1] + (3,):
        raise ValueError('Expected Object21[...,21] and hole_relative[...,3]')
    output = object21.clone()
    if preserve_endpoint:
        output[..., 7:10] = object21[..., 7:10] + object21[..., 10:13] - hole_relative
    output[..., 10:13] = hole_relative
    return output


def corrected_state38(robot15, object21, freshness2, hole_relative):
    """Use the existing State38 assembler with a geometrically consistent Object21."""
    return assemble_state38(robot15, correct_object21(object21, hole_relative), freshness2)


def rollout(gate, prediction, tcp_world, quality):
    """Unroll one episode causally without targets or teacher forcing."""
    memory = None
    positions, alphas = [], []
    for index in range(len(prediction)):
        memory, alpha = gate.step(prediction[index, 10:13], tcp_world[index], quality[index], memory)
        positions.append(memory)
        alphas.append(alpha)
    if not positions:
        raise ValueError('Cannot replay an empty episode')
    return torch.stack(positions), torch.stack(alphas)
