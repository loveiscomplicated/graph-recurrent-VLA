"""Diagnostic geometry encoders only. No recurrent module or action outputs."""
import torch
from torch import nn
from dataclasses import replace

from graph.integration_baseline.model import GraphEncoder
from graph.prototype.models import mlp
from .graph import FEATURE_DIMENSION


def canonical_robot_slots(graph):
    """Adapt named robot slots without changing the legacy encoder/readout.

    Normal builder outputs already have TCP/Left/Right first and are returned
    unchanged. Arbitrary node permutations are canonicalized using semantic
    node types, with edge endpoints remapped and all physical features retained.
    """
    expected=torch.arange(3,device=graph.node_type.device)[None].expand(graph.count,-1)
    robot_rows=graph.ptr[:-1,None]+expected
    if torch.equal(graph.node_type[robot_rows],expected):return graph
    order=torch.argsort(graph.node_graph*5+graph.node_type,stable=True)
    inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device=order.device)
    result=replace(graph,node_input=graph.node_input[order],node_type=graph.node_type[order],
                   edge_index=inverse[graph.edge_index],node_graph=graph.node_graph[order])
    if not torch.equal(result.node_type[robot_rows],expected):
        raise ValueError('one named TCP/Left/Right node per graph required')
    return result


def geometry_summary(graph):
    """Observed/input surface mean/min/max/bbox-centre/span, 15 per object.

    This is an explicit deterministic path from the identical input XYZ. It
    contains no actor origins, hole goals, meshes or extra sensor information.
    """
    result = []
    for owner in range(graph.count):
        row = []
        for kind in (3, 4):
            points = graph.node_input[(graph.node_graph == owner) & (graph.node_type == kind), :3]
            if len(points):
                lo, hi = points.amin(0), points.amax(0)
                row.extend((points.mean(0), lo, hi, (lo+hi)/2, hi-lo))
            else:
                row.extend([graph.node_input.new_zeros(3)]*5)
        result.append(torch.cat(row))
    return torch.stack(result)


class DiagnosticEncoder(nn.Module):
    def __init__(self, variant, d=64, layers=2):
        super().__init__()
        if variant not in ('E0', 'E1', 'E2'):
            raise ValueError('unknown diagnostic encoder')
        self.variant = variant
        self.graph_encoder = GraphEncoder(d=d, L=layers)
        self.graph_encoder.node_encoder = mlp([FEATURE_DIMENSION+4, d, d], final_activation=True)
        if variant == 'E1':
            self.graph_encoder.layers = nn.ModuleList()
        self.output_dimension = 5*d + (30 if variant == 'E2' else 0)

    def forward(self, graph):
        z, presence = self.graph_encoder(canonical_robot_slots(graph))
        if self.variant == 'E2':
            z = torch.cat((z, geometry_summary(graph)), -1)
        return z, presence


class GeometryRegressor(nn.Module):
    def __init__(self, variant, output_dimension=36):
        super().__init__()
        self.encoder = DiagnosticEncoder(variant)
        self.decoder = mlp([self.encoder.output_dimension, 64, output_dimension])

    def forward(self, graph):
        z, _ = self.encoder(graph)
        return self.decoder(z)
