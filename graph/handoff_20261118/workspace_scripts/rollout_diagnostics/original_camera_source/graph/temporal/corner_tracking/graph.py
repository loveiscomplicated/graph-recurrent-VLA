"""Private completed-graph contract, keeping raw observed GraphBatch unchanged."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from graph.build_graph import GraphBatch


@lru_cache(maxsize=128)
def complete_edge_indices(count, device):
    """Cache immutable topology and avoid CUDA boolean-index shape synchronization."""
    pairs = [(source, target) for source in range(count)
             for target in range(count) if source != target]
    return torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()


@dataclass
class CompletedGraph:
    """Geometry carrier plus explicit provenance for every included point.

    geometry is an internal GNN adapter: its node_valid means included support.
    It must never be serialized as a raw detector graph or used as visibility.
    """

    geometry: GraphBatch
    observed: torch.Tensor
    age: torch.Tensor
    standard_deviation: torch.Tensor

    def features(self, age_scale=32.0, coordinate_scale=0.05):
        """Five extra GNN channels: source flag, age and XYZ uncertainty."""
        return torch.cat((self.observed.float()[..., None],
                          self.age[..., None] / age_scale,
                          self.standard_deviation / coordinate_scale), dim=-1)


def build_completed_graph(positions, kinds, observed, ages, deviations, qualities,
                          quality_valid, tcp, device, *, validate=True):
    """Rebuild complete directed edges with differentiable predicted coordinates."""
    count = len(positions)
    slots = max(1, count)
    xyz = (torch.stack(positions)[None] if count else torch.zeros(1, 1, 3, device=device))
    types = torch.full((1, slots), -1, dtype=torch.long, device=device)
    valid = torch.zeros(1, slots, dtype=torch.bool, device=device)
    confidence = torch.zeros(1, slots, device=device)
    confidence_valid = torch.zeros_like(valid)
    if count:
        types[0] = torch.tensor(kinds, device=device)
        valid[0] = True
        confidence[0] = torch.tensor(qualities, device=device)
        confidence_valid[0] = torch.tensor(quality_valid, device=device)
    edge_index = complete_edge_indices(count, str(device))
    src, dst = edge_index
    relative = xyz[0, src] - xyz[0, dst]
    edge_type = torch.where(types[0, src] == types[0, dst], types[0, src], 3)
    geometry = GraphBatch(
        xyz, types, valid, torch.full_like(types, -1), torch.zeros_like(valid),
        confidence, confidence_valid, torch.as_tensor(tcp, device=device)[None],
        edge_index, edge_type, relative,
        torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    if validate:
        geometry.validate()
    source = torch.zeros_like(valid)
    age = torch.zeros(1, slots, device=device)
    std = torch.zeros(1, slots, 3, device=device)
    if count:
        source[0] = torch.tensor(observed, device=device)
        age[0] = torch.tensor(ages, device=device)
        std = torch.stack(deviations)[None]
    return CompletedGraph(geometry, source, age, std)


def batch_completed_graphs(graphs):
    """Pad independent frame graphs; edges never connect different frames."""
    slots = max(graph.geometry.xyz_world.shape[1] for graph in graphs)

    def padded(value, fill=0):
        padding = slots - value.shape[1]
        widths = (0, 0, 0, padding) if value.ndim == 3 else (0, padding)
        return torch.nn.functional.pad(value, widths, value=fill)

    node_fields = ("xyz_world", "object_type", "node_valid", "semantic_id", "semantic_valid",
                   "confidence", "confidence_valid")
    fields = {}
    for name in node_fields:
        fill = -1 if name in ("object_type", "semantic_id") else 0
        fields[name] = torch.cat([padded(getattr(graph.geometry, name), fill) for graph in graphs])
    fields["tcp_xyz_world"] = torch.cat([graph.geometry.tcp_xyz_world for graph in graphs])
    fields["edge_index"] = torch.cat([graph.geometry.edge_index + index * slots
                                       for index, graph in enumerate(graphs)], dim=1)
    for name in ("edge_type", "relative_xyz", "distance"):
        fields[name] = torch.cat([getattr(graph.geometry, name) for graph in graphs])
    geometry = GraphBatch(**fields)
    geometry.validate()
    return CompletedGraph(geometry,
                          torch.cat([padded(graph.observed) for graph in graphs]),
                          torch.cat([padded(graph.age) for graph in graphs]),
                          torch.cat([padded(graph.standard_deviation) for graph in graphs]))
