"""Existing Graph tensors and packing; no graph-generation algorithm."""
from dataclasses import dataclass
import numpy as np
import torch

NODE_TYPES = ("TCP", "Left", "Right", "Peg", "Box")
EDGE_TYPES = ("Intra-object", "Robot-Object", "Peg-Box")


@dataclass
class Graph:
    node_input: torch.Tensor  # pos3, finite F5, available3, valid3
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_type: torch.Tensor
    packet: dict  # immutable independent deep-copy from the existing receiver
    timings_ms: dict


@dataclass
class PackedGraphs:
    node_input: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_type: torch.Tensor
    node_graph: torch.Tensor
    ptr: torch.Tensor
    frame_map: torch.Tensor  # G,2: sequence batch index, timestep slot

    @property
    def count(self):
        return self.ptr.numel()-1

    def to(self, device):
        return PackedGraphs(**{name: value.to(device) for name, value in vars(self).items()})


def pack_graphs(graphs, frame_map=None):
    if not graphs:
        raise ValueError('at least one graph required')
    frames = {g.packet.get('coordinate_frame', 'EE_metres') for g in graphs}
    if len(frames) != 1:
        raise ValueError('cannot batch World and EE graphs together')
    if not frames <= {'World_metres', 'EE_metres'}:
        raise ValueError('unsupported Graph coordinate frame')
    if frame_map is None:
        frame_map = [(0, i) for i in range(len(graphs))]
    device=graphs[0].node_input.device
    if any(getattr(g,k).device!=device for g in graphs for k in ('node_input','node_type','edge_index','edge_attr','edge_type')):
        raise ValueError('graphs must share one tensor device')
    mapping = torch.as_tensor(frame_map, dtype=torch.long,device=device)
    if mapping.shape != (len(graphs), 2) or (mapping < 0).any() or len(torch.unique(mapping, dim=0)) != len(graphs):
        raise ValueError('unique nonnegative sequence/frame mapping required')
    ptr = np.r_[0, np.cumsum([len(g.node_type) for g in graphs])]
    edges = torch.cat([g.edge_index + int(ptr[i]) for i, g in enumerate(graphs)], dim=1)
    node_graph = torch.repeat_interleave(torch.arange(len(graphs),device=device), torch.tensor(np.diff(ptr),device=device))
    if edges.numel() and not torch.equal(node_graph[edges[0]], node_graph[edges[1]]):
        raise ValueError('cross-graph edge')
    return PackedGraphs(*(torch.cat([getattr(g, key) for g in graphs], dim=0) for key in ('node_input', 'node_type')),
                        edges, torch.cat([g.edge_attr for g in graphs]), torch.cat([g.edge_type for g in graphs]),
                        node_graph, torch.tensor(ptr,device=device), mapping)
