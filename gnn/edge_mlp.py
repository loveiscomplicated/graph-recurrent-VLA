"""Existing live edge-aware GNN and structured readout."""
from time import perf_counter
import torch
from torch import nn
from vla.nn import mlp


class MessageLayer(nn.Module):
    def __init__(self, d, edge_type_dim):
        super().__init__()
        self.message = mlp([2*d+4+edge_type_dim, d, d], final_activation=True)
        self.update = mlp([2*d, d, d])

    def forward(self, h, graph, edge_embedding):
        src, dst = graph.edge_index
        message = self.message(torch.cat((h[dst], h[src], graph.edge_attr, edge_embedding), -1))
        sums = torch.zeros_like(h).index_add(0, dst, message)
        degree = h.new_zeros((len(h), 1)).index_add(0, dst, h.new_ones((len(dst), 1)))
        return h + self.update(torch.cat((h, sums/degree.clamp_min(1)), -1))


def structured_readout(h, graph):
    robot_rows = graph.ptr[:-1, None] + torch.arange(3, device=h.device)
    slots = [h[robot_rows].reshape(graph.count, -1)]
    present = []
    for kind in (3, 4):
        selected = graph.node_type == kind
        owners = graph.node_graph[selected]
        sums = h.new_zeros((graph.count, h.shape[-1])).index_add(0, owners, h[selected])
        counts = h.new_zeros((graph.count, 1)).index_add(0, owners, h.new_ones((len(owners), 1)))
        slots.append(sums/counts.clamp_min(1))
        present.append(counts > 0)
    return torch.cat(slots, -1), torch.cat(present, -1).to(h.dtype)


def configure_live_encoder(encoder, *, d, variant):
    """Apply the existing Graph21/E0/E1 specialization at its original RNG point.

    Historical initialization constructs the generic encoder, then controller,
    then replaces the input layer. Preserve that order for baseline seed parity.
    """
    if variant not in ('E0', 'E1'):
        raise ValueError('E0/E1 only')
    encoder.node_encoder = mlp([21+4, d, d], final_activation=True)
    if variant == 'E1':
        encoder.layers = nn.ModuleList()


class GraphEncoder(nn.Module):
    def __init__(self, d=64, L=2, node_type_dim=4, edge_type_dim=4):
        super().__init__()
        if type(d) is not int or d < 1 or type(L) is not int or L < 1:
            raise ValueError('positive d/L required')
        self.d = d
        self.output_dim = 5 * d
        self.node_types = nn.Embedding(5, node_type_dim)
        self.edge_types = nn.Embedding(3, edge_type_dim)
        self.node_encoder = mlp([14+node_type_dim, d, d], final_activation=True)
        self.layers = nn.ModuleList([MessageLayer(d, edge_type_dim) for _ in range(L)])

    def forward(self, graph, trace=None):
        tick = perf_counter()
        h = self.node_encoder(torch.cat((graph.node_input, self.node_types(graph.node_type)), -1))
        edge_embedding = self.edge_types(graph.edge_type)
        shapes = [list(h.shape)]
        for i, layer in enumerate(self.layers):
            start = perf_counter()
            h = layer(h, graph, edge_embedding)
            if trace is not None:
                trace[f'gnn_layer_{i}_ms'] = (perf_counter()-start)*1000
            shapes.append(list(h.shape))
        after_gnn = perf_counter()
        z, presence = structured_readout(h, graph)
        if trace is not None:
            trace.update(gnn_ms=(after_gnn-tick)*1000, readout_ms=(perf_counter()-after_gnn)*1000,
                         layer_shapes=shapes, message_shape=[graph.edge_type.numel(), self.d],
                         readout_shape=list(z.shape))
        return z, presence
