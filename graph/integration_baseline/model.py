"""Mask-aware Edge MLP GNN, exact structured readout and existing GRU/BC."""
from time import perf_counter

import torch
from torch import nn

from graph.prototype.models import mlp
from graph.prototype.v2.models import bc_loss
from recurrent.models import RNNPolicy


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


class GraphEncoder(nn.Module):
    def __init__(self, d=64, L=2, node_type_dim=4, edge_type_dim=4):
        super().__init__()
        if type(d) is not int or d < 1 or type(L) is not int or L < 1:
            raise ValueError('positive d/L required')
        self.d = d
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


class IntegrationPolicy(nn.Module):
    def __init__(self, robot_schema, *, d=64, L=2, gru_hidden=144, gru_layers=1, enc_hidden=128):
        super().__init__()
        self.robot_schema = robot_schema
        self.graph_encoder = GraphEncoder(d, L)
        self.fused_dim = 5*d + 2 + robot_schema['dimension']
        self.policy = RNNPolicy(self.fused_dim, 7, cell='gru', hidden=gru_hidden,
                                layers=gru_layers, enc_hidden=enc_hidden)

    def fuse(self, graphs, robot_state, normalizer, trace=None):
        if normalizer.schema != self.robot_schema:
            raise ValueError('normalizer/model schema mismatch')
        if robot_state.shape != (graphs.count, self.robot_schema['dimension']):
            raise ValueError('one full robot state per mapped graph required')
        z, presence = self.graph_encoder(graphs, trace)
        tick = perf_counter()
        fused = torch.cat((z, presence, normalizer.apply(robot_state)), -1)
        if trace is not None:
            trace.update(fusion_ms=(perf_counter()-tick)*1000, fused_shape=list(fused.shape))
        return fused

    def forward(self, graphs, robot_state, normalizer, hidden=None, *, lengths=None, reset_before=None, trace=None):
        fused = self.fuse(graphs, robot_state, normalizer, trace)
        b = int(graphs.frame_map[:, 0].max())+1
        t = int(graphs.frame_map[:, 1].max())+1
        sequence = fused.new_zeros((b, t, self.fused_dim))
        sequence[graphs.frame_map[:, 0], graphs.frame_map[:, 1]] = fused
        occupied = torch.zeros((b, t), dtype=torch.bool, device=fused.device)
        occupied[graphs.frame_map[:, 0], graphs.frame_map[:, 1]] = True
        if lengths is None:
            lengths = occupied.sum(1)
        lengths = torch.as_tensor(lengths, device=fused.device)
        if lengths.shape != (b,) or not torch.equal(occupied, torch.arange(t, device=fused.device)[None] < lengths[:, None]):
            raise ValueError('sequence mapping must be contiguous, without holes or trailing fake frames')
        if reset_before is None and occupied.all():
            return self.policy(sequence, hidden)
        if reset_before is None:
            reset_before = torch.zeros((b, t), dtype=torch.bool, device=fused.device)
        if reset_before.shape != (b, t):
            raise ValueError('reset mask must be [B,T]')
        if hidden is None:
            hidden = fused.new_zeros((self.policy.rnn.num_layers, b, self.policy.rnn.hidden_size))
        outputs = []
        for i in range(t):
            active = occupied[:, i]
            hidden = hidden * (~(reset_before[:, i] & active))[None, :, None]
            out, next_hidden = self.policy(sequence[:, i:i+1], hidden)
            hidden = torch.where(active[None, :, None], next_hidden, hidden)
            outputs.append(out[:, 0] * active[:, None])
        return torch.stack(outputs, 1), hidden

    def step(self, graphs, robot_state, normalizer, hidden=None, reset=None):
        b = graphs.count
        if not torch.equal(graphs.frame_map, torch.stack((torch.arange(b, device=graphs.frame_map.device),
                                                         torch.zeros(b, dtype=torch.long, device=graphs.frame_map.device)), -1)):
            raise ValueError('step requires one graph per stream with timestep slot zero')
        if hidden is not None and reset is not None:
            hidden = hidden * (~reset)[None, :, None]
        action, hidden = self.forward(graphs, robot_state, normalizer, hidden)
        return action[:, 0], hidden


class EpisodeStream:
    """Single stream lifecycle; episode changes/flags/reset clear hidden state."""
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.hidden = None
        self.identity = None
        self.timestep = None

    def reset(self):
        self.hidden = self.identity = self.timestep = None

    def step(self, graphs, robot_state, episode_identity, timestep, *, reset=False,
             previous_terminated=False, previous_truncated=False):
        if reset or previous_terminated or previous_truncated or self.identity != episode_identity:
            self.reset()
        if self.timestep is not None and timestep != self.timestep+1:
            raise ValueError('noncontiguous stream observation')
        action, self.hidden = self.model.step(graphs, robot_state, self.normalizer, self.hidden)
        self.identity, self.timestep = episode_identity, timestep
        return action
