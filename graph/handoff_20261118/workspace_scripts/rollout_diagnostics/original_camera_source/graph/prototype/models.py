"""Three independent encoders with an identical 128 -> 128 -> 7 action head."""
import torch
from torch import nn

from .graph_builder import flat_features


def mlp(dims, final_activation=False):
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims)-2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ActionHead(nn.Sequential):
    def __init__(self, embedding_dim=128):
        super().__init__(nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, 7))


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = mlp([2*hidden+8, hidden, hidden], True)
        self.update = mlp([2*hidden, hidden, hidden])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, graph):
        src, dst = graph.edge_index
        msg = self.message(torch.cat((h[src], h[dst], graph.edge_attr), -1))
        sums = torch.zeros_like(h).index_add(0, dst, msg)
        counts = torch.zeros(h.shape[0], 1, device=h.device).index_add(0, dst, torch.ones_like(msg[:, :1]))
        update = self.update(torch.cat((h, sums/counts.clamp_min(1)), -1))
        return torch.relu(self.norm(h+update))


class GraphEncoder(nn.Module):
    def __init__(self, node_dim=14, context_dim=81, hidden=128, layers=3, embedding=128):
        super().__init__()
        self.node_encoder = mlp([node_dim, hidden, hidden], True)
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(layers)])
        self.readout = mlp([5*hidden+context_dim, hidden, embedding])

    def forward(self, graph):
        b, n, f = graph.x.shape
        h = self.node_encoder(graph.x.reshape(-1, f))
        for layer in self.layers:
            h = layer(h, graph)
        h = h.reshape(b, n, -1)
        readout = torch.cat((h[:, :3].flatten(1), h[:, graph.node_type == 3].mean(1),
                             h[:, graph.node_type == 4].mean(1), graph.context), -1)
        return self.readout(readout)


class FlatEncoder(nn.Module):
    def __init__(self, input_dim, width=128, embedding=128):
        super().__init__()
        self.net = mlp([input_dim, width, width, embedding])

    def forward(self, features):
        return self.net(features)


class Policy(nn.Module):
    def __init__(self, encoder, embedding=128):
        super().__init__()
        self.encoder = encoder
        self.action_head = ActionHead(embedding)

    def forward(self, inputs):
        return self.action_head(self.encoder(inputs))


def make_policy(name, cfg):
    if cfg["embedding_dim"] != 128:
        raise ValueError("prototype embedding must be 128 for shared action head")
    if name == "flat":
        dim = (3+cfg["peg_points"]+cfg["hole_points"])*14+81
        encoder = FlatEncoder(dim, cfg["flat_width"], cfg["embedding_dim"])
    elif name in ("center", "surface"):
        encoder = GraphEncoder(hidden=cfg["hidden_dim"], layers=cfg["layers"], embedding=cfg["embedding_dim"])
    else:
        raise ValueError(name)
    return Policy(encoder, cfg["embedding_dim"])


def model_inputs(name, geometry, builder=None):
    return flat_features(geometry) if name == "flat" else builder(geometry)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
