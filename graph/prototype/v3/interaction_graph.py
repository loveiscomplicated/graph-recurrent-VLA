"""B: unchanged samples/nodes/readout, explicit global and radius interactions.

Global relations use two deterministic existing surface representatives rather
than introducing extra object nodes. They are proxies, not exact pose nodes.
The existing context and v2 shared goal state continue to carry exact poses.
"""
from enum import IntEnum
from time import perf_counter
import torch
import torch.nn.functional as F
from ..graph_builder import GraphBuilder, GraphBatch, node_features
from ..models import mlp
from ..v2.models import CommonPolicy
from .geometry_audit import regions


class Relation(IntEnum):
    KINEMATIC = 0
    LOCAL_SURFACE = 1
    FINGER_PEG = 2
    PEG_HOLE = 3
    GLOBAL_GOAL = 4


class InteractionGraph:
    def __init__(self, surface_k=8, cross_k=4, radius=.05, distance_threshold=None):
        if surface_k < 1 or cross_k < 1 or radius <= 0 or (distance_threshold is not None and distance_threshold <= 0):
            raise ValueError('positive k/radius/threshold required')
        self.original = GraphBuilder(surface_k, cross_k)
        self.cross_k = cross_k
        self.radius = min(radius, distance_threshold) if distance_threshold is not None else radius

    def __call__(self, g):
        start = perf_counter()
        original = self.original(g)
        x, pos, types, ctx = node_features(g)
        b, n, _ = pos.shape
        p = g.peg_local.shape[1]
        labels = regions(g)
        keep = original.edge_type < 2
        edges = [original.edge_index[:, keep]]
        kinds = [original.edge_type[keep]]
        # All batches may have different numbers of radius-eligible edges.
        def add(src, dst, relation, batch):
            src, dst = src.reshape(-1)+batch*n, dst.reshape(-1)+batch*n
            edges.append(torch.stack((torch.cat((src, dst)), torch.cat((dst, src)))))
            kinds.append(torch.full((2*len(src),), int(relation), device=pos.device, dtype=torch.long))
        for batch in range(b):
            pg = torch.where(labels[batch] == 3)[0]
            head = torch.where(labels[batch] == 1)[0]
            hole = torch.where((labels[batch] == 4) | (labels[batch] == 5))[0]
            for source, target, relation in ((torch.tensor([1, 2], device=pos.device), pg, Relation.FINGER_PEG),
                                             (head, hole, Relation.PEG_HOLE)):
                distances = torch.cdist(pos[batch, source], pos[batch, target])
                values, neighbors = distances.topk(min(self.cross_k, len(target)), largest=False, sorted=True)
                mask = values <= self.radius
                src = source[:, None].expand_as(neighbors)[mask]
                dst = target[neighbors[mask]]
                add(src, dst, relation, batch)
            # Representatives selected locally; invariant to a common rigid transform.
            head_center = g.peg_half_size[batch].new_tensor([1., 0., 0.])*g.peg_half_size[batch, 0]
            head_rep = (g.peg_local[batch]-head_center).norm(dim=-1).argmin()+3
            tail_center = -head_center
            tail_rep = (g.peg_local[batch]-tail_center).norm(dim=-1).argmin()+3
            entry_center = g.hole_offset[batch].clone()
            entry_center[0] = -g.box_half_size[batch, 0]
            entry_ids = torch.where(labels[batch, 3+p:] == 4)[0]
            entry_rep = entry_ids[(g.hole_local[batch, entry_ids]-entry_center).norm(dim=-1).argmin()]+3+p
            add(torch.stack((head_rep, head_rep.new_tensor(0))), torch.stack((entry_rep, tail_rep)), Relation.GLOBAL_GOAL, batch)
        ei, et = torch.cat(edges, 1), torch.cat(kinds)
        encoded = torch.unique((ei[0]*(b*n)+ei[1])*len(Relation)+et, sorted=True)
        et = encoded % len(Relation)
        pair = encoded // len(Relation)
        ei = torch.stack((pair//(b*n), pair % (b*n)))
        src, dst = ei
        rel = pos.reshape(-1, 3)[dst]-pos.reshape(-1, 3)[src]
        attr = torch.cat((rel, rel.norm(dim=-1, keepdim=True), F.one_hot(et, len(Relation)).to(x.dtype)), -1)
        return GraphBatch(x, ctx, types, ei, attr, et, (perf_counter()-start)*1000)


def policy(cfg):
    """Same v2 encoder dimensions/layers/head/BC; one extra relation column."""
    model = CommonPolicy('surface', cfg, 'goal', 'regression')
    for layer in model.encoder.layers:
        layer.message = mlp([2*cfg['hidden_dim']+9, cfg['hidden_dim'], cfg['hidden_dim']], True)
    return model
