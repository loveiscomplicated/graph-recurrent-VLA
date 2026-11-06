"""Batched typed interaction graph, world axes with origin at current TCP."""
from dataclasses import dataclass
from enum import IntEnum
from time import perf_counter

import torch
from torch import Tensor
import torch.nn.functional as F

from .geometry import rotation


class NodeType(IntEnum):
    EE = 0
    LEFT_PAD = 1
    RIGHT_PAD = 2
    PEG = 3
    HOLE = 4


class EdgeType(IntEnum):
    KINEMATIC = 0
    LOCAL_SURFACE = 1
    FINGER_PEG = 2
    PEG_HOLE = 3


def context_features(g):
    origin = g.ee_pose[:, :3]
    pose_features = lambda p: torch.cat((p[:, :3]-origin, rotation(p[:, 3:]).flatten(1)), -1)
    return torch.cat((g.gripper, g.qpos, g.ee_pose[:, :3], rotation(g.ee_pose[:, 3:]).flatten(1),
        g.root_pose[:, :3], rotation(g.root_pose[:, 3:]).flatten(1),
        pose_features(g.peg_pose), g.peg_half_size,
        pose_features(g.box_pose), g.box_half_size,
        pose_features(g.hole_pose), g.hole_radius, g.hole_offset), -1)


def node_features(g, center=False):
    b = g.ee_pose.shape[0]
    robot = torch.stack((g.ee_pose[:, :3], g.left_fingertip, g.right_fingertip), 1)
    if center:
        pos = torch.cat((robot, g.peg_pose[:, None, :3], g.hole_pose[:, None, :3]), 1)
        local, normals = torch.zeros_like(pos), torch.zeros_like(pos)
        types = torch.arange(5, device=pos.device)
    else:
        pos = torch.cat((robot, g.peg_points, g.hole_points), 1)
        zeros = torch.zeros_like(robot)
        local = torch.cat((zeros, g.peg_local, g.hole_local), 1)
        normals = torch.cat((zeros, g.peg_normals, g.hole_normals), 1)
        types = torch.cat((torch.arange(3), torch.full((g.peg_points.shape[1],), 3),
                           torch.full((g.hole_points.shape[1],), 4))).to(pos.device)
    one_hot = F.one_hot(types, len(NodeType)).to(pos.dtype).expand(b, -1, -1)
    x = torch.cat((one_hot, pos-g.ee_pose[:, None, :3], local, normals), -1)
    return x, pos, types, context_features(g)


def flat_features(g):
    """Exactly the surface graph's node features and context, same order."""
    x, _, _, ctx = node_features(g)
    return torch.cat((x.flatten(1), ctx), -1)


@dataclass
class GraphBatch:
    x: Tensor                    # [B,N,14]
    context: Tensor              # [B,81]
    node_type: Tensor            # [N]
    edge_index: Tensor           # [2,E], global flattened node indices
    edge_attr: Tensor            # [E,8]: destination-source xyz, distance, 4 types
    edge_type: Tensor            # [E]
    construction_ms: float

    @property
    def stats(self):
        b, n = self.x.shape[:2]
        return {"batch_size": b, "nodes_per_graph": n,
                "edges_total": self.edge_index.shape[1],
                "edges_per_graph_mean": self.edge_index.shape[1]/b,
                "construction_ms": self.construction_ms}


class GraphBuilder:
    def __init__(self, surface_k=8, interaction_k=4, center=False):
        if surface_k < 1 or interaction_k < 1:
            raise ValueError("k must be positive")
        self.k, self.cross_k, self.center = surface_k, interaction_k, center

    def __call__(self, g):
        start = perf_counter()
        x, pos, types, ctx = node_features(g, self.center)
        b, n = pos.shape[:2]
        edges, relations = [], []
        def add(src, dst, kind):
            src = src.expand(b, -1)
            dst = dst.expand(b, -1)
            # Explicit reverse edges, reversed relative vectors computed below.
            edges.append(torch.stack((torch.cat((src, dst), 1), torch.cat((dst, src), 1)), 1))
            relations.append(torch.full((b, src.shape[1]*2), int(kind), device=pos.device))
        device = pos.device
        add(torch.tensor([[0, 0]], device=device), torch.tensor([[1, 2]], device=device), EdgeType.KINEMATIC)
        if self.center:
            add(torch.tensor([[1, 2]], device=device), torch.tensor([[3, 3]], device=device), EdgeType.FINGER_PEG)
            add(torch.tensor([[3]], device=device), torch.tensor([[4]], device=device), EdgeType.PEG_HOLE)
        else:
            p, h = g.peg_points.shape[1], g.hole_points.shape[1]
            for points, offset in ((g.peg_local, 3), (g.hole_local, 3+p)):
                m = points.shape[1]
                dist = torch.cdist(points, points)
                dist = dist.masked_fill(torch.eye(m, device=device, dtype=torch.bool)[None], torch.inf)
                neighbors = dist.topk(min(self.k, m-1), largest=False, sorted=True).indices
                src = torch.arange(m, device=device).reshape(1, m, 1).expand_as(neighbors)
                add(src.reshape(b, -1)+offset, neighbors.reshape(b, -1)+offset, EdgeType.LOCAL_SURFACE)
                # Nearest earlier-index tree guarantees one connected surface
                # even when unconstrained kNN separates thin/opposite faces.
                prefix = dist.masked_fill(torch.triu(torch.ones(m, m, device=device, dtype=torch.bool))[None], torch.inf)
                add(torch.arange(1, m, device=device)[None]+offset,
                    prefix[:, 1:].argmin(-1)+offset, EdgeType.LOCAL_SURFACE)
            dist = torch.cdist(pos[:, 1:3], g.peg_points)
            neighbors = dist.topk(min(self.cross_k, p), largest=False).indices
            src = torch.tensor([1, 2], device=device).reshape(1, 2, 1).expand_as(neighbors)
            add(src.reshape(b, -1), neighbors.reshape(b, -1)+3, EdgeType.FINGER_PEG)
            # Always present: each peg point connects to its k nearest hole
            # points. No radius cutoff can disconnect distant objects.
            dist = torch.cdist(g.peg_points, g.hole_points)
            neighbors = dist.topk(min(self.cross_k, h), largest=False).indices
            src = torch.arange(p, device=device).reshape(1, p, 1).expand_as(neighbors)
            add(src.reshape(b, -1)+3, neighbors.reshape(b, -1)+3+p, EdgeType.PEG_HOLE)
        ei = torch.cat(edges, -1) + (torch.arange(b, device=device)*n)[:, None, None]
        et = torch.cat(relations, -1)
        # Coalesce reciprocal kNN/tree duplicates, separately by relation.
        encoded = ((ei[:, 0]*b*n + ei[:, 1])*len(EdgeType)+et).flatten()
        encoded = torch.unique(encoded, sorted=True)
        et = encoded % len(EdgeType)
        pair = encoded // len(EdgeType)
        ei = torch.stack((pair//(b*n), pair%(b*n)))
        src, dst = ei
        positions = pos.reshape(-1, 3)
        rel = positions[dst]-positions[src]
        attr = torch.cat((rel, rel.norm(dim=-1, keepdim=True), F.one_hot(et, len(EdgeType)).to(x.dtype)), -1)
        return GraphBatch(x, ctx, types, ei, attr, et, (perf_counter()-start)*1000)
