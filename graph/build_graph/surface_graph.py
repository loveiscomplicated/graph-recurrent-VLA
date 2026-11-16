"""Connect detected corners and tracked surface points in image space."""

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.spatial import Delaunay
import torch

from graph.build_graph import EdgeType, GraphBatch, NodeType


@dataclass(frozen=True)
class SurfaceGraph:
    """A frame's nodes and undirected edges before GNN collation."""

    uv: np.ndarray
    xyz: np.ndarray
    kinds: np.ndarray
    is_corner: np.ndarray
    track_id: np.ndarray
    edges: np.ndarray
    tcp: np.ndarray
    target: np.ndarray


def delaunay_edges(uv):
    """Triangulate pixel centers and keep every node at a shared pixel."""
    # Depth for surface nodes is looked up at these same integer centers.
    pixels, groups = np.unique(np.rint(uv), axis=0, return_inverse=True)
    if len(pixels) < 2:
        return np.empty((0, 2), dtype=np.int64)
    if np.linalg.matrix_rank(pixels - pixels[0]) < 2:
        # The 1D Delaunay edges join consecutive unique points along the line.
        links = zip(range(len(pixels) - 1), range(1, len(pixels)))
    else:
        triangles = Delaunay(pixels).simplices
        links = {
            tuple(sorted(edge))
            for triangle in triangles
            for edge in combinations(triangle, 2)
        }

    nodes_at_pixel = [[] for _ in pixels]
    for node, group in enumerate(groups):
        nodes_at_pixel[group].append(node)
    edges = [
        tuple(sorted((a, b)))
        for left, right in links
        for a, b in product(nodes_at_pixel[left], nodes_at_pixel[right])
    ]
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def collate_surface_graphs(frames):
    """Pad nodes and convert local undirected edges to directed GraphBatch edges."""
    batch_size = len(frames)
    width = max(1, max(len(frame.xyz) for frame in frames))
    xyz = torch.zeros(batch_size, width, 3)
    kinds = torch.full((batch_size, width), NodeType.PAD, dtype=torch.long)
    valid = torch.zeros(batch_size, width, dtype=torch.bool)
    edges = []
    for index, frame in enumerate(frames):
        count = len(frame.xyz)
        xyz[index, :count] = torch.from_numpy(frame.xyz)
        kinds[index, :count] = torch.from_numpy(frame.kinds)
        valid[index, :count] = True
        links = torch.from_numpy(frame.edges.T) + index * width
        edges.extend((links, links.flip(0)))

    edge_index = torch.cat(edges, dim=1)
    source, destination = edge_index
    flattened_kinds = kinds.flatten()
    source_kinds = flattened_kinds[source]
    destination_kinds = flattened_kinds[destination]
    edge_type = torch.where(source_kinds == destination_kinds, source_kinds, EdgeType.CROSS)
    flattened_xyz = xyz.reshape(-1, 3)
    relative = flattened_xyz[source] - flattened_xyz[destination]
    graph = GraphBatch(
        xyz_world=xyz, object_type=kinds, node_valid=valid,
        semantic_id=torch.full_like(kinds, -1), semantic_valid=torch.zeros_like(valid),
        confidence=xyz.new_zeros(valid.shape), confidence_valid=torch.zeros_like(valid),
        tcp_xyz_world=torch.from_numpy(np.stack([frame.tcp for frame in frames])),
        edge_index=edge_index, edge_type=edge_type, relative_xyz=relative,
        distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    graph.validate()
    # The original GNN's one extra feature marks direct observation.
    return graph, valid.to(xyz.dtype).unsqueeze(-1), torch.from_numpy(
        np.stack([frame.target for frame in frames])
    )
