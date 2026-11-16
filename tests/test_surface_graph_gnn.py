"""Check the topology and batching used for observed surface graphs."""

import unittest

import numpy as np
import torch

from graph.build_graph import EdgeType, NodeType
from graph.build_graph.surface_graph import SurfaceGraph, collate_surface_graphs, delaunay_edges


def frame_for(uv, kinds):
    """Create a frame with a distinct XYZ value for every node."""
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    count = len(uv)
    xyz = np.column_stack((uv, np.arange(count, dtype=np.float32)))
    return SurfaceGraph(
        uv, xyz, np.asarray(kinds, dtype=np.int64),
        np.ones(count, dtype=bool), np.full(count, -1), delaunay_edges(uv),
        np.zeros(3, dtype=np.float32), np.zeros(21, dtype=np.float32),
    )


class SurfaceGraphTests(unittest.TestCase):
    def test_shared_pixel_nodes_keep_their_delaunay_neighbors(self):
        edges = delaunay_edges(np.array([[1, 1], [1.1, 1], [4, 1], [1, 4]]))
        self.assertEqual(edges.tolist(), [[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

    def test_collinear_pixels_connect_only_consecutive_nodes(self):
        edges = delaunay_edges(np.array([[5, 5], [1, 1], [3, 3]]))
        self.assertEqual(edges.tolist(), [[0, 2], [1, 2]])

    def test_batches_keep_reverse_edges_and_distinguish_cross_edges(self):
        frames = [
            frame_for([[0, 0], [4, 0], [0, 4]], [NodeType.PEG, NodeType.PEG, NodeType.BOX]),
            frame_for([[0, 0], [4, 0]], [NodeType.BOX, NodeType.BOX]),
            frame_for([], []),
        ]
        graph, observed, target = collate_surface_graphs(frames)
        self.assertEqual(graph.edge_index.shape, (2, 8))
        self.assertEqual(torch.bincount(graph.edge_type).tolist(), [2, 2, 0, 4])
        edges = set(map(tuple, graph.edge_index.T.tolist()))
        self.assertEqual(edges, {(dst, src) for src, dst in edges})
        self.assertTrue(torch.equal(observed.squeeze(-1).bool(), graph.node_valid))
        self.assertTrue(torch.all(graph.edge_type[graph.edge_index[0] >= 3] == EdgeType.BOX))
        self.assertEqual(target.shape, (3, 21))


if __name__ == "__main__":
    unittest.main()
