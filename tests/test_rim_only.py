"""Exercise filtered M0 graphs and cached family-label selection."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.box_hole_diagnostics import graph_from_arrays
from graph.build_graph import filter_nodes
from training.gnn.data import TrainingBatch, save_cached_sample
from training.gnn.vertex_family_data import VertexFamilyDataset


class RimOnlyTests(unittest.TestCase):
    def setUp(self):
        self.graph = graph_from_arrays(
            np.array([[[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]]),
            np.array([[0, 1, 1, 1]]), np.zeros((1, 3)),
        )

    def test_induced_edges_and_empty_graph(self):
        filtered = filter_nodes(self.graph, torch.tensor([[True, False, True, False]]))
        self.assertEqual(filtered.node_valid.tolist(), [[True, False, True, False]])
        self.assertEqual(filtered.object_type.tolist(), [[0, -1, 1, -1]])
        self.assertEqual(set(map(tuple, filtered.edge_index.T.tolist())), {(0, 2), (2, 0)})
        self.assertEqual(self.graph.node_valid.sum().item(), 4)
        empty = filter_nodes(self.graph, torch.zeros_like(self.graph.node_valid))
        self.assertEqual(empty.edge_index.shape, (2, 0))
        empty.validate()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            filter_nodes(self.graph, torch.ones_like(self.graph.object_type))

    def test_label_sources_keep_different_rims_without_semantic_input(self):
        sample = TrainingBatch(self.graph, {'object21': torch.zeros(1, 21)}, ['frame'])
        identity = {'sample_id': 'frame'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_cached_sample(root / 'graph.npz', sample, identity)
            np.savez_compressed(root / 'labels.npz', detector_family=[-1, 0, 1, -1],
                                oracle_family=[-1, 1, 0, -1], xyz_world=self.graph.xyz_world[0].numpy())
            row = dict(sample_id='frame', parent_id='p1', split='train', graph_path='graph.npz',
                       labels_path='labels.npz', sample_identity=identity,
                       graph_sha256=file_sha256(root / 'graph.npz'), labels_sha256=file_sha256(root / 'labels.npz'))
            manifest = dict(samples=[row])
            detector = VertexFamilyDataset(root, manifest, 'train', 'baseline', 'detector_rim')[0].graph
            oracle = VertexFamilyDataset(root, manifest, 'train', 'baseline', 'oracle_rim')[0].graph
            self.assertEqual(detector.node_valid.tolist(), [[True, False, True, False]])
            self.assertEqual(oracle.node_valid.tolist(), [[True, True, False, False]])
            self.assertFalse(detector.semantic_valid.any())
            self.assertFalse(oracle.semantic_valid.any())
            with self.assertRaisesRegex(ValueError, 'M0'):
                VertexFamilyDataset(root, manifest, 'train', 'oracle', 'oracle_rim')
