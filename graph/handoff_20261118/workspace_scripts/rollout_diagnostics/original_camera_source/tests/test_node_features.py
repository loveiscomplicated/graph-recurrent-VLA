"""Node ordering, frozen-frame alignment and temporary-hook lifecycle contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from graph.temporal.node_features import capture_pre_pool_features, ordered_node_features
from training.gnn.temporal.node_features import infer_node_data


class NodePool(nn.Module):
    def forward(self, features, *unused):
        return features


class NodeFeatureTests(unittest.TestCase):
    def setUp(self):
        # Nodes 1/2 tie exactly; node 3 is padding and frame 1 has no nodes.
        self.graph = SimpleNamespace(
            xyz_world=torch.tensor([
                [[3., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 0., 0.]],
                [[0., 0., 0.]] * 4,
            ]),
            object_type=torch.tensor([[1, 0, 0, -1], [-1, -1, -1, -1]]),
            node_valid=torch.tensor([[True, True, True, False], [False] * 4]),
        )
        self.features = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        self.model = nn.Module()
        self.model.peg_pool = NodePool()

    def test_ordering_keeps_ties_empty_frames_and_feature_gradients(self):
        features = self.features.clone().requires_grad_()
        frames = ordered_node_features(self.graph, features)
        torch.testing.assert_close(frames[0], features[0, [1, 2, 0]], rtol=0, atol=0)
        self.assertEqual(frames[1].shape, (0, 2))

        frames[0].sum().backward()
        expected = torch.zeros_like(features)
        expected[0, :3] = 1
        torch.testing.assert_close(features.grad, expected, rtol=0, atol=0)

    def test_hook_preserves_existing_observers_and_is_removed_after_failure(self):
        observed = []
        permanent = self.model.peg_pool.register_forward_pre_hook(
            lambda module, arguments: observed.append(arguments[0])
        )
        self.addCleanup(permanent.remove)
        captured = []
        with self.assertRaisesRegex(RuntimeError, "failed forward"):
            with capture_pre_pool_features(self.model, captured.append):
                self.model.peg_pool(self.features)
                raise RuntimeError("failed forward")

        self.assertEqual(len(self.model.peg_pool._forward_pre_hooks), 1)
        self.model.peg_pool(self.features)
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], self.features)

    def test_frozen_features_follow_result_ids_and_own_their_storage(self):
        batch = SimpleNamespace(graph=self.graph, sample_ids=['first', 'empty'])

        def infer(frozen, loader, *arguments):
            for source in loader:
                frozen.peg_pool(self.features, None, None)
            return {'sample_ids': np.array(['empty', 'first'])}

        with patch('training.gnn.temporal.node_features.infer_insertion_data', side_effect=infer):
            result = infer_node_data(self.model, [batch], 'cpu', {}, {}, 'train')

        self.assertFalse(self.model.peg_pool._forward_pre_hooks)
        self.assertEqual(result['node_features'][0].shape, (0, 2))
        expected = self.features[0, [1, 2, 0]].numpy().copy()
        np.testing.assert_array_equal(result['node_features'][1], expected)
        self.features.fill_(-1)
        np.testing.assert_array_equal(result['node_features'][1], expected)

    def test_duplicate_frame_rejects_data_and_removes_hook(self):
        batch = SimpleNamespace(graph=self.graph, sample_ids=['duplicate', 'duplicate'])

        def infer(frozen, loader, *arguments):
            for source in loader:
                frozen.peg_pool(self.features, None, None)

        with patch('training.gnn.temporal.node_features.infer_insertion_data', side_effect=infer):
            with self.assertRaisesRegex(ValueError, 'Duplicate graph frame'):
                infer_node_data(self.model, [batch], 'cpu', {}, {}, 'train')
        self.assertFalse(self.model.peg_pool._forward_pre_hooks)

    def test_rejects_misaligned_feature_shape(self):
        with self.assertRaisesRegex(ValueError, r'\[B,N,H\]'):
            ordered_node_features(self.graph, self.features[:, :2])


if __name__ == '__main__':
    unittest.main()
