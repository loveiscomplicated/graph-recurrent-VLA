"""Causality, masking and gradient tests for frame-node-token Mamba."""

import unittest

import numpy as np
import torch

from data_collection.temporal.node_tokens import collate_node_windows
from graph.temporal.insertion import TemporalInsertionEstimator
from graph.temporal.node_token_mamba import NodeTokenMamba
from training.gnn.temporal.config import LOSS_TERMS
from training.gnn.temporal.insertion import update_early_stopping
from training.gnn.temporal.node_token_mamba import run_epoch, should_stop_early


class NodeTokenMambaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.model = NodeTokenMamba(node_width=5, d_model=16, d_state=4, d_conv=2,
                                    head_hidden=8, backend='torch_reference')
        self.observations = torch.randn(2, 3, 64)
        self.observations[..., 3:7] = torch.tensor([1., 0., 0., 0.])
        self.observations[..., 13:17] = torch.tensor([1., 0., 0., 0.])
        self.nodes = torch.randn(2, 3, 4, 5)
        self.node_mask = torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
                                       [[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.bool)
        self.frame_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    def test_padding_does_not_change_valid_outputs(self):
        expected = self.model.forward_nodes(self.observations, self.nodes, self.node_mask, self.frame_mask)
        modified = self.nodes.clone()
        modified[~self.node_mask] = 1000
        actual = self.model.forward_nodes(self.observations, modified, self.node_mask, self.frame_mask)
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(actual[1, 2], torch.zeros(21))

    def test_future_nodes_cannot_change_earlier_frame(self):
        expected = self.model.forward_nodes(self.observations, self.nodes, self.node_mask, self.frame_mask)
        modified = self.nodes.clone()
        modified[0, 2, 0] += 100
        actual = self.model.forward_nodes(self.observations, modified, self.node_mask, self.frame_mask)
        torch.testing.assert_close(expected[0, :2], actual[0, :2])
        self.assertFalse(torch.equal(expected[0, 2], actual[0, 2]))

    def test_empty_nodes_match_original_estimator(self):
        baseline = TemporalInsertionEstimator(d_model=16, d_state=4, d_conv=2,
                                             head_hidden=8, backend='torch_reference')
        baseline.load_state_dict({key: value for key, value in self.model.state_dict().items()
                                  if not key.startswith('node_projection.')})
        actual = self.model.forward_nodes(self.observations, self.nodes,
                                          torch.zeros_like(self.node_mask), self.frame_mask)
        expected = baseline.forward_sequence(self.observations, self.frame_mask)
        torch.testing.assert_close(actual, expected)

    def test_node_projection_gets_gradient(self):
        result = self.model.forward_nodes(self.observations, self.nodes, self.node_mask, self.frame_mask)
        result[self.frame_mask].square().sum().backward()
        self.assertGreater(float(self.model.node_projection[0].weight.grad.abs().sum()), 0)

    def test_training_epoch_updates_model(self):
        targets = self.observations[0, :2, :21].clone()
        nodes = [np.ones((2, 5), dtype=np.float32), np.empty((0, 5), dtype=np.float32)]
        windows = [(self.observations[0, :2], targets, nodes)]
        effective = {'training': {'episode_batch_size': 1, 'gradient_clip_norm': 1.0},
                     'loss': {'position_unit_m': .01, 'rotation_unit_rad': .1,
                              'ablation': 'pose_relative_full',
                              'weights': {name: 1.0 for name in LOSS_TERMS}}}
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        before = self.model.node_projection[0].weight.detach().clone()
        loss = run_epoch(self.model, windows, effective, torch.device('cpu'), optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, self.model.node_projection[0].weight))

    def test_early_stopping_resets_on_improvement_and_respects_minimum(self):
        settings = {'min_epochs': 5, 'patience': 2, 'min_delta': 0.0}
        state = {'reference_loss': float('inf'), 'stale_epochs': 0}
        stopped = []
        for epoch, loss in enumerate([3.0, 4.0, 4.0, 2.0, 3.0, 4.0], 1):
            state = update_early_stopping(state, loss, settings)
            stopped.append(should_stop_early(epoch, settings, state))
        self.assertEqual(stopped, [False, False, False, False, False, True])
        self.assertFalse(should_stop_early(100, None, state))

    def test_collate_preserves_empty_frames(self):
        nodes = [np.empty((0, 5), dtype=np.float32), np.ones((2, 5), dtype=np.float32)]
        obs = self.observations[0, :2]
        batch = collate_node_windows([(obs, torch.zeros(2, 21), nodes)])
        self.assertEqual(batch[2].shape, (1, 2, 2, 5))
        self.assertEqual(batch[3].sum().item(), 2)
        self.assertEqual(batch[4].sum().item(), 2)


if __name__ == '__main__':
    unittest.main()
