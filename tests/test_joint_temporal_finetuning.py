"""Check trainable GNN observations and gradient propagation through both backbones."""

import unittest

import torch
from torch import nn

from graph.temporal.insertion import TemporalInsertionEstimator, feature_names
from graph.temporal.pose_observations import pose_input_step
from training.gnn.temporal.joint_finetuning import replace_gnn_predictions


class JointFinetuningTest(unittest.TestCase):
    def test_world_observation_matches_existing_feature_contract(self):
        torch.manual_seed(7)
        prediction = torch.randn(3, 21)
        tcp = torch.randn(3, 3)
        quality = torch.randn(3, 25)
        observations = torch.randn(2, 2, len(feature_names(True)))
        mask = torch.tensor([[True, True], [True, False]])
        output = replace_gnn_predictions(observations, mask, prediction, tcp)
        expected = pose_input_step(prediction, quality, tcp)[:, :21]
        torch.testing.assert_close(output[mask, :21], expected)
        torch.testing.assert_close(output[..., 21:], observations[..., 21:])
        self.assertTrue(torch.equal(output[~mask, :21], torch.zeros(1, 21)))

    def test_temporal_loss_reaches_and_updates_gnn_for_both_models(self):
        for architecture in ('mamba', 'transformer'):
            with self.subTest(architecture=architecture):
                torch.manual_seed(9)
                gnn = nn.Linear(5, 21)
                temporal = TemporalInsertionEstimator(
                    architecture=architecture, backend='torch_reference', d_model=16,
                    n_layers=1, head_hidden=8, n_heads=4, feedforward_dim=32,
                )
                optimizer = torch.optim.Adam(list(gnn.parameters()) + list(temporal.parameters()), lr=1e-3)
                mask = torch.tensor([[True, True], [True, False]])
                observations = torch.zeros(2, 2, len(feature_names(True)))
                prediction = gnn(torch.randn(3, 5))
                inputs = replace_gnn_predictions(observations, mask, prediction, torch.randn(3, 3))
                before = gnn.weight.detach().clone()
                result = temporal.forward_sequence(inputs, mask)
                loss = result[mask].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(gnn.weight.grad).all())
                self.assertGreater(float(gnn.weight.grad.abs().sum()), 0)
                optimizer.step()
                self.assertFalse(torch.equal(before, gnn.weight))

    def test_rejects_misaligned_predictions(self):
        with self.assertRaisesRegex(ValueError, 'one Object21'):
            replace_gnn_predictions(torch.zeros(1, 2, 65), torch.tensor([[True, True]]),
                                    torch.zeros(1, 21), torch.zeros(1, 3))


if __name__ == '__main__':
    unittest.main()
