"""Check trainable GNN observations and gradient propagation through both backbones."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from graph.temporal.insertion import TemporalInsertionEstimator, feature_names
from graph.temporal.pose_observations import pose_input_step
from graph.build_graph import GraphBuilder
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.node_token_mamba import NodeTokenMamba
from training.gnn.temporal.joint_finetuning import (
    replace_gnn_predictions, forward_joint_predictions, build_joint_temporal,
)


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



class NodeJointFinetuningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1)
        self.temporal = NodeTokenMamba(node_width=16, d_model=16, d_state=4,
                                       d_conv=2, head_hidden=8)
        points = torch.tensor([[.03, 0, .1], [0., 0, .1], [.1, .02, .1]])
        kinds = torch.tensor([0, 0, 1])
        self.corner = SimpleNamespace(xyz=points, object_id=kinds, metadata=[{}, {}, {}])
        self.graph = GraphBuilder()([self.corner] * 3, torch.zeros(3, 3))
        self.observations = torch.randn(2, 2, 64)
        self.mask = torch.tensor([[True, True], [True, False]])

    def test_joint_step_updates_both_networks(self):
        optimizer = torch.optim.Adam(list(self.gnn.parameters()) + list(self.temporal.parameters()), lr=.001)
        before_gnn = self.gnn.node_encoder[0].weight.detach().clone()
        before_mamba = self.temporal.node_projection[0].weight.detach().clone()
        prediction = forward_joint_predictions(self.gnn, self.temporal, self.graph,
                                                self.observations, self.mask)
        prediction[self.mask].square().mean().backward()
        self.assertGreater(float(self.gnn.node_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.temporal.node_projection[0].weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before_gnn, self.gnn.node_encoder[0].weight))
        self.assertFalse(torch.equal(before_mamba, self.temporal.node_projection[0].weight))
        torch.testing.assert_close(prediction[~self.mask], torch.zeros(1, 21))

    def test_node_path_reaches_gnn_without_object21_gradient(self):
        def detach_object21(module, arguments, output):
            return dict(output, object21=output['object21'].detach())
        handle = self.gnn.register_forward_hook(detach_object21)
        try:
            output = forward_joint_predictions(self.gnn, self.temporal, self.graph,
                                                 self.observations, self.mask)
            output[self.mask].square().sum().backward()
        finally:
            handle.remove()
        self.assertGreater(float(self.gnn.node_encoder[0].weight.grad.abs().sum()), 0)

    def test_slot_permutation_uses_same_coordinate_order(self):
        expected = forward_joint_predictions(self.gnn, self.temporal, self.graph,
                                               self.observations, self.mask)
        permutation = torch.tensor([2, 0, 1])
        corner = SimpleNamespace(xyz=self.corner.xyz[permutation],
                                 object_id=self.corner.object_id[permutation], metadata=[{}, {}, {}])
        graph = GraphBuilder()([corner] * 3, torch.zeros(3, 3))
        actual = forward_joint_predictions(self.gnn, self.temporal, graph,
                                             self.observations, self.mask)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertFalse(self.gnn.peg_pool._forward_pre_hooks)

    def test_restore_preserves_pretrained_node_architecture(self):
        config = {'mode': self.temporal.mode, 'with_robot': True,
                  'model': {'d_model': 16, 'd_state': 4, 'd_conv': 2, 'head_hidden': 8}}
        restored = build_joint_temporal(config, node_width=16)
        restored.load_state_dict(self.temporal.state_dict(), strict=True)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, self.temporal.state_dict()[name])

if __name__ == '__main__':
    unittest.main()
