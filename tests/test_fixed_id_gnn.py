"""Contracts for the fixed-ID temporal predictor to GNN connection."""

import unittest

import torch
from torch import nn

from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from graph.temporal.fixed_id_gnn import FixedIdCornerGNN


class ConstantTemporal(nn.Module):
    def forward(self, inputs, context_valid):
        return torch.ones((*inputs.shape[:2], 3), device=inputs.device) * 0.5


class CaptureGNN(nn.Module):
    extra_node_dim = 1

    def forward(self, graph, provenance):
        return {"object21": graph.xyz_world.new_zeros((1, 21)), "provenance": provenance}


def sample():
    return dict(
        inputs=torch.zeros(3, 2, 13),
        context_valid=torch.tensor([[True, True], [True, False], [False, False]]),
        seen=torch.tensor([True, True, False]),
        observed=torch.tensor([True, False, False]),
        detection_xyz=torch.tensor([[1., 2., 3.], [90., 90., 90.], [99., 99., 99.]]),
        last_observed_xyz=torch.tensor([[7., 7., 7.], [2., 3., 4.], [0., 0., 0.]]),
        object_types=torch.tensor([0, 2, -1]),
        tcp_xyz=torch.zeros(3),
    )


class FixedIdGNNTests(unittest.TestCase):
    def test_actual_detection_and_prediction_are_kept_separate(self):
        result = FixedIdCornerGNN(ConstantTemporal(), CaptureGNN())(**sample())
        torch.testing.assert_close(result["graph"].xyz_world[0],
                                   torch.tensor([[1., 2., 3.], [2.5, 3.5, 4.5]]))
        torch.testing.assert_close(result["gnn_output"]["provenance"],
                                   torch.tensor([[[1.], [0.]]]))
        self.assertEqual(result["selected_indices"].tolist(), [0, 1])
        self.assertEqual(result["graph"].edge_index.shape, (2, 2))

    def test_missing_without_history_is_rejected(self):
        values = sample()
        values["context_valid"][1] = False
        with self.assertRaisesRegex(ValueError, "nonempty causal history"):
            FixedIdCornerGNN(ConstantTemporal(), CaptureGNN())(**values)

    def test_real_gnn_outputs_21_and_backpropagates_through_missing_position(self):
        temporal = FixedIdCornerPredictor(architecture="transformer", width=16, layers=1, heads=4)
        gnn = VertexEdgeGatedMPNN(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1)
        model = FixedIdCornerGNN(temporal, gnn)
        result = model(**sample())
        self.assertEqual(result["object21"].shape, (1, 21))
        self.assertTrue(torch.isfinite(result["object21"]).all())
        result["object21"].square().sum().backward()
        self.assertIsNotNone(temporal.delta_head.weight.grad)
        self.assertGreater(temporal.delta_head.weight.grad.abs().sum().item(), 0)

    def test_no_seen_corners_has_empty_padded_graph(self):
        values = sample()
        values["seen"].fill_(False)
        values["observed"].fill_(False)
        result = FixedIdCornerGNN(ConstantTemporal(), CaptureGNN())(**values)
        self.assertFalse(result["graph"].node_valid.any())
        self.assertEqual(result["graph"].edge_index.shape, (2, 0))


if __name__ == "__main__":
    unittest.main()
