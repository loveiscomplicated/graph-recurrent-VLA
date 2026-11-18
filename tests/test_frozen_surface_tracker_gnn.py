"""Verify causal node eligibility and frozen-GNN input isolation."""

import unittest

import torch

from graph.frozen_tracker_graph import build_tracker_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.frozen_surface_tracker_gnn import TrackerBatch, TrackerObject21Task, state_digest


class FrozenTrackerGraphTest(unittest.TestCase):
    def test_never_seen_coordinates_do_not_affect_prediction_or_gradient(self):
        torch.manual_seed(5)
        model = VertexEdgeGatedMPNN(20, hidden=16, layers=1, extra_node_dim=2).eval()
        xyz = torch.randn(3, 20, 3, requires_grad=True)
        valid = torch.zeros(3, 20, dtype=torch.bool)
        valid[0, :4] = True
        valid[1, 8:12] = True
        observed = valid.clone()
        observed[0, 2:4] = False
        tcp = torch.zeros(3, 3)
        graph, flags = build_tracker_graph(xyz, observed, valid, tcp)
        graph.validate(20)
        before = state_digest(model)
        prediction = model(graph, flags)["object21"]
        changed = xyz.detach().clone()
        changed[~valid] = 1e8
        other_graph, other_flags = build_tracker_graph(changed, observed, valid, tcp)
        torch.testing.assert_close(prediction, model(other_graph, other_flags)["object21"], atol=0, rtol=0)
        target = prediction.detach().clone()
        target[:, 0] += 0.01
        result = TrackerObject21Task().compute(model, TrackerBatch(graph, flags, target))
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        self.assertEqual(float(xyz.grad[~valid].abs().sum()), 0)
        self.assertEqual(state_digest(model), before)
        self.assertEqual(flags[0, 2].tolist(), [0, 1])
        self.assertFalse(graph.node_valid[2].any())


if __name__ == "__main__":
    unittest.main()
