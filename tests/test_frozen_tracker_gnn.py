"""Check missing-slot eligibility and graph-to-GNN input isolation."""

import unittest

import torch

from graph.frozen_tracker_gnn import completed_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss


class FrozenTrackerGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.xyz = torch.randn(2, 20, 3) * 0.02
        self.seen = torch.zeros(2, 20, dtype=torch.bool)
        self.seen[0, [0, 2, 8, 16]] = True
        self.observed = torch.zeros_like(self.seen)
        self.observed[0, [0, 8]] = True
        self.tcp = torch.zeros(2, 3)

    def test_complete_edges_only_between_seen_nodes(self):
        graph, provenance = completed_graph(self.xyz, self.seen, self.observed, self.tcp)
        graph.validate(num_semantic_ids=24)
        self.assertEqual(graph.edge_index.shape, (2, 12))
        self.assertTrue((graph.edge_index < 20).all())
        self.assertTrue(torch.equal(graph.node_valid, self.seen))
        self.assertTrue(torch.equal(provenance[..., 0].bool(), self.observed))
        self.assertTrue((graph.xyz_world[~self.seen] == 0).all())
        self.assertEqual(graph.object_type[0, 16].item(), 1)
        self.assertFalse(graph.semantic_valid.any())

    def test_unseen_coordinates_cannot_affect_output_and_empty_graph_is_finite(self):
        model = VertexEdgeGatedMPNN(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1).eval()
        graph, provenance = completed_graph(self.xyz, self.seen, self.observed, self.tcp)
        changed = self.xyz.clone()
        changed[~self.seen] = 1234
        other, _ = completed_graph(changed, self.seen, self.observed, self.tcp)
        with torch.no_grad():
            prediction = model(graph, provenance)["object21"]
            perturbed = model(other, provenance)["object21"]
        torch.testing.assert_close(prediction, perturbed, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_real_object21_loss_updates_gnn(self):
        model = VertexEdgeGatedMPNN(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1)
        graph, provenance = completed_graph(self.xyz, self.seen, self.observed, self.tcp)
        target = torch.zeros(2, 21)
        target[:, [3, 13]] = 1
        target[:, 17:] = torch.tensor([0.04, 0.01, 0.01, 0.02])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.peg_head[-1].weight.detach().clone()
        loss = object21_loss(model(graph, provenance), target)["total"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.peg_head[-1].weight))


if __name__ == "__main__":
    unittest.main()
