"""Geometry interventions must preserve graph consistency and supervision."""

import unittest

import torch

from evaluation.gnn.vertex_geometry_diagnostics import perturb_graph


class VertexGeometryDiagnosticsTests(unittest.TestCase):
    def test_single_graph_drop_and_noise(self):
        from types import SimpleNamespace
        from graph.build_graph import GraphBuilder
        points = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [.2, .2, 0.]])
        labels = torch.tensor([0,0,0,0,1])
        graph = GraphBuilder()([SimpleNamespace(xyz=points,object_id=labels,metadata=[{}]*5)],torch.zeros(1,3))
        original = graph.xyz_world.clone()
        for condition in ('original','drop25','drop50','noise0.5mm','noise1mm'):
            changed = perturb_graph(graph, condition, 13)
            changed.validate()
            torch.testing.assert_close(changed.tcp_xyz_world, graph.tcp_xyz_world)
            torch.testing.assert_close(changed.xyz_world[0,4], graph.xyz_world[0,4])
            repeated = perturb_graph(graph, condition, 13)
            torch.testing.assert_close(changed.xyz_world, repeated.xyz_world, atol=0,rtol=0)
        self.assertEqual(int(perturb_graph(graph,'drop25',13).node_valid.sum()),4)
        self.assertEqual(int(perturb_graph(graph,'drop50',13).node_valid.sum()),3)
        torch.testing.assert_close(graph.xyz_world, original, atol=0,rtol=0)


if __name__ == '__main__':
    unittest.main()
