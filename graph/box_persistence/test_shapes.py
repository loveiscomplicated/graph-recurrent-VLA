"""Untrained tensor contracts only; no optimizer, fit, ranking or checkpoint."""
import unittest
import numpy as np,torch
from graph.integration_baseline.graph import pack_graphs
from graph.persistent_geometry.encoder import GeometryRegressor
from graph.persistent_geometry.graph import build_graph
from graph.persistent_geometry.tests.test_contracts import camera
from .tests import fixture


class TensorContracts(unittest.TestCase):
    def graph(self,records):
        return build_graph(records,np.zeros((3,3)),camera().tcp_pose,0,include_observation_status=False)

    def check_encoder(self,variant):
        graph=pack_graphs([self.graph([]),self.graph(fixture())])
        model=GeometryRegressor(variant).eval()
        with torch.inference_mode():
            output=model(graph);readout,presence=model.encoder(graph)
        self.assertEqual(tuple(output.shape),(2,36))
        self.assertEqual(tuple(readout.shape),(2,320))
        self.assertEqual(tuple(presence.shape),(2,2))
        self.assertTrue(torch.isfinite(output).all());self.assertTrue(torch.isfinite(readout).all())

    def test_E0_geometry_decoder_finite_shape(self):self.check_encoder('E0')
    def test_E1_geometry_decoder_finite_shape(self):self.check_encoder('E1')

if __name__=='__main__':unittest.main()
