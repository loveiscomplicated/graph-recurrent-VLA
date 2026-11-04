"""Fused FP64 geometry, integer convergence and canonical component parity."""
from dataclasses import replace
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FusedGateContracts(unittest.TestCase):
    def test_world_points_bitwise_match_unfused_fp64_with_strided_calibration(self):
        from graph.persistent_geometry.tensor_gate import masked_world_points
        from graph.persistent_geometry.tensor_fusion import TensorCamera
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(811)
        for seed in range(4):
            depth = torch.as_tensor(rng.uniform(.001, 5., (35, 37)))
            k = torch.tensor([[173.123456789, 0., 18.5], [0., 311.678912345, 17.5], [0., 0., 1.]], dtype=torch.float64)
            k = k.to(torch.float64).T.contiguous().T
            m = np.eye(4); m[:3,:3] = Rotation.random(random_state=rng).as_matrix()
            m[:3,3] = rng.uniform(-.5, .5, 3)
            m = torch.as_tensor(m).T.contiguous().T
            pixels = torch.arange(depth.numel(), dtype=torch.int64)[::seed+1]
            view = TensorCamera(depth, torch.zeros((2,*depth.shape), dtype=torch.bool), k, m, m)
            expected = masked_world_points(None, view, pixels)
            gpu = replace(view, depth=depth.cuda(), mask=view.mask.cuda(), intrinsic=k.cuda(),
                          camera_from_frame=m.cuda(), frame_from_camera=m.cuda())
            actual = masked_world_points(None, gpu, pixels.cuda())
            torch.testing.assert_close(actual.cpu().view(torch.int64), expected.view(torch.int64), rtol=0, atol=0)

    def test_fused_labels_match_unfused_reference_with_duplicate_edges_and_long_chains(self):
        from graph.persistent_geometry.tensor_gate import _component_labels
        rng = np.random.default_rng(17)
        for n in (2, 3, 5, 17, 257, 1031):
            for kind in ('empty', 'chain', 'shuffled', 'disconnected'):
                order = rng.permutation(n) if kind == 'shuffled' else np.arange(n)
                edges = np.stack((order[:-1], order[1:]), 1)
                if kind == 'empty': edges = edges[:0]
                if kind == 'disconnected': edges = edges[np.arange(n-1)%7 != 0]
                edges = np.concatenate((edges, edges[::-1]), 0)
                a, b = torch.as_tensor(edges.T)
                expected = _component_labels(a, b, n)
                actual = _component_labels(a.cuda(), b.cuda(), n)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_sorted_cells_preserve_scipy_labels_and_ties_in_original_point_order(self):
        from graph.persistent_geometry.tensor_gate import components
        from graph.plan_a_geometry.core import components as reference
        radius = .012
        rng = np.random.default_rng(221)
        clouds = [rng.normal(size=(1100, 3))*.03,
                  np.repeat([[.1,0.,0.], [0.,0.,0.]], 13, axis=0),
                  np.stack((np.arange(400)*radius*.8, np.zeros(400), np.zeros(400)),1)]
        for cloud in clouds:
            cloud = cloud[rng.permutation(len(cloud))]
            expected = reference(cloud, radius)
            actual, _ = components(torch.as_tensor(cloud, device='cuda'), radius)
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)


if __name__ == '__main__':
    unittest.main()
