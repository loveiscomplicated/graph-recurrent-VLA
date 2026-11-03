"""Focused equivalence checks for the shared CUDA wrist gate."""
from dataclasses import replace
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on current host')
class CudaGateContracts(unittest.TestCase):
    def test_radius_components_boundaries_ties_duplicates_and_random_clouds(self):
        from graph.plan_a_geometry.core import components as reference
        from graph.persistent_geometry.tensor_gate import components
        radius = .012
        clouds = [
            np.zeros((8, 3)),
            np.array([[0., 0., 0.], [radius, 0., 0.], [.1, 0., 0.], [.1 + radius, 0., 0.]]),
            np.array([[0., 0., 0.], [np.nextafter(radius, 0), 0., 0.],
                      [np.nextafter(radius, np.inf), 0., 0.]]),
        ]
        for seed in range(5):
            rng = np.random.default_rng(seed)
            clouds.append(rng.normal(size=(80, 3)) * .02)
        for cloud in clouds:
            expected = reference(cloud, radius)
            actual, _ = components(torch.as_tensor(cloud, device='cuda'), radius)
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)

    def test_gate_normal_conflict_overlap_invalid_depth_and_sparse_support(self):
        from graph.plan_a_geometry.tests import Contracts
        from graph.plan_a_geometry.core import wrist_gate as reference
        from graph.persistent_geometry.tensor_gate import wrist_gate_tensor
        from graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark import compare_trace
        for mode in ('normal', 'conflict', 'overlap', 'invalid_depth', 'sparse'):
            fixture = Contracts()
            fixture.setUp()
            cameras = fixture.cameras
            raw = {name: value.copy() for name, value in fixture.masks.items()}
            if mode == 'conflict':
                raw['base_camera'][1] = raw['hand_camera'][0].copy()
            if mode == 'overlap':
                raw['hand_camera'][1] = raw['hand_camera'][0].copy()
            if mode == 'invalid_depth':
                cameras = [replace(c, depth_raw_mm=np.full_like(c.depth_raw_mm, -32768)) for c in cameras]
            if mode == 'sparse':
                raw['hand_camera'][0] = False
                raw['hand_camera'][0, 10, 10] = True
            clean = {}
            for name, value in raw.items():
                clean[name] = value.copy()
                clean[name][:, value[0] & value[1]] = False
            expected, expected_trace = reference(cameras[1], raw['hand_camera'], clean['hand_camera'],
                                                cameras, clean, fixture.cfg['gate'])
            gpu_raw = {name: torch.as_tensor(value, device='cuda') for name, value in raw.items()}
            gpu_clean = {name: torch.as_tensor(value, device='cuda') for name, value in clean.items()}
            actual, trace = wrist_gate_tensor(cameras, gpu_raw, gpu_clean, fixture.cfg['gate'])
            np.testing.assert_array_equal(actual.cpu().numpy(), expected, err_msg=mode)
            compare_trace(trace, expected_trace)


if __name__ == '__main__':
    unittest.main()
