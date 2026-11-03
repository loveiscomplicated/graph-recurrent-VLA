"""The unchanged CPU gate sees original raw masks and prefiltered clean masks."""
import unittest
from unittest.mock import patch
import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on this host')
class BoundaryContracts(unittest.TestCase):
    def test_gate_keeps_raw_mask_counts_filter_runs_once_and_scope_restores(self):
        from .gpu_filter import CudaRobotFilter
        from .graph_boundary import GpuMaskBoundary
        from graph.plan_a_geometry.tests import Contracts
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        import graph.plan_a_geometry.core as core
        f=Contracts();f.setUp()
        ss=[LinkSurface('robot',cube([-.2,-.2,.1],[.2,.2,.12]),np.eye(4))]
        expected=f.system.prepare(f.cameras,f.data,f.masks,ss)
        raw={k:torch.as_tensor(v,device='cuda') for k,v in f.masks.items()}
        old=[core.PlanA.prepare,core.classify_robot,core.filtered_masks];calls=[]
        def once(c,s,m):calls.append(c.camera);return CudaRobotFilter()(c,s,m)
        with GpuMaskBoundary(once) as boundary:
            actual=f.system.prepare(f.cameras,f.data,raw,ss)
            self.assertTrue(all(m.is_cuda for m in boundary.clean_cuda.values()))
            self.assertEqual(calls,[c.camera for c in f.cameras])
        self.assertEqual(old,[core.PlanA.prepare,core.classify_robot,core.filtered_masks])
        self.assertEqual(actual['gate'],expected['gate'])
        self.assertEqual(actual['gate']['raw_pixels'],36)
        for stage in ('clean','conditions'):
            if stage=='clean':
                for name in expected[stage]:np.testing.assert_array_equal(actual[stage][name],expected[stage][name])
            else:
                for condition in expected[stage]:
                    for name in expected[stage][condition]:np.testing.assert_array_equal(actual[stage][condition][name],expected[stage][condition][name])
        with self.assertRaisesRegex(RuntimeError,'scope'):
            with GpuMaskBoundary(once):raise RuntimeError('scope')
        self.assertEqual(old,[core.PlanA.prepare,core.classify_robot,core.filtered_masks])


if __name__=='__main__':unittest.main()
