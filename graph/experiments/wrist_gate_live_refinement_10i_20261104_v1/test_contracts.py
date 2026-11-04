"""World gate/view sharing, batched convergence and live entrypoint parity."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.plan_a_geometry import tests as fixtures
from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
from graph.persistent_geometry.tensor_gate import _component_labels, prepare_tensor_gate, wrist_gate_tensor


class PreparationContracts(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.Contracts(); self.f.setUp()

    def test_world_snapshot_shares_depth_mask_intrinsic_and_matches_frame_helpers(self):
        from visual.runtime.coordinate_frames import camera_from_frame, camera_to_frame
        from types import SimpleNamespace
        pose = np.array([.12, -.04, .03, np.cos(.31), 0, 0, np.sin(.31)])
        cameras = [replace(c, tcp_pose=pose) for c in self.f.cameras]
        for frame in ('world', 'ee'):
            views = prepare_tensor_views(cameras, self.f.masks, coordinate_frame=frame, device='cpu')
            world = views.world_views()
            for c in cameras:
                t, w = views.by_name[c.camera], world.by_name[c.camera]
                for name in ('depth', 'mask', 'intrinsic'):
                    self.assertIs(getattr(t, name), getattr(w, name))
                for f, value in ((frame, t), ('world', w)):
                    np.testing.assert_array_equal(value.camera_from_frame.numpy(), camera_from_frame(c, f))
                    np.testing.assert_array_equal(value.frame_from_camera.numpy(), camera_to_frame(
                        SimpleNamespace(cam2world=c.world_from_camera, tcp_pose=c.tcp_pose), f))

    def test_numpy_adapter_uploads_each_clean_mask_once_and_honors_wrist_override(self):
        from graph.persistent_geometry.wrist_gate_dispatch import wrist_gate_cuda_numpy
        as_tensor = torch.as_tensor
        override = self.f.masks['hand_camera'].copy(); override[:] = False
        uploads = []
        def upload(value, **kwargs):
            uploads.append(value)
            return as_tensor(value)
        with patch('torch.as_tensor', side_effect=upload), patch(
                'graph.persistent_geometry.tensor_gate.wrist_gate_tensor',
                return_value=(torch.zeros((24, 24), dtype=torch.bool), {})) as gate:
            wrist_gate_cuda_numpy(self.f.cameras[1], self.f.masks['hand_camera'], override,
                                  self.f.cameras, self.f.masks, self.f.cfg['gate'])
        self.assertEqual(len(uploads), 3)  # two clean masks, one raw wrist mask
        self.assertEqual(sum(v is override for v in uploads), 1)
        self.assertFalse(bool(gate.call_args.args[2]['hand_camera'].any()))

    def test_batched_labels_match_per_iteration_reference_including_short_final_batch(self):
        for n in (2, 3, 5, 17, 65, 257):
            rng = np.random.default_rng(n)
            for mode in ('empty', 'chain', 'permuted_chain', 'disconnected'):
                order = np.arange(n) if mode != 'permuted_chain' else rng.permutation(n)
                edges = np.stack((order[:-1], order[1:]), 1)
                if mode == 'empty': edges = edges[:0]
                if mode == 'disconnected': edges = edges[np.arange(n-1) % 5 != 0]
                a, b = torch.as_tensor(edges.T)
                expected = torch.arange(n)
                for iterations in range(1, n+1):
                    old = expected.clone()
                    roots = torch.minimum(expected[a], expected[b])
                    expected.scatter_reduce_(0, a, roots, reduce='amin')
                    expected.scatter_reduce_(0, b, roots, reduce='amin')
                    expected = expected[expected]
                    if torch.equal(old, expected): break
                with patch('torch.equal', wraps=torch.equal) as checks:
                    actual = _component_labels(a, b, n)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertLessEqual(checks.call_count, (iterations+3)//4)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class LiveGateContracts(unittest.TestCase):
    def test_live_preparation_validates_once_and_gate_reuses_world_for_both_graph_frames(self):
        f = fixtures.Contracts(); f.setUp()
        from graph.persistent_geometry import tensor_fusion
        raw = {k: torch.as_tensor(v, device='cuda') for k, v in f.masks.items()}
        reference = f.system.prepare(f.cameras, f.data, f.masks, [])
        clean = {k: torch.as_tensor(v, device='cuda') for k, v in reference['clean'].items()}
        for frame in ('ee', 'world'):
            with patch.object(tensor_fusion, 'synchronized', wraps=tensor_fusion.synchronized) as validate:
                prepared, views = prepare_tensor_gate(f.cameras, f.data, raw, clean, f.cfg,
                                                     coordinate_frame=frame)
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(views.coordinate_frame, frame)
            self.assertEqual(prepared['gate'], reference['gate'])
            np.testing.assert_array_equal(prepared['conditions']['C3']['hand_camera'][0].cpu().numpy(),
                                          reference['conditions']['C3']['hand_camera'][0])
            for name, view in views.world_views().by_name.items():
                self.assertIs(view.mask, prepared['conditions']['C3'][name])

    def test_stale_view_identity_and_invalid_clean_masks_are_rejected(self):
        f = fixtures.Contracts(); f.setUp()
        raw = {k: torch.as_tensor(v, device='cuda') for k, v in f.masks.items()}
        views = prepare_tensor_views(f.cameras, raw)
        stale = replace(views, identity=('other', *views.identity[1:]))
        with self.assertRaisesRegex(ValueError, 'stale gate view'):
            wrist_gate_tensor(f.cameras, raw, raw, f.cfg['gate'], views=stale)
        mixed = [f.cameras[0], replace(f.cameras[1], timestep=f.cameras[1].timestep+1)]
        with self.assertRaisesRegex(ValueError, 'stale gate view'):
            wrist_gate_tensor(mixed, raw, raw, f.cfg['gate'], views=views)
        clean = {k: v.clone() for k, v in raw.items()}
        clean['hand_camera'][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, 'subset/overlap'):
            prepare_tensor_gate(f.cameras, f.data, raw, clean, f.cfg, coordinate_frame='ee')


if __name__ == '__main__':
    unittest.main()
