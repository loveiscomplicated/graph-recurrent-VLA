"""Torch view checks preserve source identities and strict original predicates."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from graph.persistent_geometry.contracts import SurfaceObservation, transform
from graph.persistent_geometry.tests.test_contracts import camera
from graph.persistent_geometry.fusion import consistent_views, observations_from_packet
from graph.persistent_geometry.coordinates import observation_in_frame
from graph.persistent_geometry.tensor_fusion import (prepare_tensor_views, tensor_observations,
    consistent_views_tensor, numpy_observations, depth_from_mm, observations_from_tensor_packet)
from visual.runtime.test.test_coordinate_frames import cameras_and_masks, known_observations, fixture


def assert_observations(test, actual, expected):
    test.assertEqual(len(actual), len(expected))
    for a, b in zip(actual, expected):
        for name, value in asdict(b).items():
            if isinstance(value, np.ndarray):
                test.assertEqual(getattr(a, name).dtype, value.dtype)
                np.testing.assert_array_equal(getattr(a, name), value, err_msg=name)
            else:
                test.assertEqual(getattr(a, name), value)


def predicate_case():
    a, b = camera(name='base_camera'), camera(name='hand_camera')
    n = 12
    # Exact pixel interior avoids arithmetic ambiguity except the cases whose
    # strict depth/pixel boundary is deliberately being tested.
    z = np.array([1., 1.003, .997, 1.004, np.nextafter(1.004, 1.),
                  np.nextafter(1.004, np.inf), 1., 1., 1., 1., 0., -1.])
    u = np.arange(2, 2+n); v = np.full(n, 10)
    xyz = np.c_[(u+.5-12)*z/40, (v+.5-12)*z/40, z]
    xyz[8, 0] = 100.  # outside image, even if clamped pixel has opposite mask
    normals = np.tile([0., 0., -1.], (n, 1))
    validity = np.ones((n, 3), bool)
    obs = SurfaceObservation(a.episode, a.timestep, a.camera, 0, xyz, normals, validity[:, 0],
        np.c_[normals, np.zeros((n, 2))], validity, np.arange(n), np.zeros(3)).validate()
    masks = {a.camera: np.zeros((2, 24, 24), bool), b.camera: np.zeros((2, 24, 24), bool)}
    masks[b.camera][1] = True
    masks[b.camera][0, v[6], u[6]] = True  # overlapping masks do not contradict
    masks[b.camera][:, v[7], u[7]] = False  # missing mask does not contradict
    raw = b.depth_raw_mm.copy(); raw[v[9], u[9]] = -32768  # invalid depth
    b = replace(b, depth_raw_mm=raw)
    return (a, b), masks, obs


class TorchViewContracts(unittest.TestCase):
    device = 'cpu'

    def test_signed_mm_conversion_matches_all_sensor_values(self):
        raw = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16).reshape(256, 256)
        expected = np.where(raw > 0, raw.astype(np.float64)/1000, np.nan)
        actual = depth_from_mm(torch.as_tensor(raw, device=self.device)).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_strict_depth_missing_overlap_invalid_and_outside_predicates(self):
        cameras, masks, obs = predicate_case()
        for frame in ('world', 'ee'):
            o = observation_in_frame(obs, cameras[0], frame)
            expected, audit = consistent_views([o], cameras, masks)
            views = prepare_tensor_views(cameras, masks, device=self.device, coordinate_frame=frame)
            inputs = tensor_observations([o], device=self.device)
            before = deepcopy(inputs[0].xyz)
            actual, trace = consistent_views_tensor(inputs, views)
            self.assertEqual(actual[0].xyz.device.type, torch.device(self.device).type)
            assert_observations(self, numpy_observations(actual), expected)
            self.assertEqual(int(trace[0]['unknown_conflict'].item()), audit[0]['unknown_conflict'])
            self.assertTrue(torch.equal(before, inputs[0].xyz))
            self.assertEqual(expected[0].pixels.tolist(), [3, 5, 6, 7, 8, 9, 10, 11])

    def test_nonidentity_multiview_both_frames_and_no_depth_property(self):
        cameras, masks = cameras_and_masks()
        for frame in ('world', 'ee'):
            observations = [observation_in_frame(o, c, frame) for c in cameras for o in known_observations(c)]
            expected, _ = consistent_views(observations, cameras, masks)
            with patch.object(type(cameras[0]), 'depth', new=property(lambda _: (_ for _ in ()).throw(AssertionError('CPU full depth forbidden')))):
                views = prepare_tensor_views(cameras, masks, device=self.device, coordinate_frame=frame)
                actual, _ = consistent_views_tensor(tensor_observations(observations, device=self.device), views)
            assert_observations(self, numpy_observations(actual), expected)

    def test_empty_and_bad_identity_frame_reference_or_layout(self):
        cameras, masks, observation = predicate_case()
        empty = observation.subset(np.zeros(len(observation.xyz), bool))
        views = prepare_tensor_views(cameras, masks, device=self.device, coordinate_frame='world')
        actual, _ = consistent_views_tensor(tensor_observations([empty], device=self.device), views)
        assert_observations(self, numpy_observations(actual), [empty])
        base = tensor_observations([observation], device=self.device)[0]
        for bad in (replace(base, episode='foreign'), replace(base, camera='other'),
                    replace(base, coordinate_frame='ee'), replace(base, xyz=base.xyz.float())):
            with self.assertRaises(ValueError): consistent_views_tensor([bad], views)
        ee = observation_in_frame(observation, cameras[0], 'ee')
        ee = tensor_observations([ee], device=self.device)[0]
        with self.assertRaises(ValueError):
            consistent_views_tensor([replace(ee, reference_ee_pose=(.1,0,0,1,0,0,0))],
                prepare_tensor_views(cameras, masks, device=self.device, coordinate_frame='ee'))
        with self.assertRaises(ValueError): prepare_tensor_views(cameras, {}, device=self.device)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA host required')
class CUDAViewContracts(TorchViewContracts):
    device = 'cuda'

    def test_packets_observations_structural_flags_and_no_array_readback(self):
        from visual.runtime.graph_perception import make_packet_cuda, make_packet
        c, masks = fixture('holes_overlap')
        for frame in ('world', 'ee'):
            packet = make_packet_cuda(c, torch.as_tensor(masks, device='cuda'), coordinate_frame=frame)
            views = prepare_tensor_views([c], {c.camera: torch.as_tensor(masks, device='cuda')}, coordinate_frame=frame)
            with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('readback forbidden')), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('NumPy conversion forbidden')):
                obs = observations_from_tensor_packet(packet, c)
                actual, _ = consistent_views_tensor(obs, views)
                for o in [*obs, *actual]:
                    for name in ('xyz','normals','normal_valid','f5','feature_valid','pixels','camera_origin','structural'):
                        self.assertTrue(getattr(o, name).is_cuda, name)
            host, flags = numpy_observations(actual, include_structural=True)
            source, _ = make_packet(c, masks, coordinate_frame=frame)
            expected = observations_from_packet(source, c, coordinate_frame=frame)
            for a, b in zip(host, expected):
                np.testing.assert_array_equal(a.pixels, b.pixels)
                np.testing.assert_allclose(a.xyz, b.xyz, rtol=0, atol=1e-6)
                np.testing.assert_allclose(a.f5, b.f5, rtol=0, atol=2e-5)
            for o, f in zip(host, flags):
                rows = source['object_id'] == o.object_id
                np.testing.assert_array_equal(f, ~source['provenance']['surface_selected'][rows])

    def test_default_graph_uses_tensor_views_and_matches_legacy_cpu_checker(self):
        from graph.plan_a_bc.geometry import CurrentGeometry
        from visual.runtime.test.test_coordinate_frames import robot
        cameras, masks = cameras_and_masks()
        masks['hand_camera'][1] = False
        data = dict(episode=np.array(cameras[0].episode), timestep=np.int64(0),
            tcp_pose=cameras[0].tcp_pose, robot_world_geometry=robot(cameras[0]), robot76=np.zeros(76))
        for frame in ('world', 'ee'):
            with patch('graph.plan_a_bc.geometry.RobotFK'):
                new = CurrentGeometry('unused', {}, coordinate_frame=frame,robot_filter_backend='cpu')
                old = CurrentGeometry('unused', {}, coordinate_frame=frame,robot_filter_backend='cpu')
            old.system.tensor_packet_builder = None
            for geo in (new, old):
                geo._check_mount = lambda *args: None
                geo.system.prepare = lambda *args: dict(conditions={'C3': masks}, gate=dict(accepted=True))
            expected, er, eg = old.build(cameras, data, [], masks)
            with patch('graph.plan_a_geometry.core.consistent_views', side_effect=AssertionError('CPU view check forbidden')), \
                 patch('graph.plan_a_geometry.core.current_records', side_effect=AssertionError('per-point CPU records forbidden')), \
                 patch('graph.plan_a_geometry.core.deduplicate', side_effect=AssertionError('CPU dedup forbidden')), \
                 patch('graph.plan_a_bc.geometry.budget', side_effect=AssertionError('CPU final FPS forbidden')), \
                 patch('graph.persistent_geometry.tensor_fusion.numpy_observations', side_effect=AssertionError('observation readback forbidden')):
                actual, ar, ag = new.build(cameras, data, [], masks)
            self.assertEqual(actual.packet['view_consistency_backend'], 'torch_cuda')
            self.assertEqual(actual.packet['records_dedup_backend'], 'torch_cuda')
            self.assertEqual(actual.packet['final_fps_backend'], 'torch_cuda')
            self.assertEqual([r['source_observation_ids'] for r in ar], [r['source_observation_ids'] for r in er])
            self.assertEqual(ag, eg)
            for key in ('node_type', 'edge_index', 'edge_type'):
                np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
            for key in ('node_input', 'edge_attr'):
                np.testing.assert_allclose(getattr(actual, key), getattr(expected, key), rtol=0, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
