"""Current-frame robot evidence, deferred callers and native image contracts."""
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from graph.plan_a_geometry import tests as fixtures
from graph.persistent_geometry.self_filter import LinkSurface, classify_robot
from graph.persistent_geometry.tests.test_contracts import cube
from .robot_frame import RobotFilterFrame


def sensor_fixture(device):
    f = fixtures.Contracts()
    f.setUp()
    observation = dict(agent={name: torch.zeros((1, 9), device=device) for name in ('qpos', 'qvel')},
                       sensor_data={}, sensor_param={})
    for c in f.cameras:
        depth = torch.as_tensor(c.depth_raw_mm.copy(), device=device)[None, ..., None]
        depth[0, 0, 0, 0] = -32768
        observation['sensor_data'][c.camera] = dict(rgb=torch.as_tensor(c.rgb, device=device)[None], depth=depth)
        observation['sensor_param'][c.camera] = {
            key: torch.as_tensor(getattr(c, field), device=device)[None]
            for key, field in (('intrinsic_cv', 'intrinsic'), ('extrinsic_cv', 'extrinsic_cv'), ('cam2world_gl', 'cam2world_gl'))}
    fields = dict(qpos=np.zeros(9, np.float32), qvel=np.zeros(9, np.float32), tcp_pose=f.cameras[0].tcp_pose)
    return f, observation, fields


class RobotFrameContracts(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.Contracts()
        self.f.setUp()
        self.surfaces = [LinkSurface('camera_link', cube([0., -.1, .099], [.02, .1, .105]), np.eye(4))]

    def test_full_diagnostics_preserve_correction_and_skip_classifier(self):
        f = self.f
        expected = f.system.prepare(f.cameras, f.data, f.masks, self.surfaces)
        diagnostics = {c.camera: classify_robot(c, self.surfaces) for c in f.cameras}
        gpu = {name: {k: torch.as_tensor(v) for k, v in value.items() if isinstance(v, np.ndarray)}
               for name, value in diagnostics.items()}
        frame = RobotFilterFrame(f.system, f.cameras, gpu)
        from graph.task10i_stage4a.perframe_cache import install
        memo = install(SimpleNamespace(system=f.system))
        try:
            with patch('graph.plan_a_geometry.core.classify_robot', side_effect=AssertionError('CPU classification')):
                actual = frame.prepare(f.cameras, f.data, f.masks, self.surfaces)
                again = frame.prepare(f.cameras, f.data, f.masks, self.surfaces)
            self.assertEqual(memo.misses, 0)
            self.assertEqual(actual['gate'], expected['gate'])
            for name in f.masks:
                np.testing.assert_array_equal(actual['clean'][name], expected['clean'][name])
                np.testing.assert_array_equal(again['clean'][name], expected['clean'][name])
                self.assertIsNotNone(actual['filters'][name])
                raw = torch.as_tensor(f.masks[name])
                cleaned = frame.clean({**{n: torch.as_tensor(m) for n, m in f.masks.items()}})[name]
                np.testing.assert_array_equal(cleaned.numpy(), expected['clean'][name])
                self.assertNotEqual(raw.data_ptr(), cleaned.data_ptr())
            with self.assertRaisesRegex(ValueError, 'foreign'):
                frame.prepare(f.cameras, {**f.data, 'timestep': 1}, f.masks, self.surfaces)
        finally:
            frame.close()
        with self.assertRaisesRegex(RuntimeError, 'completed frame'):
            frame.clean({n: torch.as_tensor(m) for n, m in f.masks.items()})

    def test_depth_conversion_failure_discards_worker_job(self):
        from .tensor_ingress import TensorRobotPreparation
        class Filter:
            def prepare_surfaces(self, *_):
                return None
            def depth_tensor(self, raw):
                raise RuntimeError('injected conversion error')
        with TensorRobotPreparation(Filter()) as preparation, \
                patch('graph.plan_a_bc.tensor_ingress.torch.as_tensor', side_effect=lambda raw, **_: raw):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, 'conversion error'):
                    preparation.submit(self.f.cameras, self.surfaces)
                self.assertIsNone(preparation._pending)

    def test_correction_state_machine_uses_supplied_robot_frame(self):
        from graph.task10i_stage4a2.test_contracts import PracticalIntegrationContracts
        fixture = PracticalIntegrationContracts()
        control, _, _ = fixture.policy(uncertain_output_policy='preserve_current')
        candidate, _, _ = fixture.policy(uncertain_output_policy='preserve_current')
        supplied = candidate.geometry.system.prepare
        calls = []
        def prepare(*args):
            calls.append(args[1]['timestep'])
            return supplied(*args)
        candidate.geometry.system.prepare = lambda *_: (_ for _ in ()).throw(AssertionError('CPU fallback'))
        try:
            for t in range(8):
                raw = fixture.masks(False)
                cameras = fixture.cameras(t)
                data = dict(episode='e', timestep=t)
                found = {('base_camera', 1): True}
                expected = control.process(cameras, data, [], raw, found)
                actual = candidate.process(cameras, data, [], raw, found,
                                           robot_frame=SimpleNamespace(prepare=prepare))
                for name in raw:
                    np.testing.assert_array_equal(actual[name], expected[name])
            self.assertEqual(candidate.events, control.events)
            self.assertTrue(calls)
        finally:
            candidate.close()
            control.close()

    def test_frozen_entrypoint_defers_vision_and_passes_robot_frame(self):
        from graph.task10i_stage4a2.frozen_inference import infer_frame
        f = self.f
        for correction in (False, True):
            events = []
            test = self
            class Adapter:
                found = {}
                def process(self, rgb, episode, t, *, mask_device):
                    test.assertEqual(events[-1], 'build')
                    test.assertEqual(mask_device, 'cuda')
                    return np.zeros((2, 24, 24), bool), dict(new_object_ids=[])
            marker = object()
            class Geometry:
                robot_filter_backend = 'cuda'
                last_robot_schedule = 'before_vision'
                def build(self, cameras, data, surfaces, producer, *, mask_transform=None):
                    events.append('build')
                    raw = producer()
                    self.last_raw_masks = raw if mask_transform is None else mask_transform(raw, marker)
                    return object(), [], {}
            class Policy:
                def process(self, cameras, data, surfaces, raw, found, *, robot_frame):
                    test.assertIs(robot_frame, marker)
                    events.append('correction')
                    result = {n: a.copy() for n, a in raw.items()}
                    result['hand_camera'][0, 0, 0] = True
                    return result
            adapters = {c.camera: Adapter() for c in f.cameras}
            _, _, _, state = infer_frame(Geometry(), adapters, f.cameras, f.data, [], 'ep', 0,
                                         SimpleNamespace(), 'fixture', policy=Policy() if correction else None)
            self.assertEqual(events, ['build', 'correction'] if correction else ['build'])
            self.assertEqual(bool(state['masks']['hand_camera'][0, 0, 0]), correction)
            self.assertFalse(state['baseline_masks']['hand_camera'].any())
            self.assertEqual(state['shared_robot_classification'], correction)

    def test_reach_live_entrypoint_defers_cuda_mask_producer(self):
        from graph.geometry_reach10d import evaluate
        f = self.f
        events = []
        test = self
        class Adapter:
            def process(self, rgb, identity, t, *, mask_device):
                test.assertEqual(events[-1], 'build')
                test.assertEqual(mask_device, 'cuda')
                return np.zeros((2, 24, 24), bool), {}
        class Geometry:
            def build(self, cameras, data, surfaces, producer):
                events.append('build')
                self.last_raw_masks = producer()
                return 'graph', [], {}
        with patch.object(evaluate, 'sensors', return_value=(f.cameras, f.data, self.surfaces)):
            output = evaluate.live_graph(None, None, 'ep', 0, SimpleNamespace(schema={}), Geometry(),
                                         {c.camera: Adapter() for c in f.cameras})
        self.assertEqual(output[0], 'graph')

    def test_host_sensor_snapshot_remains_available_without_cuda(self):
        from . import rollout
        f, observation, fields = sensor_fixture('cpu')
        with patch.object(rollout, 'robot_observation', return_value=(fields, np.zeros((3, 3)))), \
                patch.object(rollout, 'state_schema', return_value={}), \
                patch.object(rollout, 'flatten_state', return_value=np.zeros(76)), \
                patch.object(rollout, 'robot_surfaces', return_value=self.surfaces), \
                patch('torch.cuda.is_available', return_value=False):
            cameras, _, _ = rollout.sensors(None, observation, 'ep', 0, {})
        self.assertIsInstance(cameras[0].rgb, np.ndarray)
        self.assertEqual(cameras[0].depth_raw_mm[0, 0], -32768)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA host required')
class CudaRobotIntegration(unittest.TestCase):
    def test_native_and_host_sensor_images_reach_cuda_without_native_readback(self):
        from . import rollout
        export = rollout.numpy
        def small_only(value):
            if torch.is_tensor(value) and value.is_cuda and value.numel() > 32:
                raise AssertionError('native image was downloaded')
            return export(value)
        surfaces = [LinkSurface('camera_link', cube([0, 0, .1], [.02, .02, .11]), np.eye(4))]
        for device in ('cpu', 'cuda'):
            f, observation, fields = sensor_fixture(device)
            with patch.object(rollout, 'robot_observation', return_value=(fields, np.zeros((3, 3)))), \
                    patch.object(rollout, 'state_schema', return_value={}), \
                    patch.object(rollout, 'flatten_state', return_value=np.zeros(76)), \
                    patch.object(rollout, 'robot_surfaces', return_value=surfaces), \
                    patch.object(rollout, 'numpy', side_effect=small_only):
                cameras, _, _ = rollout.sensors(None, observation, 'ep', 0, {})
            for c in cameras:
                self.assertTrue(c.rgb.is_cuda and c.depth_raw_mm.is_cuda)
                target = observation['sensor_data'][c.camera]['depth'][0, ..., 0].cuda()
                torch.testing.assert_close(c.depth_raw_mm, target, rtol=0, atol=0)
                self.assertNotEqual(c.depth_raw_mm.data_ptr(), target.data_ptr())

    def test_shared_full_classification_handles_new_proposal_pixels(self):
        from .geometry import CurrentGeometry
        from graph.persistent_geometry.tensor_camera import tensor_camera
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
        f = fixtures.Contracts()
        f.setUp()
        surfaces = [LinkSurface('camera_link', cube([0., -.1, .099], [.02, .1, .105]), np.eye(4))]
        raw = {name: torch.as_tensor(a, device='cuda') for name, a in f.masks.items()}
        proposed = {name: a.copy() for name, a in f.masks.items()}
        proposed['hand_camera'][0, 5:8, 12:15] = True
        expected = {c.camera: classify_robot(c, surfaces) for c in f.cameras}
        with patch('graph.current.RobotFK'):
            geometry = CurrentGeometry('unused', {})
        calls = []
        original = CudaRobotFilter.filter_cameras
        depth_calls = []
        original_depth = CudaRobotFilter.depth_tensor
        def converted(self, raw):
            depth_calls.append(raw.dtype)
            return original_depth(self, raw)
        def filtered(self, *args, **kwargs):
            calls.append(kwargs.get('diagnostics', False))
            return original(self, *args, **kwargs)
        geometry._build_prepared = lambda cameras, data, shapes, masks, clean_masks=None, **_: clean_masks
        used = []
        def correct(values, frame):
            used.append(frame)
            # Use CPU camera snapshots only for the unchanged correction consumer.
            prepared = frame.prepare(f.cameras, f.data, proposed, surfaces)
            for name in proposed:
                for key in ('confirmed_robot', 'uncertain', 'candidate', 'link_owner', 'inside_known_link_volume'):
                    np.testing.assert_array_equal(prepared['filters'][name][key], expected[name][key])
            return proposed
        try:
            with patch.object(CudaRobotFilter, 'filter_cameras', filtered), \
                    patch('graph.plan_a_geometry.core.classify_robot', side_effect=AssertionError('CPU classifier')), \
                    patch.object(CudaRobotFilter, 'depth_tensor', converted):
                actual = geometry.build(tuple(tensor_camera(c) for c in f.cameras), f.data, surfaces,
                                        lambda: raw, mask_transform=correct)
            self.assertEqual(calls, [True])
            self.assertEqual(depth_calls, [torch.int16, torch.int16])
            from graph.persistent_geometry.fusion import filtered_masks
            for name in proposed:
                target, _ = filtered_masks(proposed[name], expected[name])
                np.testing.assert_array_equal(actual[name].cpu().numpy(), target)
            with self.assertRaisesRegex(RuntimeError, 'completed'):
                used[0].clean(raw)
        finally:
            geometry.close()

    def test_legacy_entrypoints_accept_native_rgb_and_keep_cuda_masks(self):
        from graph.integration_baseline.tensor_online_vision import TensorOnlineVisionAdapter
        class Tracker:
            predictor = SimpleNamespace(image_size=8)
            def start_frames(self, images, shape, boxes, frame_idx=0):
                return dict(shape=shape, ids=tuple(boxes))
            def track(self, state, start_frame=0):
                yield start_frame, {oid: torch.full(state['shape'], oid == 1, device='cuda', dtype=torch.bool)
                                    for oid in state['ids']}
            def reset(self, state):
                pass
        initial = {name: dict(box=[0., 0., 3., 3.], frame=0, score=1.) for name in ('peg', 'box')}
        modules = ('graph.graph_recurrent10g.eval_runner', 'graph.geometry_reach10d.evaluate',
                   'graph.handoff10h.runner', 'graph.predstate10i.rollout')
        for name in modules:
            cls = import_module(name).OnlineVisionAdapter
            self.assertIs(cls, TensorOnlineVisionAdapter)
            adapter = cls(None, Tracker())
            adapter.reset('ep')
            try:
                with patch('graph.integration_baseline.tensor_online_vision.detect_initial', return_value=initial):
                    for t in range(2):
                        masks, _ = adapter.process(torch.zeros((4, 4, 3), device='cuda', dtype=torch.uint8),
                                                   'ep', t, mask_device='cuda')
                        self.assertTrue(masks.is_cuda)
                        self.assertEqual(masks.dtype, torch.bool)
                        self.assertTrue(masks[0].all())
                        self.assertFalse(masks[1].any())
            finally:
                adapter.close()


if __name__ == '__main__':
    unittest.main()
