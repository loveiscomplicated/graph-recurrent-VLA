"""World/EE geometry contracts with nonidentity calibration and current TCP."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from visual.runtime.test.test_object_groups import fixture
from visual.runtime.coordinate_frames import FRAME_NAMES, camera_from_frame
from visual.runtime.graph_perception import make_packet
from visual.runtime.vision_packet import pose_matrix
from graph.vision_input import validate_packet
from graph.persistent_geometry.contracts import SurfaceObservation, transform
from graph.persistent_geometry.coordinates import (observation_in_frame, record_xyz,
    record_normal, validate_record_frames, camera_origin)
from graph.persistent_geometry.fusion import observations_from_packet, consistent_views
from graph.persistent_geometry.graph import build_graph, free_space_violation
from graph.persistent_geometry.map import current_records
from graph.plan_a_geometry.core import deduplicate, budget
from graph.plan_a_bc.geometry import CurrentGeometry


def cameras_and_masks(mode='both'):
    a, masks = fixture(mode)
    raw = a.depth_raw_mm.copy()
    # Avoid symmetric distance ties in the complete two-view pipeline fixture.
    valid = raw > 0
    raw[valid] = np.random.default_rng(7).integers(290, 315, valid.sum(), dtype=np.int16)
    a = replace(a, depth_raw_mm=raw)
    world = a.world_from_camera.astype(float)
    world[:3, 3] += world[:3, :3] @ [.012, 0, 0]
    b = replace(a, camera='hand_camera', extrinsic_cv=np.linalg.inv(world)[:3],
        cam2world_gl=world @ np.diag([1., -1., -1., 1.]))
    return (a.validate(), b.validate()), {a.camera: masks.copy(), b.camera: masks.copy()}


def robot(c):
    pose = pose_matrix(c.tcp_pose)
    return np.array([pose[:3, 3], pose[:3, 3]+pose[:3, :3] @ [0, .02, .04],
                     pose[:3, 3]+pose[:3, :3] @ [0, -.02, .04]])


def known_observations(c):
    rng = np.random.default_rng(42)
    observations = []
    for oid in (0, 1):
        camera_xyz = rng.uniform([-.02+oid*.05, -.02, .2998], [.02+oid*.05, .02, .3002], (30, 3))
        world = c.world_from_camera.astype(float)
        xyz = transform(camera_xyz, world)
        normals = np.tile(np.array([0., 0, -1.]) @ world[:3, :3].T, (len(xyz), 1))
        valid = np.ones((len(xyz), 3), bool)
        observations.append(SurfaceObservation(c.episode, c.timestep, c.camera, oid, xyz, normals,
            valid[:, 0], np.c_[normals, np.zeros((len(xyz), 2))], valid,
            np.arange(oid*100, oid*100+len(xyz)), world[:3, 3]).validate())
    return observations


class CoordinateFrames(unittest.TestCase):
    def test_packet_points_normals_and_direct_observations_both_frames(self):
        for mode in ('both', 'peg', 'box', 'empty', 'holes_overlap'):
            c, masks = fixture(mode)
            packets = {}
            for frame in ('world', 'ee'):
                with self.subTest(mode=mode, frame=frame):
                    p, _ = make_packet(c, masks, backend='cpu', coordinate_frame=frame)
                    packets[frame] = p
                    validate_packet(p)
                    self.assertEqual(p['coordinate_frame'], FRAME_NAMES[frame])
                    self.assertEqual(p['feature_fields'][0]['frame'], 'World' if frame == 'world' else 'EE')
                    matrix = c.world_from_camera if frame == 'world' else np.linalg.inv(pose_matrix(c.tcp_pose)) @ c.world_from_camera
                    np.testing.assert_allclose(p['points'], transform(p['camera_points'], matrix), rtol=0, atol=1e-6)
                    self.assertEqual('T_EE_from_camera' in p, frame == 'ee')
                    # Selected packet is already in the configured frame. No
                    # second point transformation is allowed at this boundary.
                    with patch('graph.persistent_geometry.fusion.transform', side_effect=AssertionError('extra point transform')):
                        observations = observations_from_packet(p, c, coordinate_frame=frame)
                    for oid, o in enumerate(observations):
                        take = p['object_id'] == oid
                        np.testing.assert_array_equal(o.xyz, p['points'][take])
                        np.testing.assert_array_equal(o.f5, p['geometric_features'][take])
                        np.testing.assert_allclose(o.camera_origin, matrix[:3, 3], rtol=0, atol=1e-14)
                        self.assertEqual(o.coordinate_frame, frame)
                        records = current_records([o], fuse_views=False, coordinate_frame=frame)
                        for i, r in enumerate(records):
                            np.testing.assert_array_equal(record_xyz(r), o.xyz[i])
                            self.assertNotIn('world_xyz', r)
                            self.assertEqual(r['coordinate_frame'], frame)
            a, b = packets['world'], packets['ee']
            np.testing.assert_array_equal(a['provenance']['source_index'], b['provenance']['source_index'])
            np.testing.assert_allclose(a['points'], transform(b['points'], pose_matrix(c.tcp_pose)), rtol=0, atol=1e-6)
            nv = a['feature_field_valid'][:, 0]
            np.testing.assert_allclose(a['geometric_features'][nv, :3],
                b['geometric_features'][nv, :3] @ pose_matrix(c.tcp_pose)[:3, :3].T, rtol=0, atol=2e-7)
            np.testing.assert_array_equal(a['geometric_features'][:, 3:], b['geometric_features'][:, 3:])

    def test_two_view_projection_and_free_space_use_configured_frame(self):
        cameras, masks = cameras_and_masks()
        a, b = cameras
        # One physical point with supported opposite identity in the other view.
        local = np.array([[.003, -.005, .3], [.002, .002, .28]])
        world = transform(local, a.world_from_camera)
        uv = transform(world[:1], np.linalg.inv(b.world_from_camera))
        pixel = np.floor(uv[0, :2]/uv[0, 2]*[b.intrinsic[0, 0], b.intrinsic[1, 1]]+b.intrinsic[:2, 2]).astype(int)
        for m in masks.values(): m[:] = False
        masks[b.camera][1, pixel[1], pixel[0]] = True
        depth = np.full_like(b.depth_raw_mm, 300)
        cameras = (replace(a, depth_raw_mm=depth), replace(b, depth_raw_mm=depth))
        o = known_observations(a)[0]
        o = replace(o, xyz=world, normals=o.normals[:2], normal_valid=o.normal_valid[:2],
            f5=o.f5[:2], feature_valid=o.feature_valid[:2], pixels=np.array([100, 101]))
        outcomes = []
        for frame in ('world', 'ee'):
            current = observation_in_frame(o, a, frame)
            checked, _ = consistent_views([current], cameras, masks)
            outcomes.append(checked[0].pixels)
            samples = current.xyz
            np.testing.assert_array_equal(free_space_violation(samples, cameras, coordinate_frame=frame), [False, True])
            np.testing.assert_allclose(transform(samples, camera_from_frame(a, frame)), local, rtol=0, atol=2e-8)
        np.testing.assert_array_equal(outcomes[0], [101])
        np.testing.assert_array_equal(outcomes[0], outcomes[1])

    def test_dedup_budget_graph_robot_and_edge_vectors_transform_together(self):
        c, _ = fixture()
        observations = known_observations(c)
        graphs = {}
        kept = {}
        for frame in ('world', 'ee'):
            os = [observation_in_frame(o, c, frame) for o in observations]
            records = current_records(os, fuse_views=False, coordinate_frame=frame)
            baseline_duplicates = deduplicate(records)[1]['duplicates']
            # Include a real duplicate with a distinct source identity.
            duplicate = deepcopy(records[0]);duplicate['persistent_node_id'] += ':copy'
            duplicate['source_observation_ids'] = [(c.episode, c.timestep, c.camera, 500)]
            duplicate['xyz'] = duplicate['xyz']+record_normal(duplicate)*.0008
            records, audit = deduplicate(records+[duplicate])
            self.assertEqual(audit['duplicates'], baseline_duplicates+1)
            records = budget(records, 12)
            kept[frame] = [r['source_observation_ids'] for r in records]
            g = build_graph(records, robot(c), c.tcp_pose, c.timestep, point_cap=None,
                            include_observation_status=False, coordinate_frame=frame)
            graphs[frame] = g
            for i, node_id in enumerate(g.packet['node_ids']):
                r = next(r for r in records if r['persistent_node_id'] == node_id)
                np.testing.assert_array_equal(g.node_input[i+3, :3], np.asarray(r['xyz'], np.float32))
                np.testing.assert_array_equal(g.node_input[i+3, 3:6], np.asarray(r['last_f5'][:3], np.float32))
            expected_robot = robot(c) if frame == 'world' else transform(robot(c), np.linalg.inv(pose_matrix(c.tcp_pose)))
            np.testing.assert_allclose(g.node_input[:3, :3], expected_robot, rtol=0, atol=1e-7)
        self.assertEqual(kept['world'], kept['ee'])
        a, b = graphs['world'], graphs['ee']
        for key in ('node_type', 'edge_type', 'edge_index'):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        matrix = pose_matrix(c.tcp_pose)
        np.testing.assert_allclose(a.node_input[:, :3], transform(b.node_input[:, :3], matrix), rtol=0, atol=1e-7)
        np.testing.assert_allclose(a.edge_attr[:, 0], b.edge_attr[:, 0], rtol=0, atol=1e-7)
        np.testing.assert_allclose(a.edge_attr[:, 1:], b.edge_attr[:, 1:].numpy() @ matrix[:3, :3].T, rtol=0, atol=2e-5)
        from graph.integration_baseline.graph import pack_graphs
        with self.assertRaisesRegex(ValueError, 'World and EE'):
            pack_graphs([a, b])

    def test_wrong_frames_transforms_and_stale_ee_references_are_rejected(self):
        c, masks = fixture()
        p, _ = make_packet(c, masks, backend='cpu', coordinate_frame='world')
        for mutate in (lambda p: p.update(coordinate_frame='EE_metres'),
                       lambda p: p['T_frame_from_camera'].__setitem__((0, 3), 7.),
                       lambda p: p['feature_fields'][0].update(frame='EE')):
            bad = deepcopy(p); mutate(bad)
            with self.assertRaises(ValueError): validate_packet(bad)
        with self.assertRaises(ValueError): observations_from_packet(p, c, coordinate_frame='ee')
        world = known_observations(c)[0]
        ee = observation_in_frame(world, c, 'ee')
        records = current_records([ee], fuse_views=False, coordinate_frame='ee')
        stale = replace(c, tcp_pose=c.tcp_pose+np.array([.1, 0, 0, 0, 0, 0, 0]))
        with self.assertRaises(ValueError): consistent_views([ee], [stale], {c.camera: masks})
        with self.assertRaises(ValueError): build_graph(records, robot(stale), stale.tcp_pose, 0, coordinate_frame='ee')
        other = current_records([world], fuse_views=False, coordinate_frame='world')
        with self.assertRaises(ValueError): deduplicate(records+other)
        with self.assertRaises(ValueError): validate_record_frames(records, 'world')
        with self.assertRaises(ValueError): current_records([ee])
        with self.assertRaises(ValueError): current_records([ee, replace(ee, reference_ee_pose=tuple(stale.tcp_pose))], fuse_views=False, coordinate_frame='ee')
        with self.assertRaises(ValueError): consistent_views([ee, world], [c], {c.camera: masks})
        with self.assertRaises(ValueError): build_graph(records, robot(c), c.tcp_pose, 0)
        with patch('graph.current.RobotFK'):
            for frame in ('world', 'ee'):
                g = CurrentGeometry('unused', {}, perception_backend='cpu', coordinate_frame=frame)
                self.assertEqual(g.system.coordinate_frame, frame)
                self.assertEqual(g.system.packet_builder.keywords['coordinate_frame'], frame)
            with self.assertRaises(ValueError): CurrentGeometry('unused', {}, coordinate_frame='camera')

    def test_cache_readback_preserves_world_frame_metadata(self):
        from graph.plan_a_bc.cache import graph_at
        c, _ = fixture()
        records = current_records(known_observations(c), fuse_views=False, coordinate_frame='world')
        g = build_graph(records, robot(c), c.tcp_pose, 0, coordinate_frame='world', include_observation_status=False)
        arrays = {name: getattr(g, name).numpy() for name in ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')}
        arrays.update(node_ptr=np.array([0, len(g.node_type)]), edge_ptr=np.array([0, g.edge_index.shape[1]]), coordinate_frame=np.array('World_metres'))
        self.assertEqual(graph_at(arrays, 0).packet['coordinate_frame'], 'World_metres')

    def test_public_current_geometry_keeps_configured_frame_to_return(self):
        cameras, masks = cameras_and_masks()
        masks['hand_camera'][1] = False
        data = dict(episode=np.array(cameras[0].episode), timestep=np.int64(0),
            tcp_pose=cameras[0].tcp_pose, robot_world_geometry=robot(cameras[0]), robot76=np.zeros(76))
        for frame in ('world', 'ee'):
            with patch('graph.current.RobotFK'):
                geometry = CurrentGeometry('unused', {}, perception_backend='cpu', coordinate_frame=frame)
            geometry._check_mount = lambda *args: None
            geometry.system.prepare = lambda *args: dict(conditions={'C3': masks}, gate=dict(accepted=True),
                wrist_gate_backend='cpu')
            g, records, _ = geometry.build(cameras, data, [], masks)
            self.assertEqual(g.packet['coordinate_frame'], FRAME_NAMES[frame])
            by_id = {r['persistent_node_id']: r for r in records}
            for i, name in enumerate(g.packet['node_ids']):
                r = by_id[name]
                self.assertEqual(r['coordinate_frame'], frame)
                np.testing.assert_array_equal(g.node_input[i+3, :3], np.asarray(r['xyz'], np.float32))
            expected = robot(cameras[0]) if frame == 'world' else transform(robot(cameras[0]), np.linalg.inv(pose_matrix(cameras[0].tcp_pose)))
            np.testing.assert_allclose(g.node_input[:3, :3], expected, rtol=0, atol=1e-7)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA host required')
class CUDACoordinateFrames(unittest.TestCase):
    def test_tensor_and_numpy_packets_both_frames(self):
        from visual.runtime.graph_perception import make_packet_cuda
        from visual.runtime.tensor_packet import numpy_packet
        for mode in ('both', 'empty', 'holes_overlap'):
            c, masks = fixture(mode)
            for frame in ('world', 'ee'):
                a, _ = make_packet(c, masks, coordinate_frame=frame)
                b = make_packet_cuda(c, torch.as_tensor(masks, device='cuda'), coordinate_frame=frame)
                validate_packet(b)
                self.assertTrue(b['points'].is_cuda)
                b = numpy_packet(b); validate_packet(b)
                self.assertEqual(a['coordinate_frame'], b['coordinate_frame'])
                for key in ('object_id', 'feature_field_valid'):
                    np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a['provenance']['source_index'], b['provenance']['source_index'])
                np.testing.assert_allclose(a['points'], b['points'], rtol=0, atol=1e-6)
                np.testing.assert_allclose(a['geometric_features'], b['geometric_features'], rtol=0, atol=2e-5)

    def test_current_and_resident_graphs_both_frames_without_filter_or_gate(self):
        import json
        from pathlib import Path
        if not (Path(__file__).resolve().parents[3]/'graph/experiments/cuda_ingress_10i_20261103_v1/pipeline.py').is_file():
            self.skipTest('optional CUDA ingress prototype is not present')
        from graph.plan_a_geometry.core import PlanA
        from graph.experiments.cuda_ingress_10i_20261103_v1.pipeline import TensorGeometry
        cameras, masks = cameras_and_masks()
        masks['hand_camera'][1] = False
        data = dict(episode=np.array(cameras[0].episode), timestep=np.int64(0),
            tcp_pose=cameras[0].tcp_pose, robot_world_geometry=robot(cameras[0]), robot76=np.zeros(76))
        protocol = json.loads((Path(__file__).resolve().parents[3]/'graph/plan_a_geometry/protocol.json').read_text())
        tensors = {}
        for c in cameras:
            tensor = lambda a: torch.as_tensor(np.array(a, copy=True), device='cuda')
            world = c.world_from_camera.astype(float); ee = np.linalg.inv(pose_matrix(c.tcp_pose)) @ world
            tensors[c.camera] = SimpleNamespace(camera=c, world=tensor(world), inverse=tensor(np.linalg.inv(world)),
                ee=tensor(ee), world_ee=tensor(pose_matrix(c.tcp_pose)), depth32=tensor(c.depth.astype(np.float32)),
                depth64=tensor(c.depth), intrinsic=tensor(c.intrinsic))
        staged = SimpleNamespace(cameras=cameras, tensors=tensors, data=data)
        raw = {name: torch.as_tensor(m, device='cuda') for name, m in masks.items()}
        for frame in ('world', 'ee'):
            from functools import partial
            system = PlanA('base_camera', 'hand_camera', protocol,
                packet_builder=partial(make_packet, coordinate_frame=frame), coordinate_frame=frame)
            ready = dict(conditions={'C3': masks})
            pools, _, _, _ = system.represent(cameras, ready, 'C3', ('SF',))
            records = budget(pools['SF'], 128)
            expected = system.graph(records, data, cameras)
            pipe = TensorGeometry(system, robot_filter=SimpleNamespace())
            try:
                pipe.ingress.filter = lambda *args: raw
                with patch('graph.experiments.cuda_ingress_10i_20261103_v1.pipeline.wrist_gate',
                    return_value=(raw['hand_camera'][0], dict(accepted=True))):
                    actual, out, _ = pipe.build(staged, raw)
                self.assertEqual(actual.packet['coordinate_frame'], FRAME_NAMES[frame])
                self.assertTrue(all(r['coordinate_frame'] == frame for r in out))
                self.assertEqual([r['source_observation_ids'] for r in out], [r['source_observation_ids'] for r in records])
                for key in ('node_type', 'edge_index', 'edge_type'):
                    np.testing.assert_array_equal(getattr(actual, key).cpu(), getattr(expected, key))
                for key in ('node_input', 'edge_attr'):
                    np.testing.assert_allclose(getattr(actual, key).cpu(), getattr(expected, key), rtol=0, atol=2e-5)
            finally:
                pipe.ingress.close()


if __name__ == '__main__':
    unittest.main()
