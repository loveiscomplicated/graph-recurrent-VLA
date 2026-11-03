"""Mask identities survive packet packing and per-object observation conversion."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from graph.persistent_geometry.fusion import observations_from_packet
from graph.persistent_geometry.tests.test_contracts import camera
from graph.vision_input import receive_vision_packet, validate_packet
from visual.runtime.graph_perception import make_packet
from visual.runtime.object_groups import object_rows, object_slices


def fixture(mode='both'):
    c = camera(name='base_camera', z=300)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler('xyz', [.15, -.25, .35]).as_matrix()
    world[:3, 3] = [.2, -.1, .4]
    c = replace(c, extrinsic_cv=np.linalg.inv(world)[:3],
        cam2world_gl=world @ np.diag([1., -1., -1., 1.]),
        tcp_pose=np.array([.12, -.03, .08, np.cos(.3), 0, 0, np.sin(.3)]))
    masks = np.zeros((2, *c.depth.shape), bool)
    masks[0, 3:20, 3:12] = True
    masks[1, 4:18, 14:21] = True
    if mode == 'peg':
        masks[1] = False
    elif mode == 'box':
        masks[0] = False
    elif mode == 'empty':
        masks[:] = False
    elif mode == 'holes_overlap':
        masks[1, 8:13, 8:17] = True
        raw = c.depth_raw_mm.copy()
        raw[::3, ::4] = -32768
        c = replace(c, depth_raw_mm=raw)
    return c.validate(), masks


def same_observations(test, a, b):
    for x, y in zip(a, b):
        for key, value in asdict(x).items():
            other = getattr(y, key)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other, err_msg=key)
            else:
                test.assertEqual(value, other)


class ObjectGroups(unittest.TestCase):
    def test_cpu_packets_preserve_groups_and_match_legacy_observations(self):
        for mode in ('both', 'peg', 'box', 'empty', 'holes_overlap'):
            with self.subTest(mode=mode):
                c, masks = fixture(mode)
                packet, _ = make_packet(c, masks, backend='cpu')
                validate_packet(packet)
                before = deepcopy(packet)
                n0 = int((packet['object_id'] == 0).sum())
                self.assertEqual(packet['object_slices'], object_slices(n0, len(packet['points'])-n0))
                rows = object_rows(packet)
                self.assertTrue(all(isinstance(r, slice) for r in rows))
                for oid, r in enumerate(rows):
                    self.assertTrue((packet['object_id'][r] == oid).all())
                    source = packet['provenance']['source_index'][r]
                    self.assertTrue((masks[oid] & ~masks[1-oid]).ravel()[source].all())
                    if len(source):
                        self.assertTrue(np.shares_memory(packet['points'], packet['points'][r]))
                legacy = deepcopy(packet)
                del legacy['object_slices']
                same_observations(self, observations_from_packet(packet, c), observations_from_packet(legacy, c))
                received = receive_vision_packet(packet)
                self.assertEqual(received.packet['object_slices'], packet['object_slices'])
                for key in ('points', 'object_id', 'geometric_features', 'feature_field_valid'):
                    np.testing.assert_array_equal(packet[key], before[key])
                    np.testing.assert_array_equal(received.packet[key], packet[key])

    def test_malformed_or_inconsistent_ranges_are_rejected(self):
        c, masks = fixture()
        packet, _ = make_packet(c, masks, backend='cpu')
        n = len(packet['points'])
        wrong = (None, (), ((0, n),), ((0, n), (n-1, n)), ((-1, 0), (0, n)),
                 ((0, n+1), (n+1, n)), ((False, 0), (0, n)), ((0, 0), (0, n)),
                 ((0, n), (n, n)))
        for ranges in wrong:
            with self.subTest(ranges=ranges):
                bad = {**packet, 'object_slices': ranges}
                with self.assertRaises(ValueError):
                    validate_packet(bad)
                with self.assertRaises(ValueError):
                    observations_from_packet(bad, c)

    def test_interleaved_legacy_rows_keep_each_objects_source_order(self):
        c, masks = fixture()
        packet, _ = make_packet(c, masks, backend='cpu')
        legacy = dict(packet)
        del legacy['object_slices']
        order = np.random.default_rng(42).permutation(len(packet['points']))
        for key in ('points', 'object_id', 'geometric_features', 'feature_field_valid'):
            legacy[key] = packet[key][order]
        legacy['provenance'] = {**packet['provenance'],
            'source_index': packet['provenance']['source_index'][order]}
        for oid, observation in enumerate(observations_from_packet(legacy, c)):
            expected = legacy['provenance']['source_index'][legacy['object_id'] == oid]
            np.testing.assert_array_equal(observation.pixels, expected)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA host required')
class CUDAObjectGroups(unittest.TestCase):
    def test_gpu_selection_and_tensor_packets_carry_the_same_ranges(self):
        from visual.runtime.graph_perception import make_packet_cuda
        from visual.runtime.tensor_packet import numpy_packet
        for mode in ('both', 'peg', 'box', 'empty', 'holes_overlap'):
            with self.subTest(mode=mode):
                c, masks = fixture(mode)
                packet, _ = make_packet(c, masks)
                tensor = make_packet_cuda(c, torch.as_tensor(masks, device='cuda'))
                validate_packet(tensor)
                host = numpy_packet(tensor)
                validate_packet(host)
                self.assertEqual(tensor['object_slices'], packet['object_slices'])
                self.assertEqual(host['object_slices'], packet['object_slices'])
                for rows in object_rows(tensor):
                    self.assertIsInstance(rows, slice)
                    self.assertTrue(tensor['points'][rows].is_cuda)
                for key in ('points', 'object_id', 'feature_field_valid'):
                    if key == 'points':
                        np.testing.assert_allclose(host[key], packet[key], rtol=0, atol=1e-6)
                    else:
                        np.testing.assert_array_equal(host[key], packet[key])
                np.testing.assert_array_equal(host['provenance']['source_index'], packet['provenance']['source_index'])
                np.testing.assert_allclose(host['geometric_features'], packet['geometric_features'], rtol=0, atol=2e-5)
                legacy = {k: v for k, v in packet.items() if k != 'object_slices'}
                same_observations(self, observations_from_packet(packet, c), observations_from_packet(legacy, c))
                if mode == 'both':
                    bad = dict(tensor, object_slices=((0, len(tensor['points'])), (len(tensor['points']), len(tensor['points']))))
                    with self.assertRaises(ValueError):
                        validate_packet(bad)


if __name__ == '__main__':
    unittest.main()
