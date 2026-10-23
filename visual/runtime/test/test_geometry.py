"""Numerically specified observation geometry; no simulator/model assets needed."""

from dataclasses import replace
import unittest

import numpy as np

from visual.runtime.geometry import GeometryParameters, process_geometry
from visual.runtime.rgbd import BOX, PEG, TrackedRGBDFrame


def make_frame(depth, peg=None, box=None, intrinsic=None):
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    return TrackedRGBDFrame(
        source_frame=7, camera="synthetic", rgb=np.zeros((height, width, 3), np.uint8),
        depth=depth, intrinsic=np.asarray(intrinsic if intrinsic is not None else
                                         [[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]], np.float32),
        cam2world=np.eye(4, dtype=np.float32), tcp_pose=np.array([0, 0, 0, 1, 0, 0, 0]),
        masks={PEG: np.ones_like(depth, bool) if peg is None else peg,
               BOX: np.zeros_like(depth, bool) if box is None else box},
        frame_idx=70, timestamp=0.35,
    )


def sloped_plane(height=13, width=15):
    """z = 1 + 0.2*x - 0.1*y; solve camera rays against the exact plane."""
    intrinsic = np.array([[100, 0, width / 2], [0, 120, height / 2], [0, 0, 1]], np.float32)
    v, u = np.indices((height, width))
    rx = (u + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
    ry = (v + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
    return make_frame(1 / (1 - 0.2 * rx + 0.1 * ry), intrinsic=intrinsic)


class OrganizedGeometryTests(unittest.TestCase):
    def test_exact_backprojection_and_legacy_equivalence(self):
        # Asymmetric K tests both pixel offsets and x/y ordering.
        frame = make_frame(np.full((5, 7), 2), intrinsic=[[4, 0, 2], [0, 5, 1], [0, 0, 1]])
        result = process_geometry(frame)
        v, u = np.indices(frame.depth.shape)
        expected = np.stack(((u + 0.5 - 2) * 2 / 4,
                             (v + 0.5 - 1) * 2 / 5, np.full((5, 7), 2)), axis=-1)
        np.testing.assert_allclose(result.geometry.xyz, expected, rtol=0, atol=1e-7)
        from visual.scripts.pointcloud import backproject, to_world
        np.testing.assert_array_equal(result.geometry.xyz[result.geometry.valid_depth],
                                      backproject(frame.depth, frame.intrinsic, frame.masks[PEG]))
        transform = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], np.float32)
        world = to_world(result.geometry.xyz.reshape(-1, 3), transform)
        np.testing.assert_allclose(world, expected.reshape(-1, 3) @ transform[:3, :3].T + [1, 2, 3],
                                   rtol=0, atol=3e-7)

    def test_frontoparallel_plane_normals_and_no_spurious_boundaries(self):
        result = process_geometry(make_frame(np.ones((9, 11))))
        features = result.objects[PEG]
        self.assertEqual(int(features.normal_valid.sum()), 7 * 9)
        np.testing.assert_array_equal(features.normals[features.normal_valid],
                                      np.tile([0, 0, -1], (7 * 9, 1)))
        self.assertFalse(features.depth_discontinuity.any())
        self.assertFalse(features.mask_boundary.any())
        self.assertFalse(features.depth_validity_boundary.any())
        self.assertEqual(int(features.image_boundary.sum()), 2 * 9 + 2 * 11 - 4)
        np.testing.assert_array_equal(features.depth_jump_m, 0)
        np.testing.assert_array_equal(features.normal_variation_rad[features.normal_variation_valid], 0)

    def test_sloped_plane_analytic_normal(self):
        frame = sloped_plane()
        result = process_geometry(frame)
        features = result.objects[PEG]
        expected = np.array([0.2, -0.1, -1]) / np.linalg.norm([0.2, -0.1, -1])
        np.testing.assert_allclose(features.normals[features.normal_valid],
                                   np.tile(expected, (int(features.normal_valid.sum()), 1)),
                                   rtol=0, atol=1e-5)
        self.assertFalse(features.depth_discontinuity.any())
        self.assertFalse(features.mask_boundary.any())
        self.assertLess(float(np.nanmax(features.normal_variation_rad)), 2e-5)

    def test_depth_step_marks_exact_pair_endpoints_and_blocks_normal_bridge(self):
        depth = np.ones((9, 12), np.float32)
        depth[:, 6:] = 1.2
        result = process_geometry(make_frame(depth))
        features = result.objects[PEG]
        expected = np.zeros(depth.shape, bool)
        expected[:, 5:7] = True
        np.testing.assert_array_equal(features.depth_discontinuity, expected)
        np.testing.assert_allclose(features.depth_jump_m[:, 5:7], 0.2, rtol=0, atol=6e-8)
        self.assertFalse(features.normal_valid[:, 5:7].any())
        np.testing.assert_array_equal(features.normals[features.normal_valid],
                                      np.tile([0, 0, -1], (int(features.normal_valid.sum()), 1)))
        self.assertFalse(features.mask_boundary.any())
        self.assertFalse(features.depth_validity_boundary.any())

    def test_jump_threshold_uses_metres_relative_depth_and_strict_comparison(self):
        # Exactly representable depths: dz=.25, gate=.125+.125*1=.25.
        frame = make_frame(np.tile([1, 1, 1.25, 1.25], (5, 1)))
        at_gate = process_geometry(frame, GeometryParameters(0.125, 0.125))
        self.assertFalse(at_gate.objects[PEG].depth_discontinuity.any())
        below_gate = process_geometry(frame, GeometryParameters(0.0625, 0.125))
        self.assertEqual(int(below_gate.objects[PEG].depth_discontinuity.sum()), 10)

    def test_invalid_depth_and_neighbours_never_produce_normals_or_points(self):
        for value in (np.nan, np.inf, -np.inf, 0, -1):
            with self.subTest(depth=value), np.errstate(all="raise"):
                depth = np.ones((9, 9), np.float32)
                depth[4, 4] = value
                result = process_geometry(make_frame(depth))
                features = result.objects[PEG]
                expected_boundary = np.zeros_like(depth, bool)
                expected_boundary[[3, 5], 4] = True
                expected_boundary[4, [3, 5]] = True
                np.testing.assert_array_equal(features.depth_validity_boundary, expected_boundary)
                self.assertFalse(features.valid_points[4, 4])
                self.assertTrue(np.isnan(result.geometry.xyz[4, 4]).all())
                self.assertTrue(np.isnan(features.normals[4, 4]).all())
                self.assertTrue(np.isnan(features.normal_variation_rad[4, 4]))
                self.assertFalse(features.depth_discontinuity.any())
                self.assertFalse(features.normal_valid[expected_boundary].any())
                self.assertEqual(int(features.normal_valid.sum()), 49 - 5)
                self.assertTrue(features.normal_valid[3, 3])  # Diagonal isn't part of the + stencil.

    def test_mask_cut_on_same_plane_is_observation_boundary_only(self):
        mask = np.zeros((9, 11), bool)
        mask[2:7, 3:8] = True
        result = process_geometry(make_frame(np.ones(mask.shape), peg=mask))
        features = result.objects[PEG]
        expected = mask.copy()
        expected[3:6, 4:7] = False
        np.testing.assert_array_equal(features.mask_boundary, expected)
        self.assertEqual(int(features.valid_points.sum()), 25)  # No erosion before analysis.
        self.assertEqual(int(features.normal_valid.sum()), 9)
        self.assertFalse(features.depth_discontinuity.any())
        self.assertFalse(features.depth_validity_boundary.any())
        self.assertTrue(np.isnan(features.normals[~features.normal_valid]).all())

    def test_two_object_normals_do_not_mix_even_below_jump_gate(self):
        frame = sloped_plane(height=11, width=16)
        peg = np.zeros(frame.depth.shape, bool)
        peg[:, :8] = True
        # Right object is frontoparallel, with a z difference below the jump gate.
        depth = frame.depth.copy()
        depth[:, 8:] = 1
        result = process_geometry(replace(frame, depth=depth, masks={PEG: peg, BOX: ~peg}))
        self.assertFalse(result.objects[PEG].normal_valid[:, 7:].any())
        self.assertFalse(result.objects[BOX].normal_valid[:, :9].any())
        expected = np.array([0.2, -0.1, -1]) / np.linalg.norm([0.2, -0.1, -1])
        a, b = result.objects[PEG], result.objects[BOX]
        np.testing.assert_allclose(a.normals[a.normal_valid], np.tile(expected, (int(a.normal_valid.sum()), 1)),
                                   rtol=0, atol=1e-5)
        np.testing.assert_array_equal(b.normals[b.normal_valid], np.tile([0, 0, -1], (int(b.normal_valid.sum()), 1)))
        self.assertFalse((a.valid_points & b.valid_points).any())
        self.assertFalse(a.depth_discontinuity.any())
        self.assertFalse(b.depth_discontinuity.any())

    def test_depth_jump_across_objects_is_reported_separately_from_mask_boundary(self):
        depth = np.ones((7, 10), np.float32)
        depth[:, 5:] = 2
        peg = np.zeros_like(depth, bool)
        peg[:, :5] = True
        result = process_geometry(make_frame(depth, peg=peg, box=~peg))
        for object_id, column in ((PEG, 4), (BOX, 5)):
            features = result.objects[object_id]
            expected = np.zeros_like(peg)
            expected[:, column] = True
            np.testing.assert_array_equal(features.depth_discontinuity, expected)
            np.testing.assert_array_equal(features.mask_boundary, expected)
            self.assertFalse(features.normal_valid[:, column].any())

    def test_overlap_is_preserved_but_excluded_from_both_object_supports(self):
        peg = np.zeros((9, 11), bool)
        box = peg.copy()
        peg[:, :6] = True
        box[:, 5:] = True
        frame = make_frame(np.ones(peg.shape), peg, box)
        result = process_geometry(frame)
        self.assertEqual(int(result.geometry.overlap_mask.sum()), 9)
        for object_id in (PEG, BOX):
            features = result.objects[object_id]
            np.testing.assert_array_equal(features.object_mask, frame.masks[object_id])
            self.assertFalse(features.valid_points[:, 5].any())
            self.assertFalse(features.normal_valid[:, 4:7].any())
        self.assertFalse((result.objects[PEG].valid_points & result.objects[BOX].valid_points).any())

    def test_empty_mask_all_invalid_and_tiny_images(self):
        for shape in ((1, 1), (1, 5), (5, 1), (2, 2), (7, 9)):
            for depth_value in (1, np.nan):
                with self.subTest(shape=shape, depth=depth_value):
                    empty = np.zeros(shape, bool)
                    result = process_geometry(make_frame(np.full(shape, depth_value), empty, empty))
                    for features in result.objects.values():
                        self.assertFalse(features.valid_points.any())
                        self.assertFalse(features.normal_valid.any())
                        self.assertFalse(features.depth_discontinuity.any())
                        self.assertFalse(features.mask_boundary.any())
                        self.assertTrue(np.isnan(features.normals).all())
                        self.assertTrue(np.isnan(features.depth_jump_m).all())
                        self.assertTrue(np.isnan(features.normal_variation_rad).all())
        invalid = process_geometry(make_frame(np.full((7, 9), np.nan)))
        self.assertTrue(invalid.objects[PEG].object_mask.all())
        self.assertFalse(invalid.objects[PEG].valid_points.any())

    def test_isolated_normal_has_no_variation_support(self):
        mask = np.zeros((7, 7), bool)
        mask[2:5, 2:5] = True
        features = process_geometry(make_frame(np.ones(mask.shape), peg=mask)).objects[PEG]
        self.assertEqual(int(features.normal_valid.sum()), 1)
        self.assertFalse(features.normal_variation_valid.any())
        self.assertTrue(np.isnan(features.normal_variation_rad).all())

    def test_analytic_curved_surface_has_variation_without_classification(self):
        # Smooth paraboloid z = 1 + 10*x^2, solved against each pinhole ray.
        height, width = 11, 21
        v, u = np.indices((height, width))
        ray_x = (u + 0.5 - width / 2) / 200
        depth = 2 / (1 + np.sqrt(1 - 40 * ray_x ** 2))
        frame = make_frame(depth, intrinsic=[[200, 0, width / 2], [0, 200, height / 2], [0, 0, 1]])
        result = process_geometry(frame)
        features = result.objects[PEG]
        xyz = result.geometry.xyz
        tangent_x = xyz[5, 11].astype(float) - xyz[5, 9]
        expected_centre = np.cross(tangent_x, xyz[6, 10].astype(float) - xyz[4, 10])
        expected_centre /= -np.linalg.norm(expected_centre)
        np.testing.assert_allclose(features.normals[5, 10], expected_centre, rtol=0, atol=1e-7)
        # Symmetric centre is frontoparallel. Use the exact secant of the
        # observed float32 z samples, not the unrounded ideal paraboloid:
        # rounding z to metres changes this slope by about 5e-6 radians.
        z0, z2 = float(frame.depth[5, 10]), float(frame.depth[5, 12])
        slope = (z2 - z0) / (float(ray_x[5, 12]) * z2 - float(ray_x[5, 10]) * z0)
        expected_angle = np.arctan(slope)
        self.assertAlmostEqual(float(features.normal_variation_rad[5, 10]), expected_angle, delta=1e-6)
        self.assertGreater(expected_angle, 0.09)
        self.assertFalse(features.depth_discontinuity.any())
        self.assertFalse(hasattr(features, "edge_class"))

    def test_flat_index_alignment_metadata_and_inputs_unchanged(self):
        frame = sloped_plane()
        original_depth = frame.depth.copy()
        original_mask = frame.masks[PEG].copy()
        result = process_geometry(frame)
        features = result.objects[PEG]
        selected = np.flatnonzero(features.normal_variation_valid)[[0, 9, 20, -1]]
        v, u = np.unravel_index(selected, frame.depth.shape)
        for values in (result.geometry.xyz, features.normals):
            np.testing.assert_array_equal(values.reshape(-1, 3)[selected], values[v, u])
        for values in (features.valid_points, features.depth_jump_m, features.depth_discontinuity,
                       features.mask_boundary, features.normal_variation_rad):
            np.testing.assert_array_equal(values.ravel()[selected], values[v, u])
        np.testing.assert_array_equal(result.geometry.xyz[v, u, 2], frame.depth[v, u])
        self.assertEqual((result.source_frame, result.frame_idx, result.timestamp), (7, 70, 0.35))
        self.assertEqual(result.camera, "synthetic")
        np.testing.assert_array_equal(frame.depth, original_depth)
        np.testing.assert_array_equal(frame.masks[PEG], original_mask)
        self.assertNotIn("rgb", vars(result))
        self.assertNotIn("rgb", vars(features))
        self.assertAlmostEqual(sum(result.timings_ms[k] for k in
                                   ("geometry", "normals", "discontinuity_features")),
                               result.timings_ms["total"], places=8)

    def test_malformed_geometry_inputs_raise(self):
        frame = make_frame(np.ones((5, 5)))
        for invalid in (replace(frame, depth=frame.depth.astype(np.int16)),
                        replace(frame, intrinsic=np.eye(4)),
                        replace(frame, intrinsic=np.zeros((3, 3))),
                        replace(frame, masks={7: frame.masks[PEG]}),
                        replace(frame, masks={PEG: np.ones((5, 4), bool)}),
                        replace(frame, masks={PEG: np.ones((5, 5), np.uint8)})):
            with self.subTest(frame=invalid), self.assertRaises(ValueError):
                process_geometry(invalid)
        for parameters in ((-1, 0), (0, -1), (np.nan, 0), (0, np.inf)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                GeometryParameters(*parameters)


if __name__ == "__main__":
    unittest.main()
