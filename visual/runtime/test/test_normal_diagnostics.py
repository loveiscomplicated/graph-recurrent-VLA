"""Numerical contracts for evaluation truth, quantization and scale interventions."""
from dataclasses import replace
import unittest

import numpy as np

from visual.runtime.geometry import process_geometry
from visual.runtime.test.normal_diagnostics import (
    angular_error, box_primitives, metrics, normal_blockers, oracle, quantize,
    ray_box, real_regions, rim_region, subframe, synthetic_case, synthetic_frame,
)

K = np.array([[256., 0, 256], [0, 256, 256], [0, 0, 1]])


class NormalDiagnosticsTest(unittest.TestCase):
    def test_paired_quantization_alone_creates_planar_variation(self):
        frame, truth, regions = synthetic_case('tilted', K, z0=.25, phase_m=.00025)
        before = frame.depth.copy()
        clean = metrics(process_geometry(frame).objects[0], regions['plane'], truth)
        q = replace(frame, depth=quantize(frame.depth))
        noisy = metrics(process_geometry(q).objects[0], regions['plane'], truth)
        self.assertLess(clean['angular_error_deg']['p95'], .003)
        self.assertLess(clean['variation_deg']['p95'], .005)
        self.assertGreater(noisy['angular_error_deg']['p95'], 10)
        self.assertGreater(noisy['high_20_fraction_of_valid'], .1)
        self.assertEqual(clean['normals'], noisy['normals'])
        self.assertEqual(noisy['depth_discontinuity'], 0)
        np.testing.assert_array_equal(before, frame.depth)
        delta = q.depth.astype(float)-frame.depth
        self.assertLessEqual(delta.max(), 1e-7)
        self.assertGreaterEqual(delta.min(), -.0010001)

    def test_front_plane_quantization_preserves_zero_variation(self):
        frame, truth, regions = synthetic_case('front', K, z0=.25025)
        result = process_geometry(replace(frame, depth=quantize(frame.depth))).objects[0]
        row = metrics(result, regions['plane'], truth)
        self.assertEqual(row['variation_deg']['max'], 0.)
        self.assertEqual(row['angular_error_deg']['max'], 0.)

    def test_step_preserves_jump_with_missing_normals(self):
        frame, truth, regions = synthetic_case('step', K)
        f = process_geometry(frame).objects[0]
        row = metrics(f, regions['edge_band'], truth)
        self.assertEqual(row['depth_discontinuity'], 500)
        self.assertEqual(int(f.normal_valid[131:381, 255:257].sum()), 0)
        self.assertEqual(row['high_20_count'], 0)
        self.assertEqual(row['normal_valid_fraction'], 2/3)

    def test_subsampling_keeps_source_pixel_centres_in_cv(self):
        frame, _, _ = synthetic_case('tilted', K)
        original = process_geometry(frame).geometry.xyz
        for stride in (2, 4):
            sub = subframe(frame, stride)
            np.testing.assert_array_equal(sub.depth, frame.depth[::stride, ::stride])
            np.testing.assert_allclose(process_geometry(sub).geometry.xyz,
                                       original[::stride, ::stride], atol=0, rtol=0)

    def test_ray_box_parallel_outside_and_signed_face(self):
        dirs = np.array([[0., 0, 1], [0, 1, 0], [0, 0, -1]])
        z, face = ray_box(np.array([0., 0, -2]), dirs, np.zeros(3), np.ones(3))
        self.assertEqual(z[0], 1.)
        self.assertEqual(face[0], 4)  # -z face
        self.assertTrue(np.isnan(z[1:]).all())

    def test_pose_oracle_and_plane_roi_do_not_depend_on_variation(self):
        k = np.array([[100., 0, 32], [0, 100, 32], [0, 0, 1]])
        frame = synthetic_frame(np.full((64, 64), .3, np.float32), k, np.ones((64, 64), bool))
        pose = np.array([0., 0, .4, 1, 0, 0, 0])
        half = np.array([.07, .07, .1])
        truth = oracle(frame, pose, [(np.zeros(3), half)])
        gt = np.isfinite(truth['depth'])
        result = process_geometry(frame)
        regions, _, _ = real_regions(frame, result, gt, truth, 0, half)
        self.assertGreater(regions['plane_interior'].sum(), 100)
        np.testing.assert_allclose(truth['depth'][gt], .3, atol=1e-14)
        error = angular_error(result.objects[0].normals, truth['normals'])
        self.assertLess(np.nanmax(error[regions['plane_interior']]), 1e-5)
        result.objects[0].normal_variation_rad[:] = 2.
        regions2, _, _ = real_regions(frame, result, gt, truth, 0, half)
        np.testing.assert_array_equal(regions['plane_interior'], regions2['plane_interior'])

    def test_hollow_box_oracle_has_open_hole_and_physical_rim(self):
        # Local x points toward camera, local y/z span the image.
        k = np.array([[100., 0, 64], [0, 100, 64], [0, 0, 1]])
        frame = synthetic_frame(np.full((128, 128), .4, np.float32), k, np.ones((128, 128), bool))
        # Rotation around local y by +90deg: world x<-local z, world z<- -local x.
        pose = np.array([0., 0, .5, np.sqrt(.5), 0, np.sqrt(.5), 0])
        truth = oracle(frame, pose, box_primitives(.1, .025, np.array([0., 0.])))
        self.assertTrue(np.isnan(truth['depth'][64, 64]))
        self.assertAlmostEqual(truth['depth'][64, 80], .4)
        visible = np.isfinite(truth['depth'])
        rim, _ = rim_region(frame, truth, visible, pose, .1, .025, np.array([0., 0.]))
        self.assertGreater(rim.sum(), 30)
        self.assertFalse(rim[64, 80])  # Outer face far from physical aperture edge.
        hidden, _ = rim_region(frame, truth, np.zeros_like(visible), pose, .1, .025, np.array([0., 0.]))
        self.assertEqual(hidden.sum(), 0)

    def test_mask_cut_only_removes_support_and_reasons_cover_loss(self):
        frame, truth, _ = synthetic_case('front', K, z0=.25025)
        mask = frame.masks[0].copy(); mask[:, 256:] = False
        result = process_geometry(replace(frame, masks={0: mask}))
        row = metrics(result.objects[0], mask, truth)
        self.assertEqual(row['high_5_count'], 0)
        self.assertEqual(row['depth_discontinuity'], 0)
        blockers = normal_blockers(result, 0)
        self.assertEqual(blockers['unexplained'], 0)
        self.assertEqual(blockers['unavailable'], blockers['reasons_nonexclusive']['mask'])


if __name__ == '__main__': unittest.main()
