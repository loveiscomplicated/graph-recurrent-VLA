"""Numerical and evaluation contracts for Task 02-B, no simulator or network."""
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.geometry import GeometryParameters, build_organized_geometry, estimate_surface_normals, process_geometry
from visual.runtime.test.normal_diagnostics import angular_error, metrics, quantize, synthetic_case, synthetic_frame
from visual.runtime.test.normal_estimators import METHODS, finite_difference, local_plane
from visual.runtime.test.run_normal_comparison import compare, evaluate, localization, localization_summary, response_metrics

K = np.array([[256., 0, 32], [0, 256, 32], [0, 0, 1]])


class NormalComparisonTest(unittest.TestCase):
    def test_baseline_is_bitwise_original_for_every_field_and_full_resolution(self):
        frame, _, _ = synthetic_case('crease', K, shape=(64, 64))
        frame = replace(frame, depth=quantize(frame.depth))
        originals = {k: v.copy() for k, v in frame.masks.items()}
        depth = frame.depth.copy()
        results = compare(frame)
        baseline = process_geometry(frame)
        for oid, feature in baseline.objects.items():
            for field in fields(feature):
                np.testing.assert_array_equal(getattr(feature, field.name), getattr(results['baseline'].objects[oid], field.name))
        for method in METHODS:
            f = results[method].objects[0]
            self.assertEqual(f.normals.shape, (64, 64, 3))
            self.assertEqual(f.normal_variation_rad.shape, (64, 64))
            np.testing.assert_array_equal(f.valid_points, baseline.objects[0].valid_points)
            np.testing.assert_array_equal(f.depth_discontinuity, baseline.objects[0].depth_discontinuity)
            self.assertTrue(np.isnan(f.normals[~f.normal_valid]).all())
            self.assertTrue(np.isnan(f.normal_variation_rad[~f.normal_variation_valid]).all())
            self.assertTrue((np.sum(f.normals[f.normal_valid]*results[method].geometry.xyz[f.normal_valid], axis=1) <= 0).all())
        np.testing.assert_array_equal(frame.depth, depth)
        for k, v in originals.items(): np.testing.assert_array_equal(frame.masks[k], v)

    def test_fd_one_pixel_matches_existing_normal_implementation(self):
        frame, _, _ = synthetic_case('tilted', K, shape=(64, 64))
        geometry = build_organized_geometry(frame)
        a = finite_difference(geometry, 0, 1)
        b = estimate_surface_normals(geometry, 0, GeometryParameters())
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)

    def test_float_plane_accuracy_and_quantization_improvement(self):
        frame, truth, regions = synthetic_case('tilted', K, shape=(64, 64), phase_m=.00025)
        for method, result in compare(frame).items():
            row = metrics(result.objects[0], regions['plane'], truth)
            self.assertLess(row['angular_error_deg']['p95'], .01, method)
        quantized = compare(replace(frame, depth=quantize(frame.depth)))
        base = metrics(quantized['baseline'].objects[0], regions['plane'], truth)
        for method in METHODS[1:]:
            row = metrics(quantized[method].objects[0], regions['plane'], truth)
            self.assertLess(row['angular_error_deg']['p95'], base['angular_error_deg']['p95'], method)
            self.assertLess(row['variation_deg']['p95'], base['variation_deg']['p95'], method)

    def test_fd_rejects_intermediate_mask_hole_invalid_depth_and_jump(self):
        depth = np.full((17, 17), .25, np.float32)
        mask = np.ones((17, 17), bool)
        frame = synthetic_frame(depth, K, mask)
        # Centre (8,8) and spacing-4 endpoints remain good, but u=10 blocks path.
        cases = []
        m = mask.copy(); m[8, 10] = False; cases.append(replace(frame, masks={0: m}))
        for z in (0, np.nan, np.inf, .5):
            d = depth.copy(); d[8, 10] = z; cases.append(replace(frame, depth=d))
        for observation in cases:
            _, valid = finite_difference(build_organized_geometry(observation), 0, 4)
            self.assertFalse(valid[8, 8])

    def test_pca_does_not_mix_across_depth_barrier_or_disconnected_mask(self):
        depth = np.full((17, 17), .25, np.float32); depth[:, 9:] = .5
        frame = synthetic_frame(depth, K, np.ones((17, 17), bool))
        for window in (5, 7):
            n, valid = local_plane(build_organized_geometry(frame), 0, window)
            self.assertTrue(valid[8, 8])
            np.testing.assert_allclose(n[8, 8], [0, 0, -1], atol=1e-6)
        # Even below the depth gate, a full unsupported column disconnects patches.
        depth[:, 9:] = .26
        mask = frame.masks[0].copy(); mask[:, 9] = False
        frame = replace(frame, depth=depth, masks={0: mask})
        n, valid = local_plane(build_organized_geometry(frame), 0, 7)
        self.assertTrue(valid[8, 8]); np.testing.assert_allclose(n[8, 8], [0, 0, -1], atol=1e-6)

    def test_two_objects_and_overlap_never_mix(self):
        depth = np.full((17, 17), .25, np.float32); depth[:, 9:] = .26
        peg = np.zeros((17, 17), bool); peg[:, :9] = True
        box = ~peg; box[8, 8] = True
        frame = replace(synthetic_frame(depth, K, peg), masks={0: peg, 1: box})
        g = build_organized_geometry(frame)
        n, valid = local_plane(g, 0, 7)
        self.assertFalse(valid[8, 8]); self.assertTrue(valid[7, 8])
        np.testing.assert_allclose(n[7, 8], [0, 0, -1], atol=1e-6)

    def test_insufficient_collinear_empty_and_tiny_support_are_invalid(self):
        for mode in ('empty', 'five', 'line'):
            mask = np.zeros((17, 17), bool)
            if mode == 'five': mask[8, 6:11] = True
            if mode == 'line': mask[8, :] = True
            frame = synthetic_frame(np.full((17, 17), .25, np.float32), K, mask)
            n, valid = local_plane(build_organized_geometry(frame), 0, 7)
            self.assertFalse(valid.any(), mode); self.assertTrue(np.isnan(n).all())
        tiny = synthetic_frame(np.full((2, 2), .25, np.float32), K, np.ones((2, 2), bool))
        for estimator in (lambda g: local_plane(g, 0, 5), lambda g: finite_difference(g, 0, 4)):
            self.assertFalse(estimator(build_organized_geometry(tiny))[1].any())

    def test_partial_pca_patch_recovers_normal_without_fabricating_points(self):
        frame = synthetic_frame(np.full((17, 17), .25, np.float32), K, np.ones((17, 17), bool))
        g = build_organized_geometry(frame)
        n, valid = local_plane(g, 0, 5)
        self.assertTrue(valid[0, 0]); np.testing.assert_allclose(n[0, 0], [0, 0, -1], atol=1e-6)
        baseline = estimate_surface_normals(g, 0, GeometryParameters())[1]
        self.assertFalse(baseline[0, 0])

    def test_pca_batching_is_deterministic(self):
        frame, _, _ = synthetic_case('crease', K, shape=(64, 64))
        g = build_organized_geometry(replace(frame, depth=quantize(frame.depth)))
        a = local_plane(g, 0, 7, batch_size=31)
        b = local_plane(g, 0, 7, batch_size=4096)
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)

    def test_continuous_crease_is_preserved_but_window_reduces_peak(self):
        frame, truth, regions = synthetic_case('crease', K, shape=(64, 64))
        results = compare(frame)
        base = metrics(results['baseline'].objects[0], regions['edge_band'], truth)
        for method in METHODS[1:]:
            row = metrics(results[method].objects[0], regions['edge_band'], truth)
            self.assertGreater(row['variation_deg']['max'], 1, method)
            self.assertGreater(row['angular_error_deg']['p95'], 1, method)
            self.assertLess(row['variation_deg']['max'], base['variation_deg']['max'], method)

    def test_step_complementarity_and_localization_missing_is_zero_coverage(self):
        frame, _, regions = synthetic_case('step', K, shape=(64, 64))
        for method, result in compare(frame).items():
            f = result.objects[0]
            row = response_metrics(f, regions['edge_band'])
            self.assertGreater(row['depth_discontinuity'], 0)
            self.assertEqual(row['high_20_count'], 0, method)
            self.assertEqual(row['union_20'], row['depth_discontinuity'])
        reference = np.zeros((64, 64), bool); reference[20:40, 31:33] = True
        loc = localization(result.objects[0], reference, frame.masks[0], 20)
        summary = localization_summary({'test': [loc]})['test']
        self.assertEqual(summary['coverage_2px'], 0)
        self.assertIsNone(summary['response_to_seam_px']['p95'])

    def test_parameter_validation(self):
        frame = synthetic_frame(np.ones((5, 5), np.float32), K, np.ones((5, 5), bool))
        g = build_organized_geometry(frame)
        for s in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): finite_difference(g, 0, s)
        for w in (2, 4, 1.5, True):
            with self.assertRaises(ValueError): local_plane(g, 0, w)
        with self.assertRaises(ValueError): local_plane(g, 0, 5, min_points=2)

    def test_saved_sparse_evaluation_identity_and_missing_gt_rejection(self):
        n, h, w = 3, 32, 32
        episode = {'camera/rgb': np.full((n, h, w, 3), 80, np.uint8),
                   'camera/depth': np.full((n, h, w), .3, np.float32),
                   'camera/intrinsic': np.tile([[100, 0, 16], [0, 100, 16], [0, 0, 1]], (n, 1, 1)),
                   'camera/cam2world': np.tile(np.eye(4), (n, 1, 1)),
                   'tcp_pose': np.tile([0, 0, 0, 1, 0, 0, 0], (n, 1)),
                   'camera/seg': np.ones((n, h, w), np.int16), 'peg_id': np.array(1), 'box_id': np.array(2),
                   'peg_pose': np.tile([0, 0, .4, 1, 0, 0, 0], (n, 1)),
                   'box_pose': np.tile([1, 0, .5, 1, 0, 0, 0], (n, 1)),
                   'box_hole_pose': np.tile([1, 0, .5, 1, 0, 0, 0], (n, 1)),
                   'peg_half_sizes': np.array([.1, .1, .1]), 'box_hole_radii': np.array(.02), 'seed': np.array(0)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ep = root/'episode.npz'; masks = root/'masks.npz'
            np.savez(ep, **episode)
            np.savez(masks, source_frame=np.array([0, 2]), peg=np.ones((2, h, w), bool), box=np.zeros((2, h, w), bool))
            with patch.dict(os.environ, {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root/'experiments')}):
                out = evaluate(ep, masks, camera='camera', selected=(), output_root=root/'out', synthetic=False)
            data = json.loads((out/'comparison.json').read_text())
            self.assertEqual([r['source_frame'] for r in data['frames']], [0, 2])
            self.assertEqual(data['summary']['baseline/box/all']['points'], 0)
            for m in METHODS: self.assertEqual(data['latency_ms'][m]['total']['count'], 2)
            sidecar = json.loads((out/'comparison.json.tracking.json').read_text())
            self.assertEqual(sidecar['run_id'], out.name)
            metadata = json.loads((root/'experiments'/out.name/'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'finished'); self.assertFalse(metadata['gpu_executed'])
            np.savez(ep, **{k: v for k, v in episode.items() if k != 'peg_pose'})
            with self.assertRaisesRegex(ValueError, 'GT missing'):
                evaluate(ep, masks, selected=(), camera='camera', output_root=root/'unused', synthetic=False)
            self.assertFalse((root/'unused').exists())


if __name__ == '__main__': unittest.main()
