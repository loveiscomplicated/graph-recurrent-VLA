"""Task02-C tests of pair provenance, cause separation and evaluation failure modes."""
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.geometry import build_organized_geometry, process_geometry
from visual.runtime.test.normal_diagnostics import synthetic_frame, synthetic_case, quantize
from visual.runtime.test.structural_edges import METHODS, THRESHOLDS, compare
from visual.runtime.test.run_structural_edges import Matcher, ratios, evaluate

K = np.array([[256., 0, 16], [0, 256, 16], [0, 0, 1]])


class StructuralEdgesTest(unittest.TestCase):
    def frame(self, depth=None, mask=None):
        return synthetic_frame(np.full((32, 32), .25, np.float32) if depth is None else depth,
                               K, np.ones((32, 32), bool) if mask is None else mask)

    def test_original_baseline_and_all_union_maps_are_preserved(self):
        frame, _, _ = synthetic_case('crease', K, shape=(32, 32))
        frame = replace(frame, depth=quantize(frame.depth))
        depth = frame.depth.copy(); mask = frame.masks[0].copy()
        result, evidence = compare(frame)
        original = process_geometry(frame)
        for oid in original.objects:
            for field in fields(original.objects[oid]):
                np.testing.assert_array_equal(getattr(original.objects[oid], field.name), getattr(result['baseline'].objects[oid], field.name))
        for method in METHODS:
            for oid in original.objects:
                f = result[method].objects[oid]
                for t in THRESHOLDS:
                    np.testing.assert_array_equal(evidence[method][oid].candidates(t)['union'],
                        f.depth_discontinuity | (np.rad2deg(f.normal_variation_rad) > t))
        np.testing.assert_array_equal(frame.depth, depth); np.testing.assert_array_equal(frame.masks[0], mask)

    def test_pair_indices_are_original_adjacent_pixels_without_row_wrap(self):
        _, evidence = compare(self.frame())
        p = evidence['baseline'][0].pixels
        self.assertEqual(len(p), 2*32*31)
        self.assertEqual(len(np.unique(p, axis=0)), len(p))
        delta = p[:, 1]-p[:, 0]
        self.assertTrue(np.isin(delta, [1, 32]).all())
        self.assertTrue(np.all(p[delta == 1, 0]//32 == p[delta == 1, 1]//32))

    def test_mask_cut_is_not_an_orientation_or_depth_edge(self):
        mask = np.ones((32, 32), bool); mask[:, 16] = False
        _, evidence = compare(self.frame(mask=mask))
        for method in METHODS:
            p = evidence[method][0]; c = p.candidates(5)
            self.assertTrue(p.mask_boundary.any()); self.assertTrue(c['mask'].any())
            self.assertFalse(c['union'].any(), method)
            self.assertFalse(c['invalid'].any())

    def test_invalid_depth_is_not_a_zero_depth_jump(self):
        depth = np.full((32, 32), .25, np.float32); depth[:, 16] = np.nan
        _, evidence = compare(self.frame(depth=depth))
        for method in METHODS:
            p = evidence[method][0]; c = p.candidates(5)
            self.assertTrue(c['invalid'].any()); self.assertFalse(c['depth'].any())
            self.assertFalse(c['union'].any()); self.assertTrue(np.isnan(p.signed_depth_delta_m[p.invalid_depth_boundary]).all())

    def test_depth_step_retains_both_sides_and_external_neighbor(self):
        depth = np.full((32, 32), .25, np.float32); depth[:, 16:] = .5
        _, evidence = compare(self.frame(depth=depth))
        for method in METHODS:
            p = evidence[method][0]; c = p.candidates(180)
            self.assertEqual(int(p.depth_jump.sum()), 32)
            np.testing.assert_array_equal(np.flatnonzero(c['occluding'].any(0)), [15])
            np.testing.assert_array_equal(np.flatnonzero(c['occluded'].any(0)), [16])
        mask = np.ones((32, 32), bool); mask[:, 16:] = False
        _, evidence = compare(self.frame(depth=depth, mask=mask))
        c = evidence['pca5'][0].candidates(20)
        self.assertEqual(c['depth'].sum(), 32); self.assertFalse(c['occluded'].any())

    def test_overlap_empty_support_and_threshold_validation(self):
        frame = self.frame(); frame = replace(frame, masks={0: frame.masks[0], 1: frame.masks[0]})
        _, evidence = compare(frame)
        for method in METHODS:
            p = evidence[method][0]
            self.assertTrue(p.overlap.all()); self.assertFalse(p.supported.any()); self.assertFalse(p.candidates(0)['union'].any())
            for t in (-1, np.nan, np.inf):
                with self.assertRaises(ValueError): p.candidates(t)

    def test_gt_is_not_a_detector_input_and_rgb_does_not_affect_results(self):
        frame, _, _ = synthetic_case('crease', K, shape=(32, 32))
        _, a = compare(frame)
        _, b = compare(replace(frame, rgb=np.full_like(frame.rgb, 255), tcp_pose=np.ones(7)))
        for method in METHODS:
            for field in fields(a[method][0]):
                np.testing.assert_array_equal(getattr(a[method][0], field.name), getattr(b[method][0], field.name))

    def test_projected_occluder_fails_joint_3d_matching(self):
        geometry = build_organized_geometry(self.frame())
        uv = np.array([[16., 16.]])
        xyz = geometry.xyz[16, 16:17].copy(); xyz[:, 2] += .05
        m = Matcher(geometry, 0, (uv, xyz))
        pred = np.zeros((32, 32), bool); pred[16, 16] = True
        row, _ = m.score(pred)
        self.assertEqual(row['matched_candidates'], 0); self.assertEqual(row['covered_references'], 0)

    def test_dense_noise_does_not_win_one_to_one_precision(self):
        geometry = build_organized_geometry(self.frame())
        m = Matcher(geometry, 0, (np.array([[16., 16.]]), geometry.xyz[16, 16:17]))
        sparse = np.zeros((32, 32), bool); sparse[16, 16] = True
        a = ratios(m.score(sparse)[0]); b = ratios(m.score(np.ones((32, 32), bool))[0])
        self.assertEqual(a['recall'], b['recall']); self.assertEqual(a['one_to_one_f1'], 1)
        self.assertLess(b['one_to_one_precision'], .001)

    def test_missing_prediction_and_missing_reference_remain_explicit(self):
        geometry = build_organized_geometry(self.frame())
        m = Matcher(geometry, 0, (np.array([[16., 16.]]), geometry.xyz[16, 16:17]))
        row, values = m.score(np.zeros((32, 32), bool), distances=True)
        self.assertEqual(ratios(row)['recall'], 0); self.assertEqual(values['gt_to_candidate_px'].size, 0)
        m = Matcher(geometry, 0, (np.empty((0, 2)), np.empty((0, 3))))
        row, _ = m.score(np.ones((32, 32), bool))
        self.assertEqual(row['candidates_without_visible_reference'], 1024); self.assertEqual(ratios(row)['precision'], 0)

    def test_saved_evaluation_identity_and_all_frame_guard(self):
        n, h, w = 2, 16, 16
        ep = {'camera/rgb': np.zeros((n, h, w, 3), np.uint8), 'camera/depth': np.full((n, h, w), .3, np.float32),
              'camera/intrinsic': np.tile([[100, 0, 8], [0, 100, 8], [0, 0, 1]], (n, 1, 1)),
              'camera/cam2world': np.tile(np.eye(4), (n, 1, 1)), 'tcp_pose': np.tile([0, 0, 0, 1, 0, 0, 0], (n, 1)),
              'camera/seg': np.ones((n, h, w), np.int16), 'peg_id': np.array(1), 'box_id': np.array(2),
              'peg_pose': np.tile([0, 0, .4, 1, 0, 0, 0], (n, 1)), 'box_pose': np.tile([1, 0, .5, 1, 0, 0, 0], (n, 1)),
              'box_hole_pose': np.tile([1, 0, .5, 1, 0, 0, 0], (n, 1)), 'peg_half_sizes': np.array([.1, .1, .1]),
              'box_hole_radii': np.array(.02), 'seed': np.array(0)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); episode = root/'ep.npz'; mask = root/'masks.npz'
            np.savez(episode, **ep)
            np.savez(mask, source_frame=np.arange(n), peg=np.ones((n, h, w), bool), box=np.zeros((n, h, w), bool))
            with patch.dict(os.environ, {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(root/'experiments')}):
                output = evaluate(episode, mask, root/'out', camera='camera', selected=())
            data = json.loads((output/'comparison.json').read_text())
            self.assertEqual(data['evaluated_frames'], 2)
            self.assertEqual(output.name, data['run_id'])
            self.assertEqual(len(list(output.glob('*_pairs.npz'))), 2)
            self.assertFalse(data['config']['gt_detector_input'])
            metadata = json.loads((root/'experiments'/output.name/'metadata.json').read_text())
            self.assertEqual(metadata['status'], 'finished')
            np.savez(mask, source_frame=np.array([0]), peg=np.ones((1, h, w), bool), box=np.zeros((1, h, w), bool))
            with self.assertRaises(ValueError): evaluate(episode, mask, root/'out', camera='camera', selected=())


if __name__ == '__main__': unittest.main()
