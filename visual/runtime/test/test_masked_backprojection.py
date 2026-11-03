"""Mask-only projection must preserve object coordinates and consumed evidence."""
from dataclasses import replace
from functools import partial
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.pca import build_geometry
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence, extract_candidates, SamplingConfig


def frame(depth, masks):
    h, w = depth.shape
    k = np.array([[173.7, 0, w/2+.17], [0, 231.3, h/2-.23], [0, 0, 1]])
    return TrackedRGBDFrame(0, 'base_camera', np.zeros((h, w, 3), np.uint8),
        depth, k, np.eye(4), np.array([0., 0., 0., 1., 0., 0., 0.]), masks, 0, 0.)


class MaskedBackprojection(unittest.TestCase):
    def compare_coordinates(self, f):
        original, actual = build_geometry(f), build_geometry(f, mask_only=True)
        np.testing.assert_array_equal(actual.valid_depth, original.valid_depth)
        np.testing.assert_array_equal(actual.overlap_mask, original.overlap_mask)
        # Background depth participates in boundary features even without XY.
        np.testing.assert_array_equal(actual.xyz[..., 2], original.xyz[..., 2])
        union = actual.valid_depth & (actual.object_masks[0] | actual.object_masks[1])
        np.testing.assert_array_equal(actual.xyz[union], original.xyz[union])
        self.assertTrue(np.isnan(actual.xyz[~union, :2]).all())
        for oid in (0, 1):
            np.testing.assert_array_equal(actual.observed_mask(oid), original.observed_mask(oid))
            np.testing.assert_array_equal(actual.xyz[actual.observed_mask(oid)],
                                          original.xyz[original.observed_mask(oid)])

    def test_raster_order_float32_float64_invalid_depth_and_overlap(self):
        rng = np.random.default_rng(312)
        for dtype in (np.float32, np.float64):
            depth = rng.uniform(.1, 12, (31, 37)).astype(dtype)
            depth[0, :5] = [0, -1, np.nan, np.inf, -np.inf]
            masks = {i:rng.random(depth.shape) < .3 for i in (0, 1)}
            f = frame(depth, masks)
            before = depth.copy()
            self.compare_coordinates(f)
            np.testing.assert_array_equal(depth, before)

    def test_empty_full_tiny_and_noncontiguous_inputs(self):
        for shape in ((1, 1), (2, 2), (19, 23)):
            for full in (False, True):
                depth = np.full(shape, .4, np.float32).T
                masks = {0:np.full(shape, full, bool).T}
                self.compare_coordinates(frame(depth, masks))

    def test_normal_features_pairs_and_candidates_match(self):
        rng = np.random.default_rng(5)
        depth = np.full((35, 39), .3, np.float32)
        depth[:, 19:] = .6
        depth[4, 4] = np.nan
        depth[9, 11] = 0
        peg = np.zeros_like(depth, bool); peg[2:33, 2:23] = True
        peg &= rng.random(depth.shape) > .1
        box = np.zeros_like(peg); box[10:30, 21:38] = True
        for masks in ({0:peg, 1:box}, {0:np.zeros_like(peg), 1:np.zeros_like(box)}):
            f = frame(depth, masks)
            expected = extract_sampling_evidence(f)
            with patch('visual.runtime.sampling.build_geometry', partial(build_geometry, mask_only=True)):
                actual = extract_sampling_evidence(f)
            for oid in (0, 1):
                for name in expected.frame.objects[oid].__dataclass_fields__:
                    np.testing.assert_array_equal(getattr(actual.frame.objects[oid], name),
                                                  getattr(expected.frame.objects[oid], name), err_msg=name)
                for name in expected.pairs[oid].__dataclass_fields__:
                    a, b = getattr(actual.pairs[oid], name), getattr(expected.pairs[oid], name)
                    np.testing.assert_array_equal(a, b, err_msg=name)
            for method in ('S3', 'S4'):
                a = extract_candidates(actual, SamplingConfig(), method)
                b = extract_candidates(expected, SamplingConfig(), method)
                for oid in (0, 1):
                    np.testing.assert_array_equal(a[oid].flat_index, b[oid].flat_index)
                    np.testing.assert_array_equal(a[oid].points, b[oid].points)
                    for name in b[oid].metadata:
                        np.testing.assert_array_equal(a[oid].metadata[name], b[oid].metadata[name])

    def test_invalid_contracts_remain_rejected(self):
        f = frame(np.ones((9, 9), np.float32), {0:np.ones((9, 9), bool)})
        for bad in (replace(f, masks={3:f.masks[0]}),
                    replace(f, masks={0:np.ones((8, 9), bool)}),
                    replace(f, intrinsic=np.zeros((3, 3)))):
            with self.assertRaises(ValueError):build_geometry(bad, mask_only=True)


if __name__ == '__main__':
    unittest.main()
