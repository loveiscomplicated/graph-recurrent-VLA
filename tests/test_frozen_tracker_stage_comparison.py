"""Verify GT depth boundaries and equal-mode Object21 scoring with residuals."""

import unittest

import numpy as np

from evaluation.frozen_tracker_stage_comparison import depth_stage_masks, summarize_stages


class StageComparisonTests(unittest.TestCase):
    def test_depth_boundaries_partition_pre_entry_and_deep(self):
        masks = depth_stage_masks(np.array([-1, 0, 0.001, 20, 20.001]))
        np.testing.assert_array_equal(masks["alignment_pre_insertion"], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(masks["insertion"], [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(masks["entry"], [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(masks["deep"], [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(sum(masks[key] for key in ("alignment_pre_insertion", "entry", "deep")), 1)

    def test_mode_weighting_empty_slices_and_endpoint_regularization(self):
        target = np.zeros((3, 21), dtype=np.float32)
        target[:, [3, 13]] = 1
        prediction = target.copy()
        prediction[:, 0] = [0.005, 0.020, 0.020]
        residual = np.full((3, 3), 0.01, dtype=np.float32)
        modes = np.array(["short", "long", "long"])
        masks = depth_stage_masks(np.array([-1, -1, -1]))
        scores = summarize_stages(prediction, residual, target, modes, masks, {})
        self.assertAlmostEqual(scores["all"]["mode_balanced"]["peg_position_mm"], 12.5, places=5)
        self.assertAlmostEqual(scores["all"]["pooled"]["peg_position_mm"], 15, places=5)
        self.assertIsNone(scores["insertion"]["mode_balanced"])
        perfect = summarize_stages(target, residual, target, modes, masks, {})
        self.assertAlmostEqual(perfect["all"]["mode_balanced"]["object21_loss"], 0.05, places=6)

    def test_nonfinite_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            depth_stage_masks(np.array([0, np.nan]))


if __name__ == "__main__":
    unittest.main()
