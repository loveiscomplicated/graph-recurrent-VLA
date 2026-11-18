"""Verify paired Object21 reporting and equal-mode rather than pooled means."""

import unittest

import numpy as np
import torch

from evaluation.frozen_tracker_gnn_comparison import summarize_predictions, tracker_settings
from training.gnn.loss import object21_loss


class FrozenGNNComparisonTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((3, 21), dtype=np.float32)
        self.target[:, [3, 13]] = 1
        self.target[:, 17:] = [0.04, 0.01, 0.01, 0.02]
        self.residual = np.zeros((3, 3), dtype=np.float32)
        self.modes = np.array(["short", "long", "long"])

    def test_full_vectors_and_equal_mode_aggregation(self):
        prediction = self.target.copy()
        prediction[:, :3] = [[0.003, 0.004, 0], [0.020, 0, 0], [0.020, 0, 0]]
        prediction[:, 7:10] = [-0.003, -0.004, 0]
        prediction[:, 10:13] = [0, 0.006, 0.008]
        report, errors = summarize_predictions(prediction, self.residual, self.target, self.modes, {})
        np.testing.assert_allclose(errors["tip_minus_hole_mm"], 5, atol=1e-6)
        self.assertAlmostEqual(report["mode_balanced"]["peg_position_mm"], 12.5, places=5)
        self.assertAlmostEqual(report["pooled"]["peg_position_mm"], 15, places=5)
        self.assertAlmostEqual(report["mode_balanced"]["hole_position_mm"], 10, places=5)
        direct = object21_loss(
            {"object21": torch.from_numpy(prediction),
             "endpoint_residual": torch.from_numpy(self.residual)}, torch.from_numpy(self.target)
        )["total"].item()
        self.assertEqual(report["pooled"]["object21_loss"], direct)

    def test_endpoint_regularization_is_part_of_reported_loss(self):
        residual = np.full((3, 3), 0.01, dtype=np.float32)
        report, _ = summarize_predictions(self.target, residual, self.target, self.modes, {})
        self.assertAlmostEqual(report["mode_balanced"]["object21_loss"], 0.05, places=6)

    def test_nonfinite_prediction_fails(self):
        prediction = self.target.copy()
        prediction[2, 7] = np.nan
        with self.assertRaises(ValueError):
            summarize_predictions(prediction, self.residual, self.target, self.modes, {})

    def test_shared_and_paired_tracker_mapping(self):
        new_tracker = {"checkpoint": "new.pt", "sha256": "new"}
        old_tracker = {"checkpoint": "old.pt", "sha256": "old"}
        config = {"tracker": new_tracker, "models": {"existing": {}, "retrained": {}}}
        shared = tracker_settings(config)
        self.assertIs(shared["existing"], new_tracker)
        self.assertIs(shared["retrained"], new_tracker)
        paired = tracker_settings(dict(config, existing_tracker=old_tracker))
        self.assertIs(paired["existing"], old_tracker)
        self.assertIs(paired["retrained"], new_tracker)


if __name__ == "__main__":
    unittest.main()
