"""Ensure first-observation eligibility governs loss and model selection alike."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from training.five_mode_corner_tracker import (
    add_supervision_history,
    missing_loss,
    supervision_counts,
    supervision_mask,
    validation_metrics,
)


class ObservedCornerSupervisionTests(unittest.TestCase):
    def make_batch(self, policy="after_first_observation"):
        observed = np.zeros((1, 6, 20), dtype=bool)
        observed[0, 2, 0] = True
        observed[0, 4, 1] = True
        valid = np.ones((1, 6), dtype=bool)
        valid[0, 5] = False
        arrays = {
            "observed": observed,
            "valid": valid,
            "gt": np.zeros((1, 6, 20, 3), dtype=np.float32),
            "age": np.zeros((1, 6, 20), dtype=np.int32),
            "depth_mm": np.zeros((1, 6), dtype=np.float32),
        }
        add_supervision_history(arrays, policy)
        return {key: torch.as_tensor(value) for key, value in arrays.items()}

    def test_mask_excludes_pre_first_current_observation_never_seen_and_padding(self):
        batch = self.make_batch()
        expected = torch.zeros_like(batch["observed"])
        expected[0, 3:5, 0] = True
        torch.testing.assert_close(supervision_mask(batch), expected, rtol=0, atol=0)
        counts = supervision_counts(batch)
        self.assertEqual(counts["supervised_point_frames"], 2)
        self.assertEqual(counts["body"], {"peg": 2, "box": 0, "hole": 0})

    def test_excluded_labels_and_predictions_have_no_loss_or_gradient_effect(self):
        batch = self.make_batch()
        mask = supervision_mask(batch)
        prediction = torch.ones_like(batch["gt"], requires_grad=True)
        original = missing_loss(prediction, batch)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["gt"][~mask] = 100000
        altered_prediction = torch.where(
            mask[..., None], prediction, prediction + 10000
        )
        altered = missing_loss(altered_prediction, changed)
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
        altered.backward()
        self.assertTrue(
            torch.equal(
                prediction.grad[~mask], torch.zeros_like(prediction.grad[~mask])
            )
        )
        self.assertGreater(float(prediction.grad[mask].abs().sum()), 0)

    def test_empty_supervision_returns_finite_zero_and_zero_gradients(self):
        batch = self.make_batch()
        batch["ever_seen"].zero_()
        prediction = torch.ones_like(batch["gt"], requires_grad=True)
        loss = missing_loss(prediction, batch)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_validation_and_early_stopping_score_ignore_excluded_errors(self):
        batch = self.make_batch()
        mask = supervision_mask(batch).numpy()
        errors = np.full((1, 6, 20), 10000, dtype=np.float32)
        errors[mask] = [2, 4]
        metrics = validation_metrics(errors, batch, ["fixed"])
        errors[~mask] = 1000000
        altered = validation_metrics(errors, batch, ["fixed"])
        self.assertEqual(metrics, altered)
        self.assertEqual(metrics["all"]["count"], 2)
        self.assertEqual(metrics["mode_balanced_mean_mm"], 3)
        self.assertEqual(metrics["body"]["hole"]["count"], 0)

    def test_validation_rejects_modes_without_eligible_targets(self):
        batch = self.make_batch()
        batch["ever_seen"].zero_()
        with self.assertRaisesRegex(ValueError, "no supervised missing corners"):
            validation_metrics(np.zeros((1, 6, 20)), batch, ["fixed"])

    def test_historical_all_missing_configs_keep_their_loss(self):
        batch = self.make_batch("all_missing")
        self.assertNotIn("ever_seen", batch)
        prediction = torch.ones_like(batch["gt"])
        expected_mask = batch["valid"][..., None] & ~batch["observed"]
        expected = F.smooth_l1_loss(
            prediction[expected_mask] * 100, batch["gt"][expected_mask] * 100
        )
        torch.testing.assert_close(
            missing_loss(prediction, batch), expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
