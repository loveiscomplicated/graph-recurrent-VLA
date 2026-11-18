"""Keep surface support separate from a corner's own observation history."""

import unittest

import numpy as np
import torch

from training.five_mode_corner_tracker import (
    add_supervision_history,
    missing_loss,
    supervision_mask,
    validation_metrics,
)


class ObservedSurfaceSupervisionTests(unittest.TestCase):
    def make_batch(self):
        observed = np.zeros((1, 6, 20), dtype=bool)
        observed[0, 2, 0] = True
        valid = np.ones((1, 6), dtype=bool)
        valid[0, 5] = False
        arrays = {
            "observed": observed,
            "valid": valid,
            "gt": np.zeros((1, 6, 20, 3), dtype=np.float32),
            "age": np.zeros((1, 6, 20), dtype=np.int32),
            "depth_mm": np.zeros((1, 6), dtype=np.float32),
            "surface_valid": np.ones((1, 6, 32), dtype=bool),
        }
        add_supervision_history(arrays, "after_first_observation")
        return {key: torch.as_tensor(value) for key, value in arrays.items()}

    def test_surface_presence_does_not_initialize_corner_supervision(self):
        batch = self.make_batch()
        expected = torch.zeros_like(batch["observed"])
        expected[0, 3:5, 0] = True
        torch.testing.assert_close(supervision_mask(batch), expected, rtol=0, atol=0)
        self.assertFalse(bool(batch["ever_seen"][:, :2].any()))
        self.assertFalse(bool(supervision_mask(batch)[..., 1:].any()))

        prediction = torch.ones_like(batch["gt"], requires_grad=True)
        original = missing_loss(prediction, batch)
        batch["gt"][~expected] = 100000
        altered = torch.where(expected[..., None], prediction, prediction + 100000)
        changed = missing_loss(altered, batch)
        torch.testing.assert_close(changed, original, rtol=0, atol=0)
        changed.backward()
        torch.testing.assert_close(
            prediction.grad[~expected],
            torch.zeros_like(prediction.grad[~expected]),
            rtol=0,
            atol=0,
        )

    def test_surface_validation_and_selection_ignore_unseen_corner_errors(self):
        batch = self.make_batch()
        mask = supervision_mask(batch).numpy()
        errors = np.full((1, 6, 20), 10000, dtype=np.float32)
        errors[mask] = [2, 4]
        original = validation_metrics(errors, batch, ["fixed"])
        errors[~mask] *= 10000
        changed = validation_metrics(errors, batch, ["fixed"])
        self.assertEqual(original, changed)
        self.assertEqual(changed["mode_balanced_mean_mm"], 3)
        self.assertEqual(changed["all"]["count"], 2)
        self.assertEqual(changed["peg_surface_support"]["present"]["count"], 2)
        self.assertEqual(changed["peg_surface_support"]["absent"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
