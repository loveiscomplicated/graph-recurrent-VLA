"""Contracts for same-frame, Peg-only spatial context and unchanged corner outputs."""

import unittest

import torch

from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker


class SurfaceSpatialCornerTrackerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        self.base = SpatialTemporalCornerTracker().eval()
        torch.manual_seed(8)
        self.model = SurfaceSpatialCornerTracker().eval()
        for name, value in self.base.state_dict().items():
            torch.testing.assert_close(
                self.model.state_dict()[name], value, rtol=0, atol=0
            )
        with torch.no_grad():
            self.base.output[1].weight.normal_(std=0.1)
        self.model.load_state_dict(self.base.state_dict(), strict=False)
        self.inputs = [
            torch.randn(2, 17, 20, 10),
            torch.randn(2, 17, 20, 3),
            torch.randn(2, 17, 20, 3),
            torch.rand(2, 17, 20) > 0.8,
            torch.ones(2, 17, dtype=torch.bool),
        ]
        self.surface = torch.randn(2, 17, 5, 6)
        self.valid = torch.rand(2, 17, 5) > 0.3

    def test_empty_context_matches_baseline_and_preserves_detections(self):
        with torch.no_grad():
            expected = self.base(*self.inputs)
            actual = self.model(
                *self.inputs, self.surface, torch.zeros_like(self.valid)
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-6)
        observed = self.inputs[3]
        torch.testing.assert_close(
            actual[observed], self.inputs[2][observed], rtol=0, atol=0
        )

    def test_surface_changes_peg_but_not_box_or_hole(self):
        with torch.no_grad():
            original = self.model(*self.inputs, self.surface, self.valid)
            changed = self.model(*self.inputs, self.surface + 4, self.valid)
        self.assertGreater(
            float((original[..., :8, :] - changed[..., :8, :]).abs().max()), 1e-5
        )
        torch.testing.assert_close(
            original[..., 8:, :], changed[..., 8:, :], rtol=0, atol=2e-6
        )

    def test_future_surface_does_not_change_past_predictions(self):
        future = self.surface.clone()
        future[:, 9:] += 50
        mask = self.valid.clone()
        mask[:, 9:] = ~mask[:, 9:]
        with torch.no_grad():
            original = self.model(*self.inputs, self.surface, self.valid)
            changed = self.model(*self.inputs, future, mask)
        torch.testing.assert_close(original[:, :9], changed[:, :9], rtol=0, atol=2e-6)

    def test_masked_points_and_surface_order_do_not_change_predictions(self):
        masked = torch.where(self.valid[..., None], self.surface, self.surface + 100)
        permutation = torch.tensor([4, 2, 0, 3, 1])
        with torch.no_grad():
            original = self.model(*self.inputs, self.surface, self.valid)
            changed = self.model(*self.inputs, masked, self.valid)
            shuffled = self.model(
                *self.inputs,
                self.surface[:, :, permutation],
                self.valid[:, :, permutation],
            )
        torch.testing.assert_close(original, changed, rtol=0, atol=2e-6)
        torch.testing.assert_close(original, shuffled, rtol=0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
