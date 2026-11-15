"""Check causal alignment and square fitting."""

import unittest

import numpy as np

from visual.corner_detection.rim_consensus import RimConsensus


class RimConsensusTest(unittest.TestCase):
    def test_reorders_observations_and_rejects_distant_squares(self):
        fit = RimConsensus()
        square = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]]) * 0.02
        self.assertEqual(fit.update([]), [])
        np.testing.assert_allclose(fit.update([square]), square, atol=1e-12)
        shifted = square[[2, 0, 3, 1]] + [0.002, 0, 0]
        result = fit.update([shifted, square + [0.08, 0, 0]])
        np.testing.assert_allclose(result, square + [0.001, 0, 0], atol=1e-12)
