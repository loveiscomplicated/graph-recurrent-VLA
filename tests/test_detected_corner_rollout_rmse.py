"""Permutation-invariant scoring cannot reward unmatched points silently."""

import unittest

import numpy as np
import torch
from torch import nn

from evaluation.detected_corner_rollout_rmse import match_sets, replay_episode


class ZeroPredictor(nn.Module):
    def forward(self, inputs, context):
        return torch.zeros((*inputs.shape[:2], 3), device=inputs.device)


class RolloutRMSETests(unittest.TestCase):
    def test_match_ignores_id_and_node_order_but_respects_kind(self):
        predictions = np.array([[2., 0., 0.], [0., 0., 0.], [100., 0., 0.]], dtype=np.float32)
        reference = np.zeros((24, 3), dtype=np.float32)
        reference[0], reference[8] = [0., 0., 0.], [2., 0., 0.]
        kinds = np.array([1, 0, 0])
        reference_kinds = np.array([0] * 8 + [1] * 16)
        pairs = match_sets(predictions, kinds, np.array([0, 8]), reference, reference_kinds)
        self.assertEqual(set(pairs), {(0, 8), (1, 0)})
        self.assertEqual(len(predictions) - len(pairs), 1)

    def test_rollout_scores_visible_and_missing_with_unmatched_count(self):
        ref = np.zeros((3, 24, 3), dtype=np.float32)
        ref[:, 0, 0] = [0., .01, .02]
        source = dict(
            xyz=np.array([[[0., 0., 0.]], [[0., 0., 0.]], [[.02, 0., 0.]]], dtype=np.float32),
            valid=np.array([[True], [False], [True]]), kind=np.zeros((3, 1), dtype=np.int64),
            physical_ids=np.array([[0], [-1], [0]], dtype=np.int64), reference_xyz=ref,
            reference_kind=np.array([0] * 8 + [1] * 16), robot=np.zeros((3, 9), dtype=np.float32),
        )
        result = replay_episode(ZeroPredictor(), source, "cpu", .005)["totals"]
        self.assertEqual(result["detected"]["matched"], 2)
        self.assertAlmostEqual(result["detected"]["rmse_3d_mm"], 0)
        self.assertEqual(result["predicted"]["matched"], 1)
        self.assertAlmostEqual(result["predicted"]["rmse_3d_mm"], 10, places=4)
        self.assertEqual(result["all"]["unmatched_predicted_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
