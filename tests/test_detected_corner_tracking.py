"""Detector-only matching and causal corner history contracts."""

import unittest

import numpy as np
import torch
from torch import nn

from graph.temporal.detected_corner_tracking import DetectedCornerTracker, match_positions
from evaluation.detected_corner_tracking import score_episode


class ZeroPredictor(nn.Module):
    def forward(self, inputs, context):
        return torch.zeros((*inputs.shape[:2], 3), device=inputs.device)


class DetectedCornerTrackingTests(unittest.TestCase):
    def test_assignment_is_one_to_one_and_respects_distance_and_kind(self):
        predictions = np.array([[0., 0, 0], [.008, 0, 0]], dtype=np.float32)
        detections = np.array([[.004, 0, 0], [.009, 0, 0]], dtype=np.float32)
        matches = match_positions(predictions, np.array([0, 0]), detections,
                                  np.array([0, 0]), .005)
        self.assertEqual(matches, [(0, 0), (1, 1)])
        self.assertEqual(match_positions(predictions[:1], np.array([0]), detections[:1],
                                         np.array([1]), .005), [])

    def test_reappearance_reuses_id_and_keeps_exact_detection(self):
        tracker = DetectedCornerTracker(ZeroPredictor(), .01, "cpu")
        robot = np.zeros(9, dtype=np.float32)
        first = tracker.step([[1., 2., 3.]], [0], robot)
        missing = tracker.step(np.empty((0, 3)), [], robot)
        actual = np.array([[1.002, 2., 3.]], dtype=np.float32)
        again = tracker.step(actual, [0], robot)
        self.assertEqual(first["detection_tracks"].tolist(), [0])
        self.assertEqual(again["detection_tracks"].tolist(), [0])
        self.assertFalse(missing["observed"][0])
        np.testing.assert_array_equal(again["positions"], actual)
        self.assertEqual(tracker.tracks[0].inputs[-1][3], 0.)
        np.testing.assert_array_equal(tracker.tracks[0].inputs[-1][:3], [0, 0, 0])

    def test_distant_redetection_creates_new_id(self):
        tracker = DetectedCornerTracker(ZeroPredictor(), .01, "cpu")
        robot = np.zeros(9, dtype=np.float32)
        tracker.step([[0., 0., 0.]], [0], robot)
        tracker.step(np.empty((0, 3)), [], robot)
        result = tracker.step([[.02, 0., 0.]], [0], robot)
        self.assertEqual(result["detection_tracks"].tolist(), [1])
        self.assertEqual(result["observed"].tolist(), [False, True])
        self.assertEqual(len(tracker.tracks), 2)

    def test_score_reports_reappearance_new_id_without_using_gt_for_matching(self):
        source = dict(
            xyz=np.array([[[0., 0., 0.]], [[0., 0., 0.]], [[.02, 0., 0.]]], dtype=np.float32),
            valid=np.array([[True], [False], [True]]),
            kind=np.zeros((3, 1), dtype=np.int64),
            physical_ids=np.array([[0], [-1], [0]], dtype=np.int64),
            reference_xyz=np.zeros((3, 24, 3), dtype=np.float32),
            robot=np.zeros((3, 9), dtype=np.float32),
        )
        result = score_episode(ZeroPredictor(), source, "cpu", .01)["counts"]
        self.assertEqual(result["reappearances"], 1)
        self.assertEqual(result["reidentified"], 0)
        self.assertEqual(result["reappearance_new_track"], 1)
        self.assertEqual(result["vertices_with_multiple_track_ids"], 1)


if __name__ == "__main__":
    unittest.main()
