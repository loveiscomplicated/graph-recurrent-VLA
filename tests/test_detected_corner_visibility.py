"""Verify visibility assignment from per-frame detector outputs."""

import unittest

import numpy as np

from data_collection.detected_corner_visibility import classify_episode


class DetectedCornerVisibilityTest(unittest.TestCase):
    def test_counts_rim_labels_and_fits_peg_plane(self):
        square = np.asarray([
            [0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0],
        ])
        noisy_square = square.copy()
        noisy_square[-1, 2] = 0.0005
        noncoplanar = square.copy()
        noncoplanar[-1, 2] = 0.02
        fixtures = [
            (noncoplanar, 3),
            (noisy_square, 3),
            (square[:3], 2),
            (square[:1], 0),
            (square[:0], 0),
            (square[:3], 3),
            (square[:2], 0),
            (square[:2], 1),
            (square[:1], 4),
            (noncoplanar, 0),
        ]
        xyz, kinds, rim, offsets = [], [], [], [0]
        for peg, rim_count in fixtures:
            xyz.extend(peg)
            xyz.extend(np.zeros((rim_count + 4, 3)))
            kinds.extend([0] * len(peg) + [1] * (rim_count + 4))
            rim.extend([False] * len(peg) + [True] * rim_count + [False] * 4)
            offsets.append(len(xyz))
        arrays = {
            "frame_indices": np.arange(len(fixtures)),
            "corner_offsets": np.asarray(offsets),
            "corner_world_xyz": np.asarray(xyz),
            "corner_object_id": np.asarray(kinds),
            "corner_contour_hole": np.asarray(rim),
        }
        actual = classify_episode(arrays, len(fixtures), tolerance_m=0.001)
        self.assertEqual(actual, {
            "condition_1": [0],
            "condition_2": [1, 2, 5, 7],
            "condition_3": [3, 4, 6, 8, 9],
        })
