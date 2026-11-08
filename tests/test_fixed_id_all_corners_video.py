"""All-corner frames must use first-sighting eligibility and real observations."""

import unittest

import numpy as np

from data_collection.temporal.fixed_id_corner_sequences import CornerSequence
from evaluation.fixed_id_all_corners_video import draw_all_frame, snapshot_for_frame


def track(vertex_id, start, observed):
    length = len(observed)
    targets = np.full((length, 3), vertex_id + 0.25, dtype=np.float32)
    sequence = CornerSequence(
        episode="example", parent="parent", vertex_id=vertex_id, start_frame=start,
        inputs=np.zeros((length, 13), dtype=np.float32), targets=targets,
        observed=np.asarray(observed, dtype=bool),
        prediction_valid=np.array([False] + [True] * (length - 1)),
        last_observed_xyz_world=np.zeros((length, 3), dtype=np.float32),
        target_delta_xyz=np.zeros((length, 3), dtype=np.float32),
    )
    return dict(sequence=sequence, gt=np.zeros((length, 3), dtype=np.float32),
                predictions=dict(mamba=np.full((length, 3), 1.0, dtype=np.float32),
                                 transformer=np.full((length, 3), 2.0, dtype=np.float32)))


class FixedIdAllCornersVideoTests(unittest.TestCase):
    def test_frame_excludes_unseen_and_keeps_actual_visible_coordinates(self):
        tracks = [track(0, 0, [True, False, True]), track(9, 2, [True, False])]
        first = snapshot_for_frame(tracks, 0)
        self.assertEqual([item["vertex_id"] for item in first], [0])
        self.assertTrue(first[0]["observed"])
        self.assertEqual(first[0]["measurement"], [0.25] * 3)
        self.assertEqual(first[0]["predictions"], {})
        hidden = snapshot_for_frame(tracks, 1)
        self.assertEqual([item["vertex_id"] for item in hidden], [0])
        self.assertFalse(hidden[0]["observed"])
        self.assertIsNone(hidden[0]["measurement"])
        self.assertEqual(hidden[0]["predictions"]["mamba"], [1.0] * 3)
        later = snapshot_for_frame(tracks, 2)
        self.assertEqual([item["vertex_id"] for item in later], [0, 9])
        self.assertEqual(later[1]["measurement"], [9.25] * 3)

    def test_missing_error_uses_each_model_without_mixing_observations(self):
        snapshot = snapshot_for_frame([track(0, 0, [True, False])], 1)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        intrinsic = np.array([[100, 0, 256], [0, 100, 256], [0, 0, 1]])
        extrinsic = np.hstack((np.eye(3), np.zeros((3, 1))))
        canvas, errors = draw_all_frame(image, intrinsic, extrinsic, 1, snapshot)
        self.assertEqual(canvas.shape, (704, 1024, 3))
        self.assertAlmostEqual(errors["mamba"][0], np.sqrt(3) * 1000)
        self.assertAlmostEqual(errors["transformer"][0], 2 * np.sqrt(3) * 1000)


if __name__ == '__main__':
    unittest.main()
