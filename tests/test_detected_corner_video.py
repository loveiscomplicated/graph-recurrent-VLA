"""Small stable ID colors distinguish detection from prediction."""

import unittest

import numpy as np

from evaluation.detected_corner_video import draw_frame, draw_track, track_color


class DetectedCornerVideoTests(unittest.TestCase):
    def test_track_colors_are_stable_and_differ(self):
        self.assertEqual(track_color(3), track_color(3))
        self.assertNotEqual(track_color(3), track_color(4))

    def test_detected_and_predicted_centers_share_track_color(self):
        view = np.zeros((32, 32, 3), dtype=np.uint8)
        draw_track(view, (8, 8), 2, True)
        draw_track(view, (24, 8), 2, False)
        np.testing.assert_array_equal(view[8, 8], view[8, 24])
        self.assertGreater(int(view[8, 28].sum()), int(view[8, 12].sum()))

    def test_two_model_frame_layout(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        intrinsic = np.array([[100, 0, 256], [0, 100, 256], [0, 0, 1]], dtype=np.float32)
        extrinsic = np.hstack((np.eye(3), np.zeros((3, 1)))).astype(np.float32)
        item = dict(positions=np.array([[0, 0, 1]], dtype=np.float32),
                    observed=np.array([True]), matches=0, new_tracks=1)
        result = draw_frame(image, intrinsic, extrinsic, 0, {"mamba": item, "transformer": item})
        self.assertEqual(result.shape, (704, 1024, 3))


if __name__ == "__main__":
    unittest.main()
