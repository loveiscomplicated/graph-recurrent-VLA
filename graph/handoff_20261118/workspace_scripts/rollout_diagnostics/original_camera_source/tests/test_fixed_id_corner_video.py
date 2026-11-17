"""Check camera projection and observed/predicted frame annotation semantics."""

import unittest

import numpy as np
import torch

from data_collection.temporal.fixed_id_corner_sequences import CornerSequence
from evaluation.fixed_id_corner_video import draw_frame, marker, predict, project_world


class FixedIdCornerVideoTests(unittest.TestCase):
    def setUp(self):
        self.intrinsic = np.array([[100, 0, 256], [0, 100, 256], [0, 0, 1]], dtype=float)
        self.extrinsic = np.hstack((np.eye(3), np.zeros((3, 1))))

    def test_world_projection_rejects_behind_and_offscreen_points(self):
        self.assertEqual(project_world([0, 0, 1], self.intrinsic, self.extrinsic), (256, 256))
        self.assertEqual(project_world([1, -1, 2], self.intrinsic, self.extrinsic), (306, 206))
        self.assertIsNone(project_world([0, 0, -1], self.intrinsic, self.extrinsic))
        self.assertIsNone(project_world([10, 0, 1], self.intrinsic, self.extrinsic))
        self.assertIsNone(project_world([float('nan'), 0, 1], self.intrinsic, self.extrinsic))

    def test_markers_are_small_colored_points(self):
        colors = {'gt': (70, 235, 90), 'prediction': (235, 80, 245), 'detection': (0, 230, 255)}
        for kind, color in colors.items():
            image = np.zeros((40, 40, 3), dtype=np.uint8)
            marker(image, (20, 20), kind)
            np.testing.assert_array_equal(image[20, 20], color)
            ys, xs = np.nonzero(np.any(image != 0, axis=-1))
            self.assertLessEqual(int(xs.max() - xs.min() + 1), 11)
            self.assertLessEqual(int(ys.max() - ys.min() + 1), 11)

    def test_visible_measurement_and_missing_predictions_remain_distinct(self):
        measurement = np.array([[0, 0, 1], [.01, 0, 1], [0, 0, 1]], dtype=np.float32)
        sequence = CornerSequence(
            episode="example", parent="parent", vertex_id=4, start_frame=0,
            inputs=np.zeros((3, 13), dtype=np.float32), targets=measurement,
            observed=np.array([True, True, False]), prediction_valid=np.array([False, True, True]),
            last_observed_xyz_world=measurement.copy(), target_delta_xyz=np.zeros((3, 3), dtype=np.float32),
        )
        gt = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        predicted = dict(mamba=gt + np.array([.02, 0, 0]), transformer=gt + np.array([0, .03, 0]))
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        selection = dict(vertex_id=4)
        canvas, visible = draw_frame(image, self.intrinsic, self.extrinsic, 1, selection, sequence, gt, predicted)
        self.assertEqual(canvas.shape, (704, 1024, 3))
        self.assertEqual(visible["state"], "DETECTED")
        np.testing.assert_array_equal(visible["detection_world"], measurement[1])
        self.assertAlmostEqual(visible["predictions"]["mamba"]["gt_error_mm"], 20)
        _, missing = draw_frame(image, self.intrinsic, self.extrinsic, 2, selection, sequence, gt, predicted)
        self.assertEqual(missing["state"], "NO CERTIFIED DETECTION")
        self.assertIsNone(missing["detection_world"])
        self.assertAlmostEqual(missing["predictions"]["transformer"]["gt_error_mm"], 30)
        _, first = draw_frame(image, self.intrinsic, self.extrinsic, 0, selection, sequence, gt, predicted)
        self.assertEqual(first["predictions"], {})

    def test_inference_uses_only_the_prefix_available_at_each_frame(self):
        class PrefixRecordingModel:
            def __init__(self):
                self.lengths = []

            def __call__(self, inputs, context):
                self.lengths.append(inputs.shape[1])
                output = torch.zeros((*inputs.shape[:2], 3))
                output[:, -1, 0] = inputs[:, :, 0].sum(-1)
                return output

        sequence = CornerSequence(
            episode="example", parent="parent", vertex_id=4, start_frame=0,
            inputs=np.zeros((4, 13), dtype=np.float32), targets=np.zeros((4, 3), dtype=np.float32),
            observed=np.array([True, False, True, False]),
            prediction_valid=np.array([False, True, True, True]),
            last_observed_xyz_world=np.zeros((4, 3), dtype=np.float32),
            target_delta_xyz=np.zeros((4, 3), dtype=np.float32),
        )
        sequence.inputs[:, 0] = [1, 2, 3, 100]
        model = PrefixRecordingModel()
        positions = predict(dict(transformer=model), sequence, torch.device('cpu'))["transformer"]
        self.assertEqual(model.lengths, [2, 3, 4])
        np.testing.assert_array_equal(positions[:, 0], [0, 3, 6, 106])


if __name__ == '__main__':
    unittest.main()
