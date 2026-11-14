"""Check corner matching semantics and independent reset sampling."""

import unittest
import numpy as np


class PilotCornerDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Defer optional simulator/image modules until logger tests have run.
        from visual.corner_detection.evaluation import score
        cls.score = staticmethod(score)

    def test_one_to_one_matching_does_not_credit_duplicate_detections(self):
        reference = [dict(object_id=0, kind='peg', xyz=np.zeros(3))]
        predictions = [dict(object_id=0, xyz=np.array([0.001, 0, 0])),
                       dict(object_id=0, xyz=np.array([0.002, 0, 0]))]
        peg = self.score(predictions, reference, reference, threshold=0.003)[0]
        self.assertEqual(peg['matched_3mm'], 1)
        self.assertEqual(peg['precision_3mm'], 0.5)
        self.assertEqual(peg['recall_3mm'], 1.0)

    def test_sampling_counts_each_reset_once_and_includes_terminal_frame(self):
        from evaluation.pilot_corner_detection import sample_frames
        self.assertEqual(sample_frames(dict(steps=100, mode='expert')), [0, 25, 50, 75, 100])
        self.assertEqual(sample_frames(dict(steps=100, mode='random')), [25, 50, 75, 100])
        self.assertEqual(sample_frames(dict(steps=2, mode='expert')), [0, 1, 2])

    def test_tolerance_changes_matches_only_for_eligible_reference(self):
        reference = [dict(object_id=1, kind='hole_rim', xyz=np.zeros(3))]
        prediction = [dict(object_id=1, xyz=np.array([0.004, 0, 0]))]
        self.assertEqual(self.score(prediction, reference, reference, 0.003)[1]['matched_3mm'], 0)
        self.assertEqual(self.score(prediction, reference, reference, 0.005)[1]['matched_3mm'], 1)
        self.assertEqual(self.score(prediction, [], reference, 0.005)[1]['matched_3mm'], 0)


if __name__ == '__main__':
    unittest.main()
