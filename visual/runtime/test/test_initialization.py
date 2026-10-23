"""RGB-only initialization behavior, including rejection and late acquisition."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from initialization import assign, detect_initial  # noqa: E402


class InitializationTest(unittest.TestCase):
    def test_joint_assignment_does_not_give_same_box_to_both_objects(self):
        peg = [([10, 10, 30, 30], 0.8)]
        box = [([11, 11, 31, 31], 0.9), ([50, 50, 70, 70], 0.5)]
        self.assertEqual(assign(peg, box), (peg[0], box[1]))

    def test_late_initialization_requires_two_consistent_scans(self):
        frames = np.zeros((11, 100, 100, 3), dtype=np.uint8)
        frames[:, 0, 0, 0] = np.arange(11)

        def detector(rgb, prompt):
            t = int(rgb[0, 0, 0])
            if prompt == "peg":
                return ([[10, 10, 25, 25]], [0.8]) if t == 0 else ([], [])
            if t in (5, 10):
                return ([[60, 10, 75, 25]], [0.7])
            return [], []

        initial = detect_initial(frames, detector, {"peg": "peg", "box": "box"})
        self.assertEqual(initial["peg"]["frame"], 0)
        self.assertEqual(initial["box"]["frame"], 10)
        self.assertEqual(initial["box"]["box"], [60, 10, 75, 25])


if __name__ == "__main__":
    unittest.main()
