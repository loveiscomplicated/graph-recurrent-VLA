"""Exercise timing and proximity rules for moving-Hole episodes."""

import unittest

import numpy as np

from evaluation.moving_hole import HoleMotion


class HoleMotionTests(unittest.TestCase):
    def make_motion(self, mode='continuous'):
        return HoleMotion(dict(mode=mode, start_seconds=1.0, end_seconds=3.0,
                               duration_seconds=1.0, minimum_distance_m=0.08,
                               translation_m=[0.06, 0.0, 0.0]))

    def test_continuous_motion_starts_after_delay_and_stops_at_window(self):
        motion = self.make_motion()
        peg, hole = [0, 1, 0], [0, 0, 0]
        np.testing.assert_allclose(motion.update(0.9, peg, hole), [0, 0, 0])
        np.testing.assert_allclose(motion.update(1.5, peg, hole), [0.03, 0, 0])
        np.testing.assert_allclose(motion.update(2.0, peg, [0.03, 0, 0]), [0.06, 0, 0])
        np.testing.assert_allclose(motion.update(3.0, peg, [0.06, 0, 0]), [0.06, 0, 0])
        self.assertEqual(motion.status, 'window closed')

    def test_near_peg_latches_stop_even_if_it_moves_away(self):
        motion = self.make_motion()
        np.testing.assert_allclose(motion.update(0.5, [0.07, 0, 0], [0, 0, 0]), [0, 0, 0])
        np.testing.assert_allclose(motion.update(1.5, [0, 1, 0], [0, 0, 0]), [0, 0, 0])
        self.assertTrue(motion.stopped)

    def test_jump_occurs_once_and_guard_blocks_crossing(self):
        motion = self.make_motion('jump')
        np.testing.assert_allclose(motion.update(0.9, [0, 1, 0], [0, 0, 0]), [0, 0, 0])
        np.testing.assert_allclose(motion.update(1.0, [0, 1, 0], [0, 0, 0]), [0.06, 0, 0])
        np.testing.assert_allclose(motion.update(1.1, [0, 1, 0], [0.06, 0, 0]), [0.06, 0, 0])
        blocked = self.make_motion('jump')
        np.testing.assert_allclose(blocked.update(1.0, [0.1, 0, 0], [0, 0, 0]), [0, 0, 0])
        self.assertEqual(blocked.status, 'blocked: proposed path near peg')

    def test_window_expiration_prevents_late_start(self):
        motion = self.make_motion()
        np.testing.assert_allclose(motion.update(3.0, [0, 1, 0], [0, 0, 0]), [0, 0, 0])
        np.testing.assert_allclose(motion.update(3.1, [0, 1, 0], [0, 0, 0]), [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
