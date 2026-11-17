"""Coordinate frames, quality gates and failure handling for hole-position memory."""

import unittest

import numpy as np

from graph.utils.hole_memory import (
    HolePositionMemory, MemorySettings, ObservationQuality, replace_hole_position,
)


GOOD = ObservationQuality(6, 0.9, 1.0)
MISSING = ObservationQuality(0, 0.0, 0.0)


class HoleMemoryTests(unittest.TestCase):
    def test_world_hold_tracks_current_tcp_and_preserves_peg_tip(self):
        memory = HolePositionMemory(MemorySettings(consecutive_frames=1))
        first = memory.step(0, [0.1, 0.2, 0.3], [0.5, 0, 0], GOOD)
        self.assertTrue(first.updated)
        held = memory.step(1, [9, 9, 9], [0.7, 0.1, 0.2], MISSING)
        np.testing.assert_allclose(held.hole_relative, [-0.1, 0.1, 0.1])
        self.assertTrue(held.held)
        self.assertEqual(held.age_frames, 1)
        raw = np.arange(21, dtype=float)
        corrected = replace_hole_position(raw, held.hole_relative)
        np.testing.assert_allclose(corrected[7:10] + corrected[10:13], raw[7:10] + raw[10:13])
        np.testing.assert_array_equal(corrected[13:], raw[13:])
        np.testing.assert_array_equal(raw, np.arange(21))


    def test_cold_start_returns_current_prediction_and_unknown_quality_is_rejected(self):
        memory = HolePositionMemory(MemorySettings(consecutive_frames=1))
        unknown = ObservationQuality(8, 0.0, 0.0)
        result = memory.step(0, [0.1, 0.2, 0.3], [0, 0, 0], unknown)
        self.assertFalse(result.initialized)
        self.assertEqual(result.age_frames, -1)
        np.testing.assert_allclose(result.hole_relative, [0.1, 0.2, 0.3])


    def test_stability_window_uses_world_coordinates_and_causal_median(self):
        memory = HolePositionMemory(MemorySettings())
        for frame, (relative, tcp) in enumerate(((0.5, 0.5), (0.4, 0.6), (0.302, 0.7))):
            result = memory.step(frame, [relative, 0, 0], [tcp, 0, 0], GOOD)
            self.assertEqual(result.initialized, frame == 2)
        np.testing.assert_allclose(result.hole_relative, [0.3, 0, 0], atol=1e-12)


    def test_missing_quality_or_frame_gap_breaks_consecutive_candidates(self):
        memory = HolePositionMemory(MemorySettings(consecutive_frames=2))
        memory.step(0, [1, 0, 0], [0, 0, 0], GOOD)
        memory.step(1, [1, 0, 0], [0, 0, 0], MISSING)
        self.assertFalse(memory.step(2, [1, 0, 0], [0, 0, 0], GOOD).initialized)
        self.assertFalse(memory.step(4, [1, 0, 0], [0, 0, 0], GOOD).initialized)
        self.assertTrue(memory.step(5, [1, 0, 0], [0, 0, 0], GOOD).initialized)


    def test_outlier_is_not_stored_and_reobservation_recovers(self):
        memory = HolePositionMemory(MemorySettings())
        results = [memory.step(i, [x, 0, 0], [0, 0, 0], GOOD)
                   for i, x in enumerate((0.0, 0.1, 0.0, 0.0, 0.0))]
        self.assertTrue(all(not r.initialized for r in results[:4]))
        self.assertTrue(results[4].updated)
        held = memory.step(5, [5, 5, 5], [0, 0, 0], MISSING)
        np.testing.assert_allclose(held.hole_relative, [0, 0, 0])


    def test_bad_inputs_and_reverse_time_are_rejected(self):
        memory = HolePositionMemory(MemorySettings(consecutive_frames=1))
        memory.step(2, [0, 0, 0], [0, 0, 0], GOOD)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            memory.step(1, [0, 0, 0], [0, 0, 0], GOOD)
        with self.assertRaisesRegex(ValueError, 'finite'):
            memory.step(3, [np.nan, 0, 0], [0, 0, 0], GOOD)
        with self.assertRaises(ValueError):
            MemorySettings(consecutive_frames=0)



if __name__ == '__main__':
    unittest.main()
