import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.sampling import fps_indices
from visual.runtime.test.run_priority_fps_trace import trace_order


class PriorityFPSTraceTests(unittest.TestCase):
    def test_exact_tie_and_preselection_competition(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 2., 0.]])
        order = np.array([0, 3, 1, 2])
        refs = np.array([[0, 0], [1, 0], [0, 1], [0, 0]], bool)
        with patch('visual.runtime.test.run_priority_fps_trace.np.random.default_rng',
                   side_effect=AssertionError('trace must not consume RNG')):
            trace = trace_order(points, order, np.arange(4), np.ones(4), refs)
        self.assertIsNone(trace[0]['argmax_value_m2'])
        self.assertEqual(trace[1]['rim_A']['max_min_squared_distance_m2'], 1.)
        self.assertEqual(trace[1]['uncovered_rim_B']['remaining_count'], 2)
        self.assertEqual(trace[2]['exact_tie_count'], 2)
        self.assertEqual(trace[2]['tied_local_indices'], [1, 2])
        self.assertEqual(trace[2]['first_tied_local_index'], 1)
        self.assertEqual(trace[2]['new_reference_ids'], [0])
        self.assertEqual(trace[-1]['cumulative_reference_count'], 2)

    def test_wrong_tie_winner_fails(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 2., 0.]])
        with self.assertRaisesRegex(AssertionError, 'first argmax'):
            trace_order(points, [0, 3, 2, 1], np.arange(4), np.ones(4), np.zeros((4, 0), bool))

    def test_fps_consumes_one_initial_integer_draw(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        actual_factory = np.random.default_rng
        calls = {'factory': 0, 'integers': 0}
        class Spy:
            def __init__(self, seed): self.inner = actual_factory(seed)
            def integers(self, *args, **kwargs):
                calls['integers'] += 1
                return self.inner.integers(*args, **kwargs)
        def factory(seed):
            calls['factory'] += 1
            return Spy(seed)
        with patch('visual.runtime.sampling.np.random.default_rng', side_effect=factory):
            order = fps_indices(points, 4, seed=17)
        self.assertEqual(calls, {'factory': 1, 'integers': 1})
        self.assertEqual(int(order[0]), int(actual_factory(17).integers(len(points))))

    def test_duplicate_xyz_and_annotation_independence(self):
        points = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]], np.float32)
        order = fps_indices(points, 4, seed=5)
        self.assertEqual(len(np.unique(order)), 4)
        flat = np.arange(4)
        refs_a = np.zeros((4, 2), bool)
        refs_b = np.ones((4, 2), bool)
        a = trace_order(points, order, flat, np.ones(4), refs_a)
        b = trace_order(points, order, flat, np.ones(4), refs_b, {'negative': np.ones(4, bool)})
        self.assertEqual([x['candidate_local_index'] for x in a], [x['candidate_local_index'] for x in b])
        self.assertEqual([x['argmax_value_m2'] for x in a], [x['argmax_value_m2'] for x in b])
        np.testing.assert_array_equal(order, fps_indices(points, 4, seed=5))


if __name__ == '__main__':
    unittest.main()
