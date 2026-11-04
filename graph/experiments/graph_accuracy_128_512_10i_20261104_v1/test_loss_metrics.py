import unittest
import numpy as np
from .loss_metrics import covered, transition, high_resolution_gain, retention


class InformationLossControls(unittest.TestCase):
    def test_false_point_does_not_retain_a_reference(self):
        p = np.array([[0., 0., 0.], [1., 0., 0.]])
        np.testing.assert_array_equal(covered(p, [True, False], p), [True, False])
        np.testing.assert_array_equal(covered(p[:0], [], p), [False, False])

    def test_transition_accounts_for_lost_and_recovered_samples(self):
        v = transition([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((v['lost'], v['recovered'], v['net_loss_fraction']), (1, 1, 0.))
        v = transition([1, 1], [1, 0], [False, True])
        self.assertEqual((v['reference_samples'], v['loss_fraction']), (1, 1.))

    def test_gain_uses_same_reference_and_undefined_denominator(self):
        gain = high_resolution_gain([1, 0, 0], [1, 1, 0])
        np.testing.assert_array_equal(gain, [0, 1, 0])
        self.assertEqual(retention([0, 1, 0], gain), 1.)
        self.assertIsNone(retention([0, 0], [0, 0]))
        with self.assertRaises(ValueError):
            high_resolution_gain([1], [1, 0])


if __name__ == '__main__':
    unittest.main()
