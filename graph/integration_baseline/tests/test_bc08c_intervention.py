"""Controller-boundary intervention and fixed prefix numeric gate."""
import unittest
import numpy as np

from graph.integration_baseline.bc08c_intervention import CONDITIONS, compare_arrays, intervene


class InterventionContract(unittest.TestCase):
    def test_only_requested_gripper_component_changes(self):
        raw=np.array([.1,-.2,.3,-.4,.5,-.6,.1511],np.float32)
        original=raw.copy()
        for condition in CONDITIONS:
            for t in (78,79,80,249):
                applied,active=intervene(raw,condition,t)
                np.testing.assert_array_equal(applied[:6],raw[:6])
                expected=condition==CONDITIONS[2] and t>=79 or condition==CONDITIONS[1] and t==79
                self.assertEqual(active,expected)
                self.assertEqual(applied[6],-1 if expected else raw[6])
        np.testing.assert_array_equal(raw,original)

    def test_fixed_quantization_and_physical_gate(self):
        self.assertTrue(compare_arrays(np.array([100],np.uint8),np.array([102],np.uint8),'rgb')['pass_gate'])
        self.assertFalse(compare_arrays(np.array([100],np.uint8),np.array([103],np.uint8),'rgb')['pass_gate'])
        self.assertTrue(compare_arrays(np.array([0.],np.float32),np.array([1e-6],np.float32),'qpos')['pass_gate'])
        self.assertFalse(compare_arrays(np.array([0.],np.float32),np.array([2e-6],np.float32),'qpos')['pass_gate'])


if __name__=='__main__':unittest.main()
