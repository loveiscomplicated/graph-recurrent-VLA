"""Check the two ablations change only the intended filter exclusions."""
import unittest
import numpy as np
import policies
from graph.persistent_geometry.fusion import filtered_masks


class TestPolicies(unittest.TestCase):
    def test_confirmed_only_keeps_uncertain_but_retains_overlap(self):
        masks=np.array([[[1,1,1,0],[0,1,0,0]],[[0,1,0,1],[0,0,0,0]]],bool)
        result=dict(confirmed_robot=np.array([[1,0,0,1],[0,0,0,0]],bool),
                    uncertain=np.array([[0,0,1,0],[0,1,0,0]],bool))
        original=masks.copy();full,_=filtered_masks(masks,result);part,excluded=policies.confirmed_only(masks,result)
        np.testing.assert_array_equal(masks,original)
        self.assertFalse(part[0,0,0]);self.assertFalse(part[0,0,1]);self.assertFalse(part[1,0,1])
        self.assertTrue(part[0,0,2]);self.assertFalse(full[0,0,2]);self.assertTrue(excluded[0,0])
        self.assertTrue((~full|part).all())

    def test_no_filter_preserves_overlap_exclusion(self):
        masks=np.array([[[1,1],[0,0]],[[0,1],[1,0]]],bool)
        camera=type('Camera',(),{'depth_raw_mm':np.zeros((2,2),np.int16)})()
        out,_=filtered_masks(masks,policies.no_robot_filter(camera,[]))
        np.testing.assert_array_equal(out,np.array([[[1,0],[0,0]],[[0,0],[1,0]]],bool))


if __name__=='__main__':unittest.main()
