import unittest
import numpy as np
from .depth_component import dominant_component_keep


class DepthComponentControls(unittest.TestCase):
    def test_isolated_far_depth_contamination_is_removed(self):
        surface=np.column_stack((np.arange(20)*.001,np.zeros(20),np.ones(20)))
        xyz=np.vstack((surface,[[1.,1.,2.],[1.001,1.,2.]]))
        keep,_=dominant_component_keep(xyz)
        self.assertTrue(keep[:20].all());self.assertFalse(keep[20:].any())
    def test_two_ambiguous_surfaces_are_preserved(self):
        a=np.column_stack((np.arange(10)*.001,np.zeros(10),np.ones(10)))
        keep,_=dominant_component_keep(np.vstack((a,a+[1,0,0])))
        self.assertTrue(keep.all())
    def test_empty_and_sparse_support_are_not_destroyed(self):
        for xyz in (np.empty((0,3)),np.array([[0,0,1],[1,0,1]])):
            keep,_=dominant_component_keep(xyz);self.assertTrue(keep.all())
    def test_continuous_surface_is_preserved(self):
        xyz=np.column_stack((np.arange(30)*.006,np.zeros(30),np.ones(30)))
        keep,_=dominant_component_keep(xyz);self.assertTrue(keep.all())


if __name__=='__main__':unittest.main()
