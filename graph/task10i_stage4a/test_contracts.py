import unittest
import numpy as np
from .instrumentation import support,reference_kind
from .run_pipeline import rank

class Contracts(unittest.TestCase):
    def test_source_identity_contamination(self):
        seg=np.array([[1,1,2],[3,0,0]])
        row=support(np.ones_like(seg,bool),seg,0,[3],[[1],[2]])
        self.assertEqual((row['true_positive'],row['wrong_object_pixels'],row['robot_pixels'],row['background_pixels']),(2,1,1,2))
        self.assertEqual(row['precision'],1/3)
    def test_empty_object_control(self):
        row=support(np.zeros((2,2),bool),np.zeros((2,2),int),0,[3],[[1],[2]])
        self.assertIsNone(row['recall']);self.assertIsNone(row['precision'])
    def test_false_positive_control(self):
        row=support(np.ones((2,2),bool),np.zeros((2,2),int),0,[3],[[1],[2]])
        self.assertEqual(row['false_positive'],4);self.assertEqual(row['precision'],0)
    def test_fully_visible_positive_control(self):
        row=support(np.ones((2,2),bool),np.ones((2,2),int),0,[3],[[1],[2]])
        self.assertEqual((row['precision'],row['recall'],row['iou']),(1,1,1))
    def test_no_unsupported_reference_upgrade(self):
        self.assertEqual(reference_kind({}), 'GT_UNSUPPORTED')
        self.assertEqual(reference_kind({'main':{'rgb':{'exact':False}}}),'GT_UNSUPPORTED')
    def test_incomplete_camera_reference_rejected(self):
        self.assertEqual(reference_kind({'base_camera':{},'hand_camera':{}}),'GT_UNSUPPORTED')
    def test_both_camera_reference_required(self):
        fields={k:{'exact':True} for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl')}
        self.assertEqual(reference_kind({'base_camera':fields}),'GT_UNSUPPORTED')
        self.assertEqual(reference_kind({'base_camera':fields,'hand_camera':fields}),'EXACT_GT')
    def test_pixel_alignment_rejected(self):
        with self.assertRaises(ValueError):support(np.zeros((3,2)),np.zeros((2,2)),0,[],[[1],[2]])
    def test_parent_split_deterministic(self):
        self.assertEqual(rank('parent'),rank('parent'));self.assertNotEqual(rank('parent'),rank('branch'))
if __name__=='__main__':unittest.main()
