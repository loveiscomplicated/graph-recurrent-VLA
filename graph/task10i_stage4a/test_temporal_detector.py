"""Temporal visibility and pixel-coordinate contracts independent of model."""
import unittest
import numpy as np
from .temporal_audit import runs,sequence_metrics,action_phases
from .detector_audit import pixel_boxes,bbox

class TemporalDetector(unittest.TestCase):
    def test_raw_coordinate_mapping(self):
        np.testing.assert_array_equal(pixel_boxes(np.array([[.5,.5,.5,.5]])),[[32,32,96,96]])
    def test_mask_bbox_uses_pixel_edges(self):
        mask=np.zeros((128,128),bool);mask[4:7,8:11]=True
        self.assertEqual(bbox(mask),[8,4,11,7]);self.assertIsNone(bbox(~np.ones((2,2),bool)))
    def test_absence_is_not_visible_detection_failure(self):
        m=sequence_metrics([0,0,10,10],[0,0,0,3],[True,False,False,True])
        self.assertEqual(m['visible_frames'],2);self.assertEqual(m['visible_missing_frames'],1)
        self.assertEqual(m['false_presence_absent_frames'],1);self.assertEqual(m['reappearances'][0]['delay_frames'],1)
    def test_unrecovered_reappearance_is_explicit(self):
        m=sequence_metrics([0,10,10],[0,0,0],[False,False,False])
        self.assertFalse(m['reappearances'][0]['recovered']);self.assertIsNone(m['reappearances'][0]['delay_frames'])
    def test_fragmentation_counts_segments(self):
        m=sequence_metrics([10]*5,[3,0,2,0,1],[True,False,True,False,True])
        self.assertEqual(m['temporal_fragment_count'],3);self.assertEqual(runs([True,True,False,True]),[[0,2],[3,4]])
    def test_positive_t0_is_not_reappearance(self):
        self.assertEqual(sequence_metrics([10,10],[4,4],[True,True])['reappearances'],[])
    def test_legacy_suffix_without_phase_keeps_full_raw_episode_unknown(self):
        rows=[dict(clock=t+1,segment=1 if t<24 else 2) for t in range(10,51)]
        self.assertEqual(action_phases(rows,51),['EXPERT_PHASE_UNAVAILABLE']*51)
    def test_partial_phase_labels_are_not_shifted_or_guessed(self):
        with self.assertRaises(ValueError):action_phases([dict(clock=11,phase='reach')],51)
    def test_complete_action_phase_metadata_preserves_original_labels(self):
        self.assertEqual(action_phases([dict(phase='reach'),dict(phase='close')],2),['reach','close'])
if __name__=='__main__':unittest.main()
