"""Weak-reference labels must never silently relax exact RGB-D equality."""
from copy import deepcopy
import unittest
from .approximate_reference import eligible
from .instrumentation import reference_kind
from .run_pipeline import CAMERAS


def comparisons():
    return {camera:{key:dict(exact=True,different_values=0,max_abs=0.)
        for key in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl')}
        for camera in CAMERAS}


class ApproximateReference(unittest.TestCase):
    def test_rgb_ulp_reference_remains_unsupported_for_primary(self):
        fields=comparisons()
        fields['hand_camera']['rgb'].update(exact=False,different_values=2,max_abs=1.)
        original=deepcopy(fields)
        self.assertTrue(eligible(fields))
        self.assertEqual(reference_kind(fields),'GT_UNSUPPORTED')
        self.assertEqual(fields,original)

    def test_rgb_limits_are_both_required(self):
        for count,delta in ((33,1.),(1,2.)):
            fields=comparisons()
            fields['base_camera']['rgb'].update(exact=False,different_values=count,max_abs=delta)
            self.assertFalse(eligible(fields))

    def test_depth_and_each_calibration_field_stay_exact(self):
        for key in ('depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'):
            fields=comparisons();fields['hand_camera'][key]['exact']=False
            self.assertFalse(eligible(fields))

    def test_both_cameras_are_required(self):
        fields=comparisons();del fields['hand_camera']
        self.assertFalse(eligible(fields))


if __name__=='__main__':unittest.main()
