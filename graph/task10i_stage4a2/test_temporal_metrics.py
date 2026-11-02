import unittest
from .temporal_mask_diagnostics import temporal,runs


def row(t,visible,predicted,tp,exact=True):
    return dict(timestep=t,reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',
        visible_pixels=visible,predicted_pixels=predicted,tp=tp,precision=tp/predicted if predicted else None,
        robot_pixels=predicted-tp,wrong_object_pixels=0,
        correct_mask_availability=float(visible>=8 and predicted>0 and tp/predicted>=.9 and tp/(visible+predicted-tp)>=.5))


class TemporalReferenceContracts(unittest.TestCase):
    def test_unknown_reference_breaks_clock_runs(self):
        rows=[row(0,20,0,0),row(1,20,0,0,False),row(2,20,0,0)]
        result=temporal(rows)
        self.assertEqual(result['empty_visible_runs'],[dict(start=0,end=0,frames=1),dict(start=2,end=2,frames=1)])
        self.assertEqual(result['visible_frames'],2)

    def test_initial_contamination_is_not_a_pristine_tracking_drift(self):
        result=temporal([row(0,20,20,0),row(1,20,20,20),row(2,20,20,0)])
        self.assertTrue(result['initial_wrong_identity'])
        self.assertEqual(len(result['wrong_identity_onsets_after_earlier_correct_mask']),1)
        self.assertEqual(result['wrong_identity_onsets_after_earlier_correct_mask'][0]['start'],2)

    def test_reappearance_delay_is_posthoc_and_preserves_ghost_duration(self):
        result=temporal([row(0,20,20,20),row(1,0,20,0),row(2,0,20,0),row(3,20,0,0),row(4,20,20,20)])
        self.assertEqual(result['ghost_frames'],2)
        self.assertEqual(result['occlusion_reappearance_events'][0]['delay_frames'],1)
        self.assertTrue(result['occlusion_reappearance_events'][0]['recovered'])


if __name__=='__main__':unittest.main()
