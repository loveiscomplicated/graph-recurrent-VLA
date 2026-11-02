import unittest
import numpy as np
from .registered_evaluation import composite, derived_pixel_rows, t0_both_missing, gates
from .evaluation_metrics import hierarchical_paired
from graph.task10i_stage4a.run_pipeline import read
from pathlib import Path


class RegisteredEvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.criteria=read(Path(__file__).with_name('adoption_criteria.json'))

    def test_composite_weights_slots_inside_parent_not_frame_count(self):
        def part(values):
            return dict(parent_values=[dict(parent_id=p,baseline=a,candidate=b) for p,a,b in values])
        result=composite([part([('p1',0.,1.),('p2',1.,1.)]),part([('p1',1.,1.)])], 'm', self.criteria)
        self.assertEqual(result['baseline'],.75)
        self.assertEqual(result['candidate'],1.)
        self.assertEqual(result['delta'],.25)
        self.assertEqual(result['paired_parents'],2)
        np.testing.assert_array_equal(result['ci95'],[0.,.5])

    def test_parent_and_episode_macro_ignore_long_episode_weight(self):
        rows=[]
        for episode,n,baseline in [('short',1,0.),('long',100,1.)]:
            for t in range(n):
                for condition,value in [('B0',baseline),('P',1.)]:
                    rows.append(dict(parent_id='same',episode=episode,timestep=t,condition=condition,m=value))
        result=hierarchical_paired(rows,'m',(),candidate='P',bootstrap=100)
        self.assertEqual(result[0]['baseline'],.5)
        self.assertEqual(result[0]['paired_parents'],1)

    def test_empty_prediction_damage_has_denominator_but_no_precision(self):
        shared=dict(parent_id='p',episode='e',timestep=0,camera='base_camera',object_id=0,
                    reference_kind='EXACT_GT',visible_pixels=20)
        rows=[dict(shared,condition='B0',predicted_pixels=20,precision=1.,tp=20,fp=0,correct_mask_availability=1.),
              dict(shared,condition='P',predicted_pixels=0,precision=None,tp=0,fp=0,correct_mask_availability=0.)]
        result=derived_pixel_rows(rows,'P')
        self.assertIsNone(result[1]['precision'])
        self.assertEqual(result[1]['healthy_tracking_damage'],1.)
        self.assertEqual(result[1]['visible_wrong_mask_rate'],0.)
        self.assertEqual(result[1]['false_pixels_per_visible_GT_pixel'],0.)
        self.assertEqual(hierarchical_paired(result,'precision',candidate='P'),[])
        result[0]['correct_mask_availability']=0.
        rows[0]['correct_mask_availability']=0.
        self.assertIsNone(derived_pixel_rows(rows,'P')[1]['healthy_tracking_damage'])

    def test_t0_requires_both_exact_visible_and_never_future(self):
        rows=[]
        for condition in ['B0','P']:
            for camera in ['base_camera','hand_camera']:
                rows.append(dict(episode='e',parent_id='p',condition=condition,camera=camera,timestep=0,
                    object_id=0,reference_kind='EXACT_GT',visible_pixels=8,tp=0))
        self.assertEqual(len(t0_both_missing(rows,'P')),2)
        rows[-1]['visible_pixels']=0
        self.assertEqual(len(t0_both_missing(rows,'P')),1)
        rows[0]['reference_kind']='APPROXIMATE_REFERENCE'
        self.assertEqual(t0_both_missing(rows,'P'),[])

    def test_missing_strata_do_not_silently_pass_safety(self):
        checks=gates([],dict(baseline=.8,candidate=.9,delta=.1,ci95=[.05,.15]),
                     dict(delta=.02,ci95=[.01,.03]),self.criteria)
        self.assertTrue(all(r['status']=='PASS' for r in checks[:5]))
        self.assertTrue(all(r['status']=='INCONCLUSIVE' for r in checks[5:]))


if __name__=='__main__':unittest.main()
