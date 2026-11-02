import unittest
from .stratified_evaluation import aggregation_units


class AggregationContracts(unittest.TestCase):
    def test_repeated_parent_and_unequal_episode_lengths_have_distinct_units(self):
        rows=[]
        for parent,episode,n,gain in [('p1','e1',1,1),('p1','e2',100,0),('p2','e3',1,0)]:
            for t in range(n):
                for condition,value in [('B0',0),('C',gain)]:
                    rows.append(dict(parent_id=parent,episode=episode,timestep=t,condition=condition,
                        reference_kind='EXACT_GT',metric=value))
        result=next(r for r in aggregation_units(rows,'metric',(),'C') if r['condition']=='C')
        self.assertAlmostEqual(result['frame_weighted'],1/102)
        self.assertAlmostEqual(result['episode_macro'],1/3)
        self.assertEqual(result['parent_macro'],.25)
        self.assertEqual(result['paired_parents'],2)

    def test_empty_precision_is_excluded_only_from_defined_paired_precision(self):
        rows=[dict(parent_id='p',episode='e',timestep=t,condition=c,reference_kind='EXACT_GT',
            precision=value) for t,b,a in [(0,1,None),(1,.8,.9)] for c,value in [('B0',b),('C',a)]]
        result=aggregation_units(rows,'precision',(),'C')
        self.assertEqual({r['paired_defined_frames'] for r in result},{1})
        self.assertAlmostEqual(next(r for r in result if r['condition']=='C')['parent_macro'],.9)


if __name__=='__main__':unittest.main()
