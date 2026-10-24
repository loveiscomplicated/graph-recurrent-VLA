"""Adaptive total-budget contracts; shared sampler/evaluator remain unchanged."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from visual.runtime.geometry import build_organized_geometry
from visual.runtime.rgbd import PEG, BOX
from visual.runtime.sampling import SamplingConfig, extract_candidates, extract_sampling_evidence, sample_points
from visual.runtime.test.normal_diagnostics import synthetic_frame, synthetic_case
from visual.runtime.test.sampling_evaluation import CoverageMatcher
from visual.runtime.test.run_adaptive_node_budget import (
    AreaBudget, requested_budget, matched_budget, measure_areas, exact_result,
    exact_saved, validate_selection, save_evidence, load_evidence, distance_summary,
    nearest, distribution, latency_summary, sampling_timing, preserve_check,
)

K = np.array([[100.,0,16.],[0,100.,16.],[0,0,1.]])


class AdaptiveBudgetTests(unittest.TestCase):
    def frame(self, counts=(200,200)):
        a,b=counts
        masks={PEG:np.zeros((32,32),bool),BOX:np.zeros((32,32),bool)}
        masks[PEG].ravel()[:a]=True
        masks[BOX].ravel()[512:512+b]=True
        return replace(synthetic_frame(np.full((32,32),.4,np.float32),K,masks[PEG]),masks=masks)

    def test_first_positive_reference_never_resets(self):
        p=AreaBudget()
        self.assertEqual(p.step((0,0),0),(0,(0,0)))
        self.assertIsNone(p.reference)
        self.assertEqual(p.step((100,100),1)[0],256)
        self.assertEqual(p.step((50,50),2)[0],128)
        self.assertEqual(p.step((0,0),3)[0],0)
        self.assertEqual(p.step((0,200),4)[0],256)
        self.assertEqual((p.reference,p.reference_source),(200,1))

    def test_half_up_minimum_maximum_and_invalid_inputs(self):
        # 256*81/512 = 40.5 -> 41, rather than ties-to-even 40.
        self.assertEqual(requested_budget((40,41),512)[0],41)
        self.assertEqual(requested_budget((16,16),100000)[0],32)
        self.assertEqual(requested_budget((0,20),100000),(16,(0,16)))
        self.assertEqual(requested_budget((2,3),100000),(5,(2,3)))
        self.assertEqual(requested_budget((1000,1000),100)[0],256)
        for counts in ((-1,0),(True,0),(1.2,0),(1,)):
            with self.assertRaises(ValueError):requested_budget(counts,10)
        with self.assertRaises(ValueError):requested_budget((1,0),None)
        with self.assertRaises(ValueError):requested_budget((0,0),0)
        with self.assertRaises(ValueError):SamplingConfig(num_points=16)

    def test_causal_prefix_and_future_independence(self):
        prefix=[(0,0),(100,200),(60,50),(0,0)]
        def run(sequence):
            p=AreaBudget()
            return [p.step(c,i) for i,c in enumerate(sequence)]
        self.assertEqual(run(prefix),run(prefix+[(100000,100000),(1,1)])[:len(prefix)])
        self.assertEqual(run(prefix),run(prefix+[(0,0),(0,4)])[:len(prefix)])
        values=[requested_budget((i,0),500)[0] for i in range(1000)]
        self.assertEqual(values,sorted(values))

    def test_raw_valid_partition_invalid_and_overlap(self):
        f=self.frame((200,100));m={k:v.copy() for k,v in f.masks.items()}
        m[BOX].ravel()[0:5]=True
        d=f.depth.copy();d.ravel()[5:10]=[0,-1,np.nan,np.inf,-np.inf]
        f=replace(f,depth=d,masks=m)
        e=extract_sampling_evidence(f)
        areas,count_ms,raw_ms=measure_areas(f,e.frame.geometry)
        self.assertEqual(areas['peg']['raw'],200)
        self.assertEqual(areas['peg']['valid'],190)
        self.assertEqual(areas['peg']['invalid_depth'],5)
        self.assertEqual(areas['peg']['positive_depth_overlap'],5)
        self.assertEqual(areas['box']['valid'],100)
        self.assertGreaterEqual(count_ms,0);self.assertGreaterEqual(raw_ms,0)
        c=extract_candidates(e,SamplingConfig())
        self.assertEqual(len(c[PEG].points),areas['peg']['valid'])

    def test_sampler_empty_sparse_capacity_and_alignment(self):
        for counts in ((0,0),(3,4),(0,5),(100,0),(200,200)):
            f=self.frame(counts);e=extract_sampling_evidence(f)
            n,mins=requested_budget(counts,100000)
            cfg=SamplingConfig(num_points=n,min_points=mins)
            s=sample_points(e,cfg)
            c=extract_candidates(e,cfg)
            validate_selection(s,e,cfg,c)
            self.assertEqual(len(s.points),min(n,sum(counts)))
            self.assertEqual(s.points.shape,(len(s.points),3))
            self.assertEqual(len(np.unique(s.flat_index)),len(s.flat_index))
            self.assertLessEqual(s.priority_selected.sum(),n//4)
            self.assertTrue(all(s.counts[o]['selected']>=min(16,counts[o]) for o in (PEG,BOX)))
            np.testing.assert_array_equal(f.depth,np.full((32,32),.4,np.float32))

    def test_256_reproduction_including_support_shortfall(self):
        for counts in ((200,200),(0,100),(3,4),(0,0)):
            e=extract_sampling_evidence(self.frame(counts))
            a=sample_points(e,SamplingConfig())
            b=sample_points(e,SamplingConfig(num_points=256,min_points=tuple(min(16,c) for c in counts)))
            exact_result(a,b)
            archive={f'baseline/{field}':getattr(a,field) for field in (
                'points','pixels_vu','flat_index','object_id','priority_selected','depth_priority_selected')}
            archive.update({f'baseline/metadata/{k}':v for k,v in a.metadata.items()})
            class Archive(dict):
                @property
                def files(self):return list(self)
            exact_saved(b,Archive(archive),'baseline')
            archive['baseline/points']=a.points+1
            if len(a.points):
                with self.assertRaises(AssertionError):exact_saved(b,Archive(archive),'baseline')

    def test_requested_priority_cap_and_roles(self):
        f,_,_=synthetic_case('crease',K,shape=(32,32))
        e=extract_sampling_evidence(f)
        for n in (16,41,64,256):
            cfg=SamplingConfig(num_points=n,min_points=(16,0))
            s=sample_points(e,cfg)
            validate_selection(s,e,cfg,extract_candidates(e,cfg))
            capacity=sum(int(c.metadata['structural_candidate'].sum()) for c in extract_candidates(e,cfg).values())
            self.assertEqual(int(s.priority_selected.sum()),min(n//4,capacity))
            self.assertEqual(s.counts[PEG]['selected'],s.counts[PEG]['priority']+s.counts[PEG]['surface_fps'])

    def test_gt_annotation_and_reference_cannot_affect_selection(self):
        f=self.frame();e=extract_sampling_evidence(f)
        p=AreaBudget();n,mins=p.step((200,200),0)
        cfg=SamplingConfig(num_points=n,min_points=mins)
        before=sample_points(e,cfg)
        g=e.frame.geometry
        for shift in (0.,100.):
            reference=(np.array([[0.,0.]]),g.xyz[0,0][None]+shift)
            CoverageMatcher(g,PEG,reference).score(before.flat_index[before.object_id==PEG])
        after=sample_points(e,cfg)
        exact_result(before,after)
        self.assertEqual(p.reference,400)
        with self.assertRaises(TypeError):sample_points(e,cfg,gt={})

    def test_lossless_evidence_cache_and_source_preservation(self):
        e=extract_sampling_evidence(self.frame());a=sample_points(e)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'e.npz';save_evidence(path,e);loaded=load_evidence(path)
            exact_result(a,sample_points(loaded))
            np.testing.assert_array_equal(e.frame.geometry.xyz,loaded.frame.geometry.xyz)
        from visual.runtime.test.run_adaptive_node_budget import REPO
        relative='visual/runtime/sampling.py'
        from utils.experiment_artifacts import digests
        self.assertTrue(preserve_check({'sha256':{relative:digests(REPO/relative)[1]}})['unchanged'])
        self.assertFalse(preserve_check({'sha256':{relative:'incorrect'}})['unchanged'])

    def test_fixedmatched_capacity_tie_and_count_only(self):
        m=matched_budget([1000,1000],[100,101])
        self.assertEqual(m['budget'],100)
        self.assertEqual(m['mean_error_nodes'],-.5)
        self.assertFalse(m['causal_online'])
        self.assertEqual(matched_budget([5,1000],[5,100])['budget'],100)
        self.assertEqual(matched_budget([5,5],[5,5])['budget'],32)
        self.assertEqual(matched_budget([1000],[16])['mean_error_nodes'],16)

    def test_missing_distance_ratios_distributions_and_timer_aliases(self):
        a=distance_summary(nearest(np.ones((3,3)),np.empty((0,3))))
        self.assertEqual(a['missing_count'],3);self.assertEqual(a['coverage_3mm'],0)
        self.assertIsNone(distance_summary(np.empty(0))['coverage_3mm'])
        self.assertEqual(distribution([1,2,3,None,np.nan])['mean'],2)
        self.assertEqual(distribution([1,2,3])['min'],1)
        raw=[]
        for frame in range(2):
            for repeat in range(3):
                for c in ('Fixed256','Adaptive','FixedMatched'):
                    alias='Fixed256' if c=='Adaptive' else None
                    raw.append(dict(source_frame=frame,repeat=repeat,condition=c,executed=alias is None,
                        alias_of=alias,timings_ms=None if alias else {'sampling_total':float(frame+repeat+1)}))
        s=latency_summary(raw)
        self.assertEqual(s['Adaptive']['executed_calls'],0)
        self.assertEqual(s['Adaptive']['alias_calls'],6)
        self.assertEqual(s['Fixed256']['frame_median_then_distribution']['sampling_total']['mean'],2.5)
        self.assertEqual(s['Adaptive']['frame_median_then_distribution'],s['Fixed256']['frame_median_then_distribution'])
        t=sampling_timing(sample_points(extract_sampling_evidence(self.frame())))
        self.assertGreaterEqual(t['gather_packaging_and_other_overhead'],0)


if __name__ == '__main__':
    unittest.main()
