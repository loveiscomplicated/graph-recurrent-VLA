"""Task03 denominator / joint-tolerance / visible-only rim evaluation contracts."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from visual.runtime.geometry import build_organized_geometry
from visual.runtime.rgbd import PEG, BOX
from visual.runtime.sampling import extract_sampling_evidence, sample_points, SamplingConfig
from visual.runtime.test.normal_diagnostics import synthetic_frame, oracle, box_primitives, rim_region
from visual.runtime.test.run_structural_edges import rim_reference
from visual.runtime.test.sampling_evaluation import CoverageMatcher
from visual.runtime.test.run_sampling import conditions, evaluate


class SamplingEvaluationTests(unittest.TestCase):
    def test_tracking_and_sampling_losses_have_separate_denominators(self):
        k=np.array([[100.,0,8],[0,100.,8],[0,0,1]])
        mask=np.zeros((16,16),bool);mask[5,5]=True;mask[5,6]=True
        frame=synthetic_frame(np.full((16,16),.3,np.float32),k,mask)
        g=build_organized_geometry(frame)
        uv=np.array([[5,5],[6,5],[12,12]],float)
        xyz=g.xyz[uv[:,1].astype(int),uv[:,0].astype(int)]
        matcher=CoverageMatcher(g,PEG,(uv,xyz),((.1,.0001),))
        row,_=matcher.score(np.array([5*16+5]))
        r=row['tolerance']['0.1px_0.1mm']
        self.assertEqual((r['gt_visible'],r['tracker_observed'],r['sampled']),(3,2,1))
        self.assertEqual(r['gt_coverage'],1/3);self.assertEqual(r['sampling_retention'],.5)
        row,arrays=matcher.score(np.array([],np.int64))
        self.assertEqual(row['tolerance']['0.1px_0.1mm']['sampled'],0)
        self.assertEqual(row['missing_reference_distances'],3)
        self.assertEqual(len(arrays['gt_to_sample_mm']),0)

    def test_joint_tolerance_requires_same_pair(self):
        k=np.array([[100.,0,8],[0,100.,8],[0,0,1]])
        mask=np.zeros((16,16),bool);mask[5,5]=True
        g=build_organized_geometry(synthetic_frame(np.full((16,16),.3,np.float32),k,mask))
        # One reference is near only in image; the other only in 3D.
        refs=(np.array([[5.,5.],[12.,12.]]),np.array([g.xyz[5,5]+[0,0,.1],g.xyz[5,5]]))
        row,_=CoverageMatcher(g,PEG,refs).score(np.array([85]))
        self.assertEqual(row['tolerance']['2px_3mm']['sampled'],0)

    def test_hole_reference_only_observed_entrance_not_hidden_projection(self):
        k=np.array([[100.,0,64],[0,100.,64],[0,0,1]])
        pose=np.array([0.,0,.5,np.sqrt(.5),0,np.sqrt(.5),0])
        frame=synthetic_frame(np.full((128,128),.4,np.float32),k,np.ones((128,128),bool))
        truth=oracle(frame,pose,box_primitives(.1,.025,np.zeros(2)))
        visible=np.isfinite(truth['depth'])
        rim,_=rim_region(frame,truth,visible,pose,.1,.025,np.zeros(2))
        gt={'box_pose':pose[None], 'box_hole_pose':pose[None], 'box_hole_radii':np.array(.025),'peg_half_sizes':np.array([.1,.02,.02])}
        ref=rim_reference(frame,gt,truth,rim)
        self.assertGreater(len(ref[0]),0)
        hidden=rim_reference(frame,gt,truth,np.zeros_like(rim))
        self.assertEqual(len(hidden[0]),0)
        # Hide half the actual entrance for evaluation and omit it from tracker support.
        partial=rim.copy();partial[:64]=False
        ref=rim_reference(frame,gt,truth,partial)
        self.assertTrue((ref[0][:,1]>=62.5).all())
        depth=np.where(visible,truth['depth'],0).astype(np.float32)
        frame=replace(frame,depth=depth,masks={BOX:visible & (np.indices(visible.shape)[0]>=64)})
        e=extract_sampling_evidence(frame)
        a=sample_points(e,SamplingConfig(num_points=64),method='S4')
        self.assertTrue((a.pixels_vu[:,0]>=64).all())
        self.assertTrue(visible[tuple(a.pixels_vu.T)].all())

    def test_empty_reference_is_undefined_not_perfect(self):
        k=np.eye(3);frame=synthetic_frame(np.ones((8,8),np.float32),k,np.ones((8,8),bool))
        matcher=CoverageMatcher(build_organized_geometry(frame),PEG,(np.empty((0,2)),np.empty((0,3))))
        row,_=matcher.score(np.array([0,1]))
        self.assertIsNone(row['tolerance']['2px_3mm']['gt_coverage'])
        self.assertEqual(row['tolerance']['2px_3mm']['selected'],2)

    def test_staged_sweeps_cover_all_requested_values(self):
        c=conditions();self.assertEqual(len(c),54)
        for method in ('S1','S2','S3','S4'):
            rows=[v for v in c.values() if v['method']==method]
            self.assertEqual({v['num_points'] for v in rows},{64,128,256,512})
            self.assertEqual({v['structural_ratio'] for v in rows},{0.,.125,.25,.5})
            self.assertEqual({v['orientation_threshold_deg'] for v in rows},{10,20,30,40})
            self.assertEqual({v['seed'] for v in rows},{0,1,2})

    def test_saved_evaluation_logging_all_frames_and_metadata(self):
        n,h,w=2,16,16
        ep={'camera/rgb':np.zeros((n,h,w,3),np.uint8),'camera/depth':np.full((n,h,w),.3,np.float32),
            'camera/intrinsic':np.tile([[100,0,8],[0,100,8],[0,0,1]],(n,1,1)),
            'camera/cam2world':np.tile(np.eye(4),(n,1,1)),'tcp_pose':np.tile([0,0,0,1,0,0,0],(n,1)),
            'camera/seg':np.ones((n,h,w),np.int16),'peg_id':np.array(1),'box_id':np.array(2),
            'peg_pose':np.tile([0,0,.4,1,0,0,0],(n,1)),'box_pose':np.tile([1,0,.5,1,0,0,0],(n,1)),
            'box_hole_pose':np.tile([1,0,.5,1,0,0,0],(n,1)),'peg_half_sizes':np.array([.1,.1,.1]),
            'box_hole_radii':np.array(.02),'seed':np.array(0)}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);episode=root/'ep.npz';masks=root/'masks.npz'
            np.savez(episode,**ep);np.savez(masks,source_frame=np.arange(n),peg=np.ones((n,h,w),bool),box=np.zeros((n,h,w),bool))
            with patch.dict(os.environ,{'WANDB_MODE':'disabled','EXPERIMENTS_DIR':str(root/'experiments')}):
                out=evaluate(episode,masks,root/'out',camera='camera',selected=(),seeds=(0,),make_plots=False)
            report=json.loads((out/'comparison.json').read_text())
            self.assertEqual(report['evaluated_frames'],2);self.assertEqual(report['conditions'],44)
            self.assertFalse(report['config']['gt_sampling_input'])
            self.assertEqual(len(list(out.glob('*_samples.npz'))),2)
            with np.load(out/'f0000_samples.npz',allow_pickle=False) as ar:
                self.assertIn('S4_n256_r0.25_t20_s0/metadata/depth_discontinuity',ar.files)
                self.assertEqual(len(ar['S0_n256_r0_t20_s0/points']),256)
            meta=json.loads((root/'experiments'/out.name/'metadata.json').read_text())
            self.assertEqual(meta['status'],'finished');self.assertEqual(meta['run_id'],out.name)
            np.savez(masks,source_frame=np.array([0]),peg=np.ones((1,h,w),bool),box=np.zeros((1,h,w),bool))
            with self.assertRaises(ValueError):evaluate(episode,masks,root/'out',camera='camera',selected=(),make_plots=False)


if __name__=='__main__':unittest.main()
