"""Decision-boundary, provenance and numerical comparisons with unchanged code."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,os.environ.get('TASK10I_REFERENCE_CODE',str(HERE.parent/'resolution_perception_graph_20261103_v1')))
sys.path.insert(0,str(HERE))
import native
from graph.plan_a_geometry import core
from graph.persistent_geometry.graph import farthest_rows
from visual.runtime.test import run_voxel_structural_sampling as voxel


class CPU(unittest.TestCase):
    def dedup_compare(self,xyz):
        records=[dict(world_xyz=p,object_id=i%2,structural=i%7==0,
            source_observation_ids=[('ep',0,'base',i)]) for i,p in enumerate(xyz)]
        actual,stats=native.deduplicate(records,fallback=core.deduplicate)
        expected,old=core.deduplicate(records);self.assertEqual(stats,old)
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(a['world_xyz'],b['world_xyz'])
            for k in ('object_id','structural','source_observation_ids'):self.assertEqual(a[k],b[k])
        self.assertTrue(all(len(r['source_observation_ids'])==1 for r in records))

    def test_dedup_random_objects_structural_and_chain(self):
        for seed in range(20):
            r=np.random.default_rng(seed);xyz=r.normal(0,.005,(400,3));xyz[100:130]=xyz[:30]
            self.dedup_compare(xyz)
        self.dedup_compare(np.column_stack((np.arange(100)*.0015,np.zeros((100,2)))))

    def test_strict_3mm_boundary_and_empty(self):
        self.dedup_compare(np.empty((0,3)))
        for d in (np.nextafter(.003,0),.003,np.nextafter(.003,np.inf)):
            for offset in (0.,-.1,1.):
                self.dedup_compare(np.array([[offset,0,0],[9,9,9],[offset+d,0,0]]))

    def test_fps_ties_and_stable_identity_order(self):
        for seed in range(25):
            r=np.random.default_rng(seed);xyz=r.normal(size=(300,3));xyz[20:40]=xyz[:20]
            ids=r.permutation(len(xyz)).astype(str)
            for count in (0,1,128,300,400):
                np.testing.assert_array_equal(native.farthest_rows(xyz,ids,count),farthest_rows(xyz,ids,count))
        xyz=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],float)
        np.testing.assert_array_equal(native.farthest_rows(xyz,list('edcba'),4),farthest_rows(xyz,list('edcba'),4))

    def test_voxel_all_fields_and_centroid_ties(self):
        samples=[np.empty((0,3),np.float32),np.zeros((1500,3),np.float32),
                 np.array([[.001,0,1],[.002,0,1],[.003,0,1],[.004,0,1]],np.float32)]
        for seed in range(20):
            r=np.random.default_rng(seed);samples.append(r.normal(0,.01,(800,3)).astype(np.float32))
        for xyz in samples:
            flat=np.arange(len(xyz),dtype=np.int64)*3
            for a in (.003,.03,1.):
                actual=native.representatives(flat,xyz,a);expected=voxel.representatives(flat,xyz,a)
                for k in expected:np.testing.assert_array_equal(actual[k],expected[k],err_msg=k)

    def test_partial_topk_preserves_all_cutoff_ties(self):
        from packing import global_pairs
        from graph.integration_baseline.graph import global_peg_box_pairs
        for seed in range(20):
            r=np.random.default_rng(seed);xyz=r.normal(0,.01,(256,3)).astype(np.float32)
            if seed%2:xyz[:]=0
            oid=r.integers(0,2,len(xyz));ids=r.integers(0,20,len(xyz))
            for m in (0,1,16,100):
                np.testing.assert_array_equal(global_pairs(xyz,oid,ids,m),global_peg_box_pairs(xyz,oid,ids,m))

    def test_selection_packet_identical_fields_and_provenance(self):
        from packing import selection_packet
        from visual.runtime.test import run_variable_rate_shape_first as sf
        from graph.vision_input import validate_packet
        prepared={};selected={};cases={}
        for oid in (0,1):
            x=np.arange(20,dtype=np.int64);f=np.array([1,3,7,2,11],np.int64)
            prepared[oid]=dict(C=dict(pca=np.array([1,3,5]),sil=np.array([3,7,9])))
            selected[oid]=dict(F=f,P=f[:3],T=dict(pca=f[:2],sil=f[1:3]),xyz=np.column_stack([f,f,f]).astype(np.float32))
            cases[oid]=dict(saved_flat=x,saved_features=dict(value=x.astype(np.float32),valid=x%2==0))
        expected,_=sf.make_packet(0,prepared,selected,cases);actual,_=selection_packet(0,prepared,selected,cases)
        validate_packet(actual)
        def compare(a,b):
            if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a:compare(a[k],b[k])
            else:self.assertEqual(a,b)
        compare(actual,expected)

    def test_batched_surface_edges_and_audits(self):
        import edges
        from graph.persistent_geometry.graph import surface_edges
        for seed in range(20):
            r=np.random.default_rng(seed);xyz=r.normal(0,.006,(80,3));oid=np.arange(len(xyz))%2
            normals=r.normal(size=xyz.shape);normals/=np.linalg.norm(normals,axis=1)[:,None]
            valid=r.random(len(xyz))>.3;refs={i:xyz[oid==i] for i in (0,1)}
            a,aa=edges.surface_edges(xyz,normals,valid,oid,refs)
            b,ba=surface_edges(xyz,normals,valid,oid,refs)
            np.testing.assert_array_equal(a,b);self.assertEqual(aa,ba)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA server only')
class GPU(unittest.TestCase):
    def test_context_restores_shared_functions_after_exception(self):
        from candidates import Candidate
        import graph.persistent_geometry.graph as gm
        from visual.runtime.test import run_variable_rate_shape_first as sf
        from visual.runtime.test import run_priority_fps_metric_control as prior
        owners=[(core,'deduplicate'),(core,'classify_robot'),(core.PlanA,'represent'),
                (gm,'surface_edges'),(gm,'global_peg_box_pairs'),(sf,'make_packet'),(prior,'feature_rows')]
        before=[getattr(o,n) for o,n in owners]
        with self.assertRaisesRegex(RuntimeError,'scope test'):
            with Candidate(9):raise RuntimeError('scope test')
        for (owner,name),old in zip(owners,before):self.assertIs(getattr(owner,name),old)

    def test_small_fps_initial_duplicate_tie_and_cutover(self):
        from selection import fps
        from visual.runtime.sampling import fps_indices
        for n in (1,32,513,8192,8193):
            r=np.random.default_rng(n);xyz=r.normal(0,.1,(n,3)).astype(np.float32)
            if n>20:xyz[10:20]=xyz[:10]
            for initial in ([],list(range(min(n,5)))):
                np.testing.assert_array_equal(fps(xyz,min(30,n),initial=initial),fps_indices(xyz,min(30,n),initial=initial))
        xyz=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],np.float32)
        np.testing.assert_array_equal(fps(xyz,5,initial=[0]),fps_indices(xyz,5,initial=[0]))

    def test_batched_robot_masks_owner_and_depth(self):
        from gpu_graph import OptimizedGraph
        import robot
        from graph.plan_a_geometry.tests import Contracts
        from graph.persistent_geometry.self_filter import classify_robot,LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        fixture=Contracts();fixture.setUp()
        surfaces=[LinkSurface('left',cube([-.15,-.2,1],[-.03,.2,1.1]),np.eye(4)),
                  LinkSurface('right',cube([-.08,-.2,.9],[.1,.2,1.]),np.eye(4))]
        for ss in ([],surfaces,surfaces[::-1],surfaces+surfaces):
            camera=fixture.wrist;expected=classify_robot(camera,ss)
            ctx=OptimizedGraph('cuda_fused_v5');actual=robot.classify(ctx,camera,ss)
            for k in ('confirmed_robot','uncertain','candidate','link_owner','inside_known_link_volume'):
                np.testing.assert_array_equal(actual[k],expected[k],err_msg=k)
            np.testing.assert_allclose(actual['predicted_robot_depth'],expected['predicted_robot_depth'],atol=1e-12,rtol=0)

    def test_dense_features_endpoint_maps_and_threshold_fallback(self):
        import dense
        from visual.runtime.rgbd import TrackedRGBDFrame
        from visual.runtime.sampling import extract_sampling_evidence
        from visual.runtime.test.pca_triton import fused_normals
        from visual.runtime.geometry import GeometricFeatures
        from graph.plan_a_geometry.tests import Contracts
        fixture=Contracts();fixture.setUp();camera=fixture.wrist
        for mode in ('normal','empty','overlap','holes'):
            masks=fixture.masks['hand_camera'].copy();depth=camera.depth.astype(np.float32)
            if mode=='empty':masks[:]=False
            if mode=='overlap':masks[1]=masks[0]
            if mode=='holes':depth[::3,::3]=np.nan
            frame=TrackedRGBDFrame(0,camera.camera,camera.rgb,depth,camera.intrinsic,camera.world_from_camera,
                camera.tcp_pose,dict(enumerate(masks)),0,0.)
            def normals(g,oid,p,**kw):
                n,v=fused_normals(torch.as_tensor(np.ascontiguousarray(g.xyz),device='cuda'),
                    torch.as_tensor(g.observed_mask(oid),device='cuda'),p)
                return n.cpu().numpy(),v.cpu().numpy()
            with patch('visual.runtime.sampling.local_plane_optimized',normals):expected=extract_sampling_evidence(frame)
            actual=dense.extract(frame)
            for oid in (0,1):
                for k in GeometricFeatures.__dataclass_fields__:
                    a,b=getattr(actual.frame.objects[oid],k),getattr(expected.frame.objects[oid],k)
                    if a.dtype.kind=='f':np.testing.assert_allclose(a,b,atol=1e-6,rtol=0,equal_nan=True,err_msg=k)
                    else:np.testing.assert_array_equal(a,b,err_msg=k)
                for threshold in (0,20,35):
                    a,b=actual.pairs[oid].candidates(threshold),expected.pairs[oid].candidates(threshold)
                    for k in b:np.testing.assert_array_equal(a[k],b[k],err_msg=k)


if __name__=='__main__':unittest.main()
