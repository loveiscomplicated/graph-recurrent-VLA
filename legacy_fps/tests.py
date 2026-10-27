import unittest
import numpy as np
import cv2
from scipy.spatial import cKDTree
from .adapter import extract,row_join,frame_seed,original,raw_ede

class LegacyContracts(unittest.TestCase):
    def setUp(self):
        self.depth=np.full((80,90),.8,np.float32)
        self.mask=np.zeros(self.depth.shape,bool);self.mask[5:75,5:85]=True
        self.K=np.array([[90.,0,45],[0,90,40],[0,0,1]])
        self.T=np.eye(4);self.T[:3,3]=[.1,.2,.3]
    def call(self,mask=None,seed=0,**kwargs):
        return extract(self.mask if mask is None else mask,self.depth,self.K,self.T,np.random.default_rng(seed),**kwargs)
    def test_original_sequential_rng_xyz_order_info_and_world(self):
        a=np.random.default_rng(0);b=np.random.default_rng(0)
        for n in range(5):
            mask=self.mask.copy();mask[:n*4]=False
            p=extract(mask,self.depth,self.K,self.T,a)
            c,w,info=original.extract(mask,self.depth,self.K,self.T,b)
            np.testing.assert_array_equal(p.camera_xyz,c);np.testing.assert_array_equal(p.world_xyz,w)
            for key,value in info.items():self.assertEqual(p.audit[key],value)
            self.assertEqual(a.bit_generator.state,b.bit_generator.state)
    def test_original_cluster_option(self):
        p=self.call(cluster=True)
        c,w,_=original.extract(self.mask,self.depth,self.K,self.T,np.random.default_rng(0),cluster=True)
        np.testing.assert_array_equal(p.camera_xyz,c);np.testing.assert_array_equal(p.world_xyz,w)
    def test_erosion(self):
        p=self.call();er=cv2.erode(self.mask.astype('uint8'),np.ones((3,3),'uint8'),iterations=2)>0
        np.testing.assert_array_equal(p.stages['eroded_mask'],er)
        np.testing.assert_array_equal(p.stages['backproject_source'],np.flatnonzero(er))
    def test_erosion_valid_depth_fallback(self):
        mask=np.zeros_like(self.mask);mask[10:13,10:13]=True
        p=self.call(mask);self.assertTrue(p.audit['erosion_fallback']);self.assertEqual(len(p.camera_xyz),9)
        mask=self.mask.copy();self.depth[:]=0;self.depth[10:13,10:13]=1
        self.assertTrue(self.call(mask).audit['erosion_fallback'])
    def test_sor_selection(self):
        self.depth[30,30]=20;p=self.call()
        ids=p.stages['presample_source'];xyz=original.backproject(self.depth,self.K,p.stages['selected_mask'])
        raw=p.stages['backproject_source'];lookup={int(s):i for i,s in enumerate(raw)}
        pre=xyz[[lookup[int(s)] for s in ids]]
        d=cKDTree(pre).query(pre,k=17)[0][:,1:].mean(1)
        np.testing.assert_array_equal(p.stages['sor_source'],ids[d<=d.mean()+2*d.std()])
    def test_source_backprojection_and_repeat(self):
        p=self.call();q=self.call();np.testing.assert_array_equal(p.source_pixel,q.source_pixel)
        v,u=np.divmod(p.source_pixel,self.depth.shape[1]);z=self.depth[v,u]
        xyz=np.c_[(u+.5-self.K[0,2])*z/self.K[0,0],(v+.5-self.K[1,2])*z/self.K[1,1],z].astype('float32')
        np.testing.assert_array_equal(xyz,p.camera_xyz)
        np.testing.assert_array_equal(original.to_world(xyz,self.T),p.world_xyz)
    def test_empty(self):
        p=self.call(np.zeros_like(self.mask));self.assertEqual(p.camera_xyz.shape,(0,3));self.assertEqual(len(p.source_pixel),0)
    def test_budget_never_invents(self):
        p=self.call(max_pts=128);self.assertLessEqual(len(p.source_pixel),128)
        self.assertTrue(np.isin(p.source_pixel,p.stages['sor_source']).all())
        self.assertEqual(len(np.unique(p.source_pixel)),len(p.source_pixel))
    def test_seed_identity(self):
        self.assertEqual(frame_seed('e',0,'main',0),frame_seed('e',0,'main',0))
        self.assertNotEqual(frame_seed('e',0,'main',0),frame_seed('e',0,'main',1))
        self.assertEqual(frame_seed('e',0,'main',0),frame_seed('e:branch_baseline_repeat_t0',0,'main',0))
    def test_no_approximate_row_join(self):
        with self.assertRaises(ValueError):row_join(np.zeros((2,3),'float32'),np.zeros((1,3),'float32'))
        with self.assertRaises(ValueError):row_join(np.zeros((1,3),'float32'),np.ones((1,3),'float32'))
    def test_no_inf(self):
        self.depth[5,5]=np.inf
        with self.assertRaises(ValueError):self.call()
class IntegrationContracts(unittest.TestCase):
    def setUp(self):
        from graph.persistent_geometry.contracts import CameraObservation
        h,w=32,40;K=np.array([[40.,0,w/2],[0,40,h/2],[0,0,1]])
        self.camera=CameraObservation('test',0,'main',0.,np.zeros((h,w,3),'uint8'),np.full((h,w),800,'int16'),K,np.eye(4)[:3],np.diag([1.,-1,-1,1]),np.array([0,0,0,1,0,0,0.]))
        self.masks=np.zeros((2,h,w),bool);self.masks[0,4:28,3:17]=True;self.masks[1,4:28,23:37]=True
    def test_f5_source_row_join_and_raw_ede(self):
        from .adapter import frame_points
        from visual.runtime.sampling import extract_sampling_evidence
        from visual.runtime.rgbd import TrackedRGBDFrame
        obs,points,comp=frame_points(self.camera,self.masks,f5=True)
        frame=TrackedRGBDFrame(0,'main',self.camera.rgb,self.camera.depth.astype(np.float32),self.camera.intrinsic,self.camera.world_from_camera,self.camera.tcp_pose,{i:self.masks[i] for i in (0,1)},0,0.)
        evidence=extract_sampling_evidence(frame,estimator='pca5')
        for oid,(o,p) in enumerate(zip(obs,points)):
            self.assertEqual(o.object_id,oid)
            np.testing.assert_array_equal(o.pixels,p.source_pixel)
            dense=evidence.frame.objects[o.object_id];idx=p.source_pixel
            np.testing.assert_array_equal(o.feature_valid[:,2],dense.depth_pair_valid.ravel()[idx])
            np.testing.assert_array_equal(o.f5[:,4],dense.depth_jump_m.ravel()[idx])
        edge,_,_=raw_ede(self.camera,obs,comp)
        ids=np.concatenate([np.full(len(o.xyz),o.object_id) for o in obs]);components=np.concatenate(comp)
        self.assertTrue((ids[edge[0]]==ids[edge[1]]).all());self.assertTrue((components[edge[0]]==components[edge[1]]).all())
    def test_raw_ede_rejects_past_and_wrong_camera(self):
        from dataclasses import replace
        from .adapter import frame_points
        obs,_,comp=frame_points(self.camera,self.masks,f5=True)
        with self.assertRaises(ValueError):raw_ede(self.camera,[replace(o,camera='wrist') for o in obs],comp)
        with self.assertRaises(ValueError):raw_ede(self.camera,[replace(o,timestep=1) for o in obs],comp)
    def test_missing_features_not_fabricated(self):
        from .adapter import frame_points
        obs,_,_=frame_points(self.camera,self.masks,f5=False)
        for o in obs:self.assertFalse(o.feature_valid.any());self.assertTrue(np.isnan(o.f5).all())
    def test_practical_adapter_rejects_gt(self):
        from dataclasses import replace
        from .pipeline import LegacySystem
        system=LegacySystem('main','wrist')
        with self.assertRaisesRegex(ValueError,'GT/non-sensor'):
            system.process([self.camera,replace(self.camera,camera='wrist')],dict(episode='test',timestep=0,actor_pose=np.zeros((2,7))),{'main':self.masks,'wrist':self.masks},[])
    def test_canonical_readout_same_weights_and_identity_forward(self):
        import torch
        from graph.persistent_geometry.encoder import GeometryRegressor
        from .readout_adapter import CanonicalRegressor,canonicalize_robot_slots
        from graph.integration_baseline.graph import Graph,pack_graphs
        torch.manual_seed(42);original_model=GeometryRegressor('E0').eval()
        torch.manual_seed(42);adapted=CanonicalRegressor('E0').eval()
        for k,v in original_model.state_dict().items():torch.testing.assert_close(v,adapted.state_dict()[k],atol=0,rtol=0)
        graph=Graph(torch.randn(8,21),torch.tensor([0,1,2,3,3,4,4,4]),torch.empty((2,0),dtype=torch.int64),torch.empty((0,4)),torch.empty(0,dtype=torch.int64),{}, {})
        packed=pack_graphs([graph]);self.assertIs(canonicalize_robot_slots(packed),packed)
        torch.testing.assert_close(original_model(packed),adapted(packed),atol=0,rtol=0)
        from graph.persistent_geometry.train import permutation_audit
        self.assertEqual(permutation_audit(adapted.encoder,graph,device='cpu')['status'],'PASS')
if __name__=='__main__':unittest.main()
