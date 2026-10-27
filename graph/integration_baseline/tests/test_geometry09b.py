"""Non-experiment fixtures for label frame, group leakage and probe contracts."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from graph.integration_baseline.geometry09b import target,classify,geometry_features,config,validate_splits
from graph.integration_baseline.geometry09b_probe import standardize,ridge_fit,delta,features

class ProbeContracts(unittest.TestCase):
    def test_target_actor_origin_tcp_wxyz(self):
        q=Rotation.from_euler('z',90,degrees=True).as_quat()
        tcp=np.r_[[1.,2.,3.],q[[3,0,1,2]]];peg=np.r_[[1.,2.01,3.],1.,0,0,0]
        np.testing.assert_allclose(target(peg,tcp),[.01,0,0],atol=1e-8)
        self.assertEqual(target(peg,tcp).dtype,np.float32)
        self.assertAlmostEqual(np.linalg.norm(target(peg,tcp))*1000,10.,places=5)

    def test_standardization_does_not_fit_validation(self):
        x=np.array([[1.,3.],[3.,7.],[1e9,-1e9]])
        z,mean,std=standardize(x,np.array([True,True,False]))
        np.testing.assert_array_equal(mean,[2,5]);np.testing.assert_array_equal(std,[1,2])

    def test_validity_missing_contaminated_occluded(self):
        c=config({'splits':{'train':list(range(17)),'validation':list(range(20,26))}})['validity']
        self.assertEqual(classify(dict(visible_gt_pixels=100,selected_gt_purity=None),0,c),'VISIBLE_MISSING')
        self.assertEqual(classify(dict(visible_gt_pixels=0,selected_gt_purity=0),50,c),'OCCLUDED')
        self.assertEqual(classify(dict(visible_gt_pixels=100,selected_gt_purity=.4),50,c),'GEOMETRY_CONTAMINATED')
        self.assertEqual(classify(dict(visible_gt_pixels=100,selected_gt_purity=.9),50,c),'VISIBLE_VALID')

    def test_raw_geometry_not_pose_origin(self):
        x=np.array([[1,2,3],[3,4,7]],np.float32);g=geometry_features(x)
        np.testing.assert_array_equal(g[:3],[2,3,5]);np.testing.assert_array_equal(g[3:6],[1,2,3])
        np.testing.assert_array_equal(g[6:9],[3,4,7]);self.assertEqual(g.shape,(15,))
        np.testing.assert_array_equal(geometry_features(np.zeros((0,3))),np.zeros(15))

    def test_linear_delta_recovers_geometry_and_robot_zero(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(100,5));w=rng.normal(size=(5,3));y=x@w+[1,2,3]
        ww,b=ridge_fit(x,y,.0);np.testing.assert_allclose(x@ww+b,y,atol=1e-12)
        d=delta(np.array([.01,0,0]),np.array([.005,0,0]));self.assertAlmostEqual(d['signed_gain'],.5)
        self.assertAlmostEqual(d['delta_error_mm'],5);self.assertAlmostEqual(d['direction_cosine'],1)
        self.assertIsNone(delta(np.zeros(3),np.zeros(3))['direction_cosine'])

    def test_frozen_split_no_episode_or_reset_overlap(self):
        from types import SimpleNamespace
        p={'splits':{'train':[30003]+list(range(16)),'validation':list(range(20,26))}}
        c=config(p);self.assertNotIn(30003,c['train'])
        ds=SimpleNamespace(splits=p['splits'],episodes={s:{'reset_group':str(s)} for s in range(30)})
        ds.episodes[30003]={'reset_group':'final'};validate_splits(c,ds)
        ds.episodes[20]['reset_group']='0'
        with self.assertRaises(AssertionError):validate_splits(c,ds)

    def test_p1_no_direct_robot_and_p3_exact_object_slots(self):
        data={'Full_B_readout320':np.arange(640).reshape(2,320),'robot76':np.ones((2,76)),
            'raw_geometry15':np.zeros((2,15))}
        self.assertEqual(features(data,'Full_B','P1').shape,(2,320))
        self.assertEqual(features(data,'Full_B','P2').shape,(2,396))
        np.testing.assert_array_equal(features(data,'Full_B','P3'),data['Full_B_readout320'][:,192:])


class SelectionContracts(unittest.TestCase):
    def test_final_extremes_never_change_ridge_selection(self):
        from graph.integration_baseline.geometry09b_probe import linear
        rng=np.random.default_rng(2);x=rng.normal(size=(30,4));y=x[:,:3]*.4
        train=np.arange(30)<20;val=(np.arange(30)>=20)&(np.arange(30)<25)
        pred,state,sel=linear(x,y,train,val,{'alphas':[0,.1,10]})
        xx=x.copy();yy=y.copy();xx[25:]=1e8;yy[25:]=-1e8
        pp,ss,se=linear(xx,yy,train,val,{'alphas':[0,.1,10]})
        self.assertEqual(sel,se);np.testing.assert_array_equal(pred[:25],pp[:25])

    def test_frozen_audit_does_not_clear_gradients(self):
        import torch
        from graph.integration_baseline.geometry09b import freeze
        m=torch.nn.Linear(2,1);m.eval();m.requires_grad_(False)
        models={'fixture':(m,None,None)};before=freeze(models)
        self.assertTrue(before['fixture']['no_grad'])
        m.weight.grad=torch.ones_like(m.weight)
        self.assertFalse(freeze(models)['fixture']['no_grad'])

if __name__=='__main__':unittest.main()
