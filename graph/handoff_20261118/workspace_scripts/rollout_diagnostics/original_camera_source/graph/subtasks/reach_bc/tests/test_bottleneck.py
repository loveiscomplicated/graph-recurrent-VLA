import unittest
import torch
from graph.subtasks.reach_bc.bottleneck import Probes,MODES,CONTEXT_START,FINGER_CONTEXT


class FrozenProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.probes=Probes()

    def test_feature_order_and_finger_duplicates(self):
        p=self.probes
        self.assertEqual(p.train_x.shape[1],CONTEXT_START+81)
        # Independently compare context qpos fingers and GT gripper values.
        torch.testing.assert_close(p.geometry.gripper,p.geometry.qpos[:,-2:],atol=0,rtol=0)
        self.assertEqual(FINGER_CONTEXT,(0,1,9,10))

    def test_probes_preserve_inputs_weights_and_unselected_channels(self):
        p=self.probes
        x=p.train_x[:4].clone();s=p.train_s[:4].clone()
        x[:,CONTEXT_START+22]=1000.;s[:,7:9]=1000.
        before={k:v.clone() for k,v in p.policy.model.state_dict().items()}
        original=(x.clone(),s.clone())
        for mode in MODES:
            a,b=p.modify((x,s),mode)
            self.assertEqual(a.shape,x.shape);self.assertEqual(b.shape,s.shape)
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
            if mode=='finger_mean':
                self.assertTrue((a[:,[CONTEXT_START+i for i in FINGER_CONTEXT]]==0).all())
                self.assertTrue((b[:,7:9]==0).all())
            if mode=='qvel_zero':
                torch.testing.assert_close(b[:,:9]*p.policy.sn.std[:9]+p.policy.sn.mean[:9],torch.zeros_like(b[:,:9]),atol=1e-8,rtol=0)
                torch.testing.assert_close(a,x,atol=0,rtol=0)
            if mode=='ee_floor_clip':
                self.assertLessEqual(float(a[:,CONTEXT_START+22].max()),float(p.x_max[CONTEXT_START+22]))
                torch.testing.assert_close(b,s,atol=0,rtol=0)
            if mode=='low_variance_clip':
                self.assertTrue((a[:,p.x_lowvar]<=p.x_max[p.x_lowvar]).all())
                self.assertTrue((b[:,p.s_lowvar]<=p.s_max[p.s_lowvar]).all())
                torch.testing.assert_close(a[:,~p.x_lowvar],x[:,~p.x_lowvar],atol=0,rtol=0)
                torch.testing.assert_close(b[:,~p.s_lowvar],s[:,~p.s_lowvar],atol=0,rtol=0)
        torch.testing.assert_close(x,original[0],atol=0,rtol=0);torch.testing.assert_close(s,original[1],atol=0,rtol=0)
        for k,v in before.items():torch.testing.assert_close(v,p.policy.model.state_dict()[k],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
