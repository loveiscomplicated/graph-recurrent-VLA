"""Offline numeric contracts; no simulator, policy, training or source mutation."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from graph.integration_baseline.reach_action_distribution_analysis import angle,controller,cosine,action_metrics


class AnalysisContracts(unittest.TestCase):
    def test_quaternion_sign_invariance(self):
        q=Rotation.from_euler('XYZ',[.3,-.4,.7]).as_quat(scalar_first=True)
        self.assertLess(angle(q,-q),1e-12)
        self.assertAlmostEqual(angle(q,[1,0,0,0]),angle(-q,[1,0,0,0]),12)

    def test_installed_controller_clipping_sign_and_units(self):
        a=np.array([[2,-2,.5,3,4,0,2]],np.float32)
        b=controller(a)[0]
        np.testing.assert_allclose(b,[.1,-.1,.05,-.06,-.08,0,.04],atol=2e-8)
        np.testing.assert_array_equal(a,[[2,-2,.5,3,4,0,2]])

    def test_rotation_native_error_is_not_tcp_angle(self):
        e=np.array([[0,0,0,0,0,-1.5,1]],np.float32)
        s=e.copy();s[0,5]=-2
        m=action_metrics(e,s)
        self.assertGreater(m['rotation_native_mse'],0)
        self.assertAlmostEqual(m['controller_rotation_command_geodesic_deg']['max'],0,12)

    def test_zero_translation_has_no_invented_direction(self):
        self.assertTrue(np.isnan(cosine(np.zeros((1,3)),np.ones((1,3)))[0]))
        self.assertAlmostEqual(cosine(np.array([[1.,0,0]]),np.array([[-2.,0,0]]))[0],-1)

    def test_euler_command_rotation_matches_installed_controller(self):
        import torch
        from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix
        xyz=np.array([[.04,-.07,.09],[-.1,.08,.01]],np.float32)
        installed=euler_angles_to_matrix(torch.from_numpy(xyz), 'XYZ').numpy()
        np.testing.assert_allclose(Rotation.from_euler('XYZ',xyz).as_matrix(),installed,atol=1e-7,rtol=1e-6)

    def test_root_to_world_transform(self):
        root=Rotation.from_euler('z',90,degrees=True)
        world=root.apply([[.01,0,0]])
        np.testing.assert_allclose(world,[[0,.01,0]],atol=1e-16)
        self.assertAlmostEqual(cosine(world,np.array([[0,.02,0]]))[0],1)

    def test_delta_error_is_after_action_not_shifted(self):
        before=np.array([3.,2.,4.]);after=np.array([2.,4.,1.])
        np.testing.assert_array_equal(after-before,[-1,2,-3])
        self.assertEqual(float((after-before).sum()),-2)


if __name__=='__main__':unittest.main()
