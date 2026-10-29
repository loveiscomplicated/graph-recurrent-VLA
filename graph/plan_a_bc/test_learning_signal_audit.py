"""Unit/coordinate contracts; synthetic data do not establish learning signal."""
import unittest
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from .learning_signal_audit import match,evaluate,LEVELS

class AuditContracts(unittest.TestCase):
 def fixture(self):
  r=np.zeros((4,76));r[:,27]=1;r[:,66]=1;r[:,22]=.08;r[2,0]=1 # reject physically distinct joint state
  f=pd.DataFrame(dict(split=['train']*4,episode=[1,1,2,3],t=[0]*4,phase=['early_open']*4,peg_x=[0,0,.03,.03],peg_y=[0]*4,peg_z=[0]*4,box_x=[0]*4,box_y=[0]*4,box_z=[0]*4,peg_quality=[True]*4,box_quality=[True]*4,peg_support=[128]*4,box_support=[128]*4,peg_span_mm=[60]*4,peg_center_gap_mm=[1]*4))
  return f,r,np.repeat(r[:,None],4,axis=1),np.repeat(np.eye(3)[None],4,axis=0)
 def test_cross_episode_group_and_block_constraints(self):
  f,r,h,rot=self.fixture();p=match(f,r,h,rot)
  self.assertTrue((p.episode_a!=p.episode_b).all());self.assertFalse(((p.i==2)|(p.j==2)).any())
  for level in LEVELS:self.assertEqual(int(p[p.level==level].selected.sum()),1)
 def test_command_frame_and_small_delta(self):
  f,r,h,rot=self.fixture();p=match(f,r,h,rot);e=np.zeros((4,7));e[3,0]=.1
  er,mr=evaluate(p,f,r,rot,e,{'E0':e,'E1':np.zeros_like(e)})
  self.assertTrue(np.allclose(er.expert_delta_mm,10));self.assertTrue(np.allclose(mr[mr.model=='E0'].delta_error_mm,0));self.assertTrue(np.allclose(mr[mr.model=='E1'].zero_improvement_mm,0))
  er,mr=evaluate(p,f,r,rot,e*0,{'E0':e*0});self.assertTrue(mr.direction_cosine.isna().all());self.assertTrue(mr.signed_gain.isna().all())
 def test_tcp_rotation_applied(self):
  f,r,h,rot=self.fixture();rot[:]=Rotation.from_euler('z',90,degrees=True).as_matrix();p=match(f,r,h,rot);e=np.zeros((4,7));e[3,0]=.1
  er,_=evaluate(p,f,r,rot,e,{'E0':e});self.assertTrue(np.allclose(er.expert_dy_mm,-10));self.assertTrue(np.allclose(er.expert_dx_mm,0,atol=1e-10))
if __name__=='__main__':unittest.main()
