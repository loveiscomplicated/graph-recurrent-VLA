import unittest
from types import SimpleNamespace
import numpy as np
from .depth_component import MainPegDepthComponent
from .depth_box_component import MainObjectDepthComponents


class MainObjectScopeControls(unittest.TestCase):
    def test_box_scope_extension_preserves_peg_result_and_wrist(self):
        xyz=np.zeros((2,22,3));xyz[:,:,2]=1
        xyz[:,:20,0]=np.arange(20)*.001;xyz[1,:,1]=.5
        xyz[:,20:,0]=2;xyz[:,20:,2]=2
        camera=SimpleNamespace(camera='base_camera',depth=np.ones((2,22)),world_points=lambda:xyz)
        masks={'base_camera':np.zeros((2,2,22),bool),'hand_camera':np.zeros((2,2,22),bool)}
        masks['base_camera'][0,0]=True;masks['base_camera'][1,1]=True
        geo=SimpleNamespace(system=SimpleNamespace(prepare=lambda cameras,data,mm,surfaces:dict(clean={c:m.copy() for c,m in mm.items()})))
        one=MainPegDepthComponent(geo);both=MainObjectDepthComponents(geo)
        one.reset('episode');both.reset('episode');data=dict(timestep=0,episode='episode')
        a,_=one.process([camera],data,[],masks);b,_=both.process([camera],data,[],masks)
        np.testing.assert_array_equal(a['base_camera'][0],b['base_camera'][0])
        np.testing.assert_array_equal(b['hand_camera'],masks['hand_camera'])
        self.assertEqual(a['base_camera'][1].sum(),22);self.assertEqual(b['base_camera'][1].sum(),20)
        self.assertEqual(b['base_camera'][0].sum(),20)
    def test_branch_discontinuity_is_rejected(self):
        p=MainObjectDepthComponents(None);p.reset('one')
        with self.assertRaises(ValueError):p.process([],dict(episode='other',timestep=0),[],{})


if __name__=='__main__':unittest.main()
