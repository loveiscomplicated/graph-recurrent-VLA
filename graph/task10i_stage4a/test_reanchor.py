import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graph.persistent_geometry.contracts import CameraObservation
from .reliable_reanchor import positive_main_witness,AnchorState,WristPegReanchor,refine_shared_surface,raw_support_guard
from dataclasses import replace
from graph.integration_baseline.online_vision_adapter import ObservedFrames

def camera(name,depth=1000):
    return CameraObservation('e',0,name,0.,np.zeros((4,4,3),np.uint8),np.full((4,4),depth,np.int16),np.array([[100.,0,2],[0,100,2],[0,0,1]]),np.eye(4)[:3],np.diag([1.,-1,-1,1]),np.array([0.,0,0,1,0,0,0]))

class FakeTracker:
    predictor=object()
    def start(self,path,boxes,frame_idx):
        from pathlib import Path
        assert len(list(Path(path).glob('*.jpg')))==1 and frame_idx==0
        return dict(images=[],num_frames=1)
    def track(self,state,start_frame):yield start_frame,{1:torch.ones((4,4),dtype=torch.bool)}
    def reset(self,state):state['reset']=True

class Reanchor(unittest.TestCase):
    def test_positive_witness_coordinate_alignment(self):
        mask=np.zeros((2,4,4),bool);mask[0,1:3,1:4]=True
        out=positive_main_witness([camera('base_camera'),camera('hand_camera')],{'base_camera':mask})
        np.testing.assert_array_equal(out,mask[0])
    def test_disagreeing_depth_is_not_positive_identity_evidence(self):
        mask=np.ones((2,4,4),bool)
        out=positive_main_witness([camera('base_camera'),camera('hand_camera',1020)],{'base_camera':mask})
        self.assertFalse(out.any())
    def test_invalid_depth_does_not_create_support(self):
        mask=np.ones((2,4,4),bool)
        out=positive_main_witness([camera('base_camera'),camera('hand_camera',0)],{'base_camera':mask})
        self.assertFalse(out.any())
    def test_anchor_uses_one_observed_frame_and_contiguous_local_clock(self):
        def append(self,path,predictor):self.frames.append(None);return 4,4
        with patch.object(ObservedFrames,'append_jpeg',append):
            state=AnchorState(FakeTracker(),np.zeros((4,4,3),np.uint8),[0,0,4,4],17)
            self.assertEqual(state.current().sum(),16)
            state.append(np.zeros((4,4,3),np.uint8),18);self.assertEqual(state.state['num_frames'],1)
            state.current();self.assertEqual(state.state['num_frames'],2)
            with self.assertRaises(ValueError):state.append(np.zeros((4,4,3),np.uint8),20)
            state.close()
    def refinement_inputs(self):
        def large(name):
            return replace(camera(name),rgb=np.zeros((16,16,3),np.uint8),depth_raw_mm=np.full((16,16),1000,np.int16),intrinsic=np.array([[100.,0,8],[0,100,8],[0,0,1]]))
        main,wrist=large('base_camera'),large('hand_camera');clean=np.zeros((2,16,16),bool);clean[0,5:8,5:8]=True
        raw=clean[0].copy();raw[14,14]=True
        robot={k:np.zeros((16,16),bool) for k in ('confirmed_robot','uncertain')}
        return main,wrist,clean,raw,robot
    def test_shared_visible_background_is_removed(self):
        main,wrist,clean,raw,robot=self.refinement_inputs()
        out,trace=refine_shared_surface(raw,[main,wrist],{'base_camera':clean},robot)
        self.assertEqual(out.sum(),9);self.assertEqual(trace['shared_depth_confirmed_disagreement_removed'],1)
    def test_main_occlusion_keeps_complementary_wrist_surface(self):
        main,wrist,clean,raw,robot=self.refinement_inputs();main.depth_raw_mm[14,14]=500
        out,trace=refine_shared_surface(raw,[main,wrist],{'base_camera':clean},robot)
        self.assertTrue(out[14,14]);self.assertEqual(trace['shared_depth_confirmed_disagreement_removed'],0)
    def test_robot_refinement_retains_separate_attribution(self):
        main,wrist,clean,raw,robot=self.refinement_inputs();robot['confirmed_robot'][6,6]=True
        out,trace=refine_shared_surface(raw,[main,wrist],{'base_camera':clean},robot)
        self.assertFalse(out[6,6]);self.assertEqual(trace['robot_removed'],1)
    def test_refinement_does_not_bypass_raw_clean_fraction_guard(self):
        main,wrist,clean,raw,robot=self.refinement_inputs();raw[:]=True
        robot['confirmed_robot'][:]=True;robot['confirmed_robot'][5:8,5:8]=False
        passed,trace=raw_support_guard(raw,wrist,np.zeros_like(raw),robot,{'minimum_valid_depth_fraction':.5,'minimum_clean_fraction':.25})
        self.assertFalse(passed);self.assertEqual(trace['clean_fraction'],9/256)
    def test_episode_reset_and_no_skipped_timestamps(self):
        p=WristPegReanchor(None,None,None);p.reset('branch1')
        with self.assertRaises(ValueError):p.process([],{'timestep':0,'episode':'branch2'},[],{})
        with self.assertRaises(ValueError):p.process([],{'timestep':1,'episode':'branch1'},[],{})
        p.reset('branch2');self.assertEqual(p.last_t,-1);self.assertIsNone(p.active)
if __name__=='__main__':unittest.main()
