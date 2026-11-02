import unittest
from types import SimpleNamespace
import numpy as np
from .perframe_cache import CurrentRobotMemo


class CurrentRobotMemoControls(unittest.TestCase):
    def setUp(self):
        self.calls=0
        def classify(camera,surfaces):
            self.calls+=1;return dict(marker=self.calls)
        self.memo=CurrentRobotMemo(classify)
        self.camera=SimpleNamespace(episode='one',timestep=0,timestamp=0.,camera='base_camera',
            depth_raw_mm=np.ones((2,2),np.int16),intrinsic=np.eye(3),extrinsic_cv=np.eye(4)[:3],cam2world_gl=np.eye(4),tcp_pose=np.zeros(7))
        self.surface=SimpleNamespace(link='finger',local_vertices=np.ones((8,3)),world_from_link=np.eye(4))
    def test_exact_repeat_returns_actual_classifier_result(self):
        first=self.memo(self.camera,[self.surface]);second=self.memo(self.camera,[self.surface])
        self.assertIs(first,second);self.assertEqual(self.calls,1)
    def test_depth_mutation_same_identity_invalidates(self):
        self.memo(self.camera,[self.surface]);self.camera.depth_raw_mm[0,0]+=1;self.memo(self.camera,[self.surface])
        self.assertEqual(self.calls,2)
    def test_robot_pose_and_mesh_mutation_invalidate(self):
        self.memo(self.camera,[self.surface]);self.surface.world_from_link[0,3]+=.01;self.memo(self.camera,[self.surface])
        self.surface.local_vertices[0,0]+=.01;self.memo(self.camera,[self.surface]);self.assertEqual(self.calls,3)
    def test_camera_frame_and_episode_isolation(self):
        self.memo(self.camera,[self.surface]);self.camera.camera='hand_camera';self.memo(self.camera,[self.surface])
        self.camera.timestep=1;self.memo(self.camera,[self.surface]);self.camera.episode='branch';self.memo(self.camera,[self.surface])
        self.assertEqual(self.calls,4);self.assertEqual(len(self.memo.cache),1)


if __name__=='__main__':unittest.main()
