"""Independent synthetic storage fixtures for the strict real-data gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from graph.integration_baseline.data import ReplayDataset, collate_sequences
from graph.integration_baseline.io import write_json, sha256, save_npz, save_packet
from graph.integration_baseline.tests.test_baseline import fixture_packet, robot_fixture, ROBOT
from graph.integration_baseline.robot import flatten_state


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.replay = self.root/'replay';self.vision = self.root/'vision'
        self.replay.mkdir();self.vision.mkdir()
        fields, schema = robot_fixture()
        state = flatten_state(fields, schema)
        source = self.root/'expert.h5'
        action = np.arange(21,dtype=np.float32).reshape(3,7)/30
        flags = np.array([False,False,True])
        ref = np.concatenate((fields['qpos'],fields['qvel'],fields['tcp_pose'],np.zeros(18,np.float32)))
        with h5py.File(source,'w') as f:
            g=f.create_group('traj_3');g['actions']=action;g['terminated']=np.zeros(3,bool);g['truncated']=flags
            g['obs']=np.stack([ref]*4)
        digest = sha256(source)
        frozen = self.root/'frozen.json'
        splits={'train':[3],'validation':[4],'test':[5]}
        write_json(frozen,dict(splits=splits,episodes=[dict(episode_id=e,episode_seed=e) for e in (3,4,5)]))
        raw = self.replay/'ep003.npz'
        save_npz(raw,actions=action,terminated=np.zeros(3,bool),truncated=flags,
            robot_state=np.stack([state]*4),robot_world_geometry=np.stack([ROBOT]*4),
            tcp_pose=np.stack([fields['tcp_pose']]*4),timestep=np.arange(4),timestamp=np.arange(4)/20,
            episode_id=np.full(4,3),source_sha256=np.array(digest))
        # Camera calibration is needed by the frame join; preserve the existing
        # fixture packet's source camera-to-world exactly.
        p=fixture_packet()
        with np.load(raw) as a:values={k:a[k] for k in a.files}
        values['base_camera/cam2world']=np.stack([p['camera_to_world']]*4).astype(np.float32)
        save_npz(raw,**values)
        self.rm=dict(source=str(source),source_sha256=digest,frozen_split_manifest=str(frozen),
            frozen_split_sha256=sha256(frozen),splits=splits,robot_state_schema=schema,
            episodes=[dict(episode_id=3,path='ep003.npz',sha256=sha256(raw),aligned_targets=3,control_freq=20)],
            pilot_gate=dict(replay_atol=.001),coverage=dict(complete_episodes=0,episodes=1),
            missing_episode_ids=[4,5],failures=[])
        write_json(self.replay/'manifest.json',self.rm)
        mask=self.vision/'masks.npz';save_npz(mask,peg=np.zeros((4,16,16),bool),box=np.zeros((4,16,16),bool))
        rows=[]
        for t in range(4):
            q=copy.deepcopy(p);q.update(episode=digest+':traj_3',source_frame=t,frame_idx=t,timestamp=t/20)
            q['observation_provenance']=dict(episode_id=3,timestep=t,source_sha256=digest,replay_sha256=sha256(raw))
            path=self.vision/f'f{t}.npz';save_packet(path,q)
            rows.append(dict(timestep=t,path=path.name,sha256=sha256(path),has_action_target=t<3))
        self.vm=dict(source_sha256=digest,replay_manifest_sha256=sha256(self.replay/'manifest.json'),splits=splits,
            robot_state_schema=schema,camera='base_camera',episodes=[dict(episode_id=3,replay_sha256=sha256(raw),
            mask_provenance=dict(gt_segmentation_used=False,future_observations_used=False,
                                 sam2_horizon='num_frames=t+1; fixture'),mask_path=mask.name,
            mask_sha256=sha256(mask),packets=rows)])
        write_json(self.vision/'manifest.json',self.vm)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_join_contiguous_sequence_and_last_observation_unlabelled(self):
        d=ReplayDataset(self.replay,self.vision)
        frames=d.sequence(3,0,3)
        graph,state,target,mask=collate_sequences([frames,frames[:2]])
        self.assertEqual(graph.count,5);self.assertEqual(tuple(target.shape),(2,3,7))
        self.assertFalse(mask[1,2])
        with self.assertRaises(ValueError):d.frame(3,3)

    def test_frame_episode_mismatch_and_packet_mutation_rejected(self):
        d=ReplayDataset(self.replay,self.vision)
        d.packet_rows[(3,0)]=d.packet_rows[(3,1)]
        with self.assertRaises(ValueError):d.frame(3,0)
        (self.vision/'f0.npz').write_bytes(b'corrupt')
        d=ReplayDataset(self.replay,self.vision)
        with self.assertRaises(ValueError):d.frame(3,0)

    def test_mask_future_gt_and_cross_split_rejected(self):
        for key in ('gt_segmentation_used','future_observations_used'):
            bad=copy.deepcopy(self.vm);bad['episodes'][0]['mask_provenance'][key]=True
            write_json(self.vision/'manifest.json',bad)
            with self.assertRaises(ValueError):ReplayDataset(self.replay,self.vision)
        bad=copy.deepcopy(self.vm);bad['splits']['test']=[3]
        write_json(self.vision/'manifest.json',bad)
        with self.assertRaises(ValueError):ReplayDataset(self.replay,self.vision)

    def test_terminal_crossing_rejected(self):
        d=ReplayDataset(self.replay,self.vision);d.raw[3]['truncated'][0]=True
        with self.assertRaises(ValueError):d.sequence(3,0,2)


if __name__=='__main__':
    unittest.main()
