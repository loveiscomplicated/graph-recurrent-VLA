"""Recorder lifecycle/negative gates; fixtures never establish Expert success."""
from pathlib import Path
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.integration_baseline.collect_recurrent_expert import reset_for_recording, validate_episode
from graph.integration_baseline.validate_direct_expert import DirectExpertReader


class RecorderLifecycle(unittest.TestCase):
    def test_reset_replaces_controller_and_audits_new_instance(self):
        received=[]
        def controller():
            child=SimpleNamespace(_preprocess_action=lambda a:a*2)
            def set_action(a): received.append(child._preprocess_action(a).clone())
            return SimpleNamespace(set_action=set_action,controllers={'arm':child})
        task=SimpleNamespace(agent=SimpleNamespace(controller=controller()),elapsed_steps=torch.tensor([0]))
        old=task.agent.controller
        def reset(seed):
            task.agent.controller=controller()
            return {'seed':seed},{}
        env=SimpleNamespace(unwrapped=task,reset=reset)
        with patch('graph.integration_baseline.collect_recurrent_expert.robot_observation',
                   return_value=({},np.zeros((3,3),np.float32))), \
             patch('graph.integration_baseline.collect_recurrent_expert.flatten_state',
                   return_value=np.zeros(76,np.float32)):
            obs,_,audit=reset_for_recording(env,2000,{})
            self.assertIsNot(task.agent.controller,old)
            command=torch.arange(7,dtype=torch.float32)[None]
            task.agent.controller.set_action(command)
            np.testing.assert_array_equal(audit['controller_input'],command[0].numpy())
            np.testing.assert_array_equal(audit['arm_scaled_input'],received[0][0].numpy())
            self.assertEqual(obs['seed'],2000)

    def test_aborted_real_collector_artifact_is_rejected(self):
        run=Path('experiments/5a71640e4a9f4036b109bcaea7c17824')
        if not run.exists(): self.skipTest('optional preserved Task06-B diagnostic artifact absent')
        with self.assertRaisesRegex(AssertionError,'partial collector'):
            DirectExpertReader(run)


class ActualRecordingNegativeGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_dir=Path(os.environ.get('TASK06B_RUN','experiments/9fec4f02fc9d4172a48e97d5eff90abd'))
        if not cls.run_dir.exists(): raise unittest.SkipTest('actual optional pilot artifact absent')
        cls.reader=DirectExpertReader(cls.run_dir)
        cls.eid=next(iter(cls.reader.raw))

    def arrays(self):return {k:v.copy() for k,v in self.reader.raw[self.eid].items()}

    def test_native_command_mutation_rejected(self):
        a=self.arrays();a['actions'][0,0]+=.001
        with self.assertRaises(AssertionError):validate_episode(a,self.reader.schema,self.eid)

    def test_one_frame_robot_shift_rejected(self):
        a=self.arrays();a['robot_state']=np.roll(a['robot_state'],1,axis=0)
        with self.assertRaises(AssertionError):validate_episode(a,self.reader.schema,self.eid)

    def test_episode_cross_contamination_rejected(self):
        a=self.arrays();a['episode_id'][1]+=1
        with self.assertRaises(AssertionError):validate_episode(a,self.reader.schema,self.eid)

    def test_bad_controller_scaling_rejected(self):
        a=self.arrays();a['arm_scaled_input'][0,0]+=.001
        with self.assertRaises(AssertionError):validate_episode(a,self.reader.schema,self.eid)

    def test_rgb_row_mutation_rejected_by_independent_recorder(self):
        a=self.arrays();a['base_camera/rgb'][0,0,0,0]^=1
        path=self.run_dir/self.reader.episodes[self.eid]['path']
        with self.assertRaises(AssertionError):validate_episode(a,self.reader.schema,self.eid,path.with_suffix('.h5'))

    def test_final_observation_has_no_target(self):
        t=len(self.reader.raw[self.eid]['actions'])
        with self.assertRaisesRegex(ValueError,'no action'):
            self.reader.frame(self.eid,t,Path('/not-read'))


if __name__=='__main__': unittest.main()
