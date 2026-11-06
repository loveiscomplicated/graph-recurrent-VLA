import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from graph.subtasks.generate import episode_split
from graph.subtasks.specs import NAMES


class GenerationUnitTests(unittest.TestCase):
    def test_duplicate_seed_and_split_determinism(self):
        with self.assertRaises(ValueError):
            episode_split([1,1,2])
        seeds = list(range(10000,10050))
        split = episode_split(seeds)
        self.assertEqual(split,episode_split(seeds))
        self.assertEqual({k:len(v) for k,v in split.items()},dict(train=40,validation=5,test=5))
        self.assertEqual(set(sum(split.values(),[])),set(seeds))
        self.assertEqual(len(sum(split.values(),[])),len(seeds))
        self.assertEqual({k:len(v) for k,v in episode_split([1,2,3]).items()},dict(train=1,validation=1,test=1))

    def test_nonempty_output_never_overwritten(self):
        from graph.subtasks.generate import generate_expert_data
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'preserved.txt'
            p.write_text('user data')
            with self.assertRaises(FileExistsError):
                generate_expert_data(output_dir=tmp)
            self.assertEqual(p.read_text(),'user data')

    def test_split_source_leak_rejection(self):
        from graph.subtasks.dataset import validate_splits
        def episode(eid,split,seed):
            return dict(episode_id=eid,split=split,segments=[dict(source_episode_id=eid,split=split,
                        initial_state_ref=dict(episode_seed=seed,reset_kwargs={'seed':seed,'options':None}))])
        d=dict(episodes=[episode(0,'train',4),episode(1,'test',4)],splits={'train':[0],'test':[1]})
        with self.assertRaises(ValueError):
            validate_splits(d)
        d['episodes'][1]['segments'][0]['initial_state_ref']=dict(episode_seed=5,reset_kwargs={'seed':5})
        validate_splits(d)
        d['splits']['test'].append(0)
        with self.assertRaises(ValueError):
            validate_splits(d)


@unittest.skipUnless(os.environ.get('SUBTASK_INTEGRATION')=='1','set SUBTASK_INTEGRATION=1 for actual CPU execution')
class IndependentCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_five_resets_and_success(self):
        from graph.subtasks.env import make_subtask_env
        for name in NAMES:
            with self.subTest(subtask=name):
                env = make_subtask_env(name)
                try:
                    obs,info = env.reset(options={'source_episode_id':0,'subtask':name})
                    self.assertTrue(info['reset_valid'])
                    self.assertEqual(info['subtask_step'],0)
                    self.assertTrue(env.subtask_spec.check_start(env.metrics,env.context))
                    self.assertFalse(env.subtask_spec.check_success(env.metrics,env.context))
                    start_obs = obs.clone()
                    i,j = env.segment['start_t'],env.segment['end_t']
                    for action in env.audit['actions'][i:j]:
                        obs,reward,terminated,truncated,info = env.step(action)
                        if terminated or truncated:
                            break
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertTrue(info['subtask_success'])
                    self.assertEqual(reward,1.)
                    self.assertEqual(info['subtask_step'],env.learner_steps)
                    self.assertNotEqual(info['subtask_step'],env.context.t if i else -1)
                    with self.assertRaises(RuntimeError):
                        env.step(action)
                    repeat,_ = env.reset(options={'source_episode_id':0})
                    torch.testing.assert_close(repeat,start_obs,atol=1e-6,rtol=0)
                    self.assertEqual(env.max_prefix_observation_error,0.)
                finally:
                    env.close()

    def test_fresh_seed_learner_action_and_timeout(self):
        from graph.subtasks.env import make_subtask_env
        env = make_subtask_env('reach',index=None,max_steps=1)
        try:
            obs,info = env.reset(seed=10000)
            self.assertEqual(info['initialization'],'fresh_reset')
            original = obs.clone()
            action = torch.tensor([.1,0,0,0,0,0,1.])
            obs,reward,terminated,truncated,info = env.step(action)
            self.assertFalse(torch.equal(obs,original))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['failure_reason'],'timeout')
            self.assertFalse(info['subtask_failure'])
            self.assertEqual(info['subtask_step'],1)
            repeat,_ = env.reset(seed=10000)
            torch.testing.assert_close(repeat,original,atol=1e-6,rtol=0)
        finally:
            env.close()
        with self.assertRaises(ValueError):
            make_subtask_env('reach',index=None,max_steps=0)

    def test_selection_and_invalid_start_rejection(self):
        from graph.subtasks.env import make_subtask_env,InvalidSubtaskReset
        env = make_subtask_env('insert')
        try:
            with self.assertRaises(ValueError):
                env.reset(seed=1)
            with self.assertRaises(ValueError):
                env.reset(seed=3,options={'source_episode_id':0})
            segment = env.dataset['episodes'][0]['segments'][-1]
            old = segment['start_t']
            segment['start_t'] = segment['end_t']
            with self.assertRaises(InvalidSubtaskReset):
                env.reset(options={'source_episode_id':0})
            segment['start_t'] = old
            with self.assertRaises(RuntimeError):
                env.step(torch.zeros(7))
        finally:
            env.close()
        with self.assertRaises(ValueError):
            make_subtask_env('reach',control_mode='pd_joint_pos')

    def test_failure_flags_and_action_validation(self):
        from unittest.mock import patch
        from graph.subtasks.env import make_subtask_env
        env = make_subtask_env('reach',index=None,max_steps=1)
        try:
            env.reset(seed=10000)
            for bad in (torch.zeros(8),torch.full((7,),float('nan')),torch.full((7,),2.)):
                with self.assertRaises(ValueError):
                    env.step(bad)
            self.assertEqual(env.learner_steps,0)
            # Fault injection tests wrapper routing, not a new physical threshold.
            # The real CPU simulator still executes this action.
            with patch.object(env.subtask_spec,'failure_reasons',return_value=['workspace_exit']):
                _,reward,terminated,truncated,info = env.step(torch.tensor([0.,0,0,0,0,0,1.]))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info['subtask_failure'])
            self.assertFalse(info['subtask_success'])
            self.assertEqual(info['failure_reason'],'workspace_exit')
            self.assertEqual(reward,0.)
        finally:
            env.close()


@unittest.skipUnless(os.environ.get('SUBTASK_GENERATION_INDEX'),'set SUBTASK_GENERATION_INDEX to verified generated index')
class GeneratedDatasetIntegrationTests(unittest.TestCase):
    def test_save_load_geometry_alignment_splits_and_replay(self):
        from graph.subtasks.dataset import load_segments,segment_arrays
        from graph.subtasks.env import make_subtask_env
        index_path = os.environ['SUBTASK_GENERATION_INDEX']
        index = load_segments(index_path)
        ids = sum(index['splits'].values(),[])
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{e['episode_id'] for e in index['episodes']})
        uid_split = {}
        for ep in index['episodes']:
            for s in ep['segments']:
                key = s['source_uid']
                if key in uid_split:
                    self.assertEqual(uid_split[key],s['split'])
                uid_split[key] = s['split']
                if s['status'] == 'detected':
                    arrays = segment_arrays(index,s)
                    self.assertEqual(len(arrays['states']),len(arrays['actions'])+1)
        torch.set_num_threads(4)
        for name in NAMES:
            candidates = [e for e in index['episodes'] if next(s for s in e['segments'] if s['subtask']==name)['status']=='detected']
            self.assertTrue(candidates,name)
            env = make_subtask_env(name,index=index_path)
            try:
                env.reset(options={'source_episode_id':candidates[0]['episode_id']})
                i,j = env.segment['start_t'],env.segment['end_t']
                for a in env.audit['actions'][i:j]:
                    _,_,terminated,truncated,info = env.step(a)
                    if terminated or truncated:
                        break
                self.assertTrue(info['subtask_success'],name)
                # Exercise the same policy entry point used by future learners,
                # on an explicit paired start (expert actions stand in for policy).
                from graph.subtasks.evaluate import evaluate_policy
                from graph.prototype.data import sha256
                start=dict(subtask=name,source_episode_id=candidates[0]['episode_id'],
                    seed=env.segment['initial_state_ref']['episode_seed'],
                    index_sha256=sha256(index_path),source_sha256=index['source_sha256'])
                actions=iter(env.audit['actions'][i:j])
                result=evaluate_policy(env,lambda obs:next(actions),[start])[0]
                self.assertTrue(result['subtask_success'])
                self.assertLessEqual(result['steps'],j-i)
                self.assertEqual(result['initialization'],'expert_prefix')
                with self.assertRaises(ValueError):
                    evaluate_policy(env,lambda obs:None,[dict(start,index_sha256='wrong')])
            finally:
                env.close()


if __name__=='__main__':
    unittest.main()
