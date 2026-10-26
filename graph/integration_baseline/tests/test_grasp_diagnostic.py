"""Detect real path/hidden/input mistakes with the unchanged baseline model."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from graph.integration_baseline.data import AlignedFrame
from graph.integration_baseline.grasp_diagnostic import (reference_episode,assert_same_inputs,
    compare_streams,path_checks,controller_commands,select_episodes)
from graph.integration_baseline.graph import build_graph
from graph.integration_baseline.model import IntegrationPolicy,EpisodeStream
from graph.integration_baseline.robot import RobotNormalizer
from graph.integration_baseline.tests.test_baseline import fixture_packet,robot_fixture,ROBOT
from graph.prototype.v2.state import StateNormalizer


class TinyDataset:
    def __init__(self,frames):
        self.frames=frames;self.episodes={17:dict(actions=len(frames))}

    def sequence(self,eid,start,length):
        assert eid==17
        return self.frames[start:start+length]


class GraspDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.schema=robot_fixture()[1];self.model=IntegrationPolicy(self.schema).eval()
        self.norm=RobotNormalizer(self.schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[17])
        self.states=np.random.default_rng(1).normal(size=(5,76)).astype(np.float32)
        self.frames=[AlignedFrame(17,t,fixture_packet((0,1) if t==2 else ()),self.states[t],ROBOT,
                     np.zeros(7,np.float32),False,False) for t in range(5)]
        self.config=dict(device='cpu',training=dict(sequence_length=2),graph=dict(k=4,M=16,block_rows=64))
        self.tolerance=dict(atol=1e-6,rtol=1e-5)

    def test_actual_evaluator_and_rollout_match_with_common_and_carried_hidden(self):
        ref=reference_episode(TinyDataset(self.frames),self.model,self.norm,17,self.config)
        graphs=[build_graph(f.packet,ROBOT) for f in self.frames]
        assert_same_inputs(ref,graphs,self.states)
        result=compare_streams(self.model,self.norm,graphs,self.states,ref,'fixture')
        self.assertTrue(all(c['pass_tolerance'] for c in path_checks(result,ref,self.tolerance).values()))
        self.assertEqual(ref['metrics']['count'],5)
        self.assertEqual(ref['pre_hidden'].shape,(5,1,1,144))
        self.assertTrue(np.array_equal(ref['pre_hidden'][1],ref['post_hidden'][0]))

    def test_injected_hidden_reset_is_detected(self):
        class BrokenStream(EpisodeStream):
            def step(self,*args,**kwargs):
                self.hidden=None
                return super().step(*args,**kwargs)
        ref=reference_episode(TinyDataset(self.frames),self.model,self.norm,17,self.config)
        graphs=[build_graph(f.packet,ROBOT) for f in self.frames]
        result=compare_streams(self.model,self.norm,graphs,self.states,ref,'fixture',stream_type=BrokenStream)
        checks=path_checks(result,ref,self.tolerance)
        self.assertFalse(checks['same_hidden_forward_vs_rollout']['pass_tolerance'])
        self.assertFalse(checks['chunk_reference_vs_carried_step']['pass_tolerance'])

    def test_changed_graph_is_rejected_before_comparing_outputs(self):
        ref=reference_episode(TinyDataset(self.frames),self.model,self.norm,17,self.config)
        graphs=[build_graph(f.packet,ROBOT) for f in self.frames]
        graphs[1].node_input[0,0]+=1
        with self.assertRaises(ValueError):assert_same_inputs(ref,graphs,self.states)

    def test_selection_uses_input_coverage_without_predictions(self):
        phases=[dict(before=p) for p in ['reach','descend','close','close']]
        d=SimpleNamespace(splits=dict(train=[3,2,1],validation=[12,11,10]),
            episodes={e:dict(phases=phases) for e in [1,2,3,10,11,12]},
            graph_manifest=dict(episodes=[dict(episode_id=e,packets=[dict(timestep=t,object_counts=([1,1] if e in [3,11] else [0,1])) for t in range(4)]) for e in [1,2,3,10,11,12]]))
        selected=select_episodes(d,dict(splits=['train','validation'],post_close_context_steps=8))
        self.assertEqual([e['episode_id'] for e in selected],[1,3,10,11])

    def test_controller_intent_is_raw_native_scaling(self):
        action=np.array([[2,-2,.5,0,0,2,2]],np.float32)
        np.testing.assert_allclose(controller_commands(action),[[.1,-.1,.05,0,0,-.1,.04]],atol=1e-8,rtol=0)


if __name__=='__main__':unittest.main()
