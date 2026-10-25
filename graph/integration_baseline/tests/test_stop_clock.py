"""Timing, recurrent, action, manifest and performance-report contracts."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from graph.integration_baseline.direct_runtime import read
from graph.integration_baseline.graph import build_graph,pack_graphs
from graph.integration_baseline.model import IntegrationPolicy,EpisodeStream
from graph.integration_baseline.robot import RobotNormalizer
from graph.integration_baseline.rollout_stop_clock import (PhysicsAudit,validate_config,restore_stream_hidden,
    controller_semantics,infer,validate_controller_command,execute)
from graph.integration_baseline.rollout_metrics import summarize,wilson,prepare_seeds,validate_seeds
from graph.integration_baseline.tests.test_baseline import fixture_packet,robot_fixture,ROBOT
from graph.prototype.v2.state import StateNormalizer


class StopClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_physics_contract_detects_inference_step_and_wrong_substep(self):
        scene=SimpleNamespace(step=lambda:None)
        task=SimpleNamespace(scene=scene,elapsed_steps=torch.tensor([0]))
        env=SimpleNamespace(unwrapped=task)
        audit=PhysicsAudit(env)
        try:
            token=audit.token();audit.unchanged(token)
            scene.step()
            with self.assertRaises(ValueError):audit.unchanged(token)
            with self.assertRaises(ValueError):audit.stepped(token)
            for _ in range(4):scene.step()
            task.elapsed_steps+=1
            audit.stepped(token)
        finally:audit.close()

    def test_gru_carry_empty_reset_gaps_and_sequence_equivalence(self):
        schema=robot_fixture()[1];model=IntegrationPolicy(schema).eval()
        norm=RobotNormalizer(schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[1])
        graphs=[build_graph(fixture_packet(empty),ROBOT) for empty in ((),(0,1),())]
        states=torch.randn(3,76)
        stream=EpisodeStream(model,norm)
        with torch.inference_mode():
            sequence,expected=model(pack_graphs(graphs),states,norm)
            values=[];h=[]
            for t,g in enumerate(graphs):
                values.append(stream.step(pack_graphs([g]),states[t:t+1],'ep',t))
                h.append(stream.hidden.clone())
            torch.testing.assert_close(torch.stack(values,1),sequence)
            torch.testing.assert_close(stream.hidden,expected)
            self.assertFalse(torch.equal(h[0],h[1]))
            with self.assertRaises(ValueError):stream.step(pack_graphs([graphs[0]]),states[:1],'ep',2)
            with self.assertRaises(ValueError):stream.step(pack_graphs([graphs[0]]),states[:1],'ep',4)
            stream.reset();self.assertIsNone(stream.hidden)
            zero=model.step(pack_graphs([graphs[0]]),states[:1],norm)[0]
            for flags in ({},{'previous_terminated':True},{'previous_truncated':True}):
                got=stream.step(pack_graphs([graphs[0]]),states[:1],'new',0,**flags)
                torch.testing.assert_close(got,zero)
            self.assertEqual(got.shape,(1,7));self.assertTrue(torch.isfinite(got).all())

    def test_smoke_never_reports_random_policy_success_rate(self):
        result=summarize([dict(group='smoke',status='CONTRACT_SMOKE',success=True)],smoke=True)
        self.assertIsNone(result['policy_success_rate'])
        self.assertEqual(result['status'],'ROLLOUT_NOT_EXECUTED')
        full=summarize([dict(group='heldout',status='ERROR',success=False)],smoke=False)
        self.assertIsNone(full['groups']['heldout']['success_rate'])
        self.assertEqual(full['status'],'INCOMPLETE')
        self.assertEqual(wilson(0,0),None)
        self.assertLess(wilson(0,40)[1],.1)
        self.assertGreater(wilson(40,40)[0],.9)

    def test_restorable_hidden_has_episode_timestep_identity(self):
        schema=robot_fixture()[1];model=IntegrationPolicy(schema).eval()
        norm=RobotNormalizer(schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[])
        graph=pack_graphs([build_graph(fixture_packet(),ROBOT)]);state=torch.zeros(1,76)
        stream=EpisodeStream(model,norm)
        _,timing=infer(stream,graph,state,'ep',0)
        self.assertTrue({'gnn_readout_ms','gru_ms','gnn_gru_ms'} <= set(timing))
        hidden=stream.hidden.detach().numpy().copy()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'hidden.npz'
            np.savez(path,hidden=hidden[None],timestep=[0],episode_identity='ep')
            new=EpisodeStream(model,norm)
            restore_stream_hidden(new,path,'ep',0)
            with torch.inference_mode():
                a=stream.step(graph,state,'ep',1);b=new.step(graph,state,'ep',1)
            torch.testing.assert_close(a,b)
            with self.assertRaises(ValueError):restore_stream_hidden(new,path,'other',0)
            with self.assertRaises(ValueError):restore_stream_hidden(new,path,'ep',1)

    def test_controller_semantics_preserve_values_across_install_paths(self):
        a=dict(arm=dict(urdf_path='/local/panda.urdf',rot_lower=-.1))
        b=dict(arm=dict(urdf_path='/server/panda.urdf',rot_lower=-.1))
        self.assertEqual(controller_semantics(a),controller_semantics(b))
        b['arm']['rot_lower']=.1
        self.assertNotEqual(controller_semantics(a),controller_semantics(b))

    def test_controller_scaling_rotation_sign_and_native_unclipped_values(self):
        action=np.array([2,-2,.5,0,0,2,2],np.float32)
        audit=dict(controller_input=action.copy(),
                   arm_scaled_input=np.array([.1,-.1,.05,0,0,-.1],np.float32),
                   gripper_scaled_input=np.array([.04],np.float32))
        validate_controller_command(action,audit)
        audit['arm_scaled_input'][-1]=.1
        with self.assertRaises(AssertionError):validate_controller_command(action,audit)

    def test_evaluate_without_checkpoint_and_seeds_is_never_started(self):
        args=SimpleNamespace(config=Path(__file__).parents[1]/'rollout_config.json',mode='evaluate',
                             checkpoint_manifest=None,seeds_manifest=None)
        with self.assertRaisesRegex(ValueError,'checkpoint AND frozen seeds'):
            execute(args)

    def test_config_fixes_20hz_and_rejects_50hz(self):
        c=read(Path(__file__).parents[1]/'rollout_config.json');validate_config(c)
        c['environment']['control_freq']=50
        with self.assertRaises(ValueError):validate_config(c)

    def test_seeds_overlap_tampering_and_separate_groups(self):
        c=read(Path(__file__).parents[1]/'rollout_config.json')
        p=dict(split=dict(splits=dict(train=[1],validation=[2],test=list(range(3,15))),
            reset_groups=dict(test=list(map(str,range(3,15))))),
            dataset=dict(episodes=[dict(episode_id=i,seed=i) for i in range(1,15)]),
            identity=dict(split_sha256='split'),manifest=dict(files=dict(checkpoint=dict(sha256='checkpoint'))))
        m=prepare_seeds(p,c)
        self.assertEqual(len(validate_seeds(m,p,c)),52)
        m['groups']['heldout'][0]['seed']=1
        with self.assertRaises(ValueError):validate_seeds(m,p,c)
        c['heldout_seeds'][0]=1
        with self.assertRaises(ValueError):prepare_seeds(p,c)


if __name__=='__main__':
    unittest.main()
