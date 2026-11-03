"""Shared output wrapper stays attached to live reset/step and CUDA RGB-D."""
from dataclasses import asdict,dataclass
from pathlib import Path
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from envs.maniskill.observation_device import ObservationDevice
from graph.plan_a_bc import rollout
from tests.test_observation_device import RawEnv


@dataclass
class ControllerConfig:
    mode:str='pd_ee_delta_pose'


class FakeTask(RawEnv):
    def __init__(self):
        super().__init__();config=ControllerConfig()
        self.agent=SimpleNamespace(controller=SimpleNamespace(controllers={'arm':SimpleNamespace(config=config)}))
        self.control_freq=20;self.sim_freq=100;self._sim_steps_per_control=5
        self.elapsed_steps=torch.zeros(1,dtype=torch.int64);self.device=torch.device('cpu');self.num_envs=1
    def step(self,action):
        self.elapsed_steps+=1
        return super().step(action)


class LiveOutputWiring(unittest.TestCase):
    def test_shared_wrapper_survives_custom_horizon(self):
        task=FakeTask();config={'controller_config':{'arm':asdict(ControllerConfig())}}
        with patch.object(rollout,'make_peg_insertion_env',return_value=gym.Wrapper(task)) as make, \
             patch.object(rollout,'read',return_value=config), \
             patch('envs.maniskill.observation_device.observation_to_device',side_effect=lambda obs,device:obs) as transfer:
            env=rollout.environment(Path('unused'))
            self.addCleanup(env.close)
            self.assertIsInstance(env,ObservationDevice)
            self.assertEqual(env.observation_device,torch.device('cuda'))
            self.assertIs(env.unwrapped,task);self.assertEqual(env.env._max_episode_steps,250)
            self.assertEqual(make.call_args.kwargs['device'],'cpu')
            self.assertEqual(make.call_args.kwargs['render_backend'],'cuda')
            self.assertEqual(make.call_args.kwargs['sensor_configs'],
                {name:dict(width=512,height=512) for name in ('base_camera','hand_camera')})
            env.reset(seed=42);action=torch.zeros(1);env.step(action)
            self.assertEqual(transfer.call_count,2);self.assertIs(task.action,action)
            self.assertTrue(all(call.args[1]==torch.device('cuda') for call in transfer.call_args_list))
            task.elapsed_steps[:]=99
            self.assertFalse(bool(env.step(action)[3].item()))
            task.elapsed_steps[:]=249
            self.assertTrue(bool(env.step(action)[3].item()))


def leaves(tree):
    if isinstance(tree,dict):
        for value in tree.values():yield from leaves(value)
    elif isinstance(tree,(tuple,list)):
        for value in tree:yield from leaves(value)
    else:yield tree


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get('TASK10I_LIVE_ROOT'),
    'native CUDA renderer and existing live assets required; run in a separate process')
class NativeLiveObservation(unittest.TestCase):
    def test_reset_step_and_sensor_adapter_keep_native_images(self):
        from graph.integration_baseline.robot import robot_observation,state_schema
        from graph.persistent_geometry.tensor_camera import TensorCameraObservation
        original=ObservationDevice.observation;calls=[]
        def observe(wrapper,obs):
            images={name:{k:(v.device.type,v.dtype,tuple(v.shape),v.data_ptr())
                for k,v in values.items()} for name,values in obs['sensor_data'].items()}
            calls.append(images)
            returned=original(wrapper,obs)
            for name,values in images.items():
                for k,(_,_,_,pointer) in values.items():self.assertEqual(returned['sensor_data'][name][k].data_ptr(),pointer)
            return returned
        with patch.object(ObservationDevice,'observation',observe):
            env=rollout.environment(Path(os.environ['TASK10I_LIVE_ROOT']))
            try:
                self.assertIsInstance(env,ObservationDevice)
                self.assertEqual(env.unwrapped.backend.sim_backend,'physx_cpu')
                self.assertEqual(env.unwrapped.backend.render_backend,'sapien_cuda')
                self.assertEqual(env.unwrapped.device.type,'cpu')
                obs,_=env.reset(seed=42)
                for t in range(3):
                    tensors=[x for x in leaves(obs) if isinstance(x,torch.Tensor)]
                    self.assertTrue(tensors);self.assertTrue(all(x.is_cuda for x in tensors))
                    self.assertFalse(any(isinstance(x,np.ndarray) for x in leaves(obs)))
                    schema=state_schema(robot_observation(env)[0]);image_storage={v.untyped_storage().data_ptr()
                        for values in obs['sensor_data'].values() for v in values.values()}
                    to_numpy=rollout.numpy
                    def guard(x):
                        if isinstance(x,torch.Tensor) and x.untyped_storage().data_ptr() in image_storage:
                            raise AssertionError('live image downloaded in sensors')
                        return to_numpy(x)
                    with patch.object(rollout,'numpy',side_effect=guard):
                        cameras,data,surfaces=rollout.sensors(env,obs,'native_live_test',t,schema)
                    for c in cameras:
                        self.assertIsInstance(c,TensorCameraObservation)
                        self.assertEqual(tuple(c.rgb.shape),(512,512,3))
                        self.assertEqual(tuple(c.depth_raw_mm.shape),(512,512))
                        camera=env.unwrapped._sensors[c.camera]
                        np.testing.assert_allclose(c.intrinsic,
                            camera.camera.get_intrinsic_matrix()[0].cpu().numpy(),rtol=0,atol=0)
                        np.testing.assert_allclose(c.intrinsic[:2,2],[256.,256.],rtol=0,atol=1e-5)
                        self.assertTrue(c.rgb.is_cuda);self.assertTrue(c.depth_raw_mm.is_cuda)
                        self.assertEqual(c.rgb.dtype,torch.uint8);self.assertEqual(c.depth_raw_mm.dtype,torch.int16)
                        torch.testing.assert_close(c.rgb,obs['sensor_data'][c.camera]['rgb'][0],rtol=0,atol=0)
                        torch.testing.assert_close(c.depth_raw_mm,obs['sensor_data'][c.camera]['depth'][0,...,0],rtol=0,atol=0)
                        # Own the current sensor snapshot even when renderer reuses buffers.
                        self.assertNotEqual(c.rgb.data_ptr(),obs['sensor_data'][c.camera]['rgb'].data_ptr())
                    if t<2:obs,_,_,_,_=env.step(torch.zeros((1,env.action_space.shape[-1]),device='cpu'))
                self.assertEqual(len(calls),3)
                self.assertTrue(all(value[0]=='cuda' for call in calls for values in call.values() for value in values.values()))
                type(self).proof=dict(sim_backend=env.unwrapped.backend.sim_backend,
                    render_backend=env.unwrapped.backend.render_backend,physics_device=str(env.unwrapped.device),
                    reset_calls=1,step_calls=2,all_observation_tensor_leaves_cuda=True,
                    native_images_not_reuploaded=True,sensors_did_not_download_images=True,
                    camera_snapshots_own_buffers=True,cameras={c.camera:dict(
                        rgb_device=str(c.rgb.device),rgb_dtype=str(c.rgb.dtype),rgb_shape=list(c.rgb.shape),
                        depth_device=str(c.depth_raw_mm.device),depth_dtype=str(c.depth_raw_mm.dtype),
                        depth_shape=list(c.depth_raw_mm.shape),intrinsic_cv=c.intrinsic.tolist()) for c in cameras})
            finally:env.close()


if __name__=='__main__':unittest.main()
