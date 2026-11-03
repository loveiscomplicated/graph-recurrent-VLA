"""Raw observation values/devices and independent CPU physics configuration."""
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env
from envs.maniskill.observation_device import ObservationDevice, observation_to_device


class RawEnv(gym.Env):
    observation_space=gym.spaces.Box(0,255,(1,),np.uint8)
    action_space=gym.spaces.Box(-1,1,(1,),np.float32)

    def __init__(self):
        self.raw={
            'sensor_data':{'base_camera':{
                'rgb':torch.arange(18,dtype=torch.uint8).reshape(1,2,3,3),
                'depth':torch.tensor([0,-32768,-1,1,1000,32767],dtype=torch.int16).reshape(1,2,3,1)}},
            'sensor_param':{'intrinsic_cv':torch.eye(3)[None]},
            'agent':{'qpos':torch.arange(9,dtype=torch.float32)[None]},
        }
        self.info={'elapsed_steps':torch.tensor([0])}

    def reset(self,*,seed=None,options=None):return self.raw,self.info
    def step(self,action):
        self.action=action
        return self.raw,torch.tensor([0.]),torch.tensor([False]),torch.tensor([False]),self.info


def tensor_pairs(a,b):
    if isinstance(a,dict):
        for k in a:yield from tensor_pairs(a[k],b[k])
    else:yield a,b


class OutputDeviceContracts(unittest.TestCase):
    def test_cpu_same_device_does_not_copy_or_modify(self):
        raw=RawEnv().raw;out=observation_to_device(raw,'cpu')
        for a,b in tensor_pairs(raw,out):self.assertIs(a,b)
        values=(np.array([-32768,0,32767],np.int16),[torch.ones(2),'metadata'])
        out=observation_to_device(values,'cpu')
        self.assertEqual(out[0].dtype,torch.int16);np.testing.assert_array_equal(out[0].numpy(),values[0])
        self.assertIs(out[1][0],values[1][0]);self.assertEqual(out[1][1],'metadata')

    def test_wrapper_changes_only_observation(self):
        raw=RawEnv();env=ObservationDevice(raw,'cpu');obs,info=env.reset()
        self.assertIs(info,raw.info);action=torch.zeros(1);result=env.step(action)
        self.assertIs(raw.action,action);self.assertIs(result[-1],info)
        for a,b in tensor_pairs(raw.raw,obs):self.assertIs(a,b)

    def test_factory_keeps_cpu_physics_and_renderer(self):
        raw=RawEnv()
        with patch('envs.maniskill.peg_insertion.gym.make',return_value=raw) as make:
            env=make_peg_insertion_env(device='cpu',observation_device='cpu',seed=42)
        self.assertIs(env.unwrapped,raw)
        self.assertEqual(make.call_args.kwargs['sim_backend'],'physx_cpu')
        self.assertEqual(make.call_args.kwargs['render_backend'],'cpu')
        self.assertNotIn('observation_device',make.call_args.kwargs)
        with patch('envs.maniskill.peg_insertion.gym.make',return_value=raw) as make:
            make_peg_insertion_env(device='cpu',render_backend='cuda')
        self.assertEqual(make.call_args.kwargs['sim_backend'],'physx_cpu')
        self.assertEqual(make.call_args.kwargs['render_backend'],'cuda')
        with patch('envs.maniskill.peg_insertion.torch.cuda.is_available',return_value=False), \
                patch('envs.maniskill.peg_insertion.gym.make') as make:
            with self.assertRaisesRegex(RuntimeError,'observation_device'):
                make_peg_insertion_env(device='cpu',observation_device='cuda')
            make.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on current host')
    def test_cuda_raw_values_reset_step_and_no_reupload(self):
        raw=RawEnv();env=ObservationDevice(raw,'cuda');obs,info=env.reset()
        self.assertIs(info,raw.info)
        for a,b in tensor_pairs(raw.raw,obs):
            self.assertTrue(b.is_cuda);self.assertEqual(a.shape,b.shape);self.assertEqual(a.dtype,b.dtype)
            self.assertTrue(torch.equal(a,b.cpu()))
        resident=observation_to_device(obs,'cuda')
        for a,b in tensor_pairs(obs,resident):self.assertIs(a,b)
        action=torch.zeros(1);obs,_,_,_,info=env.step(action)
        self.assertIs(action,raw.action);self.assertIs(info,raw.info)
        self.assertTrue(obs['sensor_data']['base_camera']['depth'].is_cuda)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on current host')
class LiveCudaOutputSmoke(unittest.TestCase):
    def test_cpu_simulation_returns_cuda_observations(self):
        for mode in ('state','rgb+depth'):
            env=make_peg_insertion_env(device='cpu',observation_device='cuda',obs_mode=mode)
            try:
                self.assertEqual(env.unwrapped.backend.sim_backend,'physx_cpu')
                self.assertEqual(env.unwrapped.backend.render_backend,'sapien_cpu')
                self.assertEqual(env.unwrapped.device.type,'cpu')
                obs,_=env.reset(seed=42)
                for _,value in tensor_pairs(obs,obs):self.assertTrue(value.is_cuda)
                if mode=='rgb+depth':
                    for name in ('base_camera','hand_camera'):
                        self.assertEqual(obs['sensor_data'][name]['rgb'].dtype,torch.uint8)
                        self.assertEqual(obs['sensor_data'][name]['depth'].dtype,torch.int16)
                action=torch.zeros((1,env.action_space.shape[-1]),device=env.unwrapped.device)
                obs,_,_,_,_=env.step(action)
                for _,value in tensor_pairs(obs,obs):self.assertTrue(value.is_cuda)
            finally:env.close()


if __name__=='__main__':unittest.main()
