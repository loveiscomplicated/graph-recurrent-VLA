"""Run separately: SAPIEN render-device selection is process-global."""
import unittest
from unittest.mock import patch
import torch
from envs.maniskill import make_peg_insertion_env
from envs.maniskill.observation_device import ObservationDevice


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on current host')
class NativeCudaCameraSmoke(unittest.TestCase):
    def test_cpu_physics_native_cuda_camera_output(self):
        original=ObservationDevice.observation;raw_devices={};raw_aux={}
        def observe(wrapper,obs):
            for name,values in obs['sensor_data'].items():
                for key,value in values.items():raw_devices[name+'/'+key]=value.device.type
                raw_aux[name+'/intrinsic_cv']=obs['sensor_param'][name]['intrinsic_cv'].device.type
            raw_aux['qpos']=obs['agent']['qpos'].device.type
            return original(wrapper,obs)
        with patch.object(ObservationDevice,'observation',observe):
            env=make_peg_insertion_env(device='cpu',render_backend='cuda',
                observation_device='cuda',obs_mode='rgb+depth')
            try:
                self.assertEqual(env.unwrapped.backend.sim_backend,'physx_cpu')
                self.assertEqual(env.unwrapped.backend.render_backend,'sapien_cuda')
                self.assertEqual(env.unwrapped.device.type,'cpu')
                obs,_=env.reset(seed=42)
                self.assertTrue(raw_devices);self.assertEqual(set(raw_devices.values()),{'cuda'})
                self.assertEqual(set(raw_aux.values()),{'cpu'})
                for name in ('base_camera','hand_camera'):
                    self.assertTrue(obs['sensor_data'][name]['rgb'].is_cuda)
                    self.assertTrue(obs['sensor_data'][name]['depth'].is_cuda)
                    self.assertEqual(obs['sensor_data'][name]['depth'].dtype,torch.int16)
                    self.assertTrue(obs['sensor_param'][name]['intrinsic_cv'].is_cuda)
                self.assertTrue(obs['agent']['qpos'].is_cuda)
                action=torch.zeros((1,env.action_space.shape[-1]),device='cpu')
                obs,_,_,_,_=env.step(action)
                self.assertEqual(set(raw_devices.values()),{'cuda'})
                self.assertTrue(obs['sensor_data']['hand_camera']['rgb'].is_cuda)
            finally:env.close()


if __name__=='__main__':unittest.main()
