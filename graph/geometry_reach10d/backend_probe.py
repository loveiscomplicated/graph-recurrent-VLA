"""Isolated-process real GPU simulation/vectorization capability benchmark."""
import argparse
from dataclasses import asdict
import inspect
from pathlib import Path
import time
import traceback

import torch
from envs.maniskill import make_peg_insertion_env
from recurrent.expert import ScriptedExpert
from graph.plan_a_bc.common import write


def run(path,device,num_envs,render):
    result=dict(device=device,num_envs=num_envs,render_requested=render)
    env=None;start=time.perf_counter()
    try:
        if render=='explicit_cuda':
            import gymnasium as gym
            env=gym.make('PegInsertionSide-v1',num_envs=num_envs,sim_backend='physx_'+('cuda' if device=='cuda' else 'cpu'),
                render_backend='gpu',obs_mode='rgb+depth',control_mode='pd_ee_delta_pose')
        else:
            env=make_peg_insertion_env(device=device,num_envs=num_envs,obs_mode='rgb+depth',control_mode='pd_ee_delta_pose')
        obs,_=env.reset(seed=list(range(78000,78000+num_envs)) if num_envs>1 else 78000)
        task=env.unwrapped
        result.update(backend=str(task.backend),sim_device=str(task.device),
            action_shape=list(env.action_space.shape),control_freq=task.control_freq,sim_freq=task.sim_freq,
            controller={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()})
        begin=time.perf_counter()
        for _ in range(10):env.step(torch.zeros((num_envs,7),device=task.device))
        if torch.cuda.is_available():torch.cuda.synchronize()
        result.update(step_seconds=(time.perf_counter()-begin)/10,reset_and_build_seconds=begin-start)
        try:
            expert=ScriptedExpert(env);action=expert.act()
            result.update(expert_status='PASS',expert_action_shape=list(action.shape))
        except Exception as e:
            result.update(expert_status='FAIL',expert_error=type(e).__name__+': '+str(e))
        result['status']='PASS'
    except BaseException as e:
        result.update(status='FAIL',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
        result['wall_seconds']=time.perf_counter()-start;write(path,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--num-envs',type=int,default=1);p.add_argument('--render',default='factory');a=p.parse_args()
    run(a.output,a.device,a.num_envs,a.render)
