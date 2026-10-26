"""Evaluation sidecar: replay the frozen Expert Action7, without generating data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from envs.maniskill import get_gt_state
from graph.integration_baseline.bc08_eval import environment
from graph.integration_baseline.direct_dataset import DirectDataset, load_episode
from graph.integration_baseline.direct_runtime import atomic, read
from graph.integration_baseline.rollout_stop_clock import numpy, reset_diagnostics
from graph.integration_baseline.robot import robot_observation, flatten_state


def run(plan_path):
    plan_path=Path(plan_path).resolve()
    plan=read(plan_path);root=plan_path.parent
    ds=DirectDataset(plan['dataset_directory'])
    original=ds.episodes[30003]
    raw=load_episode(ds.run/original['path'])
    env,conditions=environment(plan)
    trace=[];max_robot_difference=0.;success_steps=[]
    try:
        env.reset(seed=30003)
        reset=reset_diagnostics(env)
        if reset!=original['reset_group']:
            raise ValueError('frozen expert replay reset mismatch')
        for t,action in enumerate(raw['actions']):
            gt=get_gt_state(env)
            robot=flatten_state(robot_observation(env)[0],ds.schema)
            max_robot_difference=max(max_robot_difference,float(np.max(np.abs(robot-raw['robot_state'][t]))))
            pre={name:numpy(getattr(gt,name))[0].tolist() for name in
                 ('ee_pose','peg_pose','box_pose','left_fingertip_pos','right_fingertip_pos')}
            command=torch.from_numpy(action[None].copy()).to(env.unwrapped.device)
            _,_,terminated,truncated,info=env.step(command)
            after=get_gt_state(env)
            post={name:numpy(getattr(after,name))[0].tolist() for name in
                  ('ee_pose','peg_pose','box_pose','left_fingertip_pos','right_fingertip_pos')}
            contact=np.stack([numpy(env.unwrapped.scene.get_pairwise_contact_forces(link,env.unwrapped.peg))[0]
                for link in [env.unwrapped.agent.finger1_link,env.unwrapped.agent.finger2_link]])
            success=bool(info['success'].item())
            if success:success_steps.append(t)
            trace.append(dict(timestep=t,pre=pre,post=post,action7=action.tolist(),
                              grasp=bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg).item()),
                              finger_forces=contact.tolist(),success=success,
                              terminated=bool(terminated.item()),truncated=bool(truncated.item())))
            if terminated.item() or truncated.item():break
        if len(trace)!=len(raw['actions']) or success_steps!=[136]:
            raise ValueError('existing Expert Action7 replay does not reproduce saved success')
        first_close=int(np.flatnonzero(raw['actions'][:,6]<0)[0])
        d=trace[first_close]
        close_distance=float(np.linalg.norm(np.asarray(d['pre']['ee_pose'][:3])-np.asarray(d['pre']['peg_pose'][:3])))
        result=dict(status='PASS' if max_robot_difference<=1e-4 else 'STATE_DIVERGENCE',
                    scope='evaluation-only replay of existing saved Action7; no new Expert actions or Student inputs',
                    reset_group=reset,episode_id=30003,actions=len(trace),success_step=success_steps[0],
                    max_robot76_difference_from_saved=max_robot_difference,
                    first_close_timestep=first_close,tcp_peg_distance_before_close_m=close_distance,
                    close_before_geometry=d['pre'],conditions=conditions,trace=trace)
        atomic(root/'expert_replay_gt.json',result,immutable=True)
        return result
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.plan)
    print(json.dumps({k:v for k,v in result.items() if k not in ('conditions','trace')},indent=2),flush=True)
