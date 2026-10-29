"""Actual reset/actor intervention/fresh ScriptedExpert/open-gripper execution.

Only this generation/evaluation module reads object GT. Raw HDF5 keeps the
original Plan A camera/Robot76 schema; GT and offsets live in a separate JSON.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time

import h5py
import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose

from graph.plan_a_bc.common import read, write, sha, append_csv, now
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.collect_recurrent_expert import (
    capture, append_h5, install_audit, array)
from graph.integration_baseline.robot import robot_observation, state_schema
from graph.integration_baseline.geometry09a import physical_state, collision_audit
from graph.integration_baseline.rollout_stop_clock import validate_controller_command
from recurrent.expert import ScriptedExpert
from recurrent.common import to_action_tensor


def target(expert, phase):
    return expert.grasp * sapien.Pose([0, 0, -.05]) if phase == 'reach' else expert.grasp


def setup(env, spec):
    obs, _ = env.reset(seed=spec['seed'])
    from graph.integration_baseline.robot import flatten_state
    robot_before=robot_observation(env)[0]
    robot_schema=state_schema(robot_before)
    before = physical_state(env)
    nominal = ScriptedExpert(env)
    peg = env.unwrapped.peg.pose.raw_pose.clone()
    peg[0, :3] += peg.new_tensor(spec['offset_mm']) * .001
    env.unwrapped.peg.set_pose(Pose.create(peg))
    after = physical_state(env)
    for key in before:
        if key != 'peg_pose':
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    np.testing.assert_allclose(after['peg_pose'][:3] - before['peg_pose'][:3],
                               np.array(spec['offset_mm']) * .001, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(after['peg_pose'][3:], before['peg_pose'][3:])
    expert = ScriptedExpert(env)
    np.testing.assert_allclose(expert.grasp.p - nominal.grasp.p,
                               np.array(spec['offset_mm']) * .001, atol=2e-6, rtol=0)
    assert abs(expert.grasp.q @ nominal.grasp.q) > 1-1e-6
    assert expert.phase == 'reach' and expert.counter == 0 and expert.lift_target is None
    valid = collision_audit(env, 'peg')
    current_obs=env.unwrapped.get_obs()
    np.testing.assert_array_equal(flatten_state(robot_before,robot_schema),
        flatten_state(robot_observation(env)[0],robot_schema))
    for camera in obs['sensor_param']:
        for key in obs['sensor_param'][camera]:
            np.testing.assert_array_equal(array(obs['sensor_param'][camera][key]),array(current_obs['sensor_param'][camera][key]))
    return current_obs, expert, dict(original={k:v.tolist() for k,v in before.items()},
        perturbed={k:v.tolist() for k,v in after.items()}, robot_box_hole_exact=True,
        robot76_base_camera_calibration_exact=True,
        target_refresh='PASS', grasp_pose=np.r_[expert.grasp.p,expert.grasp.q].tolist(),
        approach_pose=np.r_[target(expert,'reach').p,target(expert,'reach').q].tolist(),
        physical_validity=valid)


def collect_episode(root,spec,env=None):
    folder=root/'raw'/str(spec['episode_id']);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'episode.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _collect_episode(root,spec,env)


def _collect_episode(root, spec, env=None):
    folder = root/'raw'/str(spec['episode_id'])
    receipt = folder/'receipt.json'
    if receipt.exists():
        old=read(receipt)
        if old['spec'] != spec or old.get('sha256') and sha(folder/'episode.h5')!=old['sha256']:
            raise ValueError('raw resume identity/hash mismatch')
        return old
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'episode.h5').exists():
        raise ValueError('committed episode has no receipt; audit before reuse')
    own=env is None
    if own:env=environment(root)
    started=time.perf_counter();n=0;rows=[];success=False;first_reach=None
    try:
        obs,expert,audit=setup(env,spec)
        schema=state_schema(robot_observation(env)[0])
        assert schema==read(root/'normalizer.json')['schema']
        controller=install_audit(env,schema)
        if not audit['physical_validity']['valid']:
            result=dict(spec=spec,status='REJECTED',reason='initial_collision_or_workspace',audit=audit)
            write(receipt,result,immutable=True);return result
        initial_peg=np.array(audit['perturbed']['peg_pose'])
        tmp=folder/'episode.h5.partial'
        # A failed partial is retained under a unique name; never truncate it.
        if tmp.exists():tmp.rename(folder/('interrupted_'+str(time.time_ns())+'.h5'))
        with h5py.File(tmp,'x') as f:
            row,_=capture(env,obs,schema)
            for k,v in row.items():append_h5(f,k,v)
            for t in range(130):
                phase=expert.phase
                assert phase in ('reach','descend')
                goal=target(expert,phase);dp,dr=expert._err(goal)
                if phase=='reach' and dp<.01 and dr<.05:first_reach=t
                tick=time.perf_counter();command=expert.act()
                if expert.phase=='close':
                    assert phase=='descend' and dp<.004 and dr<.03
                    success=True
                    # This is the open-gripper descend action from the unchanged
                    # expert; the next close action is never requested/recorded.
                assert command.shape==(7,) and command[6]==1 and np.isfinite(command).all()
                controller.clear();obs,_,term,trunc,_=env.step(to_action_tensor(command,env))
                validate_controller_command(command,controller)
                for k,v in dict(actions=command,env_step_input=command,
                    controller_input=controller['controller_input'],
                    arm_scaled_input=controller['arm_scaled_input'],
                    gripper_scaled_input=controller['gripper_scaled_input']).items():append_h5(f,k,v)
                row,_=capture(env,obs,schema)
                for k,v in row.items():append_h5(f,k,v)
                n+=1
                final_dp,final_dr=expert._err(expert.grasp)
                live=physical_state(env)
                peg_move=float(np.linalg.norm(live['peg_pose'][:3]-initial_peg[:3]))
                rows.append(dict(t=t,phase=phase,next_phase=expert.phase,
                    active_target_position_error_m=float(dp),active_target_orientation_error_rad=float(dr),
                    final_target_position_error_m=float(final_dp),final_target_orientation_error_rad=float(final_dr),
                    peg_motion_m=peg_move,tcp=live['ee_pose'].tolist(),peg=live['peg_pose'].tolist(),
                    target=np.r_[goal.p,goal.q].tolist(),action=command.tolist(),
                    step_capture_seconds=time.perf_counter()-tick))
                append_csv(root/'pipeline_profile.csv',dict(stage='simulation_capture',episode_id=spec['episode_id'],
                    t=t,seconds=rows[-1]['step_capture_seconds']))
                if success or bool(term.item()) or bool(trunc.item()):break
            f.flush()
        with h5py.File(tmp,'r') as f:
            assert len(f['actions'])==n and len(f['robot_state'])==n+1
            np.testing.assert_array_equal(f['actions'][:],f['controller_input'][:])
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1))
            np.testing.assert_array_equal(f['timestamp'][:],np.arange(n+1)/20)
            assert (f['actions'][:,6]==1).all()
        os.link(tmp,folder/'episode.h5');tmp.unlink()
        valid_motion=max(r['peg_motion_m'] for r in rows)<.002
        result=dict(spec=spec,status='ACCEPTED' if success and valid_motion else 'REJECTED',
            reason=None if success and valid_motion else 'expert_reach_timeout_or_peg_motion',
            actions=n,observations=n+1,reach_complete=success,first_approach_success=first_reach,
            final_position_error_m=float(final_dp),final_orientation_error_rad=float(final_dr),
            max_peg_motion_m=max(r['peg_motion_m'] for r in rows),
            wall_seconds=time.perf_counter()-started,sha256=sha(folder/'episode.h5'),
            raw_path=str(folder/'episode.h5'),audit=audit,created_at=now())
        result['collector_source_sha256']=sha(__file__)
        result['runtime_source_sha256']=sha(Path(__file__).with_name('runtime.py'))
        write(folder/'evaluation_trace.json',rows,immutable=True)
        write(receipt,result,immutable=True)
        append_csv(root/'expert_reach_metrics.csv',dict(episode_id=spec['episode_id'],parent_seed=spec['seed'],
            variant=spec['variant'],split=spec['split'],status=result['status'],actions=n,
            reach_complete=success,final_position_error_m=float(final_dp),
            final_orientation_error_rad=float(final_dr),max_peg_motion_m=result['max_peg_motion_m'],
            seconds=result['wall_seconds']))
        print('COLLECT',spec['episode_id'],spec['seed'],spec['variant'],result['status'],n,flush=True)
        return result
    finally:
        if own:env.close()


def run(root, manifest):
    torch.set_num_threads(1)
    env=environment(root)
    try:
        return [collect_episode(root,spec,env) for spec in manifest]
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-manifest',default='capture_manifest.json');a=p.parse_args()
    from graph.geometry_reach10d.profile import Resources
    with Resources(a.root,'collection_'+a.output_manifest):
        write(a.root/a.output_manifest,dict(episodes=run(a.root,read(a.manifest))))
