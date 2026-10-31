"""Full ScriptedExpert capture with separate, synchronized GT supervision.

Only this collection/evaluation module can read task GT. The Graph builder sees
the original sensor HDF5 schema; GT lives exclusively in separate NPZ sidecars.
"""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from graph.plan_a_bc.common import read, write, sha, npz, now
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.collect_recurrent_expert import capture, append_h5, install_audit, array
from graph.integration_baseline.robot import robot_observation, state_schema
from graph.integration_baseline.geometry09a import physical_state, collision_audit
from graph.integration_baseline.rollout_stop_clock import validate_controller_command
from recurrent.common import raw_gt, to_action_tensor, versions
from recurrent.expert import ScriptedExpert
from recurrent.features import build_features
from .contract import object_target, measured_robot, split_fields, assemble_gt38, normalize_gt38


def label(env, robot, schema):
    before=int(env.unwrapped.elapsed_steps.item())
    raw={k:v.copy() for k,v in raw_gt(env).items()}
    measured=measured_robot(robot,schema)
    errors={}
    for key in measured:
        errors[key]=float(np.max(abs(measured[key]-raw[key])))
        np.testing.assert_allclose(measured[key],raw[key],atol=1e-7,rtol=0,err_msg=key)
    gt=build_features(raw,np.float32(1),np.float32(0))
    targets=object_target(raw)
    assembled=assemble_gt38(split_fields(targets),measured)
    np.testing.assert_allclose(assembled,gt,atol=1e-7,rtol=0)
    assert before==int(env.unwrapped.elapsed_steps.item())
    return raw, targets, gt, float(np.max(abs(assembled-gt))), errors


def interventions(env, spec, t, schema, folder):
    before=physical_state(env); task=env.unwrapped
    outputs=[]; labels=[]
    for object_name in ('peg','box'):
        actor=getattr(task,object_name)
        pose=actor.pose.raw_pose.clone()
        for j,offset in enumerate(spec['paired_protocol']['offsets_mm']):
            pair_id=f't{t:03d}_{object_name}_{j}'
            moved=pose.clone();moved[0,:3]+=moved.new_tensor(offset)*.001
            actor.set_pose(Pose.create(moved))
            current=physical_state(env)
            for key in before:
                if key==object_name+'_pose' or (object_name=='box' and key in ('hole_pose','goal_pose')):continue
                np.testing.assert_array_equal(before[key],current[key],err_msg=key)
            np.testing.assert_allclose(current[object_name+'_pose'][:3]-before[object_name+'_pose'][:3],
                np.asarray(offset)*.001,atol=1e-8,rtol=0)
            valid=collision_audit(env,object_name)
            obs=task.get_obs(); row,_=capture(env,obs,schema)
            raw,target,gt,err,_=label(env,row['robot_state'],schema)
            path=folder/'paired_rgbd.h5'
            with h5py.File(path,'a') as f:
                group=f.create_group(pair_id)
                for k,v in row.items():append_h5(group,k,v)
            outputs.append(dict(pair_id=pair_id, baseline_t=t, object=object_name,offset_mm=offset,
                valid=bool(valid['valid']), physical_audit=valid, same_robot_exact=True,
                same_camera_calibration_exact=True, causal_prefix='baseline memory immediately before timestep t',
                raw_hdf5_group=pair_id, label_index=len(labels), oracle_max_abs=err))
            labels.append(dict(**{'raw_'+k:v for k,v in raw.items()},target18=target,gt38=gt))
            actor.set_pose(Pose.create(pose))
            restored=physical_state(env)
            for key in before:np.testing.assert_array_equal(restored[key],before[key],err_msg=key)
    return outputs,labels


def episode(root,spec,env):
    folder=root/'raw'/str(spec['episode_id']);receipt=folder/'receipt.json'
    if receipt.exists():
        r=read(receipt);assert r['spec']==spec
        for name,s in r['hashes'].items():assert sha(folder/name)==s
        return r
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    obs,_=env.reset(seed=spec['seed']);expert=ScriptedExpert(env)
    schema=state_schema(robot_observation(env)[0]);assert schema==read(root/'normalizer.json')['schema']
    audit=install_audit(env,schema)
    rows=[]; pair_rows=[];pair_labels=[];stages=[];oracle_errors=[];success=False
    with h5py.File(folder/'episode.h5','x') as f:
        for t in range(spec['horizon']+1):
            row,_=capture(env,obs,schema)
            assert int(row['timestep'])==t
            raw,target,gt,oracle_error,sensor_errors=label(env,row['robot_state'],schema)
            rows.append(dict(**{'raw_'+k:v for k,v in raw.items()},target18=target,gt38=gt,
                timestep=np.int64(t),timestamp=np.float64(t/20)))
            oracle_errors.append(oracle_error)
            for k,v in row.items():append_h5(f,k,v)
            if t==spec['horizon'] or success or (t and (terminated or truncated)):break
            if spec['paired'] and t in spec['paired_protocol']['anchors']:
                out,labs=interventions(env,spec,t,schema,folder)
                offset=len(pair_labels)
                for entry in out:entry['label_index']+=offset
                pair_rows+=out;pair_labels+=labs
                obs=env.unwrapped.get_obs()
                again,_=capture(env,obs,schema)
                for k in row:np.testing.assert_array_equal(row[k],again[k],err_msg=k)
            phase=expert.phase; action=expert.act()
            assert action.shape==(7,) and np.isfinite(action).all()
            audit.clear(); obs,_,term,trunc,info=env.step(to_action_tensor(action,env))
            validate_controller_command(action,audit)
            np.testing.assert_array_equal(audit['controller_pre_state'],row['robot_state'])
            assert int(audit['controller_timestep'])==t
            for k,v in dict(actions=action,controller_input=audit['controller_input'],
                           arm_scaled_input=audit['arm_scaled_input'],gripper_scaled_input=audit['gripper_scaled_input']).items():
                append_h5(f,k,v)
            success=bool(info['success'].item());terminated=bool(term.item());truncated=bool(trunc.item())
            stages.append(dict(t=t,phase=phase,next_phase=expert.phase,success_after_action=success))
        f.flush()
    n=len(stages)
    with h5py.File(folder/'episode.h5','r') as f:
        assert len(f['actions'])==n and len(f['robot_state'])==n+1
        np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1))
        np.testing.assert_array_equal(f['actions'][:],f['controller_input'][:])
    npz(folder/'gt_supervision.npz',{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    if pair_labels:
        npz(folder/'paired_gt_supervision.npz',{k:np.stack([r[k] for r in pair_labels]) for k in pair_labels[0]})
    write(folder/'paired_receipt.json',dict(pairs=pair_rows),immutable=True)
    write(folder/'expert_stages.json',stages,immutable=True)
    hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    r=dict(spec=spec,actions=n,observations=n+1,success=success,phases=sorted({r['phase'] for r in stages}),
        oracle_max_abs=max(oracle_errors),sensor_gt_agreement_max=sensor_errors,
        pair_candidates=len(pair_rows),pairs_valid=sum(r['valid'] for r in pair_rows),
        hashes=hashes,alignment='PASS',wall_seconds=time.perf_counter()-started,created_at=now())
    write(receipt,r,immutable=True)
    print('COLLECT',spec['split'],spec['seed'],n,success,'pairs',r['pairs_valid'],'/',r['pair_candidates'],flush=True)
    return r


def run(root,worker,workers):
    torch.set_num_threads(1)
    specs=read(root/'dataset_manifest.json')['episodes'][worker::workers]
    env=environment(root)
    try:
        task=env.unwrapped
        record=dict(versions=versions(),backend=str(task.backend),
            control_freq=task.control_freq,sim_freq=task.sim_freq,physics_step=task.scene.px.timestep,
            controller={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()},
            GPU_physics_executed=False,worker=worker)
        path=root/f'environment_worker{worker}.json'
        if path.exists():assert read(path)==record
        else:write(path,record,immutable=True)
        receipts=[episode(root,s,env) for s in specs]
        path=root/f'collection_worker{worker}.json';result=dict(episodes=receipts)
        if path.exists():assert read(path)==result
        else:write(path,result,immutable=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();run(a.root,a.worker,a.workers)
