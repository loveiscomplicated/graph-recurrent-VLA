"""Live same-robot pose interventions, separate from the causal base sequence.

This is diagnostic condition generation. GT actor manipulation and collision
audits here never enter the practical geometry input API.
"""
from pathlib import Path
import numpy as np

from graph.integration_baseline.geometry09b import memory,restore
from graph.integration_baseline import geometry09a as prior
from graph.integration_baseline.io import write_json,save_npz,sha256


SPECS=(('baseline_repeat','peg',[0,0,0]),('peg_x+10','peg',[.01,0,0]),('peg_y-10','peg',[0,-.01,0]),
       ('box_x-10','box',[-.01,0,0]),('box_y+10','box',[0,.01,0]))


def collect_branches(env,adapters,pre,baseline_sensor,baseline_masks,episode,timestep,folder,physics):
    from mani_skill.utils.structs.pose import Pose
    from .collect import sensor_inputs,evaluation_sidecar
    baseline_cameras,robot76,pads,schema=sensor_inputs(env,env.unwrapped.get_obs(),episode,timestep)
    original={name:getattr(env.unwrapped,name).pose.raw_pose.clone() for name in ('peg','box')}
    baseline_state=prior.physical_state(env)
    old_collision={name:prior.collision_audit(env,name) for name in ('peg','box')}
    post={name:memory(adapter) for name,adapter in adapters.items()}
    jpeg={name:(Path(adapter.directory.name)/f'{timestep:06d}.jpg').read_bytes() for name,adapter in adapters.items()}
    token=physics.token();rows=[]
    try:
        for condition,name,delta in SPECS:
            actor=getattr(env.unwrapped,name)
            raw=original[name].clone();raw[0,:3]+=raw.new_tensor(delta)
            actor.set_pose(Pose.create(raw))
            observation=env.unwrapped.get_obs()
            branch_episode=episode+f':branch_{condition}_t{timestep}'
            cameras,state,current_pads,_=sensor_inputs(env,observation,branch_episode,timestep)
            np.testing.assert_array_equal(state,robot76);np.testing.assert_array_equal(current_pads,pads)
            for camera,base in zip(cameras,baseline_cameras):
                for key in ('intrinsic','extrinsic_cv','cam2world_gl','tcp_pose'):
                    np.testing.assert_array_equal(getattr(camera,key),getattr(base,key))
            masks={};records={}
            for camera in cameras:
                adapter=adapters[camera.camera]
                restore(adapter,pre[camera.camera])
                # The causal prefix is identical, with a new explicit branch
                # identity for the current observation and its source pixels.
                adapter.identity=branch_episode
                masks[camera.camera],records[camera.camera]=adapter.process(camera.rgb,branch_episode,timestep)
            after=prior.physical_state(env)
            if condition=='baseline_repeat':
                for camera in masks:
                    np.testing.assert_array_equal(masks[camera],baseline_masks[camera])
            allowed={name+'_pose'} | ({'hole_pose','goal_pose'} if name=='box' else set())
            unchanged={key:np.array_equal(value,after[key]) for key,value in baseline_state.items() if key not in allowed}
            for key in ({'hole_pose','goal_pose'} if name=='box' else set()):
                unchanged[key+'_rigid_attachment'] = bool(np.allclose(after[key][:3]-baseline_state[key][:3],delta,atol=3e-8,rtol=0)
                    and np.array_equal(after[key][3:],baseline_state[key][3:]))
            collision=prior.collision_audit(env,name)
            new_near=set(collision['near_contacts'])-set(old_collision[name]['near_contacts'])
            valid=bool(all(unchanged.values()) and collision['valid'] and not new_near)
            output=folder/'branches'/f't{timestep:04d}'/condition;output.mkdir(parents=True,exist_ok=False)
            sensor={k:v.copy() for k,v in baseline_sensor.items()}
            sensor['episode']=np.array(branch_episode)
            for camera in cameras:
                for key in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'):
                    sensor[camera.camera+'/'+key]=getattr(camera,key)
            save_npz(output/'input.npz',**sensor)
            save_npz(output/'masks.npz',**masks)
            save_npz(output/'gt.npz',**evaluation_sidecar(env,observation))
            physics.unchanged(token)
            row=dict(condition=condition,object=name,translation_world_m=delta,timestep=timestep,
                episode=branch_episode,parent_episode=episode,parent_history_through=timestep-1,
                physical_valid=valid,robot_state_exact=True,calibration_exact=True,no_physics_step=True,
                repeat_masks_exact=True if condition=='baseline_repeat' else None,
                source=str(output.relative_to(folder)),input_sha256=sha256(output/'input.npz'),
                masks_sha256=sha256(output/'masks.npz'),gt_sha256=sha256(output/'gt.npz'))
            write_json(output/'condition.json',dict(**row,unchanged=unchanged,collision=collision,
                newly_near_contacts=sorted(new_near),vision=records,
                note='kinematic live render intervention; no fresh PhysX contact-force claim'))
            rows.append(row)
            actor.set_pose(Pose.create(original[name]))
    finally:
        for name in ('peg','box'):
            getattr(env.unwrapped,name).set_pose(Pose.create(original[name]))
        restored_obs=env.unwrapped.get_obs()
        restored,restored_robot,_,_=sensor_inputs(env,restored_obs,episode,timestep)
        np.testing.assert_array_equal(robot76,restored_robot)
        for base,current in zip(baseline_cameras,restored):
            np.testing.assert_array_equal(base.rgb,current.rgb)
            np.testing.assert_array_equal(base.depth_raw_mm,current.depth_raw_mm)
        for name,adapter in adapters.items():
            restore(adapter,post[name])
            adapter.identity=episode
            (Path(adapter.directory.name)/f'{timestep:06d}.jpg').write_bytes(jpeg[name])
        physics.unchanged(token)
    return rows,restored_obs
