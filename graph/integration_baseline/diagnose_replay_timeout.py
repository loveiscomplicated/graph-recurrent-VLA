"""Task06-A independent diagnostics. Never writes demonstrations or production code.

Run from repository root with WANDB_MODE=disabled. Continues first-three replay
past the unchanged gate for measurement only; outputs are NOT BC dataset inputs.
Snapshot probes are disconnected experiments, explicitly separate from replay.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import h5py
import numpy as np
import torch

from graph.integration_baseline.io import sha256, write_json
from graph.integration_baseline.replay import SOURCE, INITIAL, FROZEN, audit_obs
from graph.prototype.data import reset_episode, state_at
from utils.experiment_logger import ExperimentLogger

PRIOR = Path('experiments/c184b85dadf14db5953f11cdcacb6674')


def array(x):
    return x.detach().cpu().numpy().copy()


def flatten(tree, prefix=''):
    out = {}
    for key, value in tree.items():
        name = prefix + key
        if isinstance(value, dict):
            out.update(flatten(value, name + '/'))
        else:
            out[name] = array(value) if isinstance(value, torch.Tensor) else np.array(value)
    return out


def geometry(task):
    from envs.maniskill.gt_state import _finger_pad_origin, _LEFT_TIP_LOCAL, _RIGHT_TIP_LOCAL
    return array(torch.stack([task.agent.tcp.pose.p,
        _finger_pad_origin(task.agent.finger1_link, _LEFT_TIP_LOCAL),
        _finger_pad_origin(task.agent.finger2_link, _RIGHT_TIP_LOCAL)], 1))[0]


def stats(error):
    error = np.asarray(error, dtype=np.float64)
    return dict(max=float(error.max()), mean=float(error.mean()), p95=float(np.quantile(error, .95)))


def first(mask):
    rows = np.flatnonzero(mask)
    return int(rows[0]) if len(rows) else None


def summarize(actual, ref):
    d = np.abs(actual.astype(np.float64) - ref)
    fields = dict(qpos=d[:, :9], qvel=d[:, 9:18], tcp_position=d[:, 18:21],
                  tcp_quaternion_component=d[:, 21:25], peg_pose=d[:, 25:32],
                  hole_pose=d[:, 35:42], state43=d)
    out = {name: dict(**stats(e), first_nonzero=first(e.max(1)>0),
        first_gt_1e_7=first(e.max(1)>1e-7), first_gt_1e_6=first(e.max(1)>1e-6),
        first_gt_2e_5=first(e.max(1)>2e-5), first_gt_1e_3=first(e.max(1)>1e-3),
        max_location=list(map(int, np.unravel_index(e.argmax(), e.shape)))) for name,e in fields.items()}
    out['qvel_by_joint'] = [dict(joint_index=j, units='rad/s' if j<7 else 'm/s',
        **stats(d[:,9+j]), first_gt_1e_3=first(d[:,9+j]>1e-3)) for j in range(9)]
    # Stable sign-invariant relative quaternion angle, radians.
    a = actual[:,21:25].astype(np.float64); b=ref[:,21:25].astype(np.float64)
    a /= np.linalg.norm(a,axis=1,keepdims=True); b /= np.linalg.norm(b,axis=1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs((a*b).sum(1)),0,1))
    out['tcp_orientation_rad'] = stats(angle)
    pos,vel=d[:,:9].max(1),d[:,9:18].max(1)
    out['qpos_qvel_max_error_correlation'] = (float(np.corrcoef(pos,vel)[0,1])
        if np.std(pos)>0 and np.std(vel)>0 else None)
    out['first_failure_timestep'] = first(d.max(1)>1e-3)
    t=out['first_failure_timestep']
    out['failure'] = None if t is None else dict(observation_t=t, action_index=t-1,
        component=int(d[t].argmax()), error=float(d[t].max()))
    indices=sorted(set(range(90,min(111,len(ref)))) | (set(range(max(0,t-5),min(len(ref),t+6))) if t else set()))
    out['detailed_windows']=[dict(observation_t=i,qpos_max=float(d[i,:9].max()),qvel_max=float(d[i,9:18].max()),
        qvel_joint=int(d[i,9:18].argmax()),tcp_position_max=float(d[i,18:21].max()),qvel9=d[i,9:18].tolist()) for i in indices]
    return out


def replay(env, ep, group, initial, mode):
    if mode == 'source_kwargs':
        env.reset(**ep['reset_kwargs'])
    else:
        reset_episode(env, ep)
    task=env.unwrapped
    before=audit_obs(task).copy()
    if mode != 'reset_only':
        task.set_state_dict(initial)
    if mode == 'controller_reset':
        task.agent.controller.reset()
    actual=[]; geo=[]; links=[]; flags=[]; counters=[]; inputs=[]; scaled=[]; targets=[]; poses=[]; contacts=[]
    states={}
    actions=group['actions'][()]
    controller=task.agent.controller
    original_set_action=controller.set_action
    def capture_input(action):
        inputs.append(array(action)[0])
        return original_set_action(action)
    controller.set_action=capture_input
    try:
        for t in range(len(actions)+1):
            actual.append(audit_obs(task).copy());geo.append(geometry(task))
            links.append(np.stack([np.r_[array(link.pose.raw_pose)[0],array(link.linear_velocity)[0],
                array(link.angular_velocity)[0]] for link in task.agent.robot.links]))
            for k,v in flatten(task.get_state_dict()).items(): states.setdefault(k,[]).append(v[0] if v.ndim>1 else v)
            if t == len(actions):break
            a=torch.from_numpy(actions[t].copy())[None]
            scaled.append(np.r_[array(controller.controllers['arm']._preprocess_action(a[:,:6]))[0],
                array(controller.controllers['gripper']._preprocess_action(a[:,6:]))[0]])
            obs,_,term,trunc,info=env.step(a)
            flags.append([bool(term.item()),bool(trunc.item())]);counters.append(int(task.elapsed_steps.item()))
            targets.append(np.concatenate([array(c._target_qpos)[0] for c in controller.controllers.values()]))
            poses.append(array(controller.controllers['arm']._target_pose.raw_pose)[0])
            contacts.append(len(task.scene.get_contacts()))
            if mode=='rgbd':
                # Exercise same Task06 per-observation image readback (no RGB-D dataset written).
                image_obs=task.get_obs()
                for data in image_obs['sensor_data'].values():
                    for key in ('rgb','depth'):array(data[key])
    finally:
        controller.set_action=original_set_action
    arrays=dict(actual_state43=np.stack(actual),reference_state43=group['obs'][()],
        replay_geometry=np.stack(geo), replay_link_pose_velocity=np.stack(links),
        actions=actions,controller_input=np.stack(inputs),scaled_controller_action=np.stack(scaled),
        controller_target_qpos=np.stack(targets),controller_target_pose=np.stack(poses),
        replay_flags=np.array(flags),source_flags=np.stack([group['terminated'][()],group['truncated'][()]],1),
        elapsed_steps=np.array(counters),replay_contact_count=np.array(contacts),reset_before_restore=before,
        **{'replay_state/'+k:np.stack(v) for k,v in states.items()})
    summary=summarize(arrays['actual_state43'],arrays['reference_state43'])
    summary.update(episode_id=ep['episode_id'], mode=mode,source_steps=len(actions),
        controller_input_exact=bool(np.array_equal(arrays['controller_input'],actions)),
        source_flags_exact=bool(np.array_equal(arrays['replay_flags'],arrays['source_flags'])),
        counter_exact=bool(np.array_equal(arrays['elapsed_steps'],np.arange(1,len(actions)+1))),
        first_truncated_action=first(arrays['replay_flags'][:,1]),
        first_terminated_action=first(arrays['replay_flags'][:,0]),
        reset_vs_reference_error=float(np.abs(before-group['obs'][0]).max()))
    return arrays,summary


def conversion_probe(out, metadata, source, originals, mapping):
    """Recompute only existing first-three conversions in memory, without RecordEpisode."""
    import gymnasium as gym
    from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos
    raw_meta=json.loads(INITIAL.with_suffix('.json').read_text())
    original_kwargs=dict(raw_meta['env_info']['env_kwargs'])
    original_kwargs.update(obs_mode='state',sim_backend='physx_cpu',render_backend='cpu')
    target_kwargs=dict(metadata['env_info']['env_kwargs']);target_kwargs['render_backend']='cpu'
    ori=gym.make('PegInsertionSide-v1',**original_kwargs)
    target=gym.make('PegInsertionSide-v1',**target_kwargs)
    records=[]
    try:
        for ep in metadata['episodes'][:3]:
            eid=ep['episode_id'];sid=mapping[eid];raw=originals[f'traj_{sid}']
            ori.reset(**ep['reset_kwargs']);target.reset(**ep['reset_kwargs'])
            initial=state_at(raw['env_states'],0)
            ori.unwrapped.set_state_dict(initial);target.unwrapped.set_state_dict(initial)
            actions=[];actual=[audit_obs(target.unwrapped)];targets=[];native=[]
            step=target.step;ori_step=ori.step
            def capture(action):
                actions.append(array(action))
                result=step(action)
                actual.append(audit_obs(target.unwrapped))
                targets.append(array(target.unwrapped.agent.controller.controllers['arm']._target_qpos)[0])
                return result
            def native_capture(action):
                result=ori_step(action)
                native.append(flatten(ori.unwrapped.get_state_dict()))
                return result
            target.step=capture;ori.step=native_capture
            try:
                info=from_pd_joint_pos('pd_ee_delta_pose',raw['actions'][()],ori,target)
            finally:target.step=step;ori.step=ori_step
            actions=np.stack(actions);actual=np.stack(actual);reference=source[f'traj_{eid}/obs'][()]
            reference_actions=source[f'traj_{eid}/actions'][()]
            same_shape=actions.shape==reference_actions.shape
            record=dict(episode_id=eid,source_episode_id=sid,converted_steps=len(actions),stored_steps=len(reference_actions),
                action_shape_equal=same_shape, final_success=bool(info['success'].item()))
            if same_shape:
                record.update(action_error=stats(np.abs(actions-reference_actions)),
                    action_first_nonzero=first(np.abs(actions-reference_actions).max(1)>0),
                    observation=summarize(actual,reference))
            native_errors={}
            for k in native[0]:
                ref=raw['env_states/'+k][1:];live=np.stack([row[k][0] for row in native])
                native_errors[k]=stats(np.abs(live-ref))
            record['native_replay_state_errors']=native_errors
            np.savez_compressed(out/f'conversion_probe_ep{eid:03d}.npz',regenerated_actions=actions,
                regenerated_state43=actual,regenerated_arm_target_qpos=np.stack(targets),
                **{'native_state/'+k:np.stack([row[k][0] for row in native]) for k in native[0]})
            write_json(out/f'conversion_probe_ep{eid:03d}.json',record);records.append(record)
            print('conversion_probe',eid,record.get('action_error'),flush=True)
    finally:ori.close();target.close()
    return records


def run(preflight):
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
    config=dict(task='06-A', source_run_id=PRIOR.name, episodes=[0,1,2],seed=0,
        unchanged_restore_atol=2e-5,unchanged_replay_atol=1e-3,
        modes=['state','state_repeat','rgbd','threads4','source_kwargs','horizon1000','reset_only','controller_reset'],
        conversion_probe='first 3 existing raw trajectories only, in-memory; no dataset rewrite',
        prior_failed_diagnostic_run='d272fc182468487a8bb3ee1ac89ee546',
        snapshot_probes='disconnected one-step diagnostics only; not dataset inputs',
        authorization='investigation only; no production edits, training, policy evaluation, publishing')
    logger=ExperimentLogger(team='integration',config=config,seed=0,job_type='diagnosis',model='expert_replay_diagnosis')
    out=logger.directory
    print('RUN_ID='+logger.run_id,flush=True)
    Path('/tmp/task06a_run_directory').write_text(str(out))
    shutil.copytree(preflight,out/'before_snapshot')
    log=(out/'execution.log').open('w',buffering=1)
    def emit(x):print(x,flush=True);log.write(str(x)+'\n')
    metadata=json.loads(SOURCE.with_suffix('.json').read_text())
    frozen=json.loads(FROZEN.read_text());mapping={r['episode_id']:r['source_episode_id'] for r in frozen['snapshot_checks']}
    from envs.maniskill import make_peg_insertion_env
    import gymnasium as gym
    import mani_skill
    from mani_skill.utils.registration import TimeLimitWrapper
    runtime=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        packages={n:importlib.metadata.version(n) for n in ['mani_skill','sapien','gymnasium','torch','numpy','pin','h5py','matplotlib']},
        cuda_available=torch.cuda.is_available(),mps_available=torch.backends.mps.is_available(),
        executed_device='cpu',cuda='NOT_EXECUTED',mps='NOT_EXECUTED',
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        command=' '.join(sys.argv), environment={k:os.environ.get(k) for k in ['WANDB_MODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','HF_HUB_OFFLINE','MPLCONFIGDIR']})
    results=[]
    try:
        with h5py.File(SOURCE,'r') as source,h5py.File(INITIAL,'r') as originals:
            lengths=[];episodes=[]
            for ep in metadata['episodes']:
                g=source[f'traj_{ep["episode_id"]}'];n=len(g['actions']);lengths.append(n)
                term=g['terminated'][()];trunc=g['truncated'][()];success=g['success'][()]
                episodes.append(dict(episode_id=ep['episode_id'],source_episode_id=mapping[ep['episode_id']],seed=ep['episode_seed'],steps=n,
                    first_truncated_action=first(trunc),first_terminated_action=first(term),first_success_action=first(success),
                    successes_pre100=int(success[:100].sum()),successes_post100=int(success[100:].sum()),
                    truncated_count=int(trunc.sum()),terminated_count=int(term.sum())))
            inv=dict(episodes=episodes,total_actions=sum(lengths),episode_count=len(lengths),min=min(lengths),max=max(lengths),
                mean=float(np.mean(lengths)),quantiles=dict(zip(['p0','p25','p50','p75','p100'],np.quantile(lengths,[0,.25,.5,.75,1]).tolist())),
                retained_at100=sum(min(n,100) for n in lengths),lost_at100=sum(max(0,n-100) for n in lengths),
                splits={k:dict(episodes=len(ids),actions=sum(lengths[i] for i in ids),retained_at100=sum(min(lengths[i],100) for i in ids)) for k,ids in frozen['splits'].items()})
            write_json(out/'source_inventory.json',inv)
            for mode in config['modes']:
                torch.set_num_threads(4 if mode=='threads4' else 1)
                if mode=='source_kwargs':
                    kwargs=dict(metadata['env_info']['env_kwargs']);kwargs['render_backend']='cpu'
                    env=gym.make(metadata['env_info']['env_id'],**kwargs)
                else:env=make_peg_insertion_env(obs_mode='rgb+depth' if mode=='rgbd' else 'state',control_mode='pd_ee_delta_pose')
                if mode=='horizon1000':env=TimeLimitWrapper(env.unwrapped,1000)
                try:
                    if mode=='state':
                        runtime.update(backend=str(env.unwrapped.backend),sim_config=asdict(env.unwrapped.sim_config),
                            control_freq=env.unwrapped.control_freq,sim_freq=env.unwrapped._sim_freq,
                            substeps=env.unwrapped._sim_steps_per_control,controller_config={k:asdict(v) for k,v in env.unwrapped.agent.controller.configs.items()},
                            controller_state=env.unwrapped.agent.get_controller_state(),
                            joint_names=[j.name for j in env.unwrapped.agent.robot.active_joints],link_names=[l.name for l in env.unwrapped.agent.robot.links],
                            wrapper_chain=[])
                        w=env
                        while True:
                            runtime['wrapper_chain'].append(dict(type=type(w).__module__+'.'+type(w).__name__,max_episode_steps=getattr(w,'_max_episode_steps',None)))
                            if not hasattr(w,'env'):break
                            w=w.env
                        write_json(out/'runtime.json',runtime)
                    for ep in metadata['episodes'][:3]:
                        eid=ep['episode_id'];initial=state_at(originals[f'traj_{mapping[eid]}/env_states'],0)
                        arrays,summary=replay(env,ep,source[f'traj_{eid}'],initial,mode)
                        np.savez_compressed(out/f'{mode}_ep{eid:03d}.npz',**arrays)
                        write_json(out/f'{mode}_ep{eid:03d}.json',summary);results.append(summary)
                        emit(dict(mode=mode,episode=eid,failure=summary['first_failure_timestep'],qvel=summary['qvel']['max']))
                finally:env.close()
            # Reference geometry is deterministic forward kinematics from stored snapshots,
            # not a recorded fingertip measurement. No reference link velocities are invented.
            env=make_peg_insertion_env(obs_mode='state',control_mode='pd_ee_delta_pose')
            try:
                for ep in metadata['episodes'][:3]:
                    eid=ep['episode_id'];g=source[f'traj_{eid}'];n=len(g['actions']);reset_episode(env,ep)
                    initial=state_at(originals[f'traj_{mapping[eid]}/env_states'],0)
                    ref_geo=[];restored=[];one_step=[];restored_states={}
                    for t in range(n+1):
                        env.unwrapped.set_state_dict(initial if t==0 else state_at(g['env_states'],t))
                        restored.append(audit_obs(env.unwrapped));ref_geo.append(geometry(env.unwrapped))
                        for k,v in flatten(env.unwrapped.get_state_dict()).items():restored_states.setdefault(k,[]).append(v[0] if v.ndim>1 else v)
                        if t<n:
                            env.step(torch.from_numpy(g['actions'][t].copy())[None])
                            one_step.append(audit_obs(env.unwrapped))
                    np.savez_compressed(out/f'snapshot_probe_ep{eid:03d}.npz',restored_state43=np.stack(restored),
                        reconstructed_reference_geometry=np.stack(ref_geo),one_step_state43=np.stack(one_step),
                        **{'restored_state/'+k:np.stack(v) for k,v in restored_states.items()})
                    write_json(out/f'snapshot_probe_ep{eid:03d}.json',dict(episode_id=eid,
                        restore=summarize(np.stack(restored),g['obs'][()]),
                        one_step=summarize(np.stack(one_step),g['obs'][1:]),
                        warning='Independent snapshot-started one-step transitions, NOT a continuous episode; controller/PhysX caches not restored. one_step index 0 corresponds to observation 1.'))
            finally:env.close()
            conversion_results=conversion_probe(out,metadata,source,originals,mapping)
        write_json(out/'diagnostic_summary.json',dict(results=results,status='MEASURED_NOT_DATASET',tolerances_changed=False))
        logger.record(source_run_id=PRIOR.name,diagnosis_status='MEASURED_NOT_DATASET',production_modified=False)
        logger.log(dict(diagnostic_replays=len(results),episodes=3))
        for p in out.glob('*'):
            if p.is_file() and p.name not in ['metadata.json','manifest.json','metrics.json','config.yaml']:
                logger.link_output(p,kind='reports' if p.suffix in ['.npz','.log'] else 'metrics')
    except Exception as exc:
        write_json(out/'execution_failure.json',dict(type=type(exc).__name__,message=str(exc)))
        logger.finish(failed=True)
        raise
    finally:log.close()
    logger.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',type=Path,required=True)
    run(p.parse_args().preflight)
