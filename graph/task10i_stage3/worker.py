"""Independent simulator worker, reusing F1 capture/GT and approved TCP recovery."""
import argparse
import copy
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import h5py
import numpy as np
import torch

from .core import POOL, RUNTIME_ROOT, read, write, freeze, sha, lock, begin, commit, verified, branch_spec
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.collect_recurrent_expert import capture, append_h5, install_audit
from graph.integration_baseline.robot import robot_observation, state_schema
from graph.integration_baseline.bc08_codec import controller_commands
from graph.integration_baseline.geometry09a import physical_state, collision_meshes, convex_gap
from graph.gt_pretrain10f1.collect import label
from graph.gt_pretrain10f1.remediation_collect import portable, save_snapshot, replay_gate
from graph.corrective_reach10f.pilot import restore
from graph.corrective_reach10f.recovery_audit import Audit, bounded_action
from graph.corrective_reach10f.recovery_collect import SAFETY, FEEDBACK, refresh
from graph.corrective_reach10f.safety_v2 import PhysicalSafety
from recurrent.expert import ScriptedExpert, rotvec_between
from recurrent.common import versions

def initial_api():
    """Import the already executed Stage1 implementation without changing source lookup."""
    saved=sys.path.copy()
    path=Path('/workspace/task10i_stage1_20261031/code/pipeline.py')
    spec=importlib.util.spec_from_file_location('stage1_capture_readonly',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path[:]=saved
    audit='/workspace/task10i_stage1_20261031/audit'
    if audit not in sys.path: sys.path.append(audit)
    return module

def array(x):
    return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()

class CollectionAudit(Audit):
    full_contacts=False

    def __init__(self,env):
        super().__init__(env)
        original=env.step
        self.last_transition={}
        def observed_step(action):
            result=original(action)
            obs,reward,terminated,truncated,info=result
            self.last_transition=dict(reward=float(reward.item()),terminated=bool(terminated.item()),
                truncated=bool(truncated.item()),success=bool(info['success'].item()))
            return result
        env.step=observed_step

    def contacts(self):
        if not self.full_contacts:return super().contacts()
        out=[]
        for c in self.env.unwrapped.scene.px.get_contacts():
            out.append(dict(names=[b.entity.name for b in c.bodies],
                min_separation_m=min((float(p.separation) for p in c.points),default=0),
                max_impulse=float(max((np.linalg.norm(p.impulse) for p in c.points),default=0))))
        return out

def flatten(x, prefix=''):
    if isinstance(x,dict):
        result={}
        for k,v in x.items(): result.update(flatten(v,prefix+'/'+str(k)))
        return result
    return {prefix:array(x)}

def reset_gate(env,candidate,folder):
    s1=initial_api(); cid=candidate['id']; source=POOL/'candidates'/cid
    record=read(source/'record.json')
    assert record['eligible'] and record['reset_seed']==candidate['seed']
    for name,h in record['files'].items(): assert sha(source/name)==h
    actual,state,_=s1.capture(env,candidate['seed'])
    with np.load(source/'initial.npz',allow_pickle=False) as z:
        checks=s1.compare({k:z[k] for k in z.files},actual)
    with (source/'snapshot.pkl').open('rb') as f: old=pickle.load(f)
    checks['sim_state']={'exact':s1.ahash(old['sim_state'])==s1.ahash(state['sim_state'])}
    checks['controller']={'exact':s1.ahash(old['controller'])==s1.ahash(state['controller'])}
    # Compare actual NumPy RNG state, clock and process RNG, not unsupported PhysX cache.
    for key in ('main_rng','episode_rng','batched_main_rng','batched_episode_rng','main_seed','episode_seed','elapsed_steps'):
        def equal(a,b):
            if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return bool(np.array_equal(a,b))
        checks[key]={'exact':equal(old[key],state[key])}
    assert all(c['exact'] for c in checks.values()), ('INITIAL_RESET_MISMATCH',checks)
    from geometry import evaluate
    visibility,_=evaluate(s1.camera(actual,cid),actual['peg_pose'],actual['peg_half'],actual['tcp_pose'])
    assert visibility['requested_pass'], 'VISIBILITY_REGRESSION'
    physical=s1.physical(env,actual,state)
    assert physical['status']=='PHYSICAL_VALID', 'PHYSICAL_INVALID'
    write(folder/'initial_reset.json',dict(status='PASS',candidate=cid,seed=candidate['seed'],checks=checks,
        visibility=visibility,physical=physical,route='native reset(seed), no actor teleport; solver cache not serialized'))
    np.savez_compressed(folder/'initial_readback.npz',**actual)
    return env.unwrapped.get_obs()

def full_physics(env,audit,safety):
    """Keep intended grasp contact separate from the no-contact corrective protocol."""
    s=physical_state(env);tol=SAFETY['penetration_tolerance_m']
    finite=all(np.isfinite(v).all() for v in s.values())
    limits=bool((s['qpos']>=safety.limits[:,0]-1e-5).all() and (s['qpos']<=safety.limits[:,1]+1e-5).all())
    # Audit actual contact separations at every physics substep. Finger/peg
    # compliance is reported separately; no invented force limit for grasping.
    severe=[]; grasp_contacts=[]
    for sub in audit.get('substep_contacts',[]):
        for c in sub:
            finger_peg=any('finger' in n for n in c['names']) and any('peg' in n for n in c['names'])
            if finger_peg: grasp_contacts.append(c)
            elif c['min_separation_m'] < -tol: severe.append(c)
    grasp=bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg).item())
    return dict(valid=finite and limits and not severe,finite=finite,joint_limits=limits,
        severe_non_grasp_contacts=severe,grasp_contacts=grasp_contacts,is_grasping=grasp,
        peg_z_m=float(s['peg_pose'][2]),max_qvel=float(abs(s['qvel']).max()),
        scope='all5 substep robot contact separations; intended finger/peg compliance recorded separately; no force cap; no swept-volume proof')

class Recorder:
    def __init__(self, folder, env, expert, schema, audit, controller, safety, initial, corrective, prefix=()):
        self.folder=folder;self.env=env;self.expert=expert;self.schema=schema;self.audit=audit
        self.controller=controller;self.safety=safety;self.initial=initial;self.corrective=corrective
        self.f=h5py.File(folder/'episode.h5','x');self.labels=[];self.history=[];self.trace=[];self.prefix_checks=[]
        for entry in prefix: self.add_entry(copy.deepcopy(entry))

    def observation(self,obs):
        row,_=capture(self.env,obs,self.schema)
        raw,target,gt,err,_=label(self.env,row['robot_state'],self.schema)
        for k,v in flatten(self.env.unwrapped.agent.get_controller_state()).items(): row['controller_state'+k]=v
        row['gripper_opening_m']=np.float32(physical_state(self.env)['qpos'][-2:].sum())
        lab=dict(**{'raw_'+k:v for k,v in raw.items()},target18=target,gt38=gt,
            timestep=row['timestep'],timestamp=row['timestamp'])
        for k,v in physical_state(self.env).items(): lab['physics_'+k]=v
        assert err<=1e-7
        return row,lab

    def append_obs(self,row,lab):
        for k,v in row.items():append_h5(self.f,k,v)
        self.labels.append(lab)

    def add_entry(self,e):
        self.append_obs(e['row'],e['label'])
        for k,v in e['action_fields'].items():append_h5(self.f,k,v)
        self.history.append(e);self.trace.append(e['trace'])

    def step(self,obs,action,segment,supervised,phase):
        row,lab=self.observation(obs)
        assert action.shape==(7,) and action.dtype==np.float32 and np.isfinite(action).all()
        self.controller.clear()
        self.audit.full_contacts=not self.corrective
        obs,a=self.audit.step(action)
        assert len(a['substep_contacts'])==5,'CONTROLLER_ERROR: physics/control clock drift'
        np.testing.assert_array_equal(self.controller['controller_pre_state'],row['robot_state'])
        assert int(self.controller['controller_timestep'])==int(row['timestep'])
        np.testing.assert_array_equal(self.controller['controller_input'],action)
        scaled=controller_commands(action[None])[0]
        np.testing.assert_allclose(scaled[:6],self.controller['arm_scaled_input'],atol=2e-8,rtol=1e-6)
        np.testing.assert_allclose(scaled[6:],self.controller['gripper_scaled_input'],atol=1e-8,rtol=0)
        check=self.safety.evaluate(self.initial,a['substep_contacts']) if self.corrective else full_physics(self.env,a,self.safety)
        if phase in ('reach','descend'):
            robot_contacts=[[c for c in sub if any(n in self.audit.robot_names for n in c['names'])] for sub in a['substep_contacts']]
            self.prefix_checks.append(self.safety.evaluate(self.initial,robot_contacts))
            a['corrective_prefix_physical']=self.prefix_checks[-1]
        # Persist invalid steps as diagnostic evidence before refusing approval.
        a.update(physical=check,phase=phase,segment=segment,success_after=bool(self.env.unwrapped.evaluate()['success'].item()))
        assert a['success_after']==self.audit.last_transition['success']
        a['environment_transition']=self.audit.last_transition.copy()
        af=dict(actions=action,executed_action=action,controller_input=self.controller['controller_input'].copy(),
            arm_scaled_input=self.controller['arm_scaled_input'].copy(),gripper_scaled_input=self.controller['gripper_scaled_input'].copy(),
            supervision_mask=np.bool_(supervised),segment=np.uint8(segment),environment_success=np.bool_(a['success_after']),
            terminated=np.bool_(self.audit.last_transition['terminated']),truncated=np.bool_(self.audit.last_transition['truncated']),
            reward=np.float32(self.audit.last_transition['reward']))
        self.add_entry(dict(row=row,label=lab,action_fields=af,trace=a))
        crash=os.environ.get('TASK10I_ENGINEERING_CRASH_AT')
        if crash and len(self.history)==int(crash):
            assert 'engineering' in str(self.folder), 'crash injection is engineering-only'
            self.f.flush();os._exit(91)
        assert int(row['timestep'])+1==int(self.env.unwrapped.elapsed_steps.item())
        if not check['valid']:raise RuntimeError('PHYSICAL_INVALID: '+str(check))
        return obs

    def close(self,obs,metadata):
        row,lab=self.observation(obs);self.append_obs(row,lab);self.f.flush();self.f.close()
        np.savez_compressed(self.folder/'gt_supervision.npz',**{k:np.stack([l[k] for l in self.labels]) for k in self.labels[0]})
        write(self.folder/'trace.json',self.trace)
        write(self.folder/'physical_audit.json',dict(approved=True,checks=[t['physical'] for t in self.trace],
            corrective_safety_v2=self.corrective,prefix_checks=self.prefix_checks))
        n=len(self.history)
        result=dict(**metadata,actions=n,observations=n+1,approved=True,alignment='PASS')
        write(self.folder/'episode_metadata.json',result)
        validate(self.folder,self.schema)
        return result

def validate(folder,schema):
    from graph.gt_pretrain10f1.contract import measured_robot,object_target,assemble_gt38,split_fields
    from recurrent.features import build_features
    with h5py.File(folder/'episode.h5') as f,np.load(folder/'gt_supervision.npz',allow_pickle=False) as g:
        n=len(f['actions']);assert n>0 and f['actions'].shape==(n,7) and f['robot_state'].shape==(n+1,76)
        for k in ('timestep','timestamp'):
            np.testing.assert_array_equal(f[k][:],np.arange(n+1)/(20 if k=='timestamp' else 1))
            np.testing.assert_array_equal(f[k][:],g[k])
        np.testing.assert_array_equal(f['actions'][:],f['executed_action'][:])
        np.testing.assert_array_equal(f['actions'][:],f['controller_input'][:])
        assert not (f['supervision_mask'][:]& (f['segment'][:]==1)).any()
        assert (np.diff(f['segment'][:].astype(int))>=0).all()
        for camera in ('base_camera','hand_camera'):
            assert f[camera+'/rgb'].shape==(n+1,128,128,3) and f[camera+'/rgb'].dtype==np.uint8
            assert f[camera+'/depth_raw_mm'].shape==(n+1,128,128) and f[camera+'/depth_raw_mm'].dtype==np.int16
            for k in ('intrinsic','extrinsic_cv','cam2world_gl'):assert np.isfinite(f[camera+'/'+k][:]).all()
        raw={k[4:]:g[k] for k in g.files if k.startswith('raw_')}
        np.testing.assert_allclose(build_features(raw,np.ones(n+1,np.float32),np.zeros(n+1,np.float32)),g['gt38'],atol=1e-7,rtol=0)
        measured=measured_robot(f['robot_state'][:],schema)
        for k,v in measured.items(): np.testing.assert_allclose(v,raw[k],atol=1e-7,rtol=0)
        np.testing.assert_allclose(assemble_gt38(split_fields(object_target(raw)),measured),g['gt38'],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(f['tcp_pose'][:],g['raw_tcp_pose'])
        def check(k,v):
            if isinstance(v,h5py.Dataset):assert np.isfinite(v[:]).all(),k
        f.visititems(check)
    return dict(status='PASS',actions=n,observations=n+1,all_GT_robot_frames=True,terminal_label=False)

def runtime(env,schema):
    import inspect
    import mani_skill.envs.tasks.tabletop.peg_insertion_side as task_module
    path=Path(task_module.__file__)
    record=dict(versions=versions(),backend=str(env.unwrapped.backend),control_freq=env.unwrapped.control_freq,
        physics_freq=env.unwrapped.sim_freq,physics_steps=env.unwrapped._sim_steps_per_control,
        horizon=env._max_episode_steps,task_source_sha256=sha(path),
        controller={k:asdict(v.config) for k,v in env.unwrapped.agent.controller.controllers.items()},
        schema=schema,grasp_API_source=inspect.getsource(type(env.unwrapped.agent).is_grasping),
        GPU_PhysX=False,renderer_executed=True)
    original=read(POOL/'runtime_identity.json')
    assert record['task_source_sha256']==original['task_module_sha256']
    for k in ('torch','numpy','mani_skill','sapien'):assert record['versions'][k]==original['packages'][k],k
    assert 'physx_cpu' in record['backend'] and 'sapien_cuda' in record['backend']
    assert (record['control_freq'],record['physics_freq'],record['physics_steps'],record['horizon'])==(20,100,5,250)
    return record

def load_history(folder):
    receipt=verified(folder);n=receipt['actions'];history=[];trace=read(folder/'trace.json')
    with h5py.File(folder/'episode.h5') as f,np.load(folder/'gt_supervision.npz',allow_pickle=False) as g:
        obs_keys=[];action_keys=[]
        def classify(k,v):
            if isinstance(v,h5py.Dataset):
                (obs_keys if len(v)==n+1 else action_keys).append(k)
        f.visititems(classify)
        for t in range(n):
            history.append(dict(row={k:f[k][t] for k in obs_keys},label={k:g[k][t] for k in g.files},
                action_fields={k:f[k][t] for k in action_keys},trace=trace[t]))
    return receipt,history

def collect_parent(root,candidate,branches):
    cid=candidate['id'];folder=root/'parents'/cid
    folder.mkdir(parents=True,exist_ok=True)
    with lock(folder/'owner.lock'):
        if (folder/'parent_result.json').exists():
            r=read(folder/'parent_result.json')
            if r['status']=='PASS':
                for b in r['branches']:verified(folder/b)
                if set(branches)<=set(r['branches']):return r
                freeze(folder/f"parent_result_before_extension_{len(r['branches'])}.json",r)
                branches=list(dict.fromkeys(r['branches']+branches))
            else:return r
        from graph.integration_baseline.direct_train import seed_all
        seed_all(42,1)
        freeze(folder/'attempt_identity.json',dict(candidate_id=cid,seed=candidate['seed'],stochastic_task_attempt=1,
            initial_route='same-runtime reset(seed)',automatic_task_retries=0))
        write(folder/'attempt_cursor.json',dict(status='RUNNING',requested_branches=branches,stage='RESET'))
        env=environment(RUNTIME_ROOT);audit=None;current=None;started=time.perf_counter()
        try:
            obs=reset_gate(env,candidate,folder)
            expert=ScriptedExpert(env);schema=state_schema(robot_observation(env)[0])
            assert schema==read(RUNTIME_ROOT/'normalizer.json')['schema']
            freeze(root/'runtime_identity.json',runtime(env,schema))
            initial=physical_state(env);audit=CollectionAudit(env);controller=install_audit(env,schema);safety=PhysicalSafety(env,SAFETY)
            initial_check=safety.evaluate(initial,[]);assert initial_check['valid'], 'PHYSICAL_INVALID'
            fp=folder/'full_task';anchors={}
            if verified(fp):
                full,full_history=load_history(fp);close_index=full['close_index']
                checks=[e['trace']['corrective_prefix_physical'] for e in full_history[:close_index]]
                for t in (10,25):
                    sp=folder/f'anchor_t{t}.pt'
                    if sp.exists():
                        state=portable(torch.load(sp,map_location='cpu',weights_only=False),True)
                        restore(env,expert,state)
                        anchors[t]=dict(state=state,physical=physical_state(env),prefix=copy.deepcopy(full_history[:t]))
            else:
                partial=begin(fp);current=Recorder(partial,env,expert,schema,audit,controller,safety,initial,False)
                close_index=None
                for t in range(250):
                    phase=expert.phase
                    if phase=='close' and close_index is None:close_index=t
                    if t in (10,25):
                        sp=folder/f'anchor_t{t}.pt'
                        # A prior incomplete capture is archived, including its snapshot;
                        # do not replace a snapshot from a committed trajectory.
                        if sp.exists():sp.rename(folder/(sp.name+f'.interrupted.{time.time_ns()}'))
                        state=save_snapshot(sp,env,expert,schema)
                        anchors[t]=dict(state=state,physical=physical_state(env),prefix=copy.deepcopy(current.history))
                    obs=current.step(obs,expert.act(),0,True,phase)
                    if bool(env.unwrapped.evaluate()['success'].item()):break
                    if audit.last_transition['terminated'] or audit.last_transition['truncated']:break
                success=bool(env.unwrapped.evaluate()['success'].item())
                phases=[e['trace']['phase'] for e in current.history]
                if not success:
                    reason='EXPERT_GRASP_FAILURE' if not any(e['trace']['physical']['is_grasping'] for e in current.history) else 'EXPERT_INSERTION_FAILURE'
                    raise RuntimeError(reason)
                assert close_index is not None and set(('reach','descend','close','lift','align','insert'))<=set(phases)
                assert any(e['trace']['physical']['is_grasping'] for e in current.history),'EXPERT_GRASP_FAILURE'
                full=current.close(obs,dict(candidate_id=cid,branch='full_task',success=True,close_index=close_index,
                    initial_physical=initial_check,source='ScriptedExpert unchanged',official_success=True))
                full=commit(partial,fp,full);full_history=current.history;checks=current.prefix_checks;current=None
                print('FULL_TASK_APPROVED',cid,full['actions'],flush=True)
                write(folder/'attempt_cursor.json',dict(status='RUNNING',requested_branches=branches,stage='FULL_TASK_APPROVED'))
            # The nominal prefix is the actual successful full-task history.
            # It is never regenerated with a second stochastic rollout.
            if any(b!='full_task' for b in branches) and not verified(folder/'nominal'):
                assert all(c['valid'] for c in checks[:close_index]),'PHYSICAL_INVALID: corrective nominal prefix'
                dest=folder/'nominal';pp=begin(dest)
                with h5py.File(fp/'episode.h5') as src,h5py.File(pp/'episode.h5','x') as out:
                    n=full['actions']
                    def copy_ds(k,v):
                        if isinstance(v,h5py.Dataset):out.create_dataset(k,data=v[:close_index+(1 if len(v)==n+1 else 0)],compression='gzip')
                    src.visititems(copy_ds)
                with np.load(fp/'gt_supervision.npz') as g:np.savez_compressed(pp/'gt_supervision.npz',**{k:g[k][:close_index+1] for k in g.files})
                write(pp/'trace.json',[e['trace'] for e in full_history[:close_index]])
                write(pp/'physical_audit.json',dict(approved=True,checks=checks[:close_index],initial=initial_check,corrective_safety_v2=True))
                nominal=dict(candidate_id=cid,branch='nominal',actions=close_index,observations=close_index+1,approved=True,
                    source='exact successful full-task prefix',alignment='PASS')
                write(pp/'episode_metadata.json',nominal);validate(pp,schema);commit(pp,dest,nominal)
            for b in branches:
                spec=branch_spec(b)
                if spec['kind']!='recovery':continue
                if verified(folder/b):continue
                a=anchors[spec['anchor']];gate=folder/f"replay_t{spec['anchor']}.json"
                if not gate.exists():replay_gate(gate,env,expert,audit,a['state'],schema)
                obs=restore(env,expert,a['state']);start=physical_state(env)
                for k in start:np.testing.assert_allclose(start[k],a['physical'][k],atol=1e-6,rtol=0)
                goal=start['ee_pose'].copy();goal[:3]+=np.asarray(spec['offset_mm'])*.001
                dest=folder/b;pp=begin(dest)
                prefix=copy.deepcopy(a['prefix'])
                for e in prefix:e['trace']['physical']=e['trace']['corrective_prefix_physical']
                current=Recorder(pp,env,expert,schema,audit,controller,safety,initial,True,prefix)
                integral=np.zeros(3);stable=0
                for step in range(FEEDBACK['max_steps']):
                    action,integral=bounded_action(env,goal,integral,FEEDBACK)
                    obs=current.step(obs,action,1,False,'disturbance');s=physical_state(env)
                    err=float(np.linalg.norm(s['ee_pose'][:3]-goal[:3]));ang=float(np.linalg.norm(rotvec_between(s['ee_pose'][3:],goal[3:])));vel=float(abs(s['qvel']).max())
                    stable=stable+1 if err<.001 and ang<.01 and vel<.03 else 0
                    if stable>=3:break
                assert stable>=3,'PHYSICAL_INVALID: disturbance did not converge'
                disturbance=dict(requested_offset_mm=spec['offset_mm'],actual_offset_mm=((s['ee_pose'][:3]-start['ee_pose'][:3])*1000).tolist(),
                    residual_m=err,orientation_error_rad=ang,max_qvel=vel,steps=step+1)
                target=refresh(env,expert,initial['peg_pose']);steps=0
                while expert.phase!='close' and steps<130:
                    obs=current.step(obs,expert.act(),2,True,'expert_recovery');steps+=1
                dp,dr=expert._err(expert.grasp)
                assert expert.phase=='close' and dp<.004 and dr<.03,'EXPERT_ACTION_FAILURE: recovery timeout'
                assert physical_state(env)['qpos'][-2:].sum()>=.065,'PHYSICAL_INVALID: recovery opening'
                result=current.close(obs,dict(candidate_id=cid,branch=b,success=True,disturbance=disturbance,target_refresh=target,
                    recovery_steps=steps,position_error_m=float(dp),orientation_error_rad=float(dr),prefix_BC_exact=True,independent_reset_required=True))
                commit(pp,dest,result);current=None
                print('CORRECTIVE_APPROVED',cid,b,result['actions'],flush=True)
                write(folder/'attempt_cursor.json',dict(status='RUNNING',requested_branches=branches,stage='CORRECTIVE',last_completed=b))
            result=dict(status='PASS',candidate_id=cid,seed=candidate['seed'],branches=branches,
                expert_success=True,wall_seconds=time.perf_counter()-started)
        except BaseException as e:
            if current is not None:
                try:current.f.flush();current.f.close();write(current.folder/'failure_trace.json',current.trace)
                except Exception:pass
            reason=next((r for r in ('INITIAL_RESET_MISMATCH','VISIBILITY_REGRESSION','PHYSICAL_INVALID','EXPERT_GRASP_FAILURE',
                'EXPERT_INSERTION_FAILURE','EXPERT_ACTION_FAILURE','CONTROLLER_ERROR','DATA_ALIGNMENT_FAILURE') if r in str(e)),'RUNTIME_ERROR')
            result=dict(status='FAIL',candidate_id=cid,seed=candidate['seed'],reason=reason,error=str(e),traceback=traceback.format_exc(),
                approved_branches=[p.name for p in folder.iterdir() if p.is_dir() and (p/'receipt.json').exists()],
                wall_seconds=time.perf_counter()-started)
        finally:
            if audit is not None:env.unwrapped.scene.step=audit.original_step
            env.close()
        write(folder/'parent_result.json',result)
        write(folder/'attempt_cursor.json',dict(status=result['status'],requested_branches=branches,stage='TERMINAL'))
        print(cid,result['status'],result.get('reason'),round(result['wall_seconds'],2),flush=True)
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    assert os.environ.get('CONTAINER_ID')=='57987772'
    request=read(a.request)
    for s in request['parents']:collect_parent(a.root,s['candidate'],s['branches'])

if __name__=='__main__':main()
