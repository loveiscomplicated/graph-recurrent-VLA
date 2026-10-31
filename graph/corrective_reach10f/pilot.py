"""Task10-F prerequisite gate. No policy training after a failed gate.

Uses inherited capture, ScriptedExpert and runtime; all writes are task-local.
Snapshots are serialization candidates, not an assertion of PhysX warm-start
serialization. Identical-action replay explicitly tests that limitation.
"""
import argparse
import copy
import inspect
import os
from pathlib import Path
import random
import time
import traceback

import h5py
import numpy as np
import sapien
import torch

from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d.collect import target
from graph.geometry_reach10d.profile import hardware, Resources
from graph.integration_baseline.collect_recurrent_expert import capture, append_h5
from graph.integration_baseline.geometry09a import physical_state
from graph.integration_baseline.robot import robot_observation, state_schema
from graph.plan_a_bc.common import read, write, sha, append_csv
from recurrent.expert import ScriptedExpert
from recurrent.common import to_action_tensor
from utils.experiment_logger import ExperimentLogger


def snapshot(env, expert):
    task = env.unwrapped
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
    rng={}
    for k,v in vars(task).items():
        if isinstance(v,BatchedRNG):rng[k]=dict(batched=[r.get_state() for r in v.rngs])
        elif isinstance(v,np.random.RandomState):rng[k]=dict(single=v.get_state())
    return copy.deepcopy(dict(sim=task.get_state_dict(), clock=task._elapsed_steps,
        controllers={name: {k: v for k, v in vars(c).items()
                     if k.startswith(('_target', '_start', '_step'))}
                     for name, c in task.agent.controller.controllers.items()},
        expert={k:v for k,v in vars(expert).items() if k != 'env'},
        rng=rng,
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))


def restore(env, expert, state):
    state = copy.deepcopy(state)
    task = env.unwrapped
    task.set_state_dict(state['sim'])
    task.agent.set_controller_state(state['sim'].get('controller', {}))
    for name, values in state['controllers'].items():
        c = task.agent.controller.controllers[name]
        for k,v in values.items():setattr(c,k,v)
        if hasattr(c, '_target_qpos'):c.set_drive_targets(c._target_qpos)
    task._elapsed_steps = state['clock']
    for k,v in state['expert'].items():setattr(expert,k,v)
    for k,v in state['rng'].items():
        if 'batched' in v:
            for r,s in zip(getattr(task,k).rngs,v['batched']):r.set_state(s)
        else:getattr(task,k).set_state(v['single'])
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    return task.get_obs()


def deltas(a,b):
    return {k:float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in a}


def safety(env, original):
    s=physical_state(env);q=s['qpos'];v=s['qvel']
    limits=env.unwrapped.agent.robot.get_qlimits()[0].cpu().numpy()
    contacts=[]
    for c in env.unwrapped.scene.px.get_contacts():
        names=[b.entity.name for b in c.bodies]
        impulse=max((float(np.linalg.norm(p.impulse)) for p in c.points),default=0.)
        robot=any('panda' in n or n.startswith(('link','leftfinger','rightfinger')) for n in names)
        external=any('peg' in n or 'box' in n or 'table' in n for n in names)
        if robot and external and impulse>1e-5:contacts.append(dict(names=names,impulse=impulse))
    preserved={k:float(np.linalg.norm(s[k][:3]-original[k][:3])) for k in ('peg_pose','box_pose','hole_pose')}
    valid=bool(np.isfinite(q).all() and np.isfinite(v).all() and
        (q>=limits[:,0]-1e-5).all() and (q<=limits[:,1]+1e-5).all() and
        abs(s['ee_pose'][:2]).max()<.8 and .02<s['ee_pose'][2]<1.0 and
        max(preserved.values())<.002 and not contacts)
    return dict(valid=valid,object_translation_m=preserved,contacts=contacts,max_qvel=float(abs(v).max()))


def run(root):
    assert os.environ.get('CONTAINER_ID')=='57640710','wrong Vast instance'
    torch.set_num_threads(1)
    cfg=read(root/'config.json');seeds=sorted({e['spec']['seed'] for e in
        read(root/'collection_candidates.json') if e['effective_split']=='train'})[:5]
    protocol=dict(task='10-F',parents=seeds,anchors=[10,25],offsets_mm=[[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]],
        disturbance_steps=8,displacement_tolerance_m=.001,restore_atol=1e-6,
        replay_atol=1e-6,physical_object_tolerance_m=.002,
        teacher='fresh OBB target after disturbance; preserve nominal grasp orientation if independent recomputation agrees within 1e-5 rad; reset Reach phase to reacquire elevated target',
        history='full prefix + all disturbance observations + feedback recovery; no learned hidden stored',
        mask='prefix False, disturbance False, recovery True; nominal all True',
        pilot_gate='ALL reset/snapshot/replay/safety/recovery/alignment checks, then causal graph equivalence before expansion',
        student='unchanged E0 Node21 Edge4 Robot76; no teacher metadata inputs',
        full_dataset='NOT_FROZEN until engineering gates pass',final='NOT_REGISTERED; no policy evaluation allowed')
    write(root/'perturbation_protocol.json',protocol,immutable=True)
    with ExperimentLogger(team='graph',model='Task10F_corrective_expert_pilot',config=protocol,
                          seed=42,root=root/'experiments',job_type='prepare') as log,Resources(root,'engineering_pilot'):
        write(root/'run_identity.json',dict(run_id=log.run_id,instance=57640710,root=str(root)),immutable=True)
        write(root/'hardware.json',hardware(),immutable=True)
        env=environment(root);results=[];gates={};started=time.perf_counter()
        try:
            write(root/'installed_api.json',{name:inspect.getsource(getattr(env.unwrapped,name)) for name in
                ('get_state_dict','set_state_dict')},immutable=True)
            write(root/'backend.json',dict(backend=str(env.unwrapped.backend),control_freq=env.unwrapped.control_freq),immutable=True)
            for seed in seeds:
                obs,_=env.reset(seed=seed);expert=ScriptedExpert(env)
                initial=physical_state(env);schema=state_schema(robot_observation(env)[0])
                assert schema==read(root/'normalizer.json')['schema']
                obs2,_=env.reset(seed=seed)
                reset_error=deltas(initial,physical_state(env));assert max(reset_error.values())<1e-6
                expert=ScriptedExpert(env);obs=obs2;prefix=[];anchors={}
                for t in range(130):
                    if expert.phase=='close':break
                    row,_=capture(env,obs,schema)
                    if t in protocol['anchors']:
                        anchors[t]=dict(state=snapshot(env,expert),physical=physical_state(env),prefix=copy.deepcopy(prefix),row=row,phase=expert.phase)
                    phase=expert.phase;action=expert.act();prefix.append((row,action,phase))
                    obs,*_=env.step(to_action_tensor(action,env))
                assert expert.phase=='close','nominal teacher failed'
                write(root/f'parent{seed}_phases.json',[p[2] for p in prefix],immutable=True)
                for anchor in protocol['anchors']:
                    a=anchors[anchor];assert a['phase'] in ('reach','descend')
                    expert=ScriptedExpert(env)
                    # Independently restore twice; same fresh feedback commands.
                    trials=[]
                    for repeat in range(2):
                        restore(env,expert,a['state']);pre=physical_state(env)
                        commands=[];states=[]
                        for _ in range(5):
                            command=expert.act();commands.append(command.copy())
                            env.step(to_action_tensor(command,env));states.append(physical_state(env))
                        trials.append((pre,commands,states))
                    pre_error=deltas(a['physical'],trials[1][0])
                    replay_error=max(max(deltas(x,y).values()) for x,y in zip(trials[0][2],trials[1][2]))
                    audit=dict(seed=seed,anchor=anchor,phase=a['phase'],reset_max=max(reset_error.values()),
                        restore_errors=pre_error,replay_max=replay_error,
                        action_max=float(np.max(abs(np.array(trials[0][1])-np.array(trials[1][1])))))
                    write(root/f'restore_{seed}_{anchor}.json',audit,immutable=True)
                    if max(pre_error.values())>protocol['restore_atol'] or replay_error>protocol['replay_atol']:
                        gates['snapshot_restore']='FAIL';gates['identical_action_replay']='FAIL'
                        raise RuntimeError('snapshot/replay gate failed; see restore audit; no branch collection/training')
                    gates['snapshot_restore']='PASS';gates['identical_action_replay']='PASS'
                    for offset in protocol['offsets_mm']:
                        identity=f'{seed}_t{anchor}_{offset[0]}_{offset[1]}'
                        folder=root/'raw'/identity;folder.mkdir(parents=True,exist_ok=False)
                        obs=restore(env,expert,a['state']);start=physical_state(env)
                        desired=start['ee_pose'][:3]+np.asarray(offset)*.001
                        actions=[];trace=[];safe=[]
                        with h5py.File(folder/'episode.h5','x') as f:
                            for row,action,phase in a['prefix']:
                                for k,v in row.items():append_h5(f,k,v)
                                append_h5(f,'actions',action);append_h5(f,'supervision_mask',False)
                            def step(command,segment,supervise):
                                nonlocal obs
                                row,_=capture(env,obs,schema)
                                for k,v in row.items():append_h5(f,k,v)
                                append_h5(f,'actions',command);append_h5(f,'supervision_mask',supervise)
                                before=physical_state(env);dp,dr=expert._err(expert.grasp)
                                obs,*_=env.step(to_action_tensor(command,env))
                                valid=safety(env,start);safe.append(valid)
                                trace.append(dict(t=int(row['timestep']),segment=segment,action=command.tolist(),
                                    tcp=before['ee_pose'].tolist(),position_error_m=float(dp),orientation_error_rad=float(dr),safety=valid))
                                actions.append(command.copy())
                            disturbance_target=sapien.Pose(desired,start['ee_pose'][3:])
                            for _ in range(protocol['disturbance_steps']):
                                step(np.r_[expert._servo(disturbance_target,.002),1].astype(np.float32),'disturbance',False)
                            actual=physical_state(env)['ee_pose'][:3]-start['ee_pose'][:3]
                            fresh=ScriptedExpert(env)
                            target_error=float(np.linalg.norm(fresh.grasp.p-expert.grasp.p))
                            from recurrent.expert import rotvec_between
                            angle=float(np.linalg.norm(rotvec_between(fresh.grasp.q,expert.grasp.q)))
                            assert target_error<2e-6 and angle<1e-5,'refreshed teacher target changed unexpectedly'
                            expert.grasp=fresh.grasp;expert.phase='reach';expert.counter=0;expert.lift_target=None
                            for _ in range(130):
                                if expert.phase=='close':break
                                command=expert.act();step(command,'recovery',True)
                            row,_=capture(env,obs,schema)
                            for k,v in row.items():append_h5(f,k,v)
                            assert len(f['actions'])+1==len(f['robot_state'])
                            np.testing.assert_array_equal(f['timestep'][:],np.arange(len(f['robot_state'])))
                            np.testing.assert_array_equal(f['timestamp'][:],np.arange(len(f['robot_state']))/20)
                        dp,dr=expert._err(expert.grasp)
                        success=expert.phase=='close' and dp<.004 and dr<.03 and all(s['valid'] for s in safe)
                        displacement_ok=np.linalg.norm(actual-np.asarray(offset)*.001)<protocol['displacement_tolerance_m']
                        result=dict(branch_id=identity,parent=seed,anchor=anchor,offset_mm=offset,actual_offset_mm=(actual*1000).tolist(),
                            displacement_ok=bool(displacement_ok),success=bool(success),position_error_m=float(dp),orientation_error_rad=float(dr),
                            safe=all(s['valid'] for s in safe),recovery_steps=len(actions)-8,sha256=sha(folder/'episode.h5'))
                        write(folder/'trace.json',trace,immutable=True);write(folder/'receipt.json',result,immutable=True)
                        results.append(result);append_csv(root/'expert_recovery_metrics.csv',{k:v for k,v in result.items() if not isinstance(v,list)})
                        write(root/'parent_branch_manifest.json',results)
                        print('BRANCH',identity,success,displacement_ok,flush=True)
                        if not displacement_ok or not result['safe']:
                            gates['physical_disturbance']='FAIL'
                            raise RuntimeError('physical disturbance gate failed; retain branch and diagnose before more collection')
                gates['parent_reset_identity']='PASS'
            gates['physical_disturbance']='PASS' if all(r['safe'] and r['displacement_ok'] for r in results) else 'FAIL'
            gates['expert_recovery']='PASS' if all(r['success'] for r in results) else 'FAIL'
        except Exception as exc:
            traceback.print_exc()
            write(root/'engineering_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()),immutable=True)
            print('GATE FAILURE',type(exc).__name__,str(exc),flush=True)
        finally:
            env.close()
            for name in ['physical_disturbance','expert_recovery','alignment','causal_vision','graph_equivalence','gru_history','masked_loss','ab_training','closed_loop']:
                gates.setdefault(name,'NOT_TESTED')
            write(root/'verification.json',dict(checks=gates,branches=len(results),seconds=time.perf_counter()-started,
                phase1='FAIL' if 'FAIL' in gates.values() else 'INCOMPLETE',training_executed=False),immutable=True)
            for p in root.glob('*.json'):log.link_output(p,kind='reports')
            log.record(verification=gates)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
