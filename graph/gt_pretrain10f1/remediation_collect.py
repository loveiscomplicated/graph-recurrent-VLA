"""Independent corrective GT capture. Existing Task10-F data/source is read-only."""
import argparse, copy, os, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import h5py
import torch
import sapien
from mani_skill.utils.structs.pose import Pose
from graph.plan_a_bc.common import read,write,sha,npz,now
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.collect_recurrent_expert import capture,append_h5
from graph.integration_baseline.robot import robot_observation,state_schema
from graph.integration_baseline.geometry09a import physical_state
from graph.corrective_reach10f.pilot import snapshot,restore
from graph.corrective_reach10f.recovery_audit import Audit,bounded_action
from graph.corrective_reach10f.recovery_collect import SAFETY,FEEDBACK,refresh
from graph.corrective_reach10f.safety_v2 import PhysicalSafety
from recurrent.expert import ScriptedExpert,rotvec_between
from recurrent.common import versions
from utils.experiment_logger import ExperimentLogger
from .collect import label


def portable(x,decode=False):
    if decode and isinstance(x,dict) and set(x)=={'__sapien_pose__'}:
        p=x['__sapien_pose__'];return sapien.Pose(p[:3],p[3:])
    if decode and isinstance(x,dict) and set(x)=={'__maniskill_pose__'}:return Pose.create(x['__maniskill_pose__'])
    if not decode and isinstance(x,sapien.Pose):return {'__sapien_pose__':np.r_[x.p,x.q]}
    if not decode and isinstance(x,Pose):return {'__maniskill_pose__':x.raw_pose.clone()}
    if isinstance(x,dict):return {k:portable(v,decode) for k,v in x.items()}
    if isinstance(x,list):return [portable(v,decode) for v in x]
    if isinstance(x,tuple):return tuple(portable(v,decode) for v in x)
    if x is None or isinstance(x,(str,int,float,bool,np.ndarray,np.generic,torch.Tensor)):return x
    raise TypeError('unsupported snapshot member '+str(type(x)))


def save_snapshot(path,env,expert,schema):
    state=snapshot(env,expert);encoded=portable(state);tmp=path.with_suffix('.partial')
    torch.save(encoded,tmp);back=torch.load(tmp,map_location='cpu',weights_only=False)
    from graph.plan_a_bc.train import assert_nested_equal
    assert_nested_equal(encoded,back);os.link(tmp,path);tmp.unlink()
    return portable(back,True)


def row_label(env,obs,schema):
    t=int(env.unwrapped.elapsed_steps.item());row,_=capture(env,obs,schema)
    raw,target,gt,err,sensor=label(env,row['robot_state'],schema)
    assert int(row['timestep'])==t==int(env.unwrapped.elapsed_steps.item())
    return row,dict(**{'raw_'+k:v.copy() for k,v in raw.items()},target18=target,gt38=gt,
        timestep=np.int64(t),timestamp=np.float64(t/20)),err


class Record:
    def __init__(self,folder,env,expert,audit,safety,initial,schema,prefix=(),prefix_checks=()):
        folder.mkdir(parents=True,exist_ok=False);self.folder=folder;self.env=env;self.expert=expert;self.audit=audit
        self.safety=safety;self.initial=initial;self.schema=schema;self.f=h5py.File(folder/'episode.h5','x')
        self.labels=[];self.rows=[];self.checks=list(copy.deepcopy(prefix_checks));self.err=[]
        self.current_check=safety.evaluate(initial,[]);self.initial_check=copy.deepcopy(self.current_check)
        for row,lab,action,err in prefix:
            self.append(row,lab,err);self.action(action,0,False)
        self.trace=[]
    def append(self,row,lab,err):
        for k,v in row.items():append_h5(self.f,k,v)
        self.labels.append(lab);self.rows.append(row);self.err.append(err)
    def action(self,action,segment,supervised):
        for k,v in dict(actions=action,executed_action=action,supervision_mask=np.bool_(supervised),segment=np.uint8(segment)).items():append_h5(self.f,k,v)
    def step(self,obs,action,segment,supervised):
        row,lab,err=row_label(self.env,obs,self.schema);self.append(row,lab,err);self.action(action,segment,supervised)
        assert self.current_check['valid'],'invalid pre-action state'
        obs,a=self.audit.step(action);check=self.safety.evaluate(self.initial,a['substep_contacts']);self.current_check=check
        assert int(row['timestep'])+1==int(self.env.unwrapped.elapsed_steps.item())
        a.update(physical_safety=check,segment=segment);self.trace.append(a);self.checks.append(check)
        if not check['valid']:raise RuntimeError('physical safety failed; collection is not approved')
        return obs
    def close(self,obs,result,spec):
        row,lab,err=row_label(self.env,obs,self.schema);self.append(row,lab,err);self.f.flush();self.f.close()
        n=len(self.checks);assert len(self.labels)==n+1 and len(self.rows)==n+1
        npz(self.folder/'gt_supervision.npz',{k:np.stack([x[k] for x in self.labels]) for k in self.labels[0]})
        valid=np.asarray([c['valid'] for c in self.checks],bool)
        assert valid.all() and self.initial_check['valid'] and result['approved']
        npz(self.folder/'gnn_mask.npz',{'gnn_supervision_mask':valid.copy(),'physical_valid':valid})
        write(self.folder/'physical_audit.json',dict(approved=True,physical_safety=True,legacy_failed_pilot=False,
            pre_action_valid=True,initial=self.initial_check,checks=self.checks,
            indexing='mask[t] requires observed pre-action t and every physics substep/post-action t+1 valid; prefix inherits exact recorded nominal checks'),immutable=True)
        write(self.folder/'trace.json',self.trace,immutable=True)
        with h5py.File(self.folder/'episode.h5') as f,np.load(self.folder/'gt_supervision.npz') as gt:
            assert len(f['actions'])==n and f['robot_state'].shape==(n+1,76)
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1));np.testing.assert_array_equal(f['timestamp'][:],np.arange(n+1)/20)
            for k in ['timestep','timestamp']:np.testing.assert_array_equal(gt[k],f[k][:])
        out=dict(spec=spec,actions=n,observations=n+1,approved=True,physical_safety=True,oracle_max_abs=max(self.err),
            gt_alignment='PASS',**{k:v for k,v in result.items() if k not in ('approved','physical_safety')})
        out['hashes']={p.name:sha(p) for p in self.folder.iterdir() if p.is_file()}
        write(self.folder/'receipt.json',out,immutable=True)
        print('SEALED',spec['episode_id'],spec['kind'],spec['parent'],n,flush=True)
        return out


def replay_gate(path,env,expert,audit,state,schema):
    trials=[]
    for _ in range(2):
        obs=restore(env,expert,state);rows=[];commands=[];labs=[]
        for j in range(5):
            row,lab,err=row_label(env,obs,schema);rows.append(row);labs.append(lab)
            action=expert.act();commands.append(action.copy());obs,_=audit.step(action)
        trials.append((rows,labs,commands))
    maxima={}
    for category in (0,1):
        for x,y in zip(trials[0][category],trials[1][category]):
            for k in x:
                err=float(np.max(np.abs(x[k].astype(float)-y[k].astype(float))));maxima[k]=max(maxima.get(k,0),err)
                if '/rgb' in k or '/depth' in k:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
                else:np.testing.assert_allclose(x[k],y[k],atol=1e-6,rtol=0,err_msg=k)
    np.testing.assert_allclose(trials[0][2],trials[1][2],atol=1e-6,rtol=0)
    write(path,dict(status='PASS',disk_snapshot_restore=True,steps=5,repeat=2,max_abs=maxima,
        pixel_depth_exact=True,robot_GT_tolerance=1e-6),immutable=True)
    return restore(env,expert,state)


def run(root,worker,workers,limit=None):
    assert os.environ.get('CONTAINER_ID')=='57640710';torch.set_num_threads(1)
    plan=read(root/'supplement_plan.json');specs=plan['episodes'];parent_specs={}
    for s in specs:parent_specs.setdefault(s['parent'],[]).append(s)
    parents=sorted(parent_specs)[worker::workers]
    if limit:parents=parents[:limit]
    with ExperimentLogger(team='graph',model='Task10F1_independent_GT_corrective_capture',root=root/'experiments',job_type='prepare',
        config=dict(plan_sha256=sha(root/'supplement_plan.json'),worker=worker,workers=workers,original_source_read_only=True)) as log:
        write(root/'worker_identities'/f'{log.run_id}.json',{'run_id':log.run_id,'worker':worker},immutable=True)
        env=environment(root);done=[]
        try:
            for parent in parents:
                ss=parent_specs[parent];nom=next(s for s in ss if s['kind']=='nominal');branches=[s for s in ss if s['kind']=='recovery']
                if all((root/'raw'/str(s['episode_id'])/'receipt.json').exists() for s in ss):continue
                obs,_=env.reset(seed=parent);expert=ScriptedExpert(env);schema=state_schema(robot_observation(env)[0])
                assert schema==read(root/'normalizer.json')['schema'];audit=Audit(env);safety=PhysicalSafety(env,SAFETY);initial=physical_state(env)
                runtime=dict(versions=versions(),backend=str(env.unwrapped.backend),control_freq=env.unwrapped.control_freq,
                    physics_freq=env.unwrapped.sim_freq,sim_steps_per_control=env.unwrapped._sim_steps_per_control,
                    controller={k:asdict(v.config) for k,v in env.unwrapped.agent.controller.controllers.items()},
                    schema=schema,depth_unit='int16 mm; depth_float=raw/1000',camera_shapes={k:list(v['rgb'].shape) for k,v in obs['sensor_data'].items()})
                rp=root/f'runtime_worker{worker}.json'
                if rp.exists():assert read(rp)==runtime
                else:write(rp,runtime,immutable=True)
                snapshots=root/'snapshots'/str(parent);snapshots.mkdir(parents=True,exist_ok=True)
                anchors={};prefix=[];nomfolder=root/'raw'/str(nom['episode_id'])
                if (nomfolder/'receipt.json').exists():
                    nr=read(nomfolder/'receipt.json')
                    for name,h in nr['hashes'].items():assert sha(nomfolder/name)==h
                    checks=read(nomfolder/'physical_audit.json')['checks']
                    with h5py.File(nomfolder/'episode.h5') as f,np.load(nomfolder/'gt_supervision.npz') as g:
                        keys=[];f.visititems(lambda k,v:keys.append(k) if isinstance(v,h5py.Dataset) and len(v)==nr['actions']+1 else None)
                        for t in range(nr['actions']):prefix.append(({k:f[k][t] for k in keys},{k:g[k][t] for k in g.files},f['actions'][t],nr['oracle_max_abs']))
                    for t in sorted({s['anchor'] for s in branches}):
                        state=portable(torch.load(snapshots/f't{t:03d}.pt',map_location='cpu',weights_only=False),True)
                        restore(env,expert,state);anchors[t]=dict(state=state,prefix=prefix[:t],checks=checks[:t],physical=physical_state(env))
                else:
                    rec=Record(nomfolder,env,expert,audit,safety,initial,schema)
                    while expert.phase!='close' and len(prefix)<130:
                        t=len(prefix)
                        if branches and t in plan['anchors']:
                            state=save_snapshot(snapshots/f't{t:03d}.pt',env,expert,schema)
                            anchors[t]=dict(state=state,prefix=copy.deepcopy(prefix),checks=copy.deepcopy(rec.checks),physical=physical_state(env))
                        action=expert.act();obs=rec.step(obs,action,0,True)
                        prefix.append((rec.rows[-1],rec.labels[-1],action.copy(),rec.err[-1]))
                    assert expert.phase=='close','nominal reach timeout'
                    nr=rec.close(obs,dict(approved=True,phase=expert.phase),nom)
                done.append(nr)
                for anchor in sorted({s['anchor'] for s in branches}):
                    a=anchors[anchor];gate=snapshots/f't{anchor:03d}_replay.json'
                    if gate.exists():assert read(gate)['status']=='PASS'
                    else:obs=replay_gate(gate,env,expert,audit,a['state'],schema)
                    for spec in [s for s in branches if s['anchor']==anchor]:
                        if (root/'raw'/str(spec['episode_id'])/'receipt.json').exists():continue
                        obs=restore(env,expert,a['state']);start=physical_state(env)
                        np.testing.assert_allclose(start['qpos'],a['physical']['qpos'],atol=1e-6,rtol=0)
                        goal=start['ee_pose'].copy();goal[:3]+=np.asarray(spec['offset_mm'])*.001
                        rec=Record(root/'raw'/str(spec['episode_id']),env,expert,audit,safety,initial,schema,a['prefix'],a['checks'])
                        integral=np.zeros(3);stable=0
                        for step in range(FEEDBACK['max_steps']):
                            action,integral=bounded_action(env,goal,integral,FEEDBACK);obs=rec.step(obs,action,1,False);s=physical_state(env)
                            err=np.linalg.norm(s['ee_pose'][:3]-goal[:3]);ang=np.linalg.norm(rotvec_between(s['ee_pose'][3:],goal[3:]));vel=abs(s['qvel']).max()
                            stable=stable+1 if err<.001 and ang<.01 and vel<.03 else 0
                            if stable>=3:break
                        assert stable>=3,'frozen disturbance convergence failed'
                        target=refresh(env,expert,initial['peg_pose']);steps=0
                        while expert.phase!='close' and steps<130:obs=rec.step(obs,expert.act(),2,True);steps+=1
                        dp,dr=expert._err(expert.grasp);assert expert.phase=='close' and dp<.004 and dr<.03
                        assert physical_state(env)['qpos'][-2:].sum()>=.065
                        done.append(rec.close(obs,dict(approved=True,pose_recovery=True,position_error_m=float(dp),orientation_error_rad=float(dr),
                            target_refresh=target,disturbance_steps=step+1,recovery_steps=steps),spec))
                env.unwrapped.scene.step=audit.original_step
                write(root/f'collection_progress_{worker}.json',dict(completed=len(done),parents_complete=parent,at=now()))
                log.log(dict(episodes=len(done)),step=len(done))
            write(root/f'collection_worker{worker}.json',dict(status='PASS',episodes=done),immutable=True)
        finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.root,a.worker,a.workers,a.limit)
