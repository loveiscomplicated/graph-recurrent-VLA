"""Read-only inherited-source diagnosis and bounded disturbance calibration."""
import argparse,copy,inspect,os,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch,sapien
from scipy.spatial.transform import Rotation
from graph.corrective_reach10f.pilot import snapshot,restore
from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d.profile import hardware,Resources
from graph.integration_baseline.geometry09a import physical_state,collision_meshes,convex_gap
from graph.plan_a_bc.common import read,write,sha
from recurrent.expert import ScriptedExpert,rotvec_between
from recurrent.common import to_action_tensor
from utils.experiment_logger import ExperimentLogger

PARENT='706f62ce213247e7973102962858f95b'

def vec(t):return t.detach().cpu().numpy().copy()[0]

class Audit:
    def __init__(self,env):
        self.env=env;self.arm=env.unwrapped.agent.controller.controllers['arm'];self.substeps=[];self.ik=[]
        self.robot_names={l._objs[0].entity.name for l in env.unwrapped.agent.robot.get_links()}
        self.original_ik=self.arm.kinematics.compute_ik
        def ik(*args,**kwargs):
            result=self.original_ik(*args,**kwargs)
            self.ik.append(dict(success=result is not None,returned_qpos=None if result is None else vec(result).tolist()))
            return result
        self.arm.kinematics.compute_ik=ik
        self.original_step=env.unwrapped.scene.step
        def step(*args,**kwargs):
            result=self.original_step(*args,**kwargs);self.substeps.append(self.contacts());return result
        env.unwrapped.scene.step=step
    def contacts(self):
        out=[]
        for c in self.env.unwrapped.scene.px.get_contacts():
            names=[b.entity.name for b in c.bodies]
            if any(n in self.robot_names for n in names):
                out.append(dict(names=names,min_separation_m=min((float(p.separation) for p in c.points),default=0),
                    max_impulse=float(max((np.linalg.norm(p.impulse) for p in c.points),default=0))))
        return out
    def step(self,action):
        pre=physical_state(self.env);self.substeps=[];self.ik=[]
        scaled=vec(self.arm._preprocess_action(torch.tensor(action[:6][None],device=self.env.unwrapped.device)))
        obs,*_=self.env.step(to_action_tensor(action,self.env));post=physical_state(self.env)
        a=dict(action=action.tolist(),scaled_action=scaled.tolist(),qpos_before=pre['qpos'].tolist(),qpos_after=post['qpos'].tolist(),qvel_after=post['qvel'].tolist(),
            tcp_before=pre['ee_pose'].tolist(),tcp_after=post['ee_pose'].tolist(),actual_step_m=(post['ee_pose'][:3]-pre['ee_pose'][:3]).tolist(),
            controller_target_pose=vec(self.arm._target_pose.raw_pose).tolist(),controller_target_qpos=vec(self.arm._target_qpos).tolist(),
            start_qpos=vec(self.arm._start_qpos).tolist(),ik=self.ik,substep_contacts=self.substeps,clock=int(self.env.unwrapped.elapsed_steps.item()))
        return obs,a

def bounded_action(env,pose,integral,protocol):
    state=physical_state(env);error=pose[:3]-state['ee_pose'][:3]
    integral=np.clip(integral+protocol['integral_gain']*error,-protocol['integral_cap_m'],protocol['integral_cap_m'])
    dp=protocol['position_gain']*error+integral
    dp*=min(1,protocol['max_translation_command_m']/max(np.linalg.norm(dp),1e-15))
    root=vec(env.unwrapped.agent.robot.pose.raw_pose);R=Rotation.from_quat(root[[4,5,6,3]]).as_matrix()
    rv=R.T@rotvec_between(state['ee_pose'][3:],pose[3:]);rv*=min(1,protocol['max_rotation_command_rad']/max(np.linalg.norm(rv),1e-15))
    euler=Rotation.from_rotvec(rv).as_euler('XYZ')
    return np.r_[R.T@dp/.1,-euler/.1,1].astype(np.float32),integral

def diagnose(root,base):
    assert os.getenv('CONTAINER_ID')=='57640710';torch.set_num_threads(1);root.mkdir(parents=True,exist_ok=False)
    for name in ['inputs','config.json','normalizer.json','action_codec.json','renderer_gpu_gate.json','collection_candidates.json']:(root/name).symlink_to(base/name)
    config=dict(task='10-F recovery diagnosis',parent_reference=PARENT,seed=30000,anchors=[10,25],offsets_mm=[[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]],
        calibration_only=True,max_steps=60,max_translation_command_m=.006,max_rotation_command_rad=.01,position_gain=1.5,integral_gain=.2,integral_cap_m=.004,
        position_error_m=.001,orthogonal_error_m=.001,orientation_error_rad=.01,max_qvel=.03,stable_steps=3)
    write(root/'calibration_protocol.json',config,immutable=True)
    with ExperimentLogger(team='graph',model='Task10F_recovery_controller_diagnosis',root=root/'experiments',config=config,seed=30000,job_type='prepare') as log,Resources(root,'controller_diagnosis'):
        log.record(parent_reference=PARENT);write(root/'identity.json',dict(run_id=log.run_id,parent_reference=PARENT),immutable=True)
        write(root/'hardware.json',hardware(),immutable=True)
        env=environment(root);obs,_=env.reset(seed=30000);expert=ScriptedExpert(env);audit=Audit(env)
        task=env.unwrapped;write(root/'controller.json',dict(config=asdict(audit.arm.config),root_pose=vec(task.agent.robot.pose.raw_pose).tolist(),backend=str(task.backend)),immutable=True)
        initial=physical_state(env);anchors={};nominal=[];clearance=[]
        try:
            while expert.phase!='close' and len(nominal)<130:
                t=len(nominal)
                if t in config['anchors']:anchors[t]=snapshot(env,expert)
                obs,row=audit.step(expert.act());row['legacy_height_valid']=row['tcp_after'][2]>.02
                meshes=collision_meshes(env);gaps={}
                for name,values in meshes.items():
                    if name.startswith('robot:'):
                        for obj in ['table-workspace','peg_0','box_with_hole_0']:
                            gaps[name+'|'+obj]=min((convex_gap(a,b) for a in values for b in meshes[obj]),default=10.)
                row['robot_object_signed_gaps_m']=gaps
                nominal.append(row)
            write(root/'nominal_safety_audit.json',dict(phase=expert.phase,trace=nominal,initial={k:v.tolist() for k,v in initial.items()},old_height_rejects=sum(not r['legacy_height_valid'] for r in nominal)),immutable=True)
            all_results=[]
            for anchor,state in anchors.items():
                for offset in config['offsets_mm']:
                    for method in ['original8','bounded_calibration']:
                        restore(env,expert,state);start=physical_state(env);pose=start['ee_pose'].copy();pose[:3]+=np.array(offset)*.001
                        integral=np.zeros(3);trace=[];stable=0
                        for t in range(8 if method=='original8' else config['max_steps']):
                            if method=='original8':action=np.r_[expert._servo(sapien.Pose(pose[:3],pose[3:]),.002),1].astype(np.float32)
                            else:action,integral=bounded_action(env,pose,integral,config)
                            obs,row=audit.step(action);s=physical_state(env)
                            error=float(np.linalg.norm(s['ee_pose'][:3]-pose[:3]));angle=float(np.linalg.norm(rotvec_between(s['ee_pose'][3:],pose[3:])))
                            velocity=float(abs(s['qvel']).max());stable=stable+1 if error<.001 and angle<.01 and velocity<.03 else 0
                            row.update(position_error_m=error,orientation_error_rad=angle,max_qvel=velocity,integral_m=integral.tolist());trace.append(row)
                            if method!='original8' and stable>=3:break
                        result=dict(anchor=anchor,offset_mm=offset,method=method,steps=len(trace),actual_offset_mm=((s['ee_pose'][:3]-start['ee_pose'][:3])*1000).tolist(),
                            position_error_m=error,orientation_error_rad=angle,max_qvel=velocity,converged=stable>=3,ik_failures=sum(not i['success'] for r in trace for i in r['ik']),trace=trace)
                        all_results.append(result);write(root/'disturbance_controller_audit.json',all_results)
                        print('AUDIT',anchor,offset,method,len(trace),error,velocity,result['ik_failures'],flush=True)
        finally:
            env.close()
            for p in root.glob('*.json'):
                if not p.is_symlink():log.link_output(p,kind='reports')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('/workspace/task10f_corrective_reach'));a=p.parse_args();diagnose(a.root,a.base)
