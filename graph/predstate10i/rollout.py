"""Task10H engine with new policy labels, engineering caps and post-action diagnostics."""
import os
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse, copy, csv, fcntl, hashlib, importlib.metadata, json, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from graph.graph_recurrent10g.eval_runner import (load_model,vision,nested_hash,csv_write,
 sensors,evaluation_measure,video_frame,CurrentGeometry,environment,pack_graphs,
 OnlineVisionAdapter,PhysicsAudit,controller_audit,validate_controller_command,
 numpy,seed_all,snapshot,portable,ScriptedExpert,GRAPH_KEYS,tensor_identity)
from graph.plan_a_bc.common import read,write,sha,now
from graph.integration_baseline.geometry09a import collision_audit
from recurrent.common import to_sapien
from recurrent.expert import rotvec_between
from utils.experiment_logger import ExperimentLogger
import sapien

CONDITIONS=['E','B0','H1','H2']
SELECTED='8ab82f390dccf054b0bd6539dbd55f7a60af2ae7934f53234702ab66c1369ea3'

def physical(env,expert):
    state=portable(snapshot(env,expert))
    return {k:state[k] for k in ('sim','clock','controllers','rng')}

def anchor_metrics(env,expert,pre,streak,phase,previous):
    goal=to_sapien(env.unwrapped.goal_pose)
    target=goal*sapien.Pose([-0.02-expert.half_len,0,0])
    dp,dr=expert._peg_err(target)
    lift_err=float(expert._err(expert.lift_target)[0]) if expert.lift_target is not None else None
    kind='H1' if previous=='lift' and phase=='align' else 'H2' if previous=='align' and phase=='insert' else None
    finite=all(np.isfinite(numpy(x)).all() for x in (env.unwrapped.agent.robot.get_qpos(),env.unwrapped.agent.robot.get_qvel()))
    valid=bool(finite and pre['grasp'] and pre['lift_candidate'] and streak>=3)
    if kind=='H1':valid=valid and lift_err<.015
    if kind=='H2':valid=valid and dp<.002 and dr<.02
    return dict(kind=kind,valid=bool(valid),phase=phase,previous_action_phase=previous,
      align_position_error_m=float(dp),align_orientation_error_rad=float(dr),lift_tcp_error_m=lift_err,
      finite_robot=bool(finite),lift_streak=streak,**pre)

@torch.inference_mode()
def episode(old,out,model,norm,codec,detector,tracker,seed,condition,stage,policy_label,horizon=250,post_handoff_steps=None,observers=None):
    folder=out/'rollouts'/stage/f'seed{seed}'/policy_label
    if (folder/'episode.json').exists():
        row=read(folder/'episode.json')
        assert sha(folder/'trace.json')==row['trace_sha256']
        assert sha(folder/'rollout.mp4')==row['video_sha256']
        return row
    if folder.exists():folder.rename(folder.with_name(policy_label+'_failed_'+str(time.time_ns())))
    folder.mkdir(parents=True)
    seed_all(42,1);cfg=read(old/'config.json')['evaluation'];env=environment(old)
    obs,_=env.reset(seed=seed);expert=ScriptedExpert(env)
    initial=physical(env,expert);torch.save(initial,folder/'initial_snapshot.pt')
    initial_hash=nested_hash(initial);assert nested_hash(torch.load(folder/'initial_snapshot.pt',weights_only=False))==initial_hash
    identity=f'task10h:seed{seed}'
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ('base_camera','hand_camera')}
    for a in adapters.values():a.reset(identity)
    geometry=CurrentGeometry(old/'inputs',norm.schema);physics=PhysicsAudit(env);audit=controller_audit(env)
    initial_z=float(numpy(env.unwrapped.peg.pose.raw_pose)[0,2]);peg_half_sizes=numpy(env.unwrapped.peg_half_sizes)[0].tolist();hole_radius=numpy(env.unwrapped.box_hole_radii).reshape(-1).tolist()
    observer_hidden={k:None for k in (observers or {})};hidden=None;trace=[];anchors=[];first={k:None for k in ('grasp','lift','alignment','insertion')}
    previous=None;streak=0;handoff=None;reason='horizon';tick0=time.perf_counter()
    writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(768,448))
    assert writer.isOpened()
    encstart=torch.cuda.Event(enable_timing=True);encend=torch.cuda.Event(enable_timing=True)
    h1=model.graph_encoder.register_forward_pre_hook(lambda *args:encstart.record())
    h2=model.graph_encoder.register_forward_hook(lambda *args:encend.record())
    try:
        for t in range(horizon):
            token=physics.token();tick=time.perf_counter()
            cameras,data,surfaces=sensors(env,obs,identity,t,norm.schema)
            def current_masks():
                return {c.camera:adapters[c.camera].process(c.rgb,identity,t,mask_device='cuda')[0] for c in cameras}
            graph,_,gate=geometry.build(cameras,data,surfaces,current_masks)
            from visual.runtime.tensor_packet import numpy_packet
            masks=numpy_packet(geometry.last_raw_masks)
            inputs={k:numpy(getattr(graph,k)) for k in GRAPH_KEYS};inputs['robot76']=data['robot76']
            rgbd={c.camera:dict(rgb=numpy(c.rgb),depth=numpy(c.depth_raw_mm),intrinsic=c.intrinsic if hasattr(c,'intrinsic') else numpy(obs['sensor_param'][c.camera]['intrinsic_cv']),extrinsic=numpy(obs['sensor_param'][c.camera]['extrinsic_cv'])) for c in cameras}
            input_hash=nested_hash(inputs);rgbd_hash=nested_hash(rgbd);state_hash=nested_hash(physical(env,expert))
            before_hidden=nested_hash(hidden);pre=evaluation_measure(env,initial_z,cfg)
            phase=expert.phase
            anchor=anchor_metrics(env,expert,pre,streak,phase,previous)
            if anchor['kind'] and handoff is None:
                anchor.update(timestep=t,state_sha256=state_hash,input_sha256=input_hash,rgbd_sha256=rgbd_hash,hidden_before_sha256=before_hidden)
                anchors.append(anchor)
                torch.save(physical(env,expert),folder/(anchor['kind']+'_physical.pt'))
                if condition==anchor['kind'] and anchor['valid']:handoff=t
            packed=pack_graphs([graph],[(0,0)]).to('cuda');robot=torch.from_numpy(data['robot76'][None]).cuda()
            pt=time.perf_counter();raw,hidden=model.step(packed,robot,norm,hidden)
            policy=codec.decode(codec.tag(raw));torch.cuda.synchronize();policy_seconds=time.perf_counter()-pt
            inference=time.perf_counter()-tick;encoder_seconds=encstart.elapsed_time(encend)/1000
            b=numpy(policy)[0];assert np.isfinite(b).all()
            observer_trace={}
            for family,other in (observers or {}).items():
                ob=nested_hash(observer_hidden[family])
                oa,oh=other.step(packed,robot,norm,observer_hidden[family])
                observer_hidden[family]=oh
                observer_trace[family]=dict(hidden_before_sha256=ob,hidden_after_sha256=nested_hash(oh),
                    policy_action=numpy(codec.decode(codec.tag(oa)))[0].tolist())
            # GT diagnostics follow sensor-only action/hidden computation.
            predicted=getattr(model,'last_state38',None)
            prediction_diagnostic={}
            if predicted is not None:
                from recurrent.common import raw_gt
                from recurrent.features import build_features
                from graph.gt_pretrain10f1.evaluate import errors
                measured_gt={k:v[None] for k,v in raw_gt(env).items()}
                oracle=build_features(measured_gt,np.ones(1,'f'),np.zeros(1,'f'))
                p18=np.concatenate([predicted[:,15:22],predicted[:,25:36]],-1)
                g18=np.concatenate([oracle[:,15:22],oracle[:,25:36]],-1)
                prediction_diagnostic=dict(predicted_state38=predicted[0].tolist(),oracle_gt38=oracle[0].tolist(),
                    state_prediction_errors={k:float(v[0][0]) for k,v in errors(p18,g18).items()})
            # Advance only Expert bookkeeping; no teacher value reaches model.step.
            target_capture={}
            servo=expert._servo
            def record_target(target,max_trans):
                target_capture['pose']=target
                return servo(target,max_trans)
            expert._servo=record_target
            try:teacher=expert.act().copy()
            finally:del expert._servo
            target=target_capture.get('pose',to_sapien(env.unwrapped.agent.tcp.pose))
            target_pre_position=float(np.linalg.norm(target.p-np.asarray(pre['tcp_pose'][:3])))
            target_pre_rotation=float(np.linalg.norm(rotvec_between(np.asarray(pre['tcp_pose'][3:]),target.q)))
            student=condition=='B0' or handoff is not None
            action=b if student else teacher
            if t==0:initial_input=input_hash;initial_rgbd=rgbd_hash
            physics.unchanged(token);audit.clear();step_tick=time.perf_counter()
            obs,_,term,trunc,info=env.step(torch.from_numpy(action[None]).to(env.unwrapped.device));sim_seconds=time.perf_counter()-step_tick
            physics.stepped(token);validate_controller_command(action,audit)
            post=evaluation_measure(env,initial_z,cfg)
            assert post['success']==bool(info['success'].item())
            streak=streak+1 if post['lift_candidate'] else 0
            statuses=dict(grasp=post['grasp'],lift=streak>=cfg['lift_consecutive_steps'],alignment=post['alignment'],insertion=post['success'])
            for k,v in statuses.items():
                if v and first[k] is None:first[k]=t+1
            contacts=collision_audit(env,'peg')
            contact_events=[]
            for contact in env.unwrapped.scene.px.get_contacts():
                contact_events.append(dict(bodies=[b.entity.name for b in contact.bodies],max_impulse=float(max((np.linalg.norm(p.impulse) for p in contact.points),default=0.))))
            tcp_delta=np.asarray(post['tcp_pose'][:3])-pre['tcp_pose'][:3]
            tip0=float(np.linalg.norm(pre['peg_head_pos_at_hole']));tip1=float(np.linalg.norm(post['peg_head_pos_at_hole']))
            orientation=float(np.linalg.norm(rotvec_between(np.asarray(post['peg_pose'][3:]),np.asarray(post['hole_pose'][3:]))))
            missing=[not bool((graph.node_type==k).any()) for k in (3,4)]
            trace.append(dict(observers=observer_trace,**prediction_diagnostic,timestep=t,phase=phase,phase_after_act=expert.phase,control='B' if student else 'Expert',
                handoff_timestep=handoff,handoff_relative_step=t-handoff if handoff is not None else None,
                policy_action=b.tolist(),policy_raw=numpy(raw).reshape(-1).tolist(),expert_action=teacher.tolist(),action=action.tolist(),
                controller_audit={k:v.tolist() for k,v in audit.items()},input_sha256=input_hash,rgbd_sha256=rgbd_hash,state_sha256=state_hash,
                qpos=numpy(env.unwrapped.agent.robot.get_qpos())[0].tolist(),qvel=numpy(env.unwrapped.agent.robot.get_qvel())[0].tolist(),
                robot76_sha256=nested_hash(data['robot76']),hidden_before_sha256=before_hidden,hidden_after_sha256=nested_hash(hidden),hidden_norm=float(hidden.norm()),
                graph_nodes=int(len(graph.node_type)),empty_graph=len(graph.node_type)==0,missing_objects=missing,gate=gate,
                translation_magnitude=float(np.linalg.norm(action[:3])*.1),rotation_component_magnitude=float(np.linalg.norm(action[3:6])*.1),
                saturation=bool((np.abs(action)>.999).any()),arm_saturation=bool((np.abs(action[:6])>.999).any()),
                action_direction_error=bool(np.dot(b[:3],teacher[:3])<0 and np.linalg.norm(teacher[:3])>1e-6),
                action_difference_mae=float(np.abs(b-teacher).mean()),tcp_displacement_m=tcp_delta.tolist(),
                target_error_change_m=tip1-tip0,pre_tip_hole_error_m=tip0,tip_hole_error_m=tip1,
                peg_hole_orientation_error_rad=orientation,pre=pre,collision=contacts,contact_events=contact_events,
                expert_tcp_target_pose=np.r_[target.p,target.q].tolist(),tcp_target_position_error_pre_m=target_pre_position,
                tcp_target_position_error_post_m=float(np.linalg.norm(target.p-np.asarray(post['tcp_pose'][:3]))),
                tcp_target_rotation_error_pre_rad=target_pre_rotation,tcp_target_rotation_error_post_rad=float(np.linalg.norm(rotvec_between(np.asarray(post['tcp_pose'][3:]),target.q))),
                inference_seconds=inference,policy_seconds=policy_seconds,gnn_encoder_readout_seconds=encoder_seconds,
                simulator_step_wall_seconds=sim_seconds,control_dt_seconds=.05,
                terminated=bool(term.item()),truncated=bool(trunc.item()),**post))
            writer.write(video_frame(cameras,masks,policy_label,seed,getattr(model,'checkpoint_step',0),t,post))
            previous=phase
            if t%25==0:print('LIVE',stage,seed,condition,t,phase,'handoff',handoff,flush=True)
            if post_handoff_steps is not None and handoff is not None and t-handoff+1>=post_handoff_steps:
                reason='engineering_handoff_cap';break
            if post['success'] or term.item() or trunc.item():
                reason='success' if post['success'] else 'termination' if term.item() else 'time_limit';break
    finally:
        geometry.close()
        writer.release();physics.close();h1.remove();h2.remove()
        for a in adapters.values():a.close()
        detector.cache.clear();env.close();torch.cuda.empty_cache()
        write(folder/'trace.json',trace);write(folder/'anchors.json',anchors)
    labels=[];controlled=[r for r in trace if r['control']=='B']
    if condition in ('H1','H2') and handoff is None:labels.append('ANCHOR_UNREACHABLE')
    if not first['insertion']:
        if controlled and any(r['pre']['grasp'] and not r['grasp'] for r in controlled):labels.append('GRASP_LOST')
        if any(r['action'][6]>0 and r['pre']['grasp'] for r in controlled):labels.append('PREMATURE_GRIPPER_OPEN')
        if any(r['action'][6]<0 and r['pre']['tcp_peg_distance_m']>.03 and not r['pre']['grasp'] for r in controlled):labels.append('PREMATURE_GRIPPER_CLOSE')
        if any(any(r['missing_objects']) for r in controlled):labels.append('GEOMETRY_MISSING')
        if any(r['arm_saturation'] for r in controlled):labels.append('ACTION_SATURATION')
        if any(r['action_direction_error'] for r in controlled):labels.append('ACTION_DIRECTION_ERROR')
        if handoff is not None and controlled[-1]['tip_hole_error_m']>controlled[0]['pre_tip_hole_error_m']+.005:labels.append('ALIGNMENT_DIVERGENCE')
        if handoff is not None and controlled[-1]['alignment_angle_deg']>cfg['alignment_angle_deg']:labels.append('INSERTION_MISALIGNMENT')
        if reason in ('horizon','time_limit'):labels.append('TIMEOUT')
        # Budget flag denotes exhaustion, not proof that extra time would succeed.
        if handoff is not None and len(trace)==250:labels.append('HORIZON_LIMIT')
    row=dict(status='COMPLETE',stage=stage,seed=seed,condition=policy_label,engine_condition=condition,policy_family=policy_label[0],actions=len(trace),termination=reason,
      handoff_timestep=handoff,valid_anchor=handoff is not None if condition in ('H1','H2') else None,
      completion_steps=(len(trace)-handoff if handoff is not None else len(trace)) if first['insertion'] and (condition not in ('H1','H2') or handoff is not None) else None,
      **{k:bool(first[k] is not None and (k!='insertion' or condition not in ('H1','H2') or handoff is not None)) for k in first},official_episode_insertion=first['insertion'] is not None,handoff_insertion=bool(handoff is not None and first['insertion']),
      first_success_steps=first,failure_labels=labels,initial_snapshot_sha256=initial_hash,
      initial_input_sha256=initial_input,initial_rgbd_sha256=initial_rgbd,peg_half_sizes=peg_half_sizes,hole_radius=hole_radius,wall_seconds=time.perf_counter()-tick0,
      trace_sha256=sha(folder/'trace.json'),video_sha256=sha(folder/'rollout.mp4'),folder=str(folder.relative_to(out)))
    write(folder/'episode.json',row);return row
