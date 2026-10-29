"""Live RGB-D Reach evaluation and real same-state Peg interventions.

All object GT/Expert targets are evaluation-only. The policy path delegates
Plan A sensors -> causal masks -> CurrentGeometry -> original E0 step/codec.
"""
import argparse
import copy
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,npz,checkpoint_load,append_csv,tensor_identity,digest
from graph.plan_a_bc.cache import make_vision,GRAPH_KEYS
from graph.plan_a_bc.rollout import sensors
from graph.geometry_reach10d.runtime import environment
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_stop_clock import PhysicsAudit,controller_audit,validate_controller_command,numpy
from graph.integration_baseline.geometry09a import physical_state,collision_audit,expert_reference,infer
from graph.integration_baseline.geometry09a_report import response_metrics
from graph.integration_baseline.geometry09b import memory,restore
from graph.geometry_reach10d.collect import setup,target
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from graph.geometry_reach10d.profile import Resources
from recurrent.expert import ScriptedExpert,rotvec_between
from recurrent.common import to_sapien,to_action_tensor


def adapters_for(detector,tracker,identity):
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ['base_camera','hand_camera']}
    for v in adapters.values():v.reset(identity)
    return adapters


def live_graph(env,obs,identity,t,norm,geometry,adapters):
    cameras,data,surfaces=sensors(env,obs,identity,t,norm.schema)
    assert set(data)<=set(['episode','timestep','timestamp','robot76','robot_world_geometry','tcp_pose','qpos','robot_link_world'])
    masks={c.camera:adapters[c.camera].process(c.rgb,identity,t)[0] for c in cameras}
    graph,records,gate=geometry.build(cameras,data,surfaces,masks)
    return graph,data,cameras,masks,gate


def measure(env,expert,protocol):
    """Evaluation-only poses, absent from every student input dict."""
    tcp=to_sapien(env.unwrapped.agent.tcp.pose);pre=target(expert,'reach');final=expert.grasp
    a=float(np.linalg.norm(pre.p-tcp.p));ar=float(np.linalg.norm(rotvec_between(tcp.q,pre.q)))
    p=float(np.linalg.norm(final.p-tcp.p));r=float(np.linalg.norm(rotvec_between(tcp.q,final.q)))
    return dict(approach_position_error_m=a,approach_orientation_error_rad=ar,
        target_position_error_m=p,target_orientation_error_rad=r,
        approach=a<protocol['approach_position_m'] and ar<protocol['approach_orientation_rad'],
        final=p<protocol['final_position_m'] and r<protocol['final_orientation_rad'],
        tcp_pose=np.r_[tcp.p,tcp.q].tolist(),
        opening_m=float(numpy(env.unwrapped.agent.robot.get_qpos())[0,-2:].sum()))


@torch.inference_mode()
def rollout(root,folder,spec,model,norm,codec,detector,tracker,condition,step,horizon,*,smoke=False,evaluation_mode='development'):
    if (folder/'episode.json').exists():return read(folder/'episode.json')
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'trace.json').exists():raise ValueError('incomplete evaluation requires separate attempt directory')
    proto=read(root/'collection_protocol.json');threshold=proto['threshold']
    env=environment(root);obs,expert,initial=setup(env,dict(seed=spec['seed'],offset_mm=spec['offset_mm']))
    if not initial['physical_validity']['valid']:
        result=dict(condition=condition,step=step,seed=spec['seed'],variant=spec['variant'],
            status='INVALID_PHYSICAL_SETUP',reach_success=False,denominator_eligible=False)
        write(folder/'episode.json',result,immutable=True);env.close();return result
    identity=f'reach_eval:{spec["seed"]}:{spec["variant"]}'
    adapters=adapters_for(detector,tracker,identity);geometry=CurrentGeometry(root/'inputs',norm.schema)
    physics=PhysicsAudit(env);audit=controller_audit(env);hidden=None;trace=[]
    initial_peg=np.array(initial['perturbed']['peg_pose']);approach_step=None;early_close=False;success=False
    started=time.perf_counter();trajectory=[];best=float('inf');min_at=None
    writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(512,256))
    if not writer.isOpened():raise RuntimeError('video writer unavailable')
    try:
        for t in range(horizon):
            token=physics.token();tick=time.perf_counter()
            graph,data,cameras,masks,gate=live_graph(env,obs,identity,t,norm,geometry,adapters)
            raw,hidden=model.step(pack_graphs([graph],[(0,0)]).to('cuda'),
                torch.from_numpy(data['robot76'][None]).cuda(),norm,hidden)
            action=codec.decode(codec.tag(raw));command=numpy(action)[0]
            physics.unchanged(token);inference_seconds=time.perf_counter()-tick
            pre=measure(env,expert,threshold)
            if command[6]<=0 and not (approach_step is not None and pre['final']):early_close=True
            audit.clear();obs,_,term,trunc,_=env.step(action.to(env.unwrapped.device));physics.stepped(token)
            validate_controller_command(command,audit)
            post=measure(env,expert,threshold)
            if post['approach'] and approach_step is None:approach_step=t+1
            peg=numpy(env.unwrapped.peg.pose.raw_pose)[0]
            drift=float(np.linalg.norm(peg[:3]-initial_peg[:3]))
            collision=None
            candidate=approach_step is not None and post['final'] and post['opening_m']>=threshold['opening_min_m'] and not early_close and drift<threshold['max_peg_motion_m']
            if candidate:
                collision=collision_audit(env,'peg');success=collision['valid']
            if post['target_position_error_m']<best:best=post['target_position_error_m'];min_at=t+1
            trace.append(dict(timestep=t,action=command.tolist(),inference_seconds=inference_seconds,
                peg_points=int((graph.node_type==3).sum()),box_points=int((graph.node_type==4).sum()),
                early_close=early_close,peg_motion_m=drift,collision=collision,gate=gate,
                nodes=len(graph.node_type),edges=graph.edge_index.shape[1],**post))
            trajectory.append(data['robot76'])
            frame=np.concatenate([cv2.resize(c.rgb,(256,256),interpolation=cv2.INTER_NEAREST) for c in cameras],axis=1)
            cv2.putText(frame,f'{condition} {step} seed{spec["seed"]} {spec["variant"]} t{t} err{1000*post["target_position_error_m"]:.1f}mm',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1)
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            if t==0:
                npz(folder/'initial_student_input.npz',dict(robot76=data['robot76'],
                    **{k:getattr(graph,k).numpy() for k in GRAPH_KEYS}))
            if t%25==0:print('LIVE',condition,step,spec['seed'],spec['variant'],t,post['target_position_error_m'],flush=True)
            if success or bool(term.item()) or bool(trunc.item()):break
        final_collision=collision_audit(env,'peg')
        result=dict(condition=condition,step=step,seed=spec['seed'],variant=spec['variant'],status='SMOKE_COMPLETE' if smoke else 'COMPLETE',
            evaluation_mode=evaluation_mode,denominator_eligible=not smoke,reach_success=bool(success),approach_step=approach_step,
            actions=len(trace),early_gripper_close=early_close,min_target_distance_m=best,min_target_step=min_at,
            final_target_position_error_m=post['target_position_error_m'],
            final_target_orientation_error_rad=post['target_orientation_error_rad'],
            final_collision_valid=final_collision['valid'],max_peg_motion_m=max(x['peg_motion_m'] for x in trace),
            wall_seconds=time.perf_counter()-started,current_RGBD=True,expert_cache_used=False,
            GT_as_student_input=False,initial_physical_sha256=digest(initial['perturbed']),
            initial_original_sha256=digest(initial['original']),initial_state=initial,
            target_pose=np.r_[expert.grasp.p,expert.grasp.q].tolist())
        write(folder/'trace.json',trace,immutable=True);write(folder/'episode.json',result,immutable=True)
        npz(folder/'robot_action_trace.npz',dict(robot76=np.asarray(trajectory),action7=np.array([r['action'] for r in trace])))
        append_csv(root/'reach_rollout_metrics.csv',{k:v for k,v in result.items() if k not in ['initial_state','target_pose']})
        return result
    finally:
        writer.release();physics.close()
        for a in adapters.values():a.close()
        detector.cache.clear();env.close();torch.cuda.empty_cache()


@torch.inference_mode()
def interventions(root,folder,seed,model,norm,codec,detector,tracker,condition,step):
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    proto=read(root/'collection_protocol.json');env=environment(root)
    obs,history_expert,initial=setup(env,dict(seed=seed,offset_mm=[0,0,0]))
    identity=f'reach_intervention:{seed}'
    aa=adapters_for(detector,tracker,identity);geom=CurrentGeometry(root/'inputs',norm.schema)
    physics=PhysicsAudit(env);hidden=None;results=[]
    try:
        for t in range(11):
            token=physics.token();before_hidden=hidden.clone() if hidden is not None else None
            prior={c:memory(a) for c,a in aa.items()}
            baseline=physical_state(env);pose=env.unwrapped.peg.pose.raw_pose.clone()
            graph,data,cams,masks,gate=live_graph(env,obs,identity,t,norm,geom,aa)
            base,hidden=infer(model,norm,codec,graph,data['robot76'],before_hidden)
            if t in [0,10]:
                anchor_rows=[]
                anchor=ScriptedExpert(env);anchor.phase=history_expert.phase;anchor.counter=history_expert.counter
                eb,base_detail=expert_reference(env,anchor,'peg',dict(xyz_mm=[0,0,0],yaw_deg=0))
                assert base_detail['status']=='VALID_STAGE_CONDITIONED'
                npz(folder/f't{t:03d}_baseline.npz',dict(robot76=data['robot76'],action7=base['action'],
                    expert_action7=eb,previous_hidden=base['previous_hidden'],
                    **{k:getattr(graph,k).numpy() for k in GRAPH_KEYS},
                    **{c.camera+'/'+k:v for c in cams for k,v in [('rgb',c.rgb),('depth_raw_mm',c.depth_raw_mm)]}))
                for offset in proto['intervention_offsets_mm']:
                    for c,a in aa.items():restore(a,prior[c])
                    alt_pose=pose.clone();alt_pose[0,:3]+=alt_pose.new_tensor(offset)*.001
                    env.unwrapped.peg.set_pose(Pose.create(alt_pose));current=physical_state(env)
                    for key in baseline:
                        if key!='peg_pose':np.testing.assert_array_equal(baseline[key],current[key])
                    valid=collision_audit(env,'peg')
                    alt_obs=env.unwrapped.get_obs()
                    altg,altd,altcams,_,_=live_graph(env,alt_obs,identity,t,norm,geom,aa)
                    np.testing.assert_array_equal(altd['robot76'],data['robot76'])
                    alt,_=infer(model,norm,codec,altg,altd['robot76'],before_hidden)
                    ea,detail=expert_reference(env,anchor,'peg',dict(xyz_mm=offset,yaw_deg=0))
                    metric=response_metrics(base,alt,controller_commands(eb[None])[0],controller_commands(ea[None])[0])
                    npz(folder/f't{t:03d}_x{offset[0]:+d}_y{offset[1]:+d}.npz',dict(
                        robot76=altd['robot76'],action7=alt['action'],expert_action7=ea,previous_hidden=alt['previous_hidden'],
                        **{k:getattr(altg,k).numpy() for k in GRAPH_KEYS},
                        **{c.camera+'/'+k:v for c in altcams for k,v in [('rgb',c.rgb),('depth_raw_mm',c.depth_raw_mm)]}))
                    assert metric['previous_hidden_exact']
                    row=dict(condition=condition,step=step,seed=seed,timestep=t,offset_mm=offset,
                        robot_state_sha256=digest(data['robot76'].tolist()),
                        physical_valid=valid['valid'],expert_reference_valid=detail['status']=='VALID_STAGE_CONDITIONED',
                        expert_delta_mm=metric['expert_translation_mm'],model_delta_mm=metric['model_translation_mm'],
                        direction_cosine=metric['translation_direction_cosine'],signed_gain=metric['translation_signed_gain'],
                        response_error_mm=metric['translation_response_error_mm'],
                        zero_response_improvement_mm=metric['expert_translation_mm']-metric['translation_response_error_mm'],
                        previous_hidden_exact=True,history='identical observed RGB prefix; actual Peg pose intervention before current render')
                    anchor_rows.append(dict(row=row,metrics=metric,reference=detail))
                    env.unwrapped.peg.set_pose(Pose.create(pose));physics.unchanged(token)
                # Recompute original current frame from the shared prefix. This
                # also restores the JPEG on disk, needed for late detections.
                for c,a in aa.items():restore(a,prior[c])
                repeatg,repeatd,_,repeatm,_=live_graph(env,env.unwrapped.get_obs(),identity,t,norm,geom,aa)
                repeat,hidden=infer(model,norm,codec,repeatg,repeatd['robot76'],before_hidden)
                exact=all(torch.equal(getattr(graph,k),getattr(repeatg,k)) for k in GRAPH_KEYS)
                exact=exact and all(np.array_equal(base[k],repeat[k]) for k in ['action','hidden','readout'])
                exact=exact and all(np.array_equal(masks[c],repeatm[c]) for c in masks)
                for record in anchor_rows:
                    row=record['row'];row.update(baseline_repeat_exact=exact,
                        analysis_eligible=exact and row['physical_valid'] and row['expert_reference_valid'])
                    append_csv(root/'geometry_correction_metrics.csv',row)
                    results.append(dict(**row,metrics=record['metrics'],reference=record['reference']))
            physics.unchanged(token)
            if t<10:
                obs,_,_,_,_=env.step(to_action_tensor(history_expert.act(),env));physics.stepped(token)
        write(folder/'complete.json',dict(status='PASS',rows=results,seed=seed,
            repeated_baseline_graph_action_hidden_exact=all(r['baseline_repeat_exact'] for r in results),actual_student_weights_unchanged=True,
            scope='initial offsets and same-history one-step diagnostics; no recovery rollout'),immutable=True)
    finally:
        physics.close()
        for a in aa.values():a.close()
        detector.cache.clear();env.close();torch.cuda.empty_cache()


def run(root,checkpoint,condition,mode):
    proto=read(root/'collection_protocol.json')
    if mode=='final':
        assert (root/'model_selection_frozen.json').exists()
        from graph.geometry_reach10d.final_expert import run as final_expert
        final_expert(root)
    cp=checkpoint_load(checkpoint);assert cp['config_sha256']==sha(root/'config.json')
    norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    model=CachedGraphPolicy(norm.schema,'E0',**read(root/'config.json')['model']).cuda().eval()
    model.load_state_dict(cp['model']);model.requires_grad_(False);weight=tensor_identity(model.state_dict())
    step=cp['progress']['step'];paths=read(root/'runtime_paths.json')
    dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    detector=DetectionMemo(dino,condition+'_'+mode,read(root/'optimization_equivalence.json')['model_identity'])
    import graph.plan_a_geometry.core as core
    core.components=components
    folder=root/'evaluations'/mode/condition/f'step{step:06d}';folder.mkdir(parents=True,exist_ok=True)
    specs=proto['final_cases'] if mode=='final' else proto['live_development'] if mode=='development' else [proto['live_development'][0]]
    with ExperimentLogger(team='graph',model='Task10D_'+condition+'_Reach_'+mode,seed=42,
            config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],condition=condition,mode=mode,
                checkpoint_sha256=sha(checkpoint),step=step,threshold=proto['threshold']),
            root=root/'evaluation_runs',job_type='evaluate') as log,Resources(root,'evaluation_'+condition+'_'+mode):
        log.link_output(checkpoint,checkpoint=True)
        rows=[rollout(root,folder/f'{s["seed"]}_{s["variant"]}',s,model,norm,codec,detector,tracker,condition,step,
            3 if mode=='smoke' else proto['evaluation_horizon'],smoke=mode=='smoke',evaluation_mode=mode) for s in specs]
        if mode=='final' or mode=='development' and step==11487:
            for seed in proto['final_intervention_seeds'] if mode=='final' else proto['development_intervention_seeds']:
                interventions(root,folder/f'intervention_{seed}',seed,model,norm,codec,detector,tracker,condition,step)
        assert tensor_identity(model.state_dict())==weight
        write(folder/'summary.json',dict(status='COMPLETE',mode=mode,condition=condition,step=step,
            rows=rows,weight_hash_unchanged=True,checkpoint_sha256=sha(checkpoint)),immutable=True)
        log.link_output(folder/'summary.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--condition',required=True);p.add_argument('--mode',choices=['smoke','development','final'],required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    run(a.root,a.checkpoint,a.condition,a.mode)
