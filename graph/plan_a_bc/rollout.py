"""Live current RGB-D stop-the-clock evaluation, in a separate process.

There is deliberately no CachedDataset import or expert trajectory reader here.
Object GT is read solely by evaluation_measure after the student input path.
"""
import argparse
from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from envs.maniskill import make_peg_insertion_env,get_gt_state
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.integration_baseline.rollout_stop_clock import (PhysicsAudit,controller_audit,
    validate_controller_command,reset_diagnostics,controller_semantics,numpy)
from graph.integration_baseline.robot import robot_observation,state_schema,flatten_state
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.self_filter import robot_surfaces
from visual.runtime.grounding import GDino
from visual.runtime.tracking import SAM2Tracker
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .common import read,write,sha,npz,checkpoint_load,append_csv,tensor_identity,now
from .geometry import CurrentGeometry
from .model import CachedGraphPolicy,normalizer


def environment(root):
    from mani_skill.utils.registration import TimeLimitWrapper
    env=TimeLimitWrapper(make_peg_insertion_env(device='cpu',obs_mode='rgb+depth',
        control_mode='pd_ee_delta_pose').unwrapped,250)
    task=env.unwrapped;cfg=read(Path(root)/'inputs/task08_reference_conditions.json')
    controllers={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()}
    if controller_semantics(controllers)!=controller_semantics(cfg['controller_config']):
        env.close();raise ValueError('installed controller differs from original action contract')
    if (task.control_freq,task.sim_freq,task._sim_steps_per_control)!=(20,100,5):
        env.close();raise ValueError('frequency mismatch')
    return env


def sensors(env,obs,identity,t,schema):
    fields,pads=robot_observation(env)
    if state_schema(fields)!=schema:raise ValueError('live Robot76 schema changed')
    for name in ('qpos','qvel'):
        np.testing.assert_array_equal(numpy(obs['agent'][name])[0],fields[name])
    cameras=[]
    for name in ('base_camera','hand_camera'):
        c=obs['sensor_data'][name];p=obs['sensor_param'][name]
        cameras.append(CameraObservation(identity,t,name,t/20,numpy(c['rgb'])[0],
            numpy(c['depth'])[0,...,0],numpy(p['intrinsic_cv'])[0],numpy(p['extrinsic_cv'])[0],
            numpy(p['cam2world_gl'])[0],fields['tcp_pose'].copy()))
    surfaces=robot_surfaces(env)
    data=dict(episode=np.array(identity),timestep=np.int64(t),timestamp=np.float64(t/20),
        robot76=flatten_state(fields,schema),robot_world_geometry=pads,tcp_pose=fields['tcp_pose'],
        qpos=fields['qpos'],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    return cameras,data,surfaces


def evaluation_measure(env,initial_z,cfg):
    """Evaluation-only GT. No returned value is passed to policy or perception."""
    state=get_gt_state(env);task=env.unwrapped;info=task.evaluate()
    peg=numpy(state.peg_pose)[0];box=numpy(state.box_pose)[0];hole=numpy(state.hole_pose)[0]
    tcp=numpy(state.ee_pose)[0];distance=float(np.linalg.norm(tcp[:3]-peg[:3]))
    grasp=bool(task.agent.is_grasping(task.peg).item());height=float(peg[2]-initial_z)
    head=numpy(info['peg_head_pos_at_hole'])[0]
    radius=float(numpy(state.hole_radius).reshape(-1)[0])
    dot=float(np.clip(pose_matrix(peg)[:3,0]@pose_matrix(hole)[:3,0],-1,1))
    angle=float(np.degrees(np.arccos(dot)))
    return dict(reach=distance<=cfg['reach_distance_m'],grasp=grasp,
        lift_candidate=grasp and height>cfg['lift_diagnostic_height_m'],lift_height_m=height,
        alignment=bool(abs(head[1])<=radius and abs(head[2])<=radius and angle<=cfg['alignment_angle_deg']),
        success=bool(info['success'].item()),tcp_peg_distance_m=distance,alignment_angle_deg=angle,
        peg_head_pos_at_hole=head.tolist(),peg_pose=peg.tolist(),box_pose=box.tolist(),hole_pose=hole.tolist(),
        tcp_pose=tcp.tolist(),gripper_qpos=numpy(state.gripper_qpos)[0].tolist())


def video_frame(cameras,masks,variant,seed,step,t,metrics):
    views=[]
    for c in cameras:
        rgb=c.rgb.copy()
        for oid,color in [(0,np.array([255,70,70])),(1,np.array([70,255,90]))]:
            selected=masks[c.camera][oid];rgb[selected]=(rgb[selected]*.6+color*.4).astype(np.uint8)
        view=cv2.resize(rgb,(384,384),interpolation=cv2.INTER_NEAREST)
        cv2.putText(view,c.camera,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);views.append(view)
    frame=np.zeros((448,768,3),np.uint8);frame[:384]=np.concatenate(views,axis=1)
    for i,text in enumerate([f'Plan A C3 A-SF | {variant} seed={seed} checkpoint={step} t={t}',
        f'Post-action t+1 | reach={int(metrics["reach"])} grasp={int(metrics["grasp"])} lift_mm={metrics["lift_height_m"]*1000:.1f} insertion={int(metrics["success"])}']):
        cv2.putText(frame,text,(8,409+i*25),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
    return cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


def run_episode(root,folder,model,norm,codec,detector,tracker,spec,step, *, smoke=False):
    folder.mkdir(parents=True,exist_ok=False);cfg=read(root/'config.json')['evaluation']
    env=environment(root);obs,_=env.reset(seed=spec['seed'])
    actual_reset=reset_diagnostics(env)
    if spec.get('reset_group') and actual_reset!=spec['reset_group']:
        env.close();raise ValueError('evaluation seed reset identity changed')
    geometry=CurrentGeometry(root/'inputs',norm.schema);adapters={c:OnlineVisionAdapter(detector,tracker) for c in ('base_camera','hand_camera')}
    identity=f'{model.variant}:step{step}:seed{spec["seed"]}'
    for adapter in adapters.values():adapter.reset(identity)
    hidden=None;physics=PhysicsAudit(env);audit=controller_audit(env)
    gt=get_gt_state(env);initial_z=float(numpy(gt.peg_pose)[0,2]);trace=[];hidden_states=[];robot=[];action=[]
    first={k:None for k in ['reach','grasp','lift','alignment','insertion']};streak=0;max_streak=0
    writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(768,448))
    if not writer.isOpened():raise RuntimeError('video encoder unavailable')
    started=time.perf_counter();termination='smoke_limit' if smoke else 'horizon';complete=False
    try:
        for t in range(cfg['smoke_horizon'] if smoke else cfg['horizon']):
            token=physics.token();tick=time.perf_counter()
            cameras,data,surfaces=sensors(env,obs,identity,t,norm.schema)
            masks={c.camera:adapters[c.camera].process(c.rgb,identity,t)[0] for c in cameras}
            graph,_,gate=geometry.build(cameras,data,surfaces,masks)
            packed=pack_graphs([graph],[(0,0)]).to(next(model.parameters()).device)
            state=torch.from_numpy(data['robot76'][None]).to(next(model.parameters()).device)
            with torch.inference_mode():
                raw,hidden=model.step(packed,state,norm,hidden)
                command=codec.decode(codec.tag(raw))
            if not torch.isfinite(command).all():raise FloatingPointError('nonfinite native Action7')
            physics.unchanged(token);duration=time.perf_counter()-tick
            hidden_states.append(numpy(hidden));robot.append(data['robot76'].copy());action.append(numpy(command)[0])
            tensor=command.detach().to(env.unwrapped.device);audit.clear()
            obs,_,terminated,truncated,info=env.step(tensor);physics.stepped(token)
            validate_controller_command(action[-1],audit)
            measured=evaluation_measure(env,initial_z,cfg)
            if measured['success']!=bool(info['success'].item()):raise ValueError('official predicate disagreement')
            streak=streak+1 if measured['lift_candidate'] else 0;max_streak=max(streak,max_streak)
            statuses=dict(reach=measured['reach'],grasp=measured['grasp'],lift=streak>=cfg['lift_consecutive_steps'],
                alignment=measured['alignment'],insertion=measured['success'])
            for k,v in statuses.items():
                if v and first[k] is None:first[k]=t+1
            trace.append(dict(timestep=t,action=action[-1].tolist(),encoded_action=numpy(raw)[0].tolist(),
                inference_seconds=duration,nodes=len(graph.node_type),edges=graph.edge_index.shape[1],
                missing_objects=[not bool((graph.node_type==k).any()) for k in (3,4)],gate=gate,
                terminated=bool(terminated.item()),truncated=bool(truncated.item()),**measured))
            writer.write(video_frame(cameras,masks,model.variant,spec['seed'],step,t,measured))
            if t==0 or t%25==0 or any(first[k]==t+1 for k in first):
                npz(folder/f'live_graph_t{t:04d}.npz',{k:getattr(graph,k).numpy() for k in ['node_input','node_type','edge_index','edge_attr','edge_type']})
            if t%25==0:print(f'rollout {model.variant} step={step} seed={spec["seed"]} t={t} progress={statuses}',flush=True)
            if measured['success'] or terminated.item() or truncated.item():
                termination='success' if measured['success'] else 'termination' if terminated.item() else 'time_limit';break
        complete=True
    finally:
        writer.release();physics.close()
        for adapter in adapters.values():adapter.close()
        env.close()
        if robot:npz(folder/'trajectory.npz',dict(robot76=np.asarray(robot),action7=np.asarray(action),hidden=np.asarray(hidden_states),
            peg_pose=np.asarray([r['peg_pose'] for r in trace]),box_pose=np.asarray([r['box_pose'] for r in trace]),
            hole_pose=np.asarray([r['hole_pose'] for r in trace]),gripper_qpos=np.asarray([r['gripper_qpos'] for r in trace]),
            robot_timestep=np.arange(len(robot)),object_post_action_timestep=np.arange(len(trace))+1))
        write(folder/'trace.json',trace,immutable=True)
        row=dict(status='SMOKE_COMPLETE' if smoke and complete else 'COMPLETE' if complete else 'ERROR',
            variant=model.variant,step=step,seed=spec['seed'],group=spec['group'],episode_id=spec.get('episode_id',spec['seed']),
            reset_group=actual_reset,episode_length=len(trace),termination_reason=termination,
            reach=first['reach'] is not None,grasp=first['grasp'] is not None,lift=first['lift'] is not None,
            alignment=first['alignment'] is not None,insertion=first['insertion'] is not None,
            first_reach=first['reach'],first_grasp=first['grasp'],first_lift=first['lift'],first_alignment=first['alignment'],first_insertion=first['insertion'],
            first_failed_stage=next((k for k in first if first[k] is None),None),
            maximum_progress_stage=max([i for i,k in enumerate(first) if first[k] is not None],default=-1),
            max_lift_m=max([r['lift_height_m'] for r in trace],default=0.),max_lift_consecutive_steps=max_streak,
            denominator_eligible=complete and not smoke,wall_seconds=time.perf_counter()-started,
            physics_calls=physics.count,expert_cache_used=False,current_RGBD=True,
            video_path=str(folder/'rollout.mp4'),video_sha256=sha(folder/'rollout.mp4'))
        write(folder/'episode.json',row,immutable=True)
    return row


def preflight(root,device):
    root=Path(root).resolve();cfg=read(root/'config.json');frozen=read(root/'seeds_manifest.json')
    env=environment(root);records=[];original={e['seed']:e for e in read(root/'dataset_audit.json')['episodes']}
    try:
        from graph.integration_baseline.rollout_stop_clock import time_contract
        clock=time_contract(env,dict(time_contract_seed=44000))
        # No Test12 reset/evaluation before model selection.
        for seed in [30003]+cfg['evaluation']['dev_seeds']+[44001]:
            obs,_=env.reset(seed=seed);first=reset_diagnostics(env)
            obs,_=env.reset(seed=seed);second=reset_diagnostics(env)
            if first!=second:raise ValueError('same seed reset is not reproducible')
            if seed==30003 and first!=original[seed]['reset_group']:raise ValueError('original Train seed reset mismatch')
            if seed!=30003 and first in {e['reset_group'] for e in original.values()}:raise ValueError('dev reset group leakage')
            records.append(dict(seed=seed,reset_group=first,repeat_equal=True))
        # Verify actual installed robot collisions/URDF before offline FK is used.
        schema=read(root/'normalizer.json')['schema'];geometry=CurrentGeometry(root/'inputs',schema)
        cameras,data,surfaces=sensors(env,obs,'runtime',0,schema)
        calculated=geometry.fk.surfaces(data['robot76'])
        if [s.link for s in calculated]!=[s.link for s in surfaces]:raise ValueError('installed robot shapes changed')
        for expected,actual in zip(calculated,surfaces):
            np.testing.assert_allclose(expected.local_vertices,actual.local_vertices,atol=1e-8,rtol=0)
            np.testing.assert_allclose(expected.world_from_link,actual.world_from_link,atol=3e-6,rtol=0)
        conditions=dict(backend=str(env.unwrapped.backend),versions={n:importlib.metadata.version(n) for n in ['torch','numpy','mani_skill','sapien','gymnasium','transformers']},
            cuda=torch.version.cuda,device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
            executed='CPU PhysX/SAPIEN renderer; CUDA perception and policy separately gated',
            CUDA_PhysX='NOT_TESTED',MPS='NOT_TESTED')
        write(root/'runtime_verification.json',dict(status='PASS',conditions=conditions,time_contract=clock,
            seeds=records,original_train_reset_match=True,installed_robot_shape_FK_match=True,
            official_success_source=__import__('inspect').getsource(type(env.unwrapped).evaluate)),immutable=True)
    finally:env.close()


def evaluate(root,run,checkpoint,device, *, final_kind=None,pretraining_smoke=False):
    root=Path(root).resolve();run=Path(run).resolve();cp=checkpoint_load(checkpoint);cfg=read(root/'config.json')
    if cp['config_sha256']!=sha(root/'config.json'):raise ValueError('wrong experiment checkpoint')
    if cp['normalizer']!=read(root/'normalizer.json') or cp['codec']!=read(root/'action_codec.json'):
        raise ValueError('checkpoint/root normalization identity changed')
    norm=normalizer(root);codec=ActionCodec(cp['codec']);model=CachedGraphPolicy(norm.schema,cp['variant'],**cfg['model']).to(device)
    model.load_state_dict(cp['model']);model.eval()
    if tensor_identity(model.state_dict())!=tensor_identity(cp['model']):raise ValueError('wrong evaluated weights')
    runtime=read(root/'runtime_paths.json');detector=GDino(device=device,model_id=runtime['dino']);tracker=SAM2Tracker(device=device,checkpoint=runtime['sam'])
    step=cp['progress']['step'];smoke=step==500 or pretraining_smoke
    reset={e['seed']:e['reset_group'] for e in read(root/'runtime_verification.json')['seeds']}
    if final_kind:
        if not (root/'model_selection_frozen.json').exists():raise ValueError('Test gate locked')
        specs=[dict(**s,group='test12') for s in read(root/'seeds_manifest.json')['test']]
    elif smoke:specs=[dict(seed=44001,group='smoke',reset_group=reset[44001])]
    else:
        dev=cfg['evaluation']['expanded_dev_seeds'] if step in cfg['evaluation']['expanded_steps'] else cfg['evaluation']['normal_dev_seeds']
        specs=[dict(seed=30003,group='train_seed',reset_group=reset[30003])]+[dict(seed=s,group='dev',reset_group=reset[s]) for s in dev]
    tag=final_kind or ('pretraining_smoke' if pretraining_smoke else f'step{step:06d}')
    folder=run/'rollouts'/tag
    if folder.exists():raise FileExistsError('evaluation is immutable; explicit recovery required')
    with ExperimentLogger(team='integration',model=f'plan_a_{cp["variant"]}_rollout',seed=42,
        config=dict(comparison_uuid=root.name,checkpoint_sha256=sha(checkpoint),checkpoint_step=step,
            seeds=specs,smoke=smoke,final_kind=final_kind),root=root/'evaluation_runs',job_type='evaluate') as log:
        rows=[]
        for spec in specs:
            row=run_episode(root,folder/f'seed{spec["seed"]}',model,norm,codec,detector,tracker,spec,step,smoke=smoke)
            row['checkpoint_sha256']=sha(checkpoint);row['final_kind']=final_kind or ''
            rows.append(row);append_csv(run/'rollout_metrics.csv',row);log.link_output(folder/f'seed{spec["seed"]}'/'episode.json',kind='metrics')
        for group in sorted(set(r['group'] for r in rows)):
            eligible=[r for r in rows if r['group']==group and r['denominator_eligible']]
            log.log(dict(episodes=len(eligible),**{k:sum(r[k] for r in eligible)/len(eligible) if eligible else 0. for k in ['reach','grasp','lift','alignment','insertion']}),step=step,prefix=group)
        write(folder/'summary.json',dict(checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),step=step,
            planned=len(specs),executed=len(rows),denominator=sum(r['denominator_eligible'] for r in rows),
            smoke_excluded_from_performance=smoke,episodes=rows),immutable=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path)
    p.add_argument('--checkpoint');p.add_argument('--device',default='cuda');p.add_argument('--preflight',action='store_true')
    p.add_argument('--final-kind',choices=['last','validation_best']);p.add_argument('--pretraining-smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    # Existing GNN index_add kernels need deterministic cuBLAS setup; the launcher sets it.
    torch.use_deterministic_algorithms(True)
    if a.preflight:preflight(a.root,a.device)
    else:evaluate(a.root,a.run,a.checkpoint,a.device,final_kind=a.final_kind,pretraining_smoke=a.pretraining_smoke)


if __name__=='__main__':main()
