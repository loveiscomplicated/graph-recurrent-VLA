"""Current RGB-D causal closed loop; GT exclusively downstream for scoring."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import random
import time
import cv2
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,digest,npz,tensor_identity
from graph.plan_a_bc.cache import make_vision,GRAPH_KEYS
from graph.plan_a_bc.rollout import sensors,evaluation_measure,video_frame
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d.optimizations import DetectionMemo
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_stop_clock import PhysicsAudit,controller_audit,validate_controller_command,numpy,reset_diagnostics
from graph.integration_baseline.direct_train import seed_all
from graph.integration_baseline.geometry09a import collision_audit
from graph.corrective_reach10f.pilot import snapshot
from graph.geometry_reach10d.evaluate import measure as strict_measure
from graph.gt_pretrain10f1.remediation_collect import portable
from recurrent.expert import ScriptedExpert
from .train_worker import setup


def nested_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,dict):
            for k in sorted(v):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            h.update(str(type(v)).encode())
            for x in v:visit(x)
        elif torch.is_tensor(v):visit(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes())
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def load_model(root,condition,checkpoint):
    from .model import Policy
    from graph.plan_a_bc.model import normalizer
    from graph.integration_baseline.bc08_codec import ActionCodec
    cfg=read(root/'config.json');seed_all(cfg['seed'],1)
    f1=torch.load(cfg['f1_checkpoint'],map_location='cpu',weights_only=False)
    m=Policy(read(root/'normalizer.json')['schema'],f1,condition,cfg['seed']).cuda()
    norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['config_sha256']!=sha(root/'config.json') or ck['condition']!=condition:raise ValueError('evaluation checkpoint identity')
    m.load_state_dict(ck['model']);m.eval().requires_grad_(False)
    return cfg,m,norm,codec,ck


def vision(root):
    paths=read(root/'runtime_paths.json');d,t=make_vision('cuda',paths['dino'],Path(paths['sam']))
    return DetectionMemo(d,'task10g',dict(dino=paths['dino'],sam_sha256=sha(paths['sam']))),t


@torch.inference_mode()
def episode(root,folder,model,norm,codec,detector,tracker,seed,step,horizon,smoke=False):
    if (folder/'episode.json').exists():
        row=read(folder/'episode.json')
        if row['status']!='COMPLETE' or sha(folder/'rollout.mp4')!=row['video_sha256']:raise ValueError('invalid completed episode')
        if sha(folder/'trace.json')!=row['trace_sha256']:raise ValueError('trace corrupt')
        return row
    if folder.exists() and any(folder.iterdir()):
        # Preserve a failed attempt instead of appending incompatible traces.
        folder.rename(folder.with_name(folder.name+'_failed_'+str(time.time_ns())))
    folder.mkdir(parents=True,exist_ok=True)
    seed_all(42,1);cfg=read(root/'config.json')['evaluation'];env=environment(root)
    obs,_=env.reset(seed=seed);expert=ScriptedExpert(env)
    state=snapshot(env,expert);portable_state=portable(state)
    selected={k:portable_state[k] for k in ('sim','clock','controllers','rng')}
    snap_hash=nested_hash(selected);torch.save(selected,folder/'initial_snapshot.pt')
    if nested_hash(torch.load(folder/'initial_snapshot.pt',weights_only=False))!=snap_hash:raise ValueError('snapshot readback')
    identity=f'task10g:seed{seed}' # common geometry FPS namespace, independent of model/step
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ('base_camera','hand_camera')}
    for a in adapters.values():a.reset(identity)
    geometry=CurrentGeometry(root/'inputs',norm.schema);physics=PhysicsAudit(env);audit=controller_audit(env)
    initial_z=float(numpy(env.unwrapped.peg.pose.raw_pose)[0,2]);initial_expert_action=expert.act().copy()
    initial_peg=numpy(env.unwrapped.peg.pose.raw_pose)[0,:3].copy()
    strict=cfg['strict_reach_thresholds'];approach_step=None;strict_success=False;early_close=False
    encoder_start=torch.cuda.Event(enable_timing=True);encoder_end=torch.cuda.Event(enable_timing=True)
    def before_encoder(*args):encoder_start.record()
    def after_encoder(*args):encoder_end.record()
    encoder_pre=model.graph_encoder.register_forward_pre_hook(before_encoder)
    encoder_post=model.graph_encoder.register_forward_hook(after_encoder)
    hidden=None;trace=[];robots=[];commands=[];first={k:None for k in ('reach','grasp','lift','alignment','insertion')}
    streak=0;started=time.perf_counter();completed=False;reason='horizon'
    writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(768,448))
    if not writer.isOpened():raise RuntimeError('video encoder')
    try:
        for t in range(horizon):
            token=physics.token();tick=time.perf_counter()
            cameras,data,surfaces=sensors(env,obs,identity,t,norm.schema)
            masks={c.camera:adapters[c.camera].process(c.rgb,identity,t)[0] for c in cameras}
            graph,_,gate=geometry.build(cameras,data,surfaces,masks)
            packed=pack_graphs([graph],[(0,0)]).to('cuda');robot=torch.from_numpy(data['robot76'][None]).cuda()
            policy_tick=time.perf_counter();raw,hidden=model.step(packed,robot,norm,hidden)
            command=codec.decode(codec.tag(raw));torch.cuda.synchronize()
            policy_seconds=time.perf_counter()-policy_tick;duration=time.perf_counter()-tick
            encoder_seconds=encoder_start.elapsed_time(encoder_end)/1000
            pre_strict=strict_measure(env,expert,strict)
            physics.unchanged(token);a=numpy(command)[0]
            if a[6]<=0 and not (approach_step is not None and pre_strict['final']):early_close=True
            robots.append(data['robot76'].copy());commands.append(a)
            if t==0:
                inputs={k:getattr(graph,k).numpy() for k in GRAPH_KEYS};inputs['robot76']=data['robot76']
                npz(folder/'initial_student_input.npz',inputs);input_hash=nested_hash(inputs)
            audit.clear();obs,_,term,trunc,info=env.step(command.to(env.unwrapped.device));physics.stepped(token)
            validate_controller_command(a,audit)
            measured=evaluation_measure(env,initial_z,cfg)
            if measured['success']!=bool(info['success'].item()):raise ValueError('official success disagreement')
            streak=streak+1 if measured['lift_candidate'] else 0
            status=dict(reach=measured['reach'],grasp=measured['grasp'],lift=streak>=cfg['lift_consecutive_steps'],
                alignment=measured['alignment'],insertion=measured['success'])
            for k,v in status.items():
                if v and first[k] is None:first[k]=t+1
            contacts=collision_audit(env,'peg')
            post_strict=strict_measure(env,expert,strict)
            if post_strict['approach'] and approach_step is None:approach_step=t+1
            peg_drift=float(np.linalg.norm(np.asarray(measured['peg_pose'][:3])-initial_peg))
            strict_candidate=bool(approach_step is not None and post_strict['final'] and
                post_strict['opening_m']>=strict['opening_min_m'] and not early_close and
                peg_drift<strict['max_peg_motion_m'] and contacts['valid'])
            strict_success=strict_success or strict_candidate
            trace.append(dict(timestep=t,action=a.tolist(),inference_seconds=duration,policy_seconds=policy_seconds,gnn_encoder_readout_seconds=encoder_seconds,
                strict_reach=strict_candidate,strict_approach_step=approach_step,strict_premature_close=early_close,
                strict_target_position_error_m=post_strict['target_position_error_m'],
                strict_target_orientation_error_rad=post_strict['target_orientation_error_rad'],
                missing_objects=[not bool((graph.node_type==k).any()) for k in (3,4)],
                saturation=bool((abs(a)>.999).any()),premature_close=bool(a[6]<0 and measured['tcp_peg_distance_m']>.03 and not measured['grasp']),
                collision=contacts,hidden_norm=float(hidden.norm()),gate=gate,terminated=bool(term.item()),truncated=bool(trunc.item()),**measured))
            writer.write(video_frame(cameras,masks,model.condition,seed,step,t,measured))
            if t%25==0:print('LIVE',model.condition,step,seed,t,status,flush=True)
            if measured['success'] or term.item() or trunc.item():
                reason='success' if measured['success'] else 'termination' if term.item() else 'time_limit';break
        completed=True
    finally:
        writer.release();physics.close();encoder_pre.remove();encoder_post.remove()
        for a in adapters.values():a.close()
        detector.cache.clear();env.close();torch.cuda.empty_cache()
        write(folder/'trace.json',trace)
        if robots:npz(folder/'trajectory.npz',dict(robot76=np.asarray(robots),action7=np.asarray(commands)))
    if not completed or not trace:raise RuntimeError('incomplete rollout')
    if not smoke and len(trace)!=horizon and reason=='horizon':raise ValueError('short evaluation')
    row=dict(status='COMPLETE',condition=model.condition,step=step,seed=seed,actions=len(trace),
        termination=reason,engineering=smoke,**{k:first[k] is not None for k in first},
        final_tcp_peg_distance_m=trace[-1]['tcp_peg_distance_m'],
        final_tip_hole_error_m=float(np.linalg.norm(trace[-1]['peg_head_pos_at_hole'])),
        strict_reach=strict_success,strict_approach_step=approach_step,strict_premature_close=early_close,
        max_lift_height_m=max(t['lift_height_m'] for t in trace),initial_snapshot_sha256=snap_hash,
        initial_input_sha256=input_hash,initial_action_mae=float(np.abs(commands[0]-initial_expert_action).mean()),
        premature_close=sum(t['premature_close'] for t in trace),saturation=sum(t['saturation'] for t in trace),
        missing_frames=sum(any(t['missing_objects']) for t in trace),
        inference_seconds=float(np.mean([t['inference_seconds'] for t in trace])),
        gnn_encoder_readout_seconds=float(np.mean([t['gnn_encoder_readout_seconds'] for t in trace])),
        policy_seconds=float(np.mean([t['policy_seconds'] for t in trace])),
        wall_seconds=time.perf_counter()-started,physics_calls=physics.count,
        video_sha256=sha(folder/'rollout.mp4'),trace_sha256=sha(folder/'trace.json'),folder=str(folder.relative_to(root)),
        GT_student_input=False,state_head_inference=False,reset_diagnostics=reset_diagnostics if False else 'snapshot recorded')
    write(folder/'episode.json',row)
    return row


def csv_write(path,rows):
    from graph.gt_pretrain10f1.evaluate import csv_write as write_csv
    tmp=path.with_suffix('.partial');write_csv(tmp,rows);os.replace(tmp,path)


def run(root,mode):
    cfg=read(root/'config.json');detector,tracker=vision(root);rows=[]
    if mode=='final' and not (root/'model_selection_frozen.json').exists():raise ValueError('Final locked')
    with ExperimentLogger(team='graph',model='Task10G_closed_loop_'+mode,seed=42,config=cfg,
        root=root/'experiments',job_type='evaluate') as log:
        for condition in 'ABC':
            if mode=='runtime_smoke':choices=[root/'smoke'/condition/'step1.pt']
            elif mode=='development':choices=[root/('train_'+condition)/f'step{s:06d}.pt' for s in cfg['training']['checkpoint_steps']]
            elif mode=='final':choices=[root/read(root/'model_selection_frozen.json')['conditions'][condition]['path']]
            else:raise ValueError(mode)
            for cp in choices:
                _,model,norm,codec,ck=load_model(root,condition,cp);before=tensor_identity(model.state_dict())
                seeds=[cfg['evaluation']['smoke_seed']] if mode=='runtime_smoke' else cfg['evaluation']['dev_seeds' if mode=='development' else 'final_seeds']
                for seed in seeds:
                    folder=root/'rollouts'/mode/condition/f'step{ck["progress"]["step"]:06d}'/f'seed{seed}'
                    row=episode(root,folder,model,norm,codec,detector,tracker,seed,ck['progress']['step'],
                        cfg['evaluation']['smoke_horizon'] if mode=='runtime_smoke' else cfg['evaluation']['horizon'],
                        smoke=mode=='runtime_smoke' or cfg['miniature'])
                    row['checkpoint_sha256']=sha(cp);rows.append(row);log.link_output(folder/'episode.json',kind='metrics')
                if tensor_identity(model.state_dict())!=before:raise ValueError('evaluation changed weights')
                del model;torch.cuda.empty_cache()
        # Paired initial simulator/controller/RNG and current graph checks.
        for seed in set(r['seed'] for r in rows):
            selected=[r for r in rows if r['seed']==seed]
            if len({r['initial_snapshot_sha256'] for r in selected})!=1:raise ValueError('paired snapshot mismatch')
            if len({r['initial_input_sha256'] for r in selected})!=1:raise ValueError('paired RGBD/graph mismatch')
        path=root/(mode+'_rollouts.csv');csv_write(path,rows)
        write(root/(mode+'_verification.json'),dict(status='PASS',episodes=len(rows),paired_snapshots_exact=True,
            paired_initial_graphs_exact=True,smoke=mode=='runtime_smoke' or cfg['miniature']))
        log.link_output(path,kind='metrics')
    return dict(status='PASS',episodes=len(rows),files=[path.name,mode+'_verification.json']+
        [str((root/r['folder']/'episode.json').relative_to(root)) for r in rows])


def select(root):
    import csv,shutil
    cfg=read(root/'config.json');rows=list(csv.DictReader((root/'development_rollouts.csv').open()));chosen={}
    for c in 'ABC':
        candidates=[]
        for step in cfg['training']['checkpoint_steps']:
            rr=[r for r in rows if r['condition']==c and int(r['step'])==step]
            if len(rr)!=len(cfg['evaluation']['dev_seeds']):raise ValueError('development incomplete')
            success=[sum(r[k]=='True' for r in rr) for k in ('insertion','lift','grasp','alignment')]
            error=float(np.mean([float(r['final_tcp_peg_distance_m']) for r in rr]))
            val=read(root/('train_'+c)/'validation'/f'{step:06d}.json')['BC_loss']
            candidates.append((tuple([-s for s in success]+[error,val,step]),step))
        _,step=min(candidates);p=root/('train_'+c)/f'step{step:06d}.pt'
        chosen[c]=dict(step=step,path=str(p.relative_to(root)),sha256=sha(p),candidates=[dict(step=s,rank=list(rank)) for rank,s in candidates])
        shutil.copy2(p,root/('train_'+c)/'best.pt')
    write(root/'model_selection_frozen.json',dict(status='PASS',conditions=chosen,rule=cfg['evaluation']['selection'],
        development_sha256=sha(root/'development_rollouts.csv'),final_accessed=False),immutable=True)
    return dict(status='PASS',files=['model_selection_frozen.json'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',required=True)
    a=p.parse_args();run(a.root,a.mode)
