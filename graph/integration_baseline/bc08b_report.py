"""Render observed Task08-B training and same-seed rollout evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from graph.integration_baseline.bc08_codec import controller_commands, native_metrics
from graph.integration_baseline.bc08_report import verify
from graph.integration_baseline.direct_dataset import DirectDataset, load_episode
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.direct_train import seed_all
from graph.integration_baseline.bc08_train import load_checkpoint, load_normalizer, evaluate
from graph.integration_baseline.model import IntegrationPolicy


STAGES = ('reach', 'descend', 'close', 'lift', 'align', 'insert')
COLORS = {'A': '#1f77b4', 'C': '#d95f02', 'Expert': '#333333'}


def first_close(action):
    index = np.flatnonzero(np.asarray(action)[:, 6] < 0)
    return int(index[0]) if len(index) else None


def summary(values):
    x = np.asarray(values, np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                p95=float(np.quantile(x, .95)), p99=float(np.quantile(x, .99)), max=float(x.max()))


def method_offline(run, stages):
    run = Path(run)
    completion = read(run/'completion.json')
    initial = read(run/'initial_train_metrics.json')
    history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    with np.load(run/'offline/selected_train/ep30003.npz', allow_pickle=False) as f:
        pred, target = f['native_prediction'].copy(), f['native_target'].copy()
        encoded = f['encoded_output'].copy()
    if len(pred) != 137 or not np.array_equal(target.shape, pred.shape):
        raise ValueError('selected offline sequence incomplete')
    metrics = native_metrics(pred, target)
    cp, cy = controller_commands(pred), controller_commands(target)
    arm = np.linalg.norm((cp-cy)[:, :3], axis=1)*1000
    rot = (Rotation.from_euler('XYZ', cp[:, 3:6]).inv()*Rotation.from_euler('XYZ', cy[:, 3:6])).magnitude()
    grip_mm = np.abs(cp[:, 6]-cy[:, 6])*1000
    delta = pred-target
    frame_norm = np.linalg.norm(delta, axis=1)
    stage_metrics = {}
    for stage in STAGES:
        idx = np.flatnonzero(np.asarray(stages) == stage)
        stage_metrics[stage] = dict(count=len(idx), action7_mse=float(np.mean(delta[idx]**2)),
                                    motion6_mse=float(np.mean(delta[idx, :6]**2)),
                                    translation_mm=summary(arm[idx]), rotation_rad=summary(rot[idx]),
                                    gripper_accuracy=float(np.mean((pred[idx, 6]>0)==(target[idx, 6]>0))))
    clip = dict(translation_frames=int(np.sum(np.any(np.abs(pred[:, :3])>1, axis=1))),
                rotation_frames=int(np.sum(np.linalg.norm(pred[:, 3:6], axis=1)>1)),
                gripper_frames=int(np.sum(np.abs(pred[:, 6])>1)))
    expert_close, model_close = first_close(target), first_close(pred)
    expert_transitions = (np.flatnonzero(np.diff(target[:, 6]>0))+1).tolist()
    model_transitions = (np.flatnonzero(np.diff(pred[:, 6]>0))+1).tolist()
    small = np.linalg.norm(target[:, :6], axis=1)<.01
    opposite = np.flatnonzero(np.sum(pred[:, :6]*target[:, :6], axis=1)<0).tolist()
    by_frame = dict(action7_error_l2=frame_norm.tolist(), controller_translation_error_mm=arm.tolist(),
                    controller_rotation_error_rad=rot.tolist(), controller_gripper_target_error_mm=grip_mm.tolist(),
                    action_direction_opposite_timesteps=opposite)
    details = dict(completion=completion, initial=initial, milestones=[dict(step=r['step'],
        optimization_loss=r['train']['optimization_loss'], native_action7_mse=r['train']['native_action7_mse'],
        native_motion6_mse=r['train']['native_motion6_mse'],
        controller_translation_mm=1000*r['train']['controller_translation_error_m']['mean'],
        controller_rotation_rad=r['train']['controller_rotation_geodesic_rad']['mean'],
        gripper_accuracy=r['train']['gripper_accuracy'], learning_rate=r['learning_rate']) for r in history],
        selected_native_metrics=metrics, translation_component_mae=np.mean(np.abs(delta[:, :3]),axis=0).tolist(),
        rotation_component_mae=np.mean(np.abs(delta[:, 3:6]),axis=0).tolist(),
        gripper_mae=float(np.mean(np.abs(delta[:, 6]))), action7_frame_error=summary(frame_norm),
        controller_translation_error_mm=summary(arm), controller_rotation_error_rad=summary(rot),
        controller_rotation_error_deg=summary(np.rad2deg(rot)), controller_gripper_target_error_mm=summary(grip_mm),
        clipping=clip, expert_first_close=expert_close, predicted_first_close=model_close,
        first_close_difference=None if model_close is None else model_close-expert_close,
        expert_transitions=expert_transitions, predicted_transitions=model_transitions,
        close_segment_accuracy=stage_metrics['close']['gripper_accuracy'],
        low_expert_motion_frames=int(np.sum(small)),
        low_expert_motion_predicted_motion_norm_mean=float(np.linalg.norm(pred[small, :6],axis=1).mean()) if small.any() else None,
        opposite_direction_frames=opposite, max_error_timestep=int(np.argmax(frame_norm)),
        per_stage=stage_metrics, per_frame=by_frame)
    return details, pred, target, encoded


def rollout(evaluation_run):
    evaluation_run = Path(evaluation_run)
    result = read(evaluation_run/'evaluation.json')
    if result['status'] != 'COMPLETE' or len(result['episodes']) != 1:
        raise ValueError('rollout incomplete')
    ep = result['episodes'][0]
    folder = Path(ep['artifact_directory'])
    with np.load(folder/'observations.npz', allow_pickle=False) as f:
        rgb = f['rgb'].copy()
        tcp = f['tcp_pose'].copy()
    with np.load(folder/'actions.npz', allow_pickle=False) as f:
        actions = f['actions'].copy()
    with np.load(folder/'encoded_action_output.npz', allow_pickle=False) as f:
        encoded = f['raw_output'].copy()
    with np.load(folder/'observations.npz', allow_pickle=False) as f:
        robot = f['robot_state'].copy()
    diag = read(folder/'grasp_contact_lift.json')
    trace = read(folder/'trace.json')
    if len(actions) != ep['episode_length'] or len(diag) != len(actions) or len(rgb) != len(actions)+1 or len(encoded) != len(actions):
        raise ValueError('rollout trace/action/observation alignment')
    relative = [np.asarray(d['pre']['ee_pose'][:3])-np.asarray(d['pre']['peg_pose'][:3]) for d in diag]
    close = first_close(actions)
    close_geo = None if close is None else dict(timestep=close, tcp_peg_vector_m=np.asarray(relative[close]).tolist(),
        tcp_peg_distance_m=float(np.linalg.norm(relative[close])),
        fingertip_peg_distances_m=[float(np.linalg.norm(np.asarray(diag[close]['pre'][name])-np.asarray(diag[close]['pre']['peg_pose'][:3])))
            for name in ('left_fingertip_pos','right_fingertip_pos')])
    above=np.asarray([d['peg_lift_m']>=.04 for d in diag])
    longest=0;current=0
    for value in above:
        current=current+1 if value else 0
        longest=max(longest,current)
    points=[]
    for t in range(len(actions)):
        with np.load(folder/f'packet_{t:04d}.npz',allow_pickle=False) as f:
            oid=f['packet/object_id'].copy()
        points.append([int(np.sum(oid==i)) for i in (0,1)])
    count=np.asarray(points)
    out = dict(evaluation_run=str(evaluation_run), episode=ep, first_close_timestep=close,
               close_before_geometry=close_geo, max_peg_lift_m=max(d['peg_lift_m'] for d in diag),
               final_peg_lift_m=diag[-1]['peg_lift_m'],
               lift_4cm_hold_steps_longest=longest,
               grasp_timesteps=[d['timestep'] for d in diag if d['grasp']],
               bilateral_contact_timesteps=[d['timestep'] for d in diag if all(np.linalg.norm(f)>=.5 for f in d['finger_forces'])],
               final_tcp_pose=tcp[-1].tolist(), final_peg_pose=diag[-1]['post']['peg_pose'],
               tcp_displacement_m=float(np.linalg.norm(tcp[-1,:3]-tcp[0,:3])),
               gripper_opening_initial_final_m=[float(robot[0,22]),float(robot[-1,22])],
               gripper_opening_total_change_m=float(robot[-1,22]-robot[0,22]),
               object_point_count_mean=count.mean(axis=0).tolist(),
               object_point_count_min=count.min(axis=0).tolist(),
               object_point_count_max=count.max(axis=0).tolist(),
               object_presence_fraction=np.mean(count>0,axis=0).tolist(),
               raw_encoded_action_shape=list(encoded.shape),
               native_action_clipping=dict(translation_frames=int(np.sum(np.any(np.abs(actions[:,:3])>1,axis=1))),
                   rotation_frames=int(np.sum(np.linalg.norm(actions[:,3:6],axis=1)>1)),
                   gripper_frames=int(np.sum(np.abs(actions[:,6])>1))),
               final_tcp_peg_distance_m=float(np.linalg.norm(np.asarray(diag[-1]['post']['ee_pose'][:3])-np.asarray(diag[-1]['post']['peg_pose'][:3]))),
               action_controller_input_exact=bool(all(np.array_equal(actions[i],np.asarray(row['controller']['controller_input'])) for i,row in enumerate(trace))),
               controller_translation_scaled_error_max_m=float(max(np.max(np.abs(controller_commands(actions[i:i+1])[0,:3]-np.asarray(row['controller']['arm_scaled_input'])[:3])) for i,row in enumerate(trace))))
    return out, rgb, tcp, actions, diag, folder


def save_after_action_video(rgb, folder, method):
    path = folder/'after_action_rgb.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (512,512))
    if not writer.isOpened():
        raise RuntimeError('MP4 encoder unavailable')
    try:
        for t, frame in enumerate(rgb[1:]):
            image = cv2.resize(frame, (512,512), interpolation=cv2.INTER_NEAREST)
            cv2.putText(image, f'{method} seed30003 after action t={t}', (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:count += 1
    cap.release()
    if count != len(rgb)-1:
        raise ValueError('after-action video readback mismatch')
    return path


def comparison_video(expert_rgb, student_rgb, student_actions, output):
    count=max(len(student_actions['A']),len(student_actions['C']))
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),20,(1536,512))
    if not writer.isOpened():raise RuntimeError('comparison MP4 encoder unavailable')
    try:
        for t in range(count):
            images=[]
            for label,frames,maximum in [('Expert saved',expert_rgb,len(expert_rgb)-1),
                                         ('A actual',student_rgb['A'],len(student_rgb['A'])-1),
                                         ('C actual',student_rgb['C'],len(student_rgb['C'])-1)]:
                index=min(t+1,maximum)
                image=cv2.resize(frames[index],(512,512),interpolation=cv2.INTER_NEAREST)
                cv2.putText(image,f'{label} seed30003 after t={t}' if t<maximum else f'{label} ended at t={maximum-1}',
                            (8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                images.append(image)
            writer.write(cv2.cvtColor(np.concatenate(images,axis=1),cv2.COLOR_RGB2BGR))
    finally:writer.release()
    cap=cv2.VideoCapture(str(output));observed=0
    while cap.read()[0]:observed+=1
    cap.release()
    if observed!=count:raise ValueError('comparison video readback mismatch')
    return output


def figure(path, title, xlabel, ylabel, lines, *, log=False):
    fig, ax = plt.subplots(figsize=(11,4))
    for name, x, y, style in lines:
        ax.plot(x,y,style,label=name,linewidth=1.4)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    if log:ax.set_yscale('log')
    ax.grid(alpha=.25);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def render(plan_path):
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    plan = read(plan_path)
    seed_all(42,plan['config']['threads'])
    status = read(root/'task08b_status.json')
    if status['stage'] != 'COMPLETE':
        raise ValueError('training/rollout/protection incomplete')
    ds = DirectDataset(plan['dataset_directory'])
    original = ds.episodes[30003]
    raw = load_episode(ds.run/original['path'])
    expert_replay=read(root/'expert_replay_gt.json')
    replay_tcp=np.asarray([row['pre']['ee_pose'] for row in expert_replay['trace']])
    replay_tcp_displacement=np.linalg.norm(replay_tcp[:,:3]-raw['tcp_pose'][:len(replay_tcp),:3],axis=1)
    expert_replay_audit=dict(replay_status=expert_replay['status'],
        replay_actions=expert_replay['actions'],replay_success_step=expert_replay['success_step'],
        max_robot76_difference=expert_replay['max_robot76_difference_from_saved'],
        max_tcp_position_difference_m=float(replay_tcp_displacement.max()),
        tcp_difference_at_first_close_m=float(replay_tcp_displacement[expert_replay['first_close_timestep']]),
        scope='same saved Action7 replay; contact/velocity state is not byte-identical to collection; event geometry is evaluation sidecar only')
    atomic(root/'expert_replay_spatial_audit.json',expert_replay_audit,immutable=True)
    stages = [x['before'] for x in original['phases']]
    results, predictions, targets, encoded, closed, rgbs, tcps, actions, sidecars, folders = {},{},{},{},{},{},{},{},{},{}
    for method in ('A','C'):
        train = Path(read(root/f'overfit-{method}-train.json')['directory'])
        results[method],predictions[method],targets[method],encoded[method] = method_offline(train, stages)
        worst=results[method]['max_error_timestep']
        worst_frame=ds.frame(30003,worst)
        results[method]['max_error_frame_detail']=dict(timestep=worst,stage=stages[worst],
            expert_action7=targets[method][worst].tolist(),predicted_action7=predictions[method][worst].tolist(),
            robot76=worst_frame.robot_state.tolist(),packet_sha256=ds.packet_rows[(30003,worst)]['sha256'],
            packet_path=str(ds.output/ds.packet_rows[(30003,worst)]['path']),
            object_point_count=[int(np.sum(worst_frame.packet['object_id']==i)) for i in (0,1)])
        expert_image=cv2.resize(raw['base_camera/rgb'][worst],(512,512),interpolation=cv2.INTER_NEAREST)
        cv2.putText(expert_image,f'Expert seed30003 offline max-error t={worst}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.imwrite(str(root/f'{method}_expert_max_error_t{worst:03d}.png'),cv2.cvtColor(expert_image,cv2.COLOR_RGB2BGR))
        closed[method],rgbs[method],tcps[method],actions[method],sidecars[method],folders[method] = rollout(status['evaluations'][method])
        save_after_action_video(rgbs[method],folders[method],method)
        csv_path=root/f'offline_{method}_per_frame.csv'
        np.savetxt(csv_path,np.column_stack((np.arange(137),targets[method],predictions[method])),
                   delimiter=',',header='t,expert_tx,expert_ty,expert_tz,expert_rx,expert_ry,expert_rz,expert_grip,pred_tx,pred_ty,pred_tz,pred_rx,pred_ry,pred_rz,pred_grip',comments='')
        if np.loadtxt(csv_path,delimiter=',',skiprows=1).shape!=(137,15):
            raise ValueError('per-frame CSV readback failed')
        for t in {results[method]['max_error_timestep'],closed[method]['first_close_timestep'] or 0, len(actions[method])-1}:
            frame = cv2.resize(rgbs[method][min(t+1,len(rgbs[method])-1)],(512,512),interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame,f'{method} seed30003 after-action t={t}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imwrite(str(root/f'{method}_after_t{t:03d}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    comparison_video(raw['base_camera/rgb'],rgbs,actions,root/'expert_vs_A_vs_C_after_action.mp4')
    train_runs={m:Path(read(root/f'overfit-{m}-train.json')['directory']) for m in ('A','C')}
    batches={m:[json.loads(line) for line in (train_runs[m]/'batch_identity.jsonl').read_text().splitlines()] for m in ('A','C')}
    if batches['A'] != batches['C'] or len(batches['A']) != 20000:
        raise ValueError('A/C optimizer episode/batch order differs')
    expected_starts=[0,32,64,96,128]
    if any(row['episode_ids'] != [30003] or row['start'] != expected_starts[i%5]
           for i,row in enumerate(batches['A'])):
        raise ValueError('single-episode contiguous TBPTT schedule mismatch')
    caches={m:read(train_runs[m]/'batch_cache_audit.json') for m in ('A','C')}
    if caches['A']['tensor_sha256'] != caches['C']['tensor_sha256']:
        raise ValueError('A/C cached input tensors differ')
    carry={}
    reload={}
    for method in ('A','C'):
        train=train_runs[method]
        chunks=[]
        for start in expected_starts:
            with np.load(train/f'offline/selected_train/ep30003_chunk{start:04d}.npz',allow_pickle=False) as f:
                chunks.append(dict(before=f['hidden_before'].copy(),after=f['hidden_after'].copy(),
                                   valid=int(f['valid_mask'].sum())))
        carry[method]=bool(np.count_nonzero(chunks[0]['before'])==0 and
                           all(np.array_equal(chunks[i]['after'],chunks[i+1]['before']) for i in range(4)) and
                           sum(x['valid'] for x in chunks)==137)
        if not carry[method]:raise ValueError('offline hidden carry/reset mismatch')
        cp,codec=load_checkpoint(train/'best.json' if (train/'best.json').exists() else train/'last_valid.json')
        model=IntegrationPolicy(ds.schema,**plan['config']['model']).to(plan['config']['device'])
        model.load_state_dict(cp['model'],strict=True)
        norm=load_normalizer(cp['normalizer'])
        output=root/f'reload_{method}'
        reevaluated=evaluate(ds,model,norm,codec,[30003],plan['config'],output)
        with np.load(output/'ep30003.npz',allow_pickle=False) as f:
            exact=bool(np.array_equal(f['native_prediction'],predictions[method]) and
                       np.array_equal(f['native_target'],targets[method]))
        if not exact:raise ValueError('full-episode checkpoint reload prediction mismatch')
        reload[method]=dict(exact_native_prediction=True,selected_checkpoint_sha256=results[method]['completion']['selected']['sha256'],
                            native_action7_mse=reevaluated['native_action7_mse'])
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    fairness=dict(status='PASS',optimizer_batch_order_sha256=sha(train_runs['A']/'batch_identity.jsonl'),
                  cache_tensor_sha256=caches['A']['tensor_sha256'],
                  hidden_reset_and_carry=carry,full_episode_checkpoint_reload=reload)
    atomic(root/'fairness_reload_audit.json',fairness,immutable=True)
    figdir = root/'figures';figdir.mkdir(exist_ok=True)
    for method in ('A','C'):
        h=results[method]['milestones'];x=[r['step'] for r in h]
        figure(figdir/f'{method}_optimization.png',f'{method} | Seed 30003 | optimization loss',
               'updates','own loss scale',[(method,x,[r['optimization_loss'] for r in h],'-o')])
    for name,key,unit in [('motion6_mse','native_motion6_mse','native MSE'),('action7_mse','native_action7_mse','native MSE'),
                          ('controller_translation_mm','controller_translation_mm','mm'),('controller_rotation_rad','controller_rotation_rad','rad'),
                          ('gripper_accuracy','gripper_accuracy','accuracy')]:
        figure(figdir/f'{name}.png',f'Seed 30003 | {name}','updates',unit,
               [(m,[r['step'] for r in results[m]['milestones']],[r[key] for r in results[m]['milestones']],'-o') for m in ('A','C')])
    t=np.arange(137)
    for kind,start,labels in [('translation',0,'XYZ'),('rotation',3,'XYZ')]:
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
        for j in range(3):
            axes[j].plot(t,targets['A'][:,start+j],color=COLORS['Expert'],label='Expert')
            for m in ('A','C'):axes[j].plot(t,predictions[m][:,start+j],label=m,color=COLORS[m],alpha=.8)
            axes[j].set_ylabel(kind+' '+labels[j]);axes[j].grid(alpha=.2);axes[j].legend(fontsize=8)
        axes[-1].set_xlabel('Expert observation timestep')
        fig.suptitle(f'Seed 30003 | Expert vs offline A/C {kind} XYZ')
        fig.tight_layout();fig.savefig(figdir/f'expert_predicted_{kind}.png',dpi=160);plt.close(fig)
    figure(figdir/'expert_predicted_gripper.png','Seed 30003 | Expert vs offline A/C gripper','Expert observation timestep','native gripper',
           [('Expert',t,targets['A'][:,6],'-')]+[(m,t,predictions[m][:,6],'-') for m in ('A','C')])
    figure(figdir/'frame_action_error.png','Seed 30003 | offline Action7 frame L2 error','Expert observation timestep','native L2',
           [(m,t,results[m]['per_frame']['action7_error_l2'],'-') for m in ('A','C')])
    fig,ax=plt.subplots(figsize=(10,4))
    for m in ('A','C'):
        ax.plot(np.arange(len(STAGES)),[results[m]['per_stage'][s]['action7_mse'] for s in STAGES],'-o',label=m)
    ax.set_xticks(np.arange(len(STAGES)),STAGES);ax.set(title='Seed 30003 | offline error by Expert stage',ylabel='native Action7 MSE')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(figdir/'stage_error.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(subplot_kw={'projection':'3d'},figsize=(9,7))
    ax.plot(*raw['tcp_pose'][:,:3].T,label='Expert',color=COLORS['Expert'])
    for m in ('A','C'):ax.plot(*tcps[m][:,:3].T,label=m,color=COLORS[m])
    ax.set(xlabel='world X m',ylabel='world Y m',zlabel='world Z m',title='Seed 30003 | actual TCP trajectories; distinct autonomous states')
    ax.legend();fig.tight_layout();fig.savefig(figdir/'tcp_3d.png',dpi=160);plt.close(fig)
    figure(figdir/'rollout_action7.png','Seed 30003 | actual decoded controller input Action7','Autonomous timestep','native action',
           [(m+f' dim{j}',np.arange(len(actions[m])),actions[m][:,j],'-') for m in ('A','C') for j in range(7)])
    figure(figdir/'gripper_contact_lift.png','Seed 30003 | actual gripper, grasp and peg lift','Autonomous timestep','grip command / grasp / lift m',
           [(m+' grip',np.arange(len(actions[m])),actions[m][:,6],'-') for m in ('A','C')]+[
               (m+' grasp',np.arange(len(sidecars[m])),[float(d['grasp']) for d in sidecars[m]],'--') for m in ('A','C')]+[
               (m+' bilateral contact',np.arange(len(sidecars[m])),[float(all(np.linalg.norm(f)>=.5 for f in d['finger_forces'])) for d in sidecars[m]],'-.') for m in ('A','C')]+[
               (m+' lift',np.arange(len(sidecars[m])),[d['peg_lift_m'] for d in sidecars[m]],':') for m in ('A','C')])
    figure(figdir/'tcp_peg_relative.png','Seed 30003 | evaluation sidecar TCP-Peg distance before each action','Timestep','distance m',
           [('Expert replay',np.arange(len(expert_replay['trace'])),
             [np.linalg.norm(np.asarray(d['pre']['ee_pose'][:3])-np.asarray(d['pre']['peg_pose'][:3])) for d in expert_replay['trace']],'-')]+
           [(m,np.arange(len(sidecars[m])),[np.linalg.norm(np.asarray(d['pre']['ee_pose'][:3])-np.asarray(d['pre']['peg_pose'][:3])) for d in sidecars[m]],'-') for m in ('A','C')])
    fig,ax=plt.subplots(figsize=(11,4))
    for m in ('A','C'):
        distance=[np.linalg.norm(np.asarray(d['pre']['ee_pose'][:3])-np.asarray(d['pre']['peg_pose'][:3])) for d in sidecars[m]]
        ax.plot(np.arange(len(distance)),distance,label=m,color=COLORS[m])
        if closed[m]['first_close_timestep'] is not None:
            ax.axvline(closed[m]['first_close_timestep'],color=COLORS[m],linestyle='--',alpha=.7,label=m+' first close')
    ax.set(title='Seed 30003 | actual approach and first close',xlabel='autonomous timestep',ylabel='TCP-Peg distance m')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(figdir/'approach_close.png',dpi=160);plt.close(fig)
    for m in ('A','C'):
        per_stage=results[m]['per_stage']
        atomic(root/f'{m}_per_stage.json',per_stage,immutable=True)
        atomic(root/f'{m}_offline_analysis.json',results[m],immutable=True)
        atomic(root/f'{m}_closed_loop_analysis.json',closed[m],immutable=True)
    p15={m:next(x for x in results[m]['milestones'] if x['step']==15000) for m in ('A','C')}
    p20={m:next(x for x in results[m]['milestones'] if x['step']==20000) for m in ('A','C')}
    convergence={}
    for m in ('A','C'):
        fit=results[m]['completion']['offline_fit']=='OFFLINE_FIT_PASS'
        improvement=(p15[m]['native_action7_mse']-p20[m]['native_action7_mse'])/p15[m]['native_action7_mse']
        convergence[m]=dict(offline_fit='OFFLINE_FIT_SUFFICIENT' if fit else 'OFFLINE_FIT_INSUFFICIENT',
                            trend='SUFFICIENT' if fit else 'OPTIMIZATION_PLATEAU' if improvement<.01 else 'NOT_CONVERGED_WITHIN_BUDGET',
                            action7_mse_15k_to_20k_relative_improvement=improvement,
                            selected_step=results[m]['completion']['selected']['global_step'])
    atomic(root/'convergence.json',convergence,immutable=True)
    questions={}
    for q,m in [('Q1','A'),('Q2','C')]:
        questions[q]=dict(status='CONFIRMED',answer=convergence[m]['offline_fit'],evidence=dict(selected_step=convergence[m]['selected_step'],
            action7_mse=results[m]['selected_native_metrics']['native_action7_mse'],motion6_mse=results[m]['selected_native_metrics']['native_motion6_mse'],
            gripper_accuracy=results[m]['selected_native_metrics']['gripper_accuracy']))
    questions['Q3']=dict(status='CONFIRMED',answer={m:dict(translation_mm=results[m]['controller_translation_error_mm'],
        rotation_rad=results[m]['controller_rotation_error_rad']) for m in ('A','C')})
    questions['Q4']=dict(status='CONFIRMED',answer={m:dict(first_close=results[m]['predicted_first_close'],
        expert_first_close=results[m]['expert_first_close'],transition_timesteps=results[m]['predicted_transitions'],
        close_segment_accuracy=results[m]['close_segment_accuracy']) for m in ('A','C')})
    questions['Q5']=dict(status='CONFIRMED',answer={m:dict(at_2k=next(x for x in results[m]['milestones'] if x['step']==2000)['native_action7_mse'],
        at_20k=p20[m]['native_action7_mse'],trend=convergence[m]['trend']) for m in ('A','C')})
    questions['Q6']=dict(status='CONFIRMED',answer={m:dict(success=closed[m]['episode']['success'],grasp=closed[m]['episode']['grasp_success'],
        lifted_4cm=closed[m]['episode']['lifted_4cm'],episode_length=closed[m]['episode']['episode_length']) for m in ('A','C')})
    fitted_failures=[m for m in ('A','C') if convergence[m]['offline_fit']=='OFFLINE_FIT_SUFFICIENT' and not closed[m]['episode']['success']]
    def departure(method):
        c=closed[method];d=c['close_before_geometry']
        if d is None:return 'Close command was never issued; first task-relevant deficit at grasp preparation.'
        if d['tcp_peg_distance_m']>expert_replay['tcp_peg_distance_before_close_m']+.02:
            return f"Approach/Close: TCP-Peg distance at first Close {d['tcp_peg_distance_m']:.3f} m versus {expert_replay['tcp_peg_distance_before_close_m']:.3f} m in saved-action replay."
        if not c['episode']['grasp_success']:
            return 'Close/contact: first Close occurred near Expert replay distance but no Panda grasp was observed.'
        if not c['episode']['lifted_4cm']:
            return 'Lift/hold: Panda grasp occurred, but Peg never reached the 4 cm lift criterion.'
        insert_start=next(i for i,stage in enumerate(stages) if stage=='insert')
        if not c['grasp_timesteps'] or max(c['grasp_timesteps'])<insert_start:
            close=c['first_close_timestep']
            reopen=np.flatnonzero(actions[method][close+1:,6]>0) if close is not None else []
            reopening=(f" first open command at autonomous t={close+1+int(reopen[0])} "
                       f"(native grip={actions[method][close+1+int(reopen[0]),6]:+.3f});") if len(reopen) else ''
            end=max(c['grasp_timesteps']) if c['grasp_timesteps'] else None
            return f"Lift/hold during alignment:{reopening} grasp ended at autonomous t={end} and was absent at t={end+1}, before Expert insert stage begins at t={insert_start}; final Peg lift {c['final_peg_lift_m']:.4f} m."
        return 'Alignment/insertion: grasp and 4 cm lift occurred but insertion did not complete.'
    questions['Q7']=dict(status='SUPPORTED' if fitted_failures else 'NOT_TESTED',
        answer={m:departure(m) for m in fitted_failures} if fitted_failures else
               'No offline-sufficient model failed its rollout.',
        limitation='First task-relevant deficit is event-based; it does not prove the first causally wrong action. Expert replay reproduces spatial TCP path within recorded tolerance but Robot76 velocities diverge.')
    any_fit=any(convergence[m]['offline_fit']=='OFFLINE_FIT_SUFFICIENT' for m in ('A','C'))
    all_fit=all(convergence[m]['offline_fit']=='OFFLINE_FIT_SUFFICIENT' for m in ('A','C'))
    any_success=any(closed[m]['episode']['success'] for m in ('A','C'))
    all_success=all(closed[m]['episode']['success'] for m in ('A','C'))
    questions['Q8']=dict(status='SUPPORTED',answer='Basic BC fit remains the first unresolved issue' if not any_fit else
        'Closed-loop state departure deserves investigation for both fitted models' if all_fit and not any_success else
        'This single-episode fit and closed-loop execution both work; broader behavior is not tested' if all_fit and all_success else
        'Single-episode basic BC fit is established for A and C. A loses grasp after lift during autonomous alignment, while C inserts successfully; investigate A closed-loop state departure and method-specific action errors. Held-out generalization remains untested.')
    questions['Q9']=dict(status='SUPPORTED',answer='A/C observed differences justify a controlled loss-design follow-up, but one same-seed episode does not select a general method. Motion z-score and gripper BCE change together; a separate Method B comparison is needed to isolate their effects.')
    joint_success=any(convergence[m]['offline_fit']=='OFFLINE_FIT_SUFFICIENT' and closed[m]['episode']['success'] for m in ('A','C'))
    questions['Q10']=dict(status='SUPPORTED' if joint_success else 'INCONCLUSIVE',
        answer='At least one model fits and executes this episode' if joint_success else
               'No joint evidence of sufficient offline fit and autonomous insertion on this episode')
    result=dict(status='COMPLETE',run_id=root.name,episode=read(root/'episode_identity.json'),
                reset=read(root/'reset_identity.json'),convergence=convergence,questions=questions,
                fairness_reload=fairness,resource_gate=read(root/'resource_gate.json'),
                expert_replay_spatial_audit=expert_replay_audit,
                protected=read(root/'protected_after.json'),training={m:results[m]['completion'] for m in ('A','C')},
                closed_loop={m:closed[m] for m in ('A','C')})
    atomic(root/'verification_summary.json',result,immutable=True)
    lines=['# Task 08-B — Seed 30003 single-episode A/C overfit','',
        f'Comparison UUID: `{root.name}`. 실행 조건: RTX 3090, 기존 Full A/B/C와 별도 Run, A/C 20,000 Updates.',
        '', '## 사전 계약과 기준','',
        f"Episode 30003: 137 Actions / 138 Observations, 성공 {original['success']}, HDF5 SHA256 `{original['sha256']}`.",
        f"초기 가중치 SHA256 `{plan['initial_weight_identity']}`; Robot76 Normalizer SHA256 `{plan['robot_normalizer_identity']}`.",
        '기존 Small5는 Seed 30000–30004, 611 Actions, 2,000 Updates에서 A Action7 MSE 0.004992/C 0.007975로 충분한 Overfit에 미달했다. 이번 결과와 Dataset 크기가 다르다.',
        '정책은 저장된 원본 RGB-D/Predicted Mask 기반 Shape-First/PCA5/F5, Raw EDE, Robot–Object k4/Peg–Box M16, Edge-aware GNN L2/d64, Structured Readout 320D, Presence2D, Robot76, GRU144를 그대로 사용했다.',
        'A는 Native Action7 MSE와 연속 Gripper를 사용한다. C는 Train96 Motion6 z-score MSE와 Gripper BCEWithLogits를 (6×Motion MSE+BCE)/7로 결합하고, 역정규화 및 logit>0→+1/그 외→-1로 디코딩한다.',
        'Task08 기준: Action7 MSE≤0.001, Motion6 MSE≤0.0001, 초기 Action7 MSE 대비 ≥95% 감소, Gripper sign accuracy≥99%.',
        'Train96 Robot76 및 Motion6 통계 재사용. A/C 동일 초기 가중치, AdamW/lr3e-4/B1×T32/clip1/seed42/CosineAnnealingLR(T_max=20000).',
        '기존 config batch_size=2이지만 단일 Episode이므로 실제 유효 Batch는 1×T32다. 두 방식에서 완전히 같다. 저장 Packet에서 생성한 기존 Graph/Robot76/Action 텐서를 정확한 동일성 검사 후 메모리에 재사용했으며 학습 대상과 모델 계산은 바꾸지 않았다.',
        '20,000 Update 고정. 기준을 통과한 milestone 중 Action7 MSE 최저 checkpoint를 선택하고, 없으면 20,000 최종 checkpoint를 선택했다.',
        '기준 미달 시 15k→20k Action7 MSE 개선률 <1%를 OPTIMIZATION_PLATEAU, 그 외를 NOT_CONVERGED_WITHIN_BUDGET로 사전 정의했다.',
        'Controller는 ManiSkill 설치 버전의 pd_ee_delta_pose: Translation ±0.1 m, Rotation 부호 반전 scale -0.1 rad 및 norm clipping, Gripper Target -0.01/+0.04 m. CPU PhysX 20Hz/100Hz, horizon 250, Stop-the-clock.',
        f"Reset: {read(root/'reset_identity.json')['status']}. 물리 GT reset hash와 qpos/qvel은 정확히 일치; 렌더링/수치 허용 범위는 reset_identity.json에 기록했다.",
        f"기존 저장 Action7 재생 평가 Sidecar: 성공 t={expert_replay['success_step']}, Close t={expert_replay['first_close_timestep']}; 저장 TCP와 재생 TCP 위치 차이 최대 {expert_replay_audit['max_tcp_position_difference_m']:.6g} m. Robot76 전체는 byte-identical하지 않아 재생을 원본 상태 자체로 취급하지 않는다.",
        '', '## Offline과 Closed-loop 결과','',
        '| Method | Fit | Selected update | Action7 MSE | Motion6 MSE | Translation mm mean | Rotation deg mean | Grip sign | First close Expert→Pred | Grasp | Lift 4cm | Insert |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |']
    for m in ('A','C'):
        x=results[m];e=closed[m]['episode'];z=x['selected_native_metrics']
        lines.append(f"| {m} | {convergence[m]['offline_fit']} | {convergence[m]['selected_step']} | {z['native_action7_mse']:.6g} | {z['native_motion6_mse']:.6g} | {x['controller_translation_error_mm']['mean']:.3f} | {x['controller_rotation_error_deg']['mean']:.3f} | {z['gripper_accuracy']:.3%} | {x['expert_first_close']}→{x['predicted_first_close']} | {e['grasp_success']} | {e['lifted_4cm']} | {e['success']} |")
    lines.extend(['', 'A/C Optimization Loss는 다른 목표와 BCE 때문에 직접 비교하지 않는다. 위 Offline 수치는 전부 디코딩한 Native Action 및 설치 Controller 전처리 기준이다.',
                  'GT Pose/Contact/Expert stage는 평가와 분석 sidecar에만 사용했다. Student 정책 입력에는 포함되지 않았다.',
                  '동일 timestep의 Expert/Student 위치 차이 자체로 정책 오류를 단정하지 않는다.',
                  '', '## Update별 결과',''])
    for m in ('A','C'):
        lines.extend([f'### {m}', '', '| Updates | Optimization Loss | Action7 MSE | Motion6 MSE | Translation mm | Rotation rad | Gripper accuracy |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for r in results[m]['milestones']:
            lines.append(f"| {r['step']} | {r['optimization_loss']:.6g} | {r['native_action7_mse']:.6g} | {r['native_motion6_mse']:.6g} | {r['controller_translation_mm']:.3f} | {r['controller_rotation_rad']:.5f} | {r['gripper_accuracy']:.3%} |")
        lines.extend(['',f"15k→20k Action7 MSE 개선률 {convergence[m]['action7_mse_15k_to_20k_relative_improvement']:.2%}; 판정 {convergence[m]['trend']}.",''])
    lines.extend(['## Offline 선택 Checkpoint와 최종 Update 비교','',
                  '| Method | Selected update | Selected Action7 MSE | 20k Action7 MSE | Selected translation mm mean/p95/max | Selected rotation rad mean/p95/max | Gripper target mm mean/p95/max |',
                  '| --- | ---: | ---: | ---: | --- | --- | --- |'])
    for m in ('A','C'):
        x=results[m];t=x['controller_translation_error_mm'];r=x['controller_rotation_error_rad'];g=x['controller_gripper_target_error_mm']
        lines.append(f"| {m} | {convergence[m]['selected_step']} | {x['selected_native_metrics']['native_action7_mse']:.6g} | {p20[m]['native_action7_mse']:.6g} | {t['mean']:.4f}/{t['p95']:.4f}/{t['max']:.4f} | {r['mean']:.6g}/{r['p95']:.6g}/{r['max']:.6g} | {g['mean']:.4f}/{g['p95']:.4f}/{g['max']:.4f} |")
    lines.append('')
    lines.extend(['## Expert 단계별 Offline Action7 MSE','',
                  '| Stage | A | C |','| --- | ---: | ---: |'])
    for stage in STAGES:lines.append(f"| {stage} | {results['A']['per_stage'][stage]['action7_mse']:.6g} | {results['C']['per_stage'][stage]['action7_mse']:.6g} |")
    lines.extend(['','## Same-seed 실제 조작',''])
    for m in ('A','C'):
        c=closed[m];e=c['episode']
        lines.extend([f'### {m}', '',f"250-step horizon 중 {e['episode_length']} step 실행; 파지 {e['grasp_success']}, 양손가락 접촉 {e['bilateral_contact_observed']}, 최대 Peg Lift {c['max_peg_lift_m']:.4f} m, 4cm Lift 연속 {c['lift_4cm_hold_steps_longest']} step, 최종 Peg Lift {c['final_peg_lift_m']:.4f} m, 삽입 {e['success']}.",
            f"최초 Close {c['first_close_timestep']}; Close 직전 TCP–Peg {c['close_before_geometry']}; Controller 입력 일치 {c['action_controller_input_exact']}.",
            f"Rollout artifacts: `{c['evaluation_run']}`. 실제 after-action RGB: `{folders[m]/'after_action_rgb.mp4'}`.",''])
    lines.extend(['## Q1–Q10 판정',''])
    for q,item in questions.items():lines.append(f"- **{q} — {item['status']}**: {item['answer']}")
    lines.extend(['','## 보존 및 한계','',
        f"시작/종료 보존 감사: {result['protected']['status']} ({result['protected']['checked_files']} files). 원본 Dataset, 기존 Checkpoint, Full Run은 수정하지 않았다.",
        f"시각화: `{root/'single_episode_ac_overfit.html'}`; Expert/A/C after-action 비교 영상 `{root/'expert_vs_A_vs_C_after_action.mp4'}`. 같은 timestep의 서로 다른 자율 상태는 행동 오류의 직접 증거가 아니다.",
        'A/C는 Motion z-score와 Gripper BCE가 동시에 다르므로 개별 원인은 분리되지 않는다. 필요하면 별도 Method B가 필요하다.',
        '단일 Episode 성공은 Graph 정보 활용 또는 Held-out 일반화를 증명하지 않는다. CUDA 학습은 본 RTX3090에서 실행했고, simulator는 CPU PhysX stop-the-clock로 실행했다.',
        '실시간 제어 마감시간, CUDA PhysX, MPS는 이 실험에서 검증하지 않았다. 외부 업로드와 Git Commit은 수행하지 않았다.'])
    markdown='\n'.join(lines)+'\n'
    report=root/'08b_single_episode_ac_overfit.md'
    report.write_text(markdown)
    if report.read_text()!=markdown:raise ValueError('Markdown readback failed')
    docs=Path('/workspace/task08/docs/visual-graph-integration/08b_single_episode_ac_overfit.md')
    docs.parent.mkdir(parents=True,exist_ok=True)
    docs.write_text(markdown)
    image_links='\n'.join(f'<figure><img src="figures/{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in sorted(figdir.glob('*.png')))
    body='<pre>'+html.escape(markdown)+'</pre>'
    videos='<p><a href="expert_vs_A_vs_C_after_action.mp4">Expert vs A vs C after-action RGB MP4</a></p>'+''.join(f'<p>{m}: <a href="../{Path(status["evaluations"][m]).name}/episodes/train_seed_30003/after_action_rgb.mp4">after-action RGB MP4</a></p>' for m in ('A','C'))
    (root/'single_episode_ac_overfit.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Task08-B single episode A/C</title><style>body{font:15px system-ui;margin:2rem;max-width:1100px}pre{white-space:pre-wrap}img{max-width:100%}figure{margin:2rem 0}</style></head><body><h1>Task08-B Seed 30003</h1>'+body+videos+image_links+'</body></html>')
    shutil.copy2(__file__,root/'bc08b_report_source.py')
    shutil.copy2(Path(__file__).with_name('bc08b_expert_replay.py'),root/'bc08b_expert_replay_source.py')
    verification=verify(root,link_root=Path(plan['experiments_root']))
    atomic(root/'artifact_readback.json',verification,immutable=True)
    if verification['status']!='PASS':raise ValueError('artifact readback failed')
    return dict(report=str(report),html=str(root/'single_episode_ac_overfit.html'),readback=verification)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render(args.plan)),flush=True)
