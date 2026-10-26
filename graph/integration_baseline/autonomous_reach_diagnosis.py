"""Task07-E: fixed seed causal reach/close diagnostic. Opt-in; never policy SR.

All policy inputs use the unchanged observation whitelist and RGB-only streaming
vision. GT state, segmentation, Expert stage and grasp pose are diagnostics or
Expert control only. Every trial resets independently; no restored state/actions.
"""
import argparse
from dataclasses import asdict
import inspect
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch
from transforms3d.quaternions import quat2mat, qinverse, qmult

from graph.integration_baseline.grasp_intervention import atomic, digest, retention
from graph.integration_baseline.direct_runtime import read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet
from graph.integration_baseline.model import EpisodeStream
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_checkpoint import load_bundle, required_sources
from graph.integration_baseline.rollout_stop_clock import (PhysicsAudit, observation,
    controller_audit, validate_controller_command, controller_semantics, infer, numpy,
    overlay, reset_diagnostics)
from visual.runtime.rgbd import TrackedRGBDFrame

CONDITIONS = ('positive_control', 'A', 'B', 'C')
SCOPE = 'MIXED_CONTROL_CAUSAL_DIAGNOSTIC_NOT_POLICY_SUCCESS_RATE'


def rotation_distance(q1, q2):
    a = np.asarray(q1, np.float64); b = np.asarray(q2, np.float64)
    a /= np.linalg.norm(a); b /= np.linalg.norm(b)
    return float(2*np.arccos(np.clip(abs(np.dot(a,b)), 0, 1)))


def relative_pose(parent, child):
    p, c = np.asarray(parent), np.asarray(child)
    return np.r_[(c[:3]-p[:3]) @ quat2mat(p[3:]), qmult(qinverse(p[3:]),c[3:])]


def box_surface_distance(points, pose, half):
    """Signed point-centre to oriented cuboid surface distance, metres."""
    local = (np.asarray(points)-pose[:3]) @ quat2mat(pose[3:])
    delta = np.abs(local)-np.asarray(half)
    return np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(delta.max(axis=-1),0)


def pad_segment_intersects_peg(pads, pose, half):
    """Necessary centre-line geometry indicator only; finite pads/contact excluded."""
    p = (np.asarray(pads)-pose[:3]) @ quat2mat(pose[3:])
    delta = p[1]-p[0]; lo, hi = 0., 1.
    for x,d,h in zip(p[0],delta,half):
        if abs(d)<1e-12:
            if abs(x)>h:return False
        else:
            a,b=sorted(((-h-x)/d,(h-x)/d));lo=max(lo,a);hi=min(hi,b)
    return bool(lo<=hi)


def geometry(env, t_valid):
    """Read-only independent physical diagnosis; never a policy input."""
    from envs.maniskill import get_gt_state
    g=get_gt_state(env); task=env.unwrapped
    out={k:numpy(getattr(g,k))[0] for k in ('ee_pose','peg_pose','box_pose',
         'left_finger_pose','right_finger_pose','left_fingertip_pos','right_fingertip_pos',
         'peg_half_sizes','gripper_qpos','qpos','qvel')}
    for name,link in [('tcp',task.agent.tcp),('peg',task.peg),('box',task.box)]:
        for key in ('linear_velocity','angular_velocity'):
            out[name+'_'+key]=numpy(getattr(link,key))[0]
    out['peg_relative_tcp_pose']=relative_pose(out['peg_pose'],out['ee_pose'])
    pads=np.stack([out['left_fingertip_pos'],out['right_fingertip_pos']])
    out['fingertip_centre_to_peg_surface_m']=box_surface_distance(pads,out['peg_pose'],out['peg_half_sizes'])
    out['pad_centre_segment_intersects_peg']=pad_segment_intersects_peg(pads,out['peg_pose'],out['peg_half_sizes'])
    out['gripper_opening_m']=float(out['gripper_qpos'].sum())
    out['contact_valid']=t_valid
    out['finger_forces']=np.stack([numpy(task.scene.get_pairwise_contact_forces(link,task.peg))[0]
        for link in (task.agent.finger1_link,task.agent.finger2_link)]) if t_valid else np.zeros((2,3),np.float32)
    out['bilateral_force']=bool(t_valid and np.all(np.linalg.norm(out['finger_forces'],axis=1)>=.5))
    out['grasp']=bool(task.agent.is_grasping(task.peg).item()) if t_valid else False
    return out


def diagnostic_segmentation(env, camera):
    # Read the existing current render texture only: no take_picture, render or step.
    sensor=env.unwrapped._sensors[camera]
    return numpy(sensor.get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'])[0,...,0]


def treatment_action(predicted, condition):
    a=np.asarray(predicted)
    if condition not in CONDITIONS or a.shape!=(7,) or a.dtype!=np.float32 or not np.isfinite(a).all():
        raise ValueError('finite native FP32 Action7 and prescribed condition required')
    return np.r_[np.zeros(6),-1].astype(np.float32) if condition in ('B','positive_control') else a.copy()


def validate_close_event(t, predicted, c):
    event=c['model_close_timestep']
    if t<event and predicted[6]<0:raise ValueError('CONTRACT_MISMATCH: negative gripper before fixed t27')
    if t==event and predicted[6]>=0:raise ValueError('CONTRACT_MISMATCH: no first negative gripper at fixed t27')


def compare_prefix(reference, candidate, config):
    a,b=reference['prefix'],candidate['prefix']
    if len(a)!=len(b) or reference['reset_group']!=candidate['reset_group']:
        raise ValueError('paired reset/length mismatch')
    maximum_action=maximum_hidden=maximum_physical=0.
    for x,y in zip(a,b):
        for k in ('timestep','rgbd_calibration_robot_graph_mask_sha256','expert_action_sha256'):
            if x[k]!=y[k]:raise ValueError('paired exact prefix mismatch: '+k)
        for k in ('action','hidden'):
            np.testing.assert_allclose(x[k],y[k],atol=config['atol'],rtol=config['rtol'])
        maximum_action=max(maximum_action,float(np.max(np.abs(np.array(x['action'])-y['action']))))
        maximum_hidden=max(maximum_hidden,float(np.max(np.abs(np.array(x['hidden'])-y['hidden']))))
        for k in ('qpos','qvel','ee_pose','peg_pose','box_pose','left_finger_pose','right_finger_pose',
                  'left_fingertip_pos','right_fingertip_pos','tcp_linear_velocity','tcp_angular_velocity',
                  'peg_linear_velocity','peg_angular_velocity','box_linear_velocity','box_angular_velocity'):
            xa,ya=np.array(x['physical'][k]),np.array(y['physical'][k])
            np.testing.assert_allclose(xa,ya,atol=config['physical_atol'],rtol=0)
            maximum_physical=max(maximum_physical,float(np.max(np.abs(xa-ya))))
    return dict(status='PASS',frames=len(a),observation_graph_mask_robot_source_exact=True,
                maximum_action_absolute_difference=maximum_action,maximum_hidden_absolute_difference=maximum_hidden,
                maximum_physical_absolute_difference=maximum_physical,physical_atol=config['physical_atol'])


def run_trial(env, spec, condition, config, model, norm, vision, folder):
    import sapien
    from recurrent.expert import ScriptedExpert
    from graph.integration_baseline.vision import packet_from_frame
    folder.mkdir(parents=True, exist_ok=False)
    identity = folder.name
    obs, _ = env.reset(seed=spec['seed'])
    reset_group = reset_diagnostics(env)
    expert = ScriptedExpert(env)
    stream = EpisodeStream(model, norm); stream.reset(); vision.reset(identity)
    physics = PhysicsAudit(env); audit = controller_audit(env)
    device = next(model.parameters()).device
    rows, trace, masks_out, hidden_out, videos, prefix = [], [], [], [], [], []
    gt_segmentations = []
    grasp_reference = np.r_[expert.grasp.p, expert.grasp.q].astype(np.float32)
    initial_diagnostic = geometry(env, t_valid=False)
    reference_relative = relative_pose(initial_diagnostic['peg_pose'], grasp_reference)
    expert_approach = condition in ('A', 'positive_control')
    anchor = None; close_start = None; target = None
    tick = perf_counter()
    try:
        for t in range(250):
            before = physics.token()
            if before[1] != t or stream.timestep != (t-1 if t else None):
                raise ValueError('live observation/hidden clock discontinuity')
            if close_start is None and ((expert_approach and expert.phase == 'close') or
                                         (not expert_approach and t == config['model_close_timestep'])):
                close_start = t
                anchor = geometry(env, t_valid=t > 0)
                target = sapien.Pose(anchor['ee_pose'][:3] + [0, 0, config['lift_height_m']], anchor['ee_pose'][3:])
                if expert_approach:
                    dp, dr = expert._err(expert.grasp)
                    if dp >= .006 or dr >= .04:
                        raise ValueError('Expert did not reach the prescribed takeover pose')
            if close_start is None:
                stage = expert.phase if expert_approach else 'student_approach'
                if expert_approach and stage not in ('reach', 'descend'):
                    raise ValueError('unexpected Expert approach phase')
                expert_action = expert.act() if expert_approach else None
            else:
                offset = t - close_start
                stage = ('close' if offset < config['close_steps'] else 'lift' if
                    offset < config['close_steps']+config['lift_steps'] else 'hold')
                if offset >= config['close_steps']+config['lift_steps']+config['hold_steps']:
                    break
                expert_action = None
            diagnostic_before = geometry(env, t_valid=t > 0)
            row = observation(env, obs, norm.schema, config['camera'])
            rows.append(row)
            gt_segmentations.append(diagnostic_segmentation(env, config['camera']))
            masks, timing = vision.process(row['rgb'], identity, t)
            masks_out.append(masks)
            frame = TrackedRGBDFrame(t, config['camera'], row['rgb'], row['depth'],
                row['intrinsic'], row['cam2world'], row['tcp_pose'], {0:masks[0], 1:masks[1]}, t, t/20)
            packet, _ = packet_from_frame(frame, identity, keep_ratio=.5)
            save_packet(folder/f'packet_{t:04d}.npz', packet)
            graph = build_graph(packet, row['robot_ee_geometry'], **config['graph'])
            save_npz(folder/f'graph_{t:04d}.npz', **{k:numpy(getattr(graph, k)) for k in
                ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')}, source_index=packet['provenance']['source_index'])
            command, model_timing = infer(stream, pack_graphs([graph]).to(device),
                torch.from_numpy(row['robot_state'][None]).to(device), identity, t)
            predicted = numpy(command)[0]; hidden_out.append(numpy(stream.hidden))
            if not expert_approach:
                validate_close_event(t, predicted, config)
            if close_start is None or t == close_start:
                prefix.append(dict(timestep=t,
                    rgbd_calibration_robot_graph_mask_sha256=digest(row['rgb'], row['depth_raw_mm'], row['intrinsic'],
                        row['cam2world'], row['robot_state'], row['robot_world_geometry'], row['robot_ee_geometry'], masks, packet['provenance']['source_index'], *[numpy(getattr(graph, k)) for k in
                        ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')]),
                    expert_action_sha256=digest(expert_action) if expert_action is not None else 'close_boundary',
                    action=predicted.tolist(), hidden=hidden_out[-1].tolist(), physical=diagnostic_before))
            if stage == 'close':
                action = treatment_action(predicted, condition)
            elif stage in ('lift', 'hold'):
                # Common robot-only servo, fixed target anchored before intervention.
                # Full closing after the 8-frame treatment is intentional and recorded.
                action = np.r_[expert._servo(target, expert.max_trans), -1].astype(np.float32)
            else:
                action = expert_action if expert_approach else predicted.copy()
            physics.unchanged(before)
            audit.clear()
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(action[None]).to(env.unwrapped.device))
            physics.stepped(before); validate_controller_command(action, audit)
            after = geometry(env, t_valid=True)
            presence = [bool((packet['object_id'] == i).any()) for i in (0, 1)]
            trace.append(dict(timestep=t, stage=stage, condition=condition,
                simulation_time_before=t/20, simulation_time_after=(t+1)/20,
                predicted_action=predicted, applied_action=action, controller=dict(audit),
                before=diagnostic_before, point_counts=[int((packet['object_id']==i).sum()) for i in (0,1)],
                mask_overlap_pixels=int(np.logical_and(masks[0],masks[1]).sum()),
                presence=presence, mask_present=masks.any(axis=(1, 2)), after=after,
                vision=timing, model_timing=model_timing,
                physics_steps_before=before[0], physics_steps_after=physics.count,
                terminated=bool(terminated.item()), truncated=bool(truncated.item()),
                full_task_success=bool(info['success'].item())))
            picture = overlay(row, masks, graph, t, action, False, presence)
            cv2.putText(picture, condition+' / '+stage, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            videos.append(picture)
            if t % 20 == 0 or stage == 'close':
                print(f"FRAME {identity} t={t} stage={stage} grip={action[6]:.4f} grasp={after['grasp']}", flush=True)
            if terminated.item() or truncated.item():
                raise ValueError('environment terminated before prescribed hold completion')
        rows.append(observation(env, obs, norm.schema, config['camera']))
        if close_start is None:
            raise ValueError('Expert never reached close before horizon')
        result = dict(spec, condition=condition, status='COMPLETE', close_start=close_start,
            steps=len(trace), physics_steps=physics.count, hidden_steps=len(hidden_out),
            hidden_reset_events=[dict(timestep=0, reason='fresh_trial_reset')],
            hidden_at_close_reset=False, reset_group=reset_group, anchor=anchor, grasp_reference=grasp_reference, reference_relative=reference_relative,
            initial_diagnostic=initial_diagnostic, prefix=prefix, outcome=retention(trace, anchor, config),
            first_grasp_after_action_t=next((r['timestep'] for r in trace if r['after']['grasp']), None),
            first_bilateral_force_after_action_t=next((r['timestep'] for r in trace if r['after']['bilateral_force']), None),
            first_lift_grasp_after_action_t=next((r['timestep'] for r in trace if r['stage']=='lift' and r['after']['grasp']), None),
            close_presence_mean=np.mean([x['presence'] for x in trace if x['stage']=='close'], axis=0),
            wall_seconds=perf_counter()-tick, simulation_seconds=len(trace)/20,
            grasp_semantics='installed Panda.is_grasping: bilateral force>=0.5N and contact angle<=85deg',
            policy_success_rate=None, scope=config['scope'])
        return result
    finally:
        physics.close()
        if trace:
            atomic(folder/'trace.json', trace)
            save_npz(folder/'actions.npz', predicted=np.stack([r['predicted_action'] for r in trace]),
                applied=np.stack([r['applied_action'] for r in trace]),
                controller_input=np.stack([r['controller']['controller_input'] for r in trace]),
                arm_scaled_input=np.stack([r['controller']['arm_scaled_input'] for r in trace]),
                gripper_scaled_input=np.stack([r['controller']['gripper_scaled_input'] for r in trace]),
                timestep=np.arange(len(trace)), stage=np.asarray([r['stage'] for r in trace]))
        if gt_segmentations:
            save_npz(folder/'gt_segmentation_diagnostic_only.npz', segmentation=np.stack(gt_segmentations),
                     actor_ids=np.asarray([[env.unwrapped.peg._objs[0].per_scene_id,env.unwrapped.box._objs[0].per_scene_id]],np.int64))
        if rows:
            save_npz(folder/'observations.npz', **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        if masks_out:
            save_npz(folder/'masks.npz', masks=np.stack(masks_out))
            save_npz(folder/'hidden.npz', hidden=np.stack(hidden_out), timestep=np.arange(len(hidden_out)),
                     episode_identity=np.asarray(identity), restoration='post_observation_pre_action')
        atomic(folder/'vision_initialization.json',dict(found=vision.found,detector_calls=vision.detector_calls))
        if videos:
            writer = cv2.VideoWriter(str(folder/'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (512, 512))
            if not writer.isOpened():
                raise RuntimeError('video writer unavailable')
            try:
                for image in videos:
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            cv2.imwrite(str(folder/'representative.png'), cv2.cvtColor(videos[-1], cv2.COLOR_RGB2BGR))


def validate_config(c):
    if (c['seed']!=41005 or tuple(c['conditions'])!=CONDITIONS or c['model_close_timestep']!=27 or
        (c['close_steps'],c['lift_steps'],c['hold_steps'])!=(8,20,20) or
        (c['lift_height_m'],c['minimum_peg_lift_m'],c['minimum_tcp_lift_m'])!=(.08,.04,.06) or
        c['scope']!=SCOPE or (c['atol'],c['rtol'],c['physical_atol'])!=(1e-6,1e-5,1e-6)):
        raise ValueError('fixed diagnostic protocol changed')


def file_inventory(paths):
    return {str(p):sha(p) for p in sorted(set(paths)) if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint-manifest',type=Path,required=True)
    p.add_argument('--original-episode',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args(); c=read(args.config);validate_config(c)
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    torch.set_num_threads(1);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if torch.cuda.mem_get_info()[0]<c['minimum_cuda_free_bytes']:
        raise RuntimeError('BLOCKED_RESOURCE: no existing process is interrupted')
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'])
    model,norm,provenance=load_bundle(args.checkpoint_manifest,device='cuda')
    if provenance['manifest']['files']['checkpoint']['sha256']!=c['checkpoint_sha256']:
        raise ValueError('fixed checkpoint differs')
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.expert import ScriptedExpert
    from recurrent.common import versions
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from utils.experiment_logger import ExperimentLogger
    if sha(inspect.getfile(ScriptedExpert))!=provenance['dataset']['expert_sha256']:
        raise ValueError('original Expert source differs')
    # Existing sources, model, normalizer, training data and previous result files are protected.
    protected=list(required_sources())
    protected=[Path(x) for x in protected]
    manifest=read(args.checkpoint_manifest)
    protected += [args.checkpoint_manifest, args.original_episode/'trace.json']
    protected += [x for x in args.original_episode.rglob('*') if x.is_file()]
    bundle=args.checkpoint_manifest.parent
    protected += [x for x in bundle.rglob('*') if x.is_file()]
    for folder in (Path('/workspace/graph-recurrent-VLA/experiments/2be5e62357a44b14a5651a1a346edc7a'),
                   Path('/workspace/task07-grasp-intervention/experiments/d046e3a679a3430aab5e69f279c3f0a7')):
        protected += [x for x in folder.rglob('*') if x.is_file()]
    before=file_inventory(protected)
    weights=digest(*[numpy(x) for x in model.parameters()]);normalizer=digest(numpy(norm.stats.mean),numpy(norm.stats.std))
    with ExperimentLogger(team='integration',model='autonomous_reach_causal_diagnosis',seed=c['seed'],
        job_type='evaluate',config=c,root=args.output_root) as logger:
        run=logger.directory;print('RUN_DIR='+str(run),flush=True)
        atomic(run/'protected_before.json',before);atomic(run/'checkpoint_provenance.json',provenance)
        source={}
        paths=[Path(__file__),args.config]+[Path(inspect.getfile(o)) for o in
            (ScriptedExpert,PhysicsAudit,EpisodeStream,OnlineVisionAdapter,make_peg_insertion_env,build_graph)]
        paths += [Path(x) for x in required_sources()]
        for path in set(paths):
            if not path.is_file():continue
            dest=run/'source_snapshot'/path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else run/'source_snapshot'/path.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);source[str(path)]=sha(path)
        atomic(run/'source_hashes.json',source);shutil.copy2(args.config,run/'protocol.json')
        report=dict(run_id=logger.run_id,status='RUNNING',scope=SCOPE,protocol=c,trials=[],paired_checks={},
            checkpoint_sha256=c['checkpoint_sha256'],policy_success_rate=None,original_episode=str(args.original_episode))
        atomic(run/'results.json',report)
        vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'))
        results={}
        try:
            for condition in CONDITIONS:
                env=TimeLimitWrapper(make_peg_insertion_env(obs_mode='rgb+depth',control_mode='pd_ee_delta_pose',device='cpu').unwrapped,250)
                try:
                    task=env.unwrapped
                    actual=dict(control_freq=task.control_freq,physics_freq=task.sim_freq,physics_substeps=task._sim_steps_per_control,
                        horizon=env._max_episode_steps,robot=task.agent.uid,versions=versions(),backend=str(task.backend),
                        controller_config={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()})
                    trained=provenance['conditions']
                    for key in ('control_freq','physics_freq','horizon','robot'):
                        if actual[key]!=trained[key]:raise ValueError('environment mismatch: '+key)
                    for key in ('mani_skill','sapien','gymnasium'):
                        if actual['versions'][key]!=trained['versions'][key]:raise ValueError('package mismatch: '+key)
                    if actual['physics_substeps']!=5 or controller_semantics(actual['controller_config'])!=controller_semantics(trained['controller_config']):
                        raise ValueError('physics/controller mismatch')
                    atomic(run/'executed_conditions.json',actual)
                    result=run_trial(env,dict(seed=41005,episode_id=41005,split='diagnostic_only'),condition,c,model,norm,vision,run/'trials'/condition)
                finally:env.close()
                results[condition]=result
                atomic(run/'trials'/condition/'trial.json',result)
                report['trials'].append({k:v for k,v in result.items() if k!='prefix'})
                if condition=='A':report['paired_checks']['positive_control_A']=compare_prefix(results['positive_control'],result,c)
                if condition=='C':report['paired_checks']['B_C']=compare_prefix(results['B'],result,c)
                atomic(run/'results.json',report)
                logger.log(dict(retained=int(result['outcome']['retained']),close_end_grasp=int(result['outcome']['close_end_grasp'])),prefix=condition)
            report['status']='COMPLETE'
        except Exception as exc:
            report.update(status='ERROR',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            vision.close()
            report['weights_and_normalizer_unchanged']=(weights==digest(*[numpy(x) for x in model.parameters()]) and
                normalizer==digest(numpy(norm.stats.mean),numpy(norm.stats.std)))
            changed=[path for path,h in before.items() if not Path(path).is_file() or sha(Path(path))!=h]
            atomic(run/'protection.json',dict(status='PASS' if not changed else 'FAIL',protected_file_count=len(before),changed=changed))
            atomic(run/'results.json',report)
            logger.record(source_run_id=provenance['manifest']['source_run_id'],diagnostic_status=report['status'],
                checkpoint_sha256=c['checkpoint_sha256'],policy_success_rate=None)
            for path in sorted(run.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json'):
                    logger.link_output(path,kind='rollouts' if path.suffix=='.mp4' else 'trajectory' if path.suffix=='.npz' else 'reports')
        print('RESULT='+report['status'],flush=True)


if __name__=='__main__':main()
