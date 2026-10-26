"""Task 08-C: independent physical rollouts with one controller-boundary gripper override."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch

from graph.integration_baseline.bc08_codec import ActionCodec, DecodedPolicy
from graph.integration_baseline.bc08_train import load_checkpoint, load_normalizer
from graph.integration_baseline.bc08b_eval import environment, validate_checkpoint_plan
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet
from graph.integration_baseline.model import IntegrationPolicy, EpisodeStream
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.robot import flatten_state, robot_observation
from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
from graph.integration_baseline.rollout_stop_clock import (
    PhysicsAudit, controller_audit, infer, numpy, observation, reset_diagnostics,
    validate_controller_command,
)
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import TrackedRGBDFrame


CONDITIONS = ('A0_ORIGINAL', 'A1_SINGLE_STEP_CLOSE', 'A2_SUSTAINED_CLOSE')
GATE = dict(fp32_atol=1e-6, rgb_lsb=2, depth_mm=1, exact_integer_inputs=True,
            reset_hash_exact=True, before_action_t=79, prefix_last_action_t=78)


def intervene(raw, condition, t):
    raw = np.asarray(raw)
    if raw.shape != (7,) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('finite native FP32 Action7 required')
    if condition not in CONDITIONS:
        raise ValueError('unknown intervention')
    applied = raw.copy()
    active = t == 79 and condition != CONDITIONS[0] or t >= 79 and condition == CONDITIONS[2]
    if active:
        applied[6] = -1.
    np.testing.assert_array_equal(applied[:6], raw[:6])
    if t < 79:
        np.testing.assert_array_equal(applied, raw)
    return applied, bool(active)


def diagnostics(env):
    from envs.maniskill import get_gt_state
    g = get_gt_state(env)
    task = env.unwrapped
    names = ('qpos', 'qvel', 'ee_pose', 'left_fingertip_pos', 'right_fingertip_pos',
             'left_finger_pose', 'right_finger_pose', 'peg_pose', 'box_pose',
             'gripper_qpos')
    result = {name: numpy(getattr(g, name))[0].tolist() for name in names}
    result['peg_linear_velocity'] = numpy(task.peg.linear_velocity)[0].tolist()
    result['peg_angular_velocity'] = numpy(task.peg.angular_velocity)[0].tolist()
    result['box_linear_velocity'] = numpy(task.box.linear_velocity)[0].tolist()
    result['box_angular_velocity'] = numpy(task.box.angular_velocity)[0].tolist()
    result['grasp'] = bool(task.agent.is_grasping(task.peg).item())
    result['finger_forces'] = [numpy(task.scene.get_pairwise_contact_forces(link, task.peg))[0].tolist()
                               for link in (task.agent.finger1_link, task.agent.finger2_link)]
    return result


def digest_array(a):
    x = np.ascontiguousarray(a)
    return hashlib.sha256(str((x.dtype.str, x.shape)).encode()+x.tobytes()).hexdigest()


def observed_inputs(row, masks, packet, graph):
    values = {key: row[key] for key in ('rgb', 'depth_raw_mm', 'intrinsic', 'cam2world',
                                        'robot_state', 'robot_world_geometry', 'robot_ee_geometry')}
    values['masks'] = masks
    for key in ('object_id', 'points', 'colors'):
        if key in packet:
            values['packet_'+key] = packet[key]
    for key in ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type'):
        values['graph_'+key] = numpy(getattr(graph, key))
    return values


def run_trial(env, policy, norm, vision, spec, config, folder, condition):
    from graph.integration_baseline.vision import packet_from_frame
    folder.mkdir(parents=True, exist_ok=False)
    obs, _ = env.reset(seed=spec['seed'])
    reset_hash = reset_diagnostics(env)
    if reset_hash != spec['reset_group']:
        raise ValueError('BLOCKED_RESET_IDENTITY')
    identity = 'train_seed_30003'  # identical causal Vision and GRU identity in every fresh trial
    stream = EpisodeStream(policy, norm)
    stream.reset()
    vision.reset(identity)
    physics = PhysicsAudit(env)
    audit = controller_audit(env)
    device = next(policy.parameters()).device
    states, input_frames, masks_out, hidden_before, hidden_after, raw_actions = [], [], [], [], [], []
    applied_actions, after_rgb, trace = [], [], []
    start = perf_counter()
    result = dict(condition=condition, seed=spec['seed'], status='RUNNING', reset_group=reset_hash,
                  first_success_step=None, success=False, episode_length=0)
    try:
        states.append(diagnostics(env))
        for t in range(250):
            token = physics.token()
            if token[1] != t:
                raise ValueError('observation clock discontinuity')
            row = observation(env, obs, norm.schema, config['vision']['camera'])
            masks, vision_timing = vision.process(row['rgb'], identity, t)
            frame = TrackedRGBDFrame(t, config['vision']['camera'], row['rgb'], row['depth'],
                row['intrinsic'], row['cam2world'], row['tcp_pose'], {0:masks[0], 1:masks[1]}, t, t/20)
            packet, _ = packet_from_frame(frame, identity, keep_ratio=.5)
            save_packet(folder/f'packet_{t:04d}.npz', packet)
            graph = build_graph(packet, row['robot_ee_geometry'], **config['graph'])
            save_npz(folder/f'graph_{t:04d}.npz', **{key:numpy(getattr(graph, key)) for key in
                ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')},
                source_index=packet['provenance']['source_index'])
            input_frames.append(observed_inputs(row, masks, packet, graph))
            masks_out.append(masks)
            hidden_before.append(None if stream.hidden is None else numpy(stream.hidden))
            command, timing = infer(stream, pack_graphs([graph]).to(device),
                torch.from_numpy(row['robot_state'][None]).to(device), identity, t)
            raw = numpy(command)[0]
            applied, active = intervene(raw, condition, t)
            hidden_after.append(numpy(stream.hidden))
            raw_actions.append(raw)
            applied_actions.append(applied)
            np.testing.assert_array_equal(flatten_state(robot_observation(env)[0], norm.schema), row['robot_state'])
            physics.unchanged(token)
            before = states[-1]
            audit.clear()
            action_tensor = torch.from_numpy(applied[None]).to(env.unwrapped.device)
            obs, _, terminated, truncated, info = env.step(action_tensor)
            physics.stepped(token)
            validate_controller_command(applied, audit)
            gripper_controller = env.unwrapped.agent.controller.controllers['gripper']
            target = numpy(gripper_controller._target_qpos)[0].tolist()
            after = diagnostics(env)
            states.append(after)
            success = bool(info['success'].item())
            trace.append(dict(t=t, observation_index=t, action_index=t, next_observation_index=t+1,
                model_raw_action7=raw.tolist(), applied_action7=applied.tolist(),
                model_raw_gripper=float(raw[6]), applied_gripper=float(applied[6]),
                controller_scaled_translation=audit['arm_scaled_input'][:3].tolist(),
                controller_scaled_rotation=audit['arm_scaled_input'][3:].tolist(),
                controller_gripper_target=target, controller_gripper_scaled_input=audit['gripper_scaled_input'].tolist(),
                gru_hidden_before=None if hidden_before[-1] is None else hidden_before[-1].tolist(),
                gru_hidden_after=hidden_after[-1].tolist(),
                intervention_active=active, intervention_condition=condition,
                before=before, after=after, success=success,
                terminated=bool(terminated.item()), truncated=bool(truncated.item()),
                physics_steps_before=token[0], physics_steps_after=physics.count,
                vision_ms=vision_timing['elapsed_ms'], inference_ms=timing['gnn_gru_ms']))
            # Actual Observation[t+1] image; also retained in full observation arrays below.
            next_row = observation(env, obs, norm.schema, config['vision']['camera'])
            after_rgb.append(next_row['rgb'])
            result.update(episode_length=t+1, success=success)
            if success:
                result['first_success_step'] = t+1
            if t % 20 == 0 or 75 <= t <= 90:
                print(f'{condition} t={t} raw_grip={raw[6]:+.4f} applied={applied[6]:+.4f} grasp={after["grasp"]} success={success}', flush=True)
            if success or terminated.item() or truncated.item():
                break
        result.update(status='COMPLETE', wall_seconds=perf_counter()-start, physics_steps=physics.count,
                      terminated=bool(terminated.item()), truncated=bool(truncated.item()))
    finally:
        physics.close()
        if trace:
            atomic(folder/'trace.json', trace)
            atomic(folder/'episode.json', result)
            save_npz(folder/'actions.npz', model_raw_action7=np.asarray(raw_actions),
                     applied_action7=np.asarray(applied_actions), timestep=np.arange(len(trace)))
            save_npz(folder/'hidden.npz', hidden_before=np.stack([np.zeros_like(hidden_after[0]) if h is None else h for h in hidden_before]),
                     hidden_after=np.stack(hidden_after), timestep=np.arange(len(trace)))
            save_npz(folder/'observations.npz', **{key:np.stack([x[key] for x in input_frames]) for key in
                ('rgb', 'depth_raw_mm', 'intrinsic', 'cam2world', 'robot_state',
                 'robot_world_geometry', 'robot_ee_geometry')},
                after_rgb=np.stack(after_rgb), masks=np.stack(masks_out), timestep=np.arange(len(trace)))
            atomic(folder/'state_trace.json', states)
            writer = cv2.VideoWriter(str(folder/'after_action_rgb.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (512,512))
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
            for t, image in enumerate(after_rgb):
                image = cv2.resize(image, (512,512), interpolation=cv2.INTER_NEAREST)
                cv2.putText(image, f'{condition} after Action[{t}]', (8,24),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            writer.release()
            if len(after_rgb)>79:
                cv2.imwrite(str(folder/'after_t079.png'), cv2.cvtColor(after_rgb[79], cv2.COLOR_RGB2BGR))
        stream.reset()
        vision.reset(None)
    return result


def compare_arrays(a, b, key):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(pass_gate=False, reason='shape_or_dtype', a_shape=list(a.shape), b_shape=list(b.shape))
    if np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_:
        allowed = GATE['rgb_lsb'] if key == 'rgb' else GATE['depth_mm'] if key == 'depth_raw_mm' else 0
    else:
        allowed = GATE['fp32_atol']
    difference = np.abs(a.astype(np.float64)-b.astype(np.float64))
    maximum = float(difference.max(initial=0))
    return dict(pass_gate=maximum <= allowed, max_abs=maximum, tolerance=allowed,
                differing_count=int(np.count_nonzero(difference)), exact=digest_array(a)==digest_array(b))


def prefix_audit(reference, candidate, *, old=False):
    a, b = Path(reference), Path(candidate)
    answer = dict(reference=str(a), candidate=str(b), scope='Observation[0..79], Action[0..78], before Action[79]',
                  fixed_gate=GATE, old_task08b=old, fields={}, pass_gate=True)
    with np.load(a/'observations.npz', allow_pickle=False) as x, np.load(b/'observations.npz', allow_pickle=False) as y:
        for key in ('rgb','depth_raw_mm','intrinsic','cam2world','robot_state'):
            # Existing Task08-B observations contain t=79. New run stores each pre-action frame.
            for t in range(80):
                out = compare_arrays(x[key][t], y[key][t], key)
                if not out['pass_gate'] or t in (0, 78, 79):
                    answer['fields'][f'{key}[{t}]'] = out
                answer['pass_gate'] &= out['pass_gate']
        for t in range(79):
            out = compare_arrays(x['masks'][t] if 'masks' in x else np.load(a/'masks.npz')['masks'][t],
                                 y['masks'][t], 'masks')
            if not out['pass_gate']:
                answer['fields'][f'masks[{t}]'] = out
            answer['pass_gate'] &= out['pass_gate']
    with np.load(a/'actions.npz',allow_pickle=False) as x, np.load(b/'actions.npz',allow_pickle=False) as y:
        aa=x['actions'] if old else x['model_raw_action7']
        bb=y['model_raw_action7']
        for t in range(80):
            out=compare_arrays(aa[t],bb[t], 'action')
            if not out['pass_gate'] or t in (78,79):answer['fields'][f'action[{t}]']=out
            answer['pass_gate'] &= out['pass_gate']
    with np.load(a/'hidden.npz',allow_pickle=False) as x, np.load(b/'hidden.npz',allow_pickle=False) as y:
        aa=x['hidden'] if old else x['hidden_after']
        for t in range(80):
            out=compare_arrays(aa[t],y['hidden_after'][t], 'hidden')
            if not out['pass_gate'] or t==79:answer['fields'][f'hidden_after[{t}]']=out
            answer['pass_gate'] &= out['pass_gate']
    for t in range(80):
        for key in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            with np.load(a/f'graph_{t:04d}.npz',allow_pickle=False) as x, np.load(b/f'graph_{t:04d}.npz',allow_pickle=False) as y:
                out=compare_arrays(x[key],y[key],key)
            if not out['pass_gate'] or t==79:answer['fields'][f'{key}[{t}]']=out
            answer['pass_gate'] &= out['pass_gate']
    if not old:
        xa=read(a/'state_trace.json'); xb=read(b/'state_trace.json')
        for t in range(80):
            for key in ('qpos','qvel','ee_pose','left_fingertip_pos','right_fingertip_pos',
                        'peg_pose','box_pose','peg_linear_velocity','peg_angular_velocity'):
                out=compare_arrays(np.asarray(xa[t][key],np.float32),np.asarray(xb[t][key],np.float32),key)
                if not out['pass_gate'] or t==79:answer['fields'][f'{key}[{t}]']=out
                answer['pass_gate'] &= out['pass_gate']
    answer['status']='PASS' if answer['pass_gate'] else 'INVALID_PREFIX_MATCH'
    return answer


def summarize(folder, baseline_z):
    t=read(folder/'trace.json')
    grasp=[r['after']['grasp'] for r in t]
    lift=np.array([r['after']['peg_pose'][2]-baseline_z for r in t])
    sustained=[]; n=0
    for value in lift>=.04:
        n=n+1 if value else 0;sustained.append(n)
    release=next((r['t'] for r in t[79:] if not r['after']['grasp']),None)
    return dict(condition=folder.name, steps=len(t), success=any(r['success'] for r in t),
        first_success_step=next((r['t']+1 for r in t if r['success']),None),
        before_t79_grasp=t[79]['before']['grasp'], after_t79_grasp=t[79]['after']['grasp'],
        after_t80_grasp=t[80]['after']['grasp'], first_post79_grasp_loss_action=release,
        grasp_steps_after_79=int(sum(grasp[79:])), max_lift_m=float(lift.max()),
        lift_4cm_steps=int(sum(lift>=.04)), lift_4cm_longest_steps=max(sustained),
        first_positive_model_gripper_after79=next((r['t'] for r in t[80:] if r['model_raw_gripper']>0),None),
        raw_gripper_t79=t[79]['model_raw_gripper'], applied_gripper_t79=t[79]['applied_gripper'],
        final_peg_pose=t[-1]['after']['peg_pose'], final_tcp_pose=t[-1]['after']['ee_pose'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--original-a',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    plan=read(args.plan); receipt=read(args.receipt)
    cp,codec=load_checkpoint(args.receipt)
    validate_checkpoint_plan(cp,codec,receipt,plan,args.plan,'overfit','A')
    if receipt['global_step'] != 20000 or cp['identity']['train_ids'] != [30003]:
        raise ValueError('wrong Task08-B single-episode checkpoint')
    if sha(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint hash mismatch')
    if vision_runtime_identity()!=plan['reference_provenance']['vision_runtime']:
        raise ValueError('vision runtime identity mismatch')
    torch.set_num_threads(1); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=IntegrationPolicy(cp['normalizer']['schema'],**plan['config']['model']).cuda()
    model.load_state_dict(cp['model'],strict=True);model.eval()
    norm=load_normalizer(cp['normalizer']);policy=DecodedPolicy(model,codec)
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'))
    spec=read(args.plan.parent/'seeds_manifest.json')['groups']['train_seed'][0]
    source_files=['graph/integration_baseline/bc08c_intervention.py','graph/integration_baseline/bc08b_eval.py',
                  'graph/integration_baseline/rollout_stop_clock.py','graph/integration_baseline/bc08_codec.py',
                  'graph/integration_baseline/model.py']
    source_hashes={name:sha(Path(name)) for name in source_files}
    protected={name:sha(path) for name,path in [('checkpoint',Path(receipt['path'])),('plan',args.plan),
        ('original_actions',args.original_a/'actions.npz'),('original_hidden',args.original_a/'hidden.npz')]}
    with ExperimentLogger(team='integration', model='task08c_a_gripper_intervention',seed=30003,
            config=dict(task='08-C',conditions=list(CONDITIONS),gate=GATE,checkpoint=receipt,
                        original_a=str(args.original_a),source_hashes=source_hashes),
            root=args.output_root,job_type='evaluate') as logger:
        run=logger.directory
        atomic(run/'protocol.json',dict(time_index='Observation[t] -> Action[t] -> Observation[t+1]',
            gate=GATE,conditions=list(CONDITIONS),checkpoint=receipt,normalizer_sha256=plan['robot_normalizer_identity'],
            source_hashes=source_hashes,protected_before=protected))
        shutil.copy2(__file__,run/'bc08c_intervention_source.py')
        trials={}; audits={}
        for condition in CONDITIONS:
            env,conditions=environment(plan)
            try:
                trial=run/condition
                result=run_trial(env,policy,norm,vision,spec,plan['config'],trial,condition)
                trials[condition]=result
                logger.log({f'{condition}/success':int(result['success']),f'{condition}/steps':result['episode_length']})
            finally:env.close()
        for condition in CONDITIONS[1:]:
            audits[condition]=prefix_audit(run/CONDITIONS[0],run/condition)
        audits['old_A_vs_new_A0']=prefix_audit(args.original_a,run/CONDITIONS[0],old=True)
        reset_all={c:trials[c]['reset_group'] for c in CONDITIONS}
        audits['reset_identity']=dict(values=reset_all,pass_gate=len(set(reset_all.values()))==1==bool(spec['reset_group']==next(iter(reset_all.values()))))
        atomic(run/'prefix_audit.json',audits)
        summaries={c:summarize(run/c,read(run/c/'state_trace.json')[0]['peg_pose'][2]) for c in CONDITIONS}
        atomic(run/'summary.json',summaries)
        protected_after={name:sha(path) for name,path in [('checkpoint',Path(receipt['path'])),('plan',args.plan),
            ('original_actions',args.original_a/'actions.npz'),('original_hidden',args.original_a/'hidden.npz')]}
        atomic(run/'protection_audit.json',dict(before=protected,after=protected_after,pass_gate=protected==protected_after))
        for name in ('protocol.json','prefix_audit.json','summary.json','protection_audit.json'):
            logger.link_output(run/name,kind='metrics')
        print(json.dumps(dict(run=str(run),summary=summaries,audits={k:v['status'] if 'status' in v else v['pass_gate'] for k,v in audits.items()}),indent=2),flush=True)


if __name__=='__main__':main()
