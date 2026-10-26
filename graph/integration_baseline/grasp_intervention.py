"""Four paired live close-window interventions; never a policy success evaluation.

Each trial executes a fresh real Expert approach, observing live RGB-D exactly
once per step. Only eight close actions are intervened on. The subsequent robot
servo and full-close gripper protocol are common to all conditions. No state or
hidden from another trial is restored, and GT diagnostics never enter the model.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic as _atomic, read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet, serial
from graph.integration_baseline.model import EpisodeStream
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_checkpoint import load_bundle
from graph.integration_baseline.rollout_stop_clock import (
    PhysicsAudit, observation, controller_audit, validate_controller_command,
    controller_semantics, infer, numpy, overlay, reset_diagnostics)
from visual.runtime.rgbd import TrackedRGBDFrame

CONDITIONS = ('expert', 'model', 'replace_gripper', 'stop_arm')


def atomic(path, value):
    _atomic(path, serial(value))


def digest(*values):
    h = hashlib.sha256()
    for a in values:
        a = np.asarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def close_action(predicted, condition):
    a = np.asarray(predicted)
    if a.shape != (7,) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('finite native FP32 Action7 required')
    if condition not in CONDITIONS:
        raise ValueError('unknown intervention')
    a = a.copy()
    if condition in ('expert', 'stop_arm'):
        a[:6] = 0
    if condition in ('expert', 'replace_gripper'):
        a[6] = -1
    return a


def diagnostics(env, contact_valid):
    """Independent outcome accessor. No return value is passed to the model."""
    from envs.maniskill import get_gt_state
    g = get_gt_state(env); t = env.unwrapped
    values = {k: numpy(getattr(g, k))[0] for k in ('peg_pose', 'ee_pose', 'gripper_qpos')}
    values.update(contact_valid=contact_valid,
        grasp=bool(t.agent.is_grasping(t.peg).item()) if contact_valid else False,
        finger_forces=np.stack([numpy(t.scene.get_pairwise_contact_forces(link, t.peg))[0]
            for link in (t.agent.finger1_link, t.agent.finger2_link)]) if contact_valid else np.zeros((2, 3), np.float32))
    return values


def retention(rows, anchor, config):
    close = [r for r in rows if r['stage'] == 'close']
    held = [r for r in rows if r['stage'] == 'hold']
    if len(close) != config['close_steps'] or len(held) != config['hold_steps']:
        raise ValueError('incomplete intervention/lift/hold cannot produce a grasp verdict')
    peg = [float(r['after']['peg_pose'][2] - anchor['peg_pose'][2]) for r in held]
    tcp = [float(r['after']['ee_pose'][2] - anchor['ee_pose'][2]) for r in held]
    contact = [bool(r['after']['grasp']) for r in held]
    enough_tcp = all(z >= config['minimum_tcp_lift_m'] for z in tcp)
    elevated = all(z >= config['minimum_peg_lift_m'] for z in peg)
    return dict(retained=bool(enough_tcp and elevated and all(contact)),
        lift_execution_valid=bool(enough_tcp), peg_elevated_all_hold=bool(elevated),
        grasp_all_hold=bool(all(contact)), hold_contact_fraction=float(np.mean(contact)),
        minimum_hold_peg_lift_m=min(peg), minimum_hold_tcp_lift_m=min(tcp),
        close_end_grasp=bool(close[-1]['after']['grasp']),
        close_contact_frames=sum(bool(r['after']['grasp']) for r in close),
        close_end_finger_forces=close[-1]['after']['finger_forces'],
        final_gripper_qpos=held[-1]['after']['gripper_qpos'])


def compare_prefix(reference, candidate, config):
    """Compare only data through the first close inference, before intervention."""
    a, b = reference['prefix'], candidate['prefix']
    if len(a) != len(b):
        raise ValueError('paired Expert prefixes have different lengths')
    for x, y in zip(a, b):
        for k in ('timestep', 'rgbd_calibration_robot_graph_mask_sha256', 'expert_action_sha256'):
            if x[k] != y[k]:
                raise ValueError('paired prefix mismatch: '+k)
        for k in ('action', 'hidden'):
            np.testing.assert_allclose(x[k], y[k], atol=config['atol'], rtol=config['rtol'])
    for k in ('peg_pose', 'ee_pose', 'gripper_qpos'):
        np.testing.assert_allclose(reference['anchor'][k], candidate['anchor'][k],
                                   atol=config['atol'], rtol=0)
    if reference['reset_group'] != candidate['reset_group']:
        raise ValueError('paired reset differs')
    return dict(status='PASS', frames=len(a), observation_graph_mask_robot_exact=True,
                hidden_action_within_fixed_tolerance=True, close_start_state_within_1um=True)


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
    anchor = None; close_start = None; target = None
    tick = perf_counter()
    try:
        for t in range(250):
            before = physics.token()
            if before[1] != t or stream.timestep != (t-1 if t else None):
                raise ValueError('live observation/hidden clock discontinuity')
            if close_start is None and expert.phase == 'close':
                close_start = t
                anchor = diagnostics(env, t > 0)
                target = sapien.Pose(anchor['ee_pose'][:3] + [0, 0, config['lift_height_m']], anchor['ee_pose'][3:])
                # Verify actual Expert approach tolerance at takeover; no GT peg pose input.
                dp, dr = expert._err(expert.grasp)
                if dp >= .006 or dr >= .04:
                    raise ValueError('Expert approach did not reach a valid takeover pose')
            if close_start is None:
                stage = expert.phase
                if stage not in ('reach', 'descend'):
                    raise ValueError('unexpected Expert phase before takeover')
                expert_action = expert.act()
            else:
                offset = t - close_start
                stage = ('close' if offset < config['close_steps'] else 'lift' if
                    offset < config['close_steps']+config['lift_steps'] else 'hold')
                if offset >= config['close_steps']+config['lift_steps']+config['hold_steps']:
                    break
                expert_action = None
            row = observation(env, obs, norm.schema, config['camera'])
            rows.append(row)
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
            if close_start is None or t == close_start:
                prefix.append(dict(timestep=t,
                    rgbd_calibration_robot_graph_mask_sha256=digest(row['rgb'], row['depth_raw_mm'], row['intrinsic'],
                        row['cam2world'], row['robot_state'], masks, *[numpy(getattr(graph, k)) for k in
                        ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')]),
                    expert_action_sha256=digest(expert_action) if expert_action is not None else 'close_boundary',
                    action=predicted.tolist(), hidden=hidden_out[-1].tolist()))
            if stage == 'close':
                action = close_action(predicted, condition)
            elif stage in ('lift', 'hold'):
                # Common robot-only servo, fixed target anchored before intervention.
                # Full closing after the 8-frame treatment is intentional and recorded.
                action = np.r_[expert._servo(target, expert.max_trans), -1].astype(np.float32)
            else:
                action = expert_action
            physics.unchanged(before)
            audit.clear()
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(action[None]).to(env.unwrapped.device))
            physics.stepped(before); validate_controller_command(action, audit)
            after = diagnostics(env, True)
            presence = [bool((packet['object_id'] == i).any()) for i in (0, 1)]
            trace.append(dict(timestep=t, stage=stage, condition=condition,
                simulation_time_before=t/20, simulation_time_after=(t+1)/20,
                predicted_action=predicted, applied_action=action, controller=dict(audit),
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
            hidden_at_close_reset=False, reset_group=reset_group, anchor=anchor,
            prefix=prefix, outcome=retention(trace, anchor, config),
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
    if tuple(c['conditions']) != CONDITIONS or c['close_steps'] != 8 or c['lift_steps'] != 20 or c['hold_steps'] != 20:
        raise ValueError('predefined four-condition eight-frame protocol changed')
    ids = [s['episode_id'] for s in c['episodes']]
    if len(ids) != len(set(ids)) or len(ids) > 4 or not ids:
        raise ValueError('fixed small diagnostic selection required')
    if c['scope'] != 'MIXED_CONTROL_CAUSAL_DIAGNOSTIC_NOT_POLICY_SUCCESS_RATE':
        raise ValueError('diagnostic must not be reported as policy evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(); c = read(args.config); validate_config(c)
    os.environ.update(WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    torch.set_num_threads(1); torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.mem_get_info()[0] < c['minimum_cuda_free_bytes']:
        raise RuntimeError('BLOCKED_RESOURCE: no other job is interrupted or modified')
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'])
    model, norm, provenance = load_bundle(args.checkpoint_manifest, device='cuda')
    if provenance['manifest']['files']['checkpoint']['sha256'] != c['checkpoint_sha256']:
        raise ValueError('fixed checkpoint changed')
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.expert import ScriptedExpert
    from recurrent.common import versions
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from utils.experiment_logger import ExperimentLogger
    if sha(inspect.getfile(ScriptedExpert)) != provenance['dataset']['expert_sha256']:
        raise ValueError('original collection Expert source differs')
    for spec in c['episodes']:
        e = next(e for e in provenance['dataset']['episodes'] if e['episode_id'] == spec['episode_id'])
        if e['seed'] != spec['seed'] or spec['episode_id'] not in provenance['split']['splits'][spec['split']]:
            raise ValueError('selection seed/split mismatch')
    with ExperimentLogger(team='integration', model='live_grasp_close_intervention', seed=c['seed'],
        job_type='evaluate', config=c, root=args.output_root) as logger:
        run = logger.directory; print('RUN_DIR='+str(run), flush=True)
        shutil.copy2(__file__, run/'intervention_source.py'); shutil.copy2(args.config,run/'protocol.json')
        source = {}
        for obj in (ScriptedExpert, PhysicsAudit, EpisodeStream, OnlineVisionAdapter, make_peg_insertion_env, build_graph):
            path = Path(inspect.getfile(obj)); dest = run/'source_snapshot'/path.name
            dest.parent.mkdir(exist_ok=True); shutil.copy2(path,dest); source[str(path)] = sha(path)
        atomic(run/'source_hashes.json', source)
        atomic(run/'checkpoint_provenance.json',provenance)
        weights = digest(*[numpy(p) for p in model.parameters()]); normalizer = digest(numpy(norm.stats.mean),numpy(norm.stats.std))
        report = dict(run_id=logger.run_id, status='RUNNING', scope=c['scope'], trials=[], paired_checks={},
            policy_success_rate=None, checkpoint_sha256=c['checkpoint_sha256'], protocol=c)
        vision = OnlineVisionAdapter(GDino(device='cuda'), SAM2Tracker(device='cuda'))
        try:
            for spec in c['episodes']:
                reference = None
                for condition in CONDITIONS:
                    name = f"{spec['split']}_{spec['episode_id']}_{condition}"
                    env = TimeLimitWrapper(make_peg_insertion_env(obs_mode='rgb+depth', control_mode='pd_ee_delta_pose',device='cpu').unwrapped,250)
                    try:
                        task = env.unwrapped
                        conditions = dict(control_freq=task.control_freq, physics_freq=task.sim_freq,
                            physics_substeps=task._sim_steps_per_control, horizon=env._max_episode_steps,
                            robot=task.agent.uid, versions=versions(), backend=str(task.backend),
                            controller_config={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()})
                        trained = provenance['conditions']
                        for k in ('control_freq','physics_freq','horizon','robot'):
                            if conditions[k] != trained[k]:raise ValueError('live environment contract mismatch: '+k)
                        for k in ('mani_skill','sapien','gymnasium'):
                            if conditions['versions'][k] != trained['versions'][k]:raise ValueError('simulator version mismatch: '+k)
                        if conditions['physics_substeps'] != 5 or controller_semantics(conditions['controller_config']) != controller_semantics(trained['controller_config']):
                            raise ValueError('live physics/controller contract mismatch')
                        atomic(run/'executed_conditions.json',conditions)
                        print('TRIAL '+name,flush=True)
                        result = run_trial(env,spec,condition,c,model,norm,vision,run/'trials'/name)
                    finally:
                        env.close()
                    if reference is None:reference=result
                    report['paired_checks'][name]=compare_prefix(reference,result,c)
                    atomic(run/'trials'/name/'trial.json',result)
                    # Large exact prefix hidden states remain in the per-trial file.
                    report['trials'].append({k:v for k,v in result.items() if k != 'prefix'})
                    atomic(run/'results.json',report)
                    logger.log(dict(retained=int(result['outcome']['retained']),
                        close_end_grasp=int(result['outcome']['close_end_grasp'])),prefix=name)
            report['status']='COMPLETE'
            report['weights_and_normalizer_unchanged']=(weights==digest(*[numpy(p) for p in model.parameters()]) and
                normalizer==digest(numpy(norm.stats.mean),numpy(norm.stats.std)))
            if not report['weights_and_normalizer_unchanged']:raise ValueError('weights/normalizer changed')
        except Exception as exc:
            report.update(status='ERROR', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            vision.close(); atomic(run/'results.json',report)
            logger.record(source_run_id=provenance['manifest']['source_run_id'],
                diagnostic_status=report['status'], policy_success_rate=None, checkpoint_sha256=c['checkpoint_sha256'])
            for p in sorted(run.rglob('*')):
                if p.is_file() and not p.name.endswith('.tracking.json') and 'source_snapshot' not in p.parts:
                    logger.link_output(p,kind='rollouts' if p.suffix=='.mp4' else 'trajectory' if p.suffix=='.npz' else 'reports')
        print('RESULT='+report['status'],flush=True)


if __name__ == '__main__':
    main()
