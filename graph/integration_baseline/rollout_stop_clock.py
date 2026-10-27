"""Opt-in single-environment stop-the-clock evaluation. Never starts training."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet
from graph.integration_baseline.model import IntegrationPolicy, EpisodeStream
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.robot import (robot_observation, state_schema, flatten_state,
                                              geometry_in_ee, RobotNormalizer)
from graph.integration_baseline.rollout_checkpoint import load_bundle, vision_runtime_identity
from graph.integration_baseline.rollout_metrics import OFFICIAL_NAME, summarize, prepare_seeds, validate_seeds
from graph.prototype.v2.state import StateNormalizer
from visual.runtime.rgbd import TrackedRGBDFrame


def numpy(value):
    return value.detach().cpu().numpy().copy()


def controller_semantics(config):
    """URDF installation directory is host-specific; controller values are exact."""
    value = json.loads(json.dumps(config))
    for item in value.values():
        if 'urdf_path' in item:
            item['urdf_path'] = Path(item['urdf_path']).name
    return value


def observation(env, obs, schema, camera):
    """Whitelist only current camera/proprioception; task extras never copied."""
    fields, pads = robot_observation(env)
    if state_schema(fields) != schema:
        raise ValueError('live Robot76 schema differs from checkpoint')
    for key in ('qpos', 'qvel'):
        np.testing.assert_array_equal(numpy(obs['agent'][key])[0], fields[key])
    image, params = obs['sensor_data'][camera], obs['sensor_param'][camera]
    ext = numpy(params['extrinsic_cv'])[0]
    raw = numpy(image['depth'])[0, ..., 0]
    return dict(rgb=numpy(image['rgb'])[0], depth_raw_mm=raw,
                depth=raw.astype(np.float32)/1000,
                intrinsic=numpy(params['intrinsic_cv'])[0],
                cam2world=np.linalg.inv(np.vstack((ext, [0, 0, 0, 1]))).astype(np.float32),
                tcp_pose=fields['tcp_pose'], robot_state=flatten_state(fields, schema),
                robot_world_geometry=pads,
                robot_ee_geometry=geometry_in_ee(pads, fields['tcp_pose']))


class PhysicsAudit:
    """Observe actual scene.step calls, including any accidental inference step."""
    def __init__(self, env):
        self.task = env.unwrapped
        self.count = 0
        self.scene = self.task.scene
        self.original = self.scene.step

        def step(*args, **kwargs):
            result = self.original(*args, **kwargs)
            self.count += 1
            return result

        self.scene.step = step

    def token(self):
        return self.count, int(self.task.elapsed_steps.item())

    def unchanged(self, before):
        if self.token() != before:
            raise ValueError('physics advanced during observation/inference')

    def stepped(self, before):
        if self.token() != (before[0]+5, before[1]+1):
            raise ValueError('one action must advance exactly 5 physics / 1 control steps')

    def close(self):
        self.scene.step = self.original


def controller_audit(env):
    """Record entry and actual installed preprocessing, without modifying action."""
    controller = env.unwrapped.agent.controller
    record = {}
    original = controller.set_action

    def set_action(action):
        record['controller_input'] = numpy(action)[0]
        return original(action)

    controller.set_action = set_action
    for name, child in controller.controllers.items():
        original_preprocess = child._preprocess_action

        def preprocess(action, _original=original_preprocess, _name=name):
            result = _original(action)
            record[_name+'_scaled_input'] = numpy(result)[0]
            return result

        child._preprocess_action = preprocess
    return record


def validate_controller_command(action, actual):
    """Audit the installed collection convention, including its rotation sign."""
    np.testing.assert_array_equal(actual['controller_input'], action)
    arm = action[:6].copy()
    arm[:3] = np.clip(arm[:3], -1, 1)*.1
    arm[3:] /= max(float(np.linalg.norm(arm[3:])), 1.)
    arm[3:] *= -.1
    gripper = (np.clip(action[6:], -1, 1)+1)*.025-.01
    np.testing.assert_allclose(actual['arm_scaled_input'], arm, atol=2e-8, rtol=1e-6)
    np.testing.assert_allclose(actual['gripper_scaled_input'], gripper, atol=1e-8, rtol=0)


def reset_diagnostics(env):
    """Evaluation-only reset identity; return value never enters policy inputs."""
    from envs.maniskill import get_gt_state
    gt = get_gt_state(env)
    data = np.concatenate([numpy(getattr(gt, name)).ravel() for name in
                           ('qpos', 'peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes')])
    return hashlib.sha256(data.tobytes()).hexdigest()


def restore_stream_hidden(stream, path, episode_identity, timestep):
    """Restore GRU only, after observation t; simulator restoration is separate."""
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['episode_identity']) != episode_identity or not np.array_equal(saved['timestep'], np.arange(len(saved['hidden']))):
            raise ValueError('hidden episode/timestep identity mismatch')
        if not 0 <= timestep < len(saved['hidden']):
            raise ValueError('hidden timestep missing')
        value = saved['hidden'][timestep].copy()
    rnn = stream.model.policy.rnn
    if value.shape != (rnn.num_layers, 1, rnn.hidden_size) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('invalid saved GRU state')
    stream.hidden = torch.from_numpy(value).to(next(stream.model.parameters()).device)
    stream.identity, stream.timestep = episode_identity, timestep


def infer(stream, graph, state, identity, timestep):
    """Time existing modules with temporary hooks; preserve EpisodeStream API."""
    device = next(stream.model.parameters()).device
    times, starts, handles = {}, {}, []

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def begin(name):
        def hook(module, args):
            sync(); starts[name] = perf_counter()
        return hook

    def end(name):
        def hook(module, args, output):
            sync(); times[name] = (perf_counter()-starts[name])*1000
        return hook

    for name, module in [('gnn_readout_ms', stream.model.graph_encoder), ('gru_ms', stream.model.policy.rnn)]:
        handles.extend([module.register_forward_pre_hook(begin(name)), module.register_forward_hook(end(name))])
    sync(); start = perf_counter()
    try:
        with torch.inference_mode():
            action = stream.step(graph, state, identity, timestep)
        sync()
        times['gnn_gru_ms'] = (perf_counter()-start)*1000
        return action, times
    finally:
        for handle in handles:
            handle.remove()


def overlay(row, masks, graph, timestep, action, success, presence):
    image = row['rgb'].copy()
    for oid, color in enumerate(((240, 80, 80), (60, 160, 255))):
        image[masks[oid]] = (image[masks[oid]]*.55+np.array(color)*.45).astype(np.uint8)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, f't={timestep} grip={action[6]:+.3f} success={success}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    cv2.putText(image, 'presence='+str(presence), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    # Explicit EE x/y graph inset, independent from camera mask projection.
    inset = np.full((180, 180, 3), 20, np.uint8)
    xyz = graph.node_input[:, :3].numpy()
    uv = np.clip(np.round(xyz[:, :2]*150+90), 0, 179).astype(int)
    for a, b in graph.edge_index.numpy().T:
        cv2.line(inset, tuple(uv[a]), tuple(uv[b]), (70,70,70), 1)
    for point, kind in zip(uv, graph.node_type.numpy()):
        cv2.circle(inset, tuple(point), 2 if kind < 3 else 1,
                   ((240,240,240) if kind < 3 else (240,80,80) if kind == 3 else (60,160,255)), -1)
    image[-180:, -180:] = inset
    cv2.putText(image, 'EE graph x/y', (332, 325), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    return image


def run_episode(env, model, normalizer, vision, spec, config, folder, *, smoke, forbidden_groups=()):
    from graph.integration_baseline.vision import packet_from_frame
    folder.mkdir(parents=True, exist_ok=False)
    obs, _ = env.reset(seed=spec['seed'])
    reset_group = reset_diagnostics(env)  # diagnostics kept out of row/graph
    if spec.get('reset_group') is not None and reset_group != spec['reset_group']:
        raise ValueError('frozen-test reset group differs on this simulator')
    if spec['group'] == 'heldout' and reset_group in forbidden_groups:
        raise ValueError('held-out reset group overlaps frozen dataset')
    identity = folder.name
    stream = EpisodeStream(model, normalizer)
    stream.reset()
    vision.reset(identity)
    audit = controller_audit(env)
    physics = PhysicsAudit(env)
    device = next(model.parameters()).device
    trace, rows, masks_saved, hidden_saved, videos, attempts = [], [], [], [], [], []
    result = dict(spec, status='RUNNING', success=False, first_success_step=None, episode_length=0,
                  terminated=False, truncated=False, timeout=False, reset_group=reset_group,
                  hidden_reset_events=[dict(episode=identity, timestep=0, reason='episode_reset')],
                  collision=None, task_failure=None, unsupported_metrics=['collision', 'task_failure'])
    tick = perf_counter()
    try:
        limit = config['smoke_steps'] if smoke else 250
        for t in range(limit):
            before = physics.token()
            if before[1] != t:
                raise ValueError('noncontiguous environment observation')
            row = observation(env, obs, normalizer.schema, config['vision']['camera'])
            rows.append(row)
            masks, timing = vision.process(row['rgb'], identity, t)
            masks_saved.append(masks)
            frame = TrackedRGBDFrame(t, config['vision']['camera'], row['rgb'], row['depth'],
                row['intrinsic'], row['cam2world'], row['tcp_pose'], {0:masks[0], 1:masks[1]}, t, t/20)
            packet, geometry_timing = packet_from_frame(frame, identity, keep_ratio=.5)
            packet['observation_provenance'] = dict(episode_identity=identity, timestep=t,
                rgb_sha256=hashlib.sha256(row['rgb'].tobytes()).hexdigest(),
                mask_source='live_prefix_causal_rgb_gdino_forward_sam2', future_observations_used=False)
            save_packet(folder/f'packet_{t:04d}.npz', packet)
            graph = build_graph(packet, row['robot_ee_geometry'], **config['graph'])
            save_npz(folder/f'graph_{t:04d}.npz', **{name:numpy(getattr(graph, name)) for name in
                ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')},
                source_index=packet['provenance']['source_index'])
            graph_batch = pack_graphs([graph]).to(device)
            state = torch.from_numpy(row['robot_state'][None]).to(device)
            command, model_timing = infer(stream, graph_batch, state, identity, t)
            action = numpy(command)[0]
            if action.shape != (7,) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError('finite FP32 Action7 required')
            # Read-only probe verifies same robot state as well as no physics.
            current = flatten_state(robot_observation(env)[0], normalizer.schema)
            np.testing.assert_array_equal(current, row['robot_state'])
            physics.unchanged(before)
            hidden_saved.append(numpy(stream.hidden))
            tensor = command.detach().to(env.unwrapped.device)  # no manual action clipping/conversion
            audit.clear()
            attempts.append(dict(timestep=t, command=action.tolist(), step_returned=False))
            atomic(folder/'action_attempts.json', attempts)
            obs, _, terminated, truncated, info = env.step(tensor)
            attempts[-1].update(step_returned=True, controller={k:v.tolist() for k,v in audit.items()})
            atomic(folder/'action_attempts.json', attempts)
            physics.stepped(before)
            validate_controller_command(action, audit)
            success = bool(info['success'].item())
            presence = [bool((packet['object_id']==i).any()) for i in (0,1)]
            trace.append(dict(timestep=t, simulation_time_before=t/20, simulation_time_after=(t+1)/20,
                physics_steps_before=before[0], physics_steps_after=physics.count,
                actions=action.tolist(), env_step_input=numpy(tensor)[0].tolist(),
                controller={k:v.tolist() for k,v in audit.items()}, presence=presence,
                mask_present=masks.any(axis=(1,2)).tolist(), vision=timing,
                timings_ms=dict(vision_ms=timing['elapsed_ms'], **geometry_timing,
                                graph_ms=graph.timings_ms['graph_construction_total'], **model_timing),
                success=success, terminated=bool(terminated.item()), truncated=bool(truncated.item())))
            videos.append(overlay(row, masks, graph, t, action, success, presence))
            result.update(episode_length=t+1, success=success, terminated=bool(terminated.item()),
                          truncated=bool(truncated.item()), timeout=bool(truncated.item()))
            if success:
                result['first_success_step'] = t+1
            if success or result['terminated'] or result['truncated']:
                break
        rows.append(observation(env, obs, normalizer.schema, config['vision']['camera']))
        result.update(status='CONTRACT_SMOKE' if smoke else 'COMPLETE',
                      end_reason='success' if result['success'] else 'timeout' if result['truncated'] else
                      'termination' if result['terminated'] else 'smoke_limit' if smoke else 'horizon')
    except Exception as exc:
        result.update(status='ERROR', error_type=type(exc).__name__, error=str(exc),
                      end_reason='runtime_error')
    finally:
        physics.close()
        if rows:
            save_npz(folder/'observations.npz', **{k:np.stack([r[k] for r in rows]) for k in rows[0]},
                     timestep=np.arange(len(rows)), simulation_time=np.arange(len(rows))/20)
        if masks_saved:
            save_npz(folder/'masks.npz', masks=np.stack(masks_saved), timestep=np.arange(len(masks_saved)))
        if hidden_saved:
            save_npz(folder/'hidden.npz', hidden=np.stack(hidden_saved), timestep=np.arange(len(hidden_saved)),
                     episode_identity=np.array(identity), restoration='post_observation_pre_action')
        if trace:
            save_npz(folder/'actions.npz', actions=np.array([r['actions'] for r in trace], np.float32),
                     controller_input=np.array([r['controller']['controller_input'] for r in trace], np.float32))
        atomic(folder/'trace.json', trace)
        atomic(folder/'vision_initialization.json', dict(found=vision.found, detector_calls=vision.detector_calls))
        if videos:
            writer = cv2.VideoWriter(str(folder/'rollout.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (512,512))
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
            try:
                for image in videos:
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            cv2.imwrite(str(folder/'representative.png'), cv2.cvtColor(videos[-1], cv2.COLOR_RGB2BGR))
        if trace:
            result.update(mean_presence=np.mean([r['presence'] for r in trace], axis=0).tolist(),
                missing_mask_fraction=(1-np.mean([r['mask_present'] for r in trace], axis=0)).tolist(),
                latency_ms={key:float(np.mean([r['timings_ms'][key] for r in trace])) for key in trace[0]['timings_ms']},
                robot_state_changed=not np.array_equal(rows[0]['robot_state'], rows[-1]['robot_state']))
        result.update(wall_seconds=perf_counter()-tick, simulation_seconds=len(trace)/20,
                      wall_control_steps_per_second=len(trace)/max(perf_counter()-tick, 1e-9),
                      physics_calls=physics.count, hidden_steps=len(hidden_saved),
                      dino_initializations=sum(len(r['new_object_ids']) for r in vision.records),
                      dino_detector_calls=vision.detector_calls,
                      dino_initialization_scan_events=vision.scan_events,
                      dino_research_scan_events=max(0,vision.scan_events-1))
        stream.reset()
        vision.reset(None)
        result['hidden_reset_events'].append(dict(episode=identity,timestep=len(trace),reason='episode_end'))
        atomic(folder/'episode.json', result)
    return result


def time_contract(env, config):
    """Actual 250-step reset/TimeLimit probe, without perception or an expert."""
    env.reset(seed=config['time_contract_seed'])
    physics = PhysicsAudit(env)
    try:
        from time import sleep
        token = physics.token()
        sleep(.07)  # deliberately longer than 50ms, no deadline or env.step
        physics.unchanged(token)
        for t in range(250):
            before = physics.token()
            _, _, terminated, truncated, _ = env.step(torch.zeros((1,7), device=env.unwrapped.device))
            physics.stepped(before)
            if bool(truncated.item()) != (t == 249):
                raise ValueError('evaluation TimeLimit must truncate at 250')
            if terminated.item():
                raise ValueError('unexpected termination in time-contract probe')
        return dict(status='PASS', steps=250, actual_scene_step_calls=physics.count,
                    simulation_seconds=12.5, inference_delay_probe_ms=70,
                    inference_physics_steps=0, horizon=env._max_episode_steps)
    finally:
        physics.close()


def compare_saved_sequence(rgb, detector, tracker, output):
    """Recorded RGB is revealed one frame at a time; separate from live smoke."""
    from graph.integration_baseline.vision import causal_masks
    online = OnlineVisionAdapter(detector, tracker)
    values = []
    try:
        online.reset('recorded_prefix')
        for t in range(len(rgb)):
            values.append(online.process(rgb[t], 'recorded_prefix', t)[0])
        initialization = online.found.copy()
    finally:
        online.close()
    offline, meta = causal_masks(rgb, detector, tracker)
    current = np.stack(values)
    differences = np.count_nonzero(current != offline, axis=(2,3))
    presence_diff = current.any(axis=(2,3)) != offline.any(axis=(2,3))
    save_npz(output/'recorded_vision_comparison.npz', online=current, offline=offline,
             differing_pixels=differences, presence_differences=presence_diff)
    result = dict(status='PASS' if not presence_diff.any() and initialization == meta['initialization'] else 'CONTRACT_FAILURE',
                  frames=len(rgb), differing_pixels=differences.tolist(), presence_differences=presence_diff.tolist(),
                  initialization_equal=initialization == meta['initialization'],
                  scope='recorded real RGB revealed incrementally; not live simulator evidence')
    # A second independent stream sees a changed suffix only after the prefix.
    changed = OnlineVisionAdapter(detector, tracker)
    try:
        changed.reset('suffix_probe')
        prefix_values = []
        for t in range(len(rgb)):
            image = rgb[t] if t < len(rgb)-1 else np.zeros_like(rgb[t])
            prefix_values.append(changed.process(image, 'suffix_probe', t)[0])
        prefix_diff = int(np.count_nonzero(np.stack(prefix_values)[:-1] != current[:-1]))
        result['changed_suffix_prefix_differing_pixels'] = prefix_diff
        if prefix_diff:
            result['status'] = 'CONTRACT_FAILURE'
    finally:
        changed.close()
    result['late_object_actual_sam2'] = late_object_probe(rgb, initialization, detector, tracker, output)
    if result['late_object_actual_sam2']['status'] == 'CONTRACT_FAILURE':
        result['status'] = 'CONTRACT_FAILURE'
    atomic(output/'recorded_vision_comparison.json', result)
    return result


def late_object_probe(rgb, initialization, detector, tracker, output):
    """Controlled prompt fixture tests real SAM2 memory, not detection quality."""
    from copy import deepcopy
    from unittest.mock import patch
    from graph.integration_baseline.vision import causal_masks
    if any(initialization.get(name,{}).get('frame') != 0 for name in ('peg','box')):
        return dict(status='NOT_EXECUTED', reason='fixture needs both RGB detections at frame zero')
    fixture = np.stack([rgb[t % len(rgb)] for t in range(7)])

    def initial(frames, detector):
        result = deepcopy(initialization)
        result['peg']['frame'] = 0
        result['box']['frame'] = 5 if len(frames) >= 6 else None
        if len(frames) < 6:
            result['box']['box'] = None
        return result

    online = OnlineVisionAdapter(detector, tracker)
    values = []
    retained = True
    try:
        online.reset('late_object_fixture')
        with patch('graph.integration_baseline.online_vision_adapter.detect_initial', initial):
            original_state = original_memory = None
            old_keys = set()
            for t, image in enumerate(fixture):
                values.append(online.process(image, 'late_object_fixture', t)[0])
                state = online.states[(1,)]
                memory = state['output_dict_per_obj'][0]
                keys = set(memory['cond_frame_outputs']) | set(memory['non_cond_frame_outputs'])
                if t == 4:
                    original_state, original_memory, old_keys = state, memory, keys
                if t == 5:
                    retained = state is original_state and memory is original_memory and old_keys <= keys
            groups_equal = set(online.states) == {(1,),(2,)}
    finally:
        online.close()
    with patch('graph.integration_baseline.vision.detect_initial', initial):
        offline, _ = causal_masks(fixture, detector, tracker)
    values = np.stack(values)
    differences = np.count_nonzero(values != offline, axis=(2,3))
    presence_equal = np.array_equal(values.any(axis=(2,3)), offline.any(axis=(2,3)))
    success = retained and groups_equal and presence_equal and not values[:5,1].any()
    save_npz(output/'late_object_sam2_fixture.npz', online=values, offline=offline,
             differing_pixels=differences)
    return dict(status='PASS' if success else 'CONTRACT_FAILURE', previous_object_memory_retained=retained,
                object_groups_equal=groups_equal, presence_equal=presence_equal,
                differing_pixels=differences.tolist(),
                scope='REAL_SAM2_WITH_CONTROLLED_PROMPT_FIXTURE; seven repeated recorded RGB frames; '
                      'past frame-zero DINO boxes; delayed box prompt at t5; not live detection-quality evidence')


def validate_config(config):
    expected = dict(horizon=250, control_mode='pd_ee_delta_pose', sim_backend='physx_cpu',
                    control_freq=20, physics_freq=100)
    vision = dict(config['vision'])
    backend = vision.pop('backend', 'legacy_fp32')
    if backend not in ('legacy_fp32', 'visual_stream'):
        raise ValueError('unknown vision backend: ' + str(backend))
    if backend == 'visual_stream' and not config['vision_device'].startswith('cuda'):
        raise ValueError('visual_stream requires CUDA')
    if config['environment'] != expected or vision != dict(camera='base_camera', keep_ratio=.5, point_budget=None):
        raise ValueError('fixed simulator/Vision contract changed')
    if config['graph'] != dict(k=4, M=16, block_rows=64) or config['model'] != dict(d=64,L=2,gru_hidden=144,gru_layers=1,enc_hidden=128):
        raise ValueError('fixed graph/model contract changed')
    if not 1 <= config['smoke_steps'] <= 10 or not 1 <= len(config['smoke_seeds']) <= 3:
        raise ValueError('contract smoke limited to <=3 episodes, <=10 steps')
    all_smoke = config['smoke_seeds'] + [config['time_contract_seed']]
    if len(all_smoke) != len(set(all_smoke)) or set(all_smoke) & set(config['heldout_seeds']):
        raise ValueError('smoke and evaluation seeds must be distinct')
    groups = config.get('evaluation_groups', ['frozen_test', 'heldout'])
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)) or not set(groups) <= {'frozen_test','heldout'}:
        raise ValueError('evaluation_groups must explicitly select distinct frozen_test/heldout groups')


def execute(args):
    from utils.experiment_logger import ExperimentLogger
    from graph.integration_baseline import rollout_report
    os.environ.update(WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/task07-mpl')
    config = read(args.config)
    validate_config(config)
    smoke = args.mode == 'smoke'
    if not smoke and (args.checkpoint_manifest is None or args.seeds_manifest is None):
        raise ValueError('evaluate requires checkpoint AND frozen seeds manifests')
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    if config['model_device'].startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with ExperimentLogger(team='integration', model='stop_clock_graph_gnn_gru', seed=config['seed'],
                          config=dict(config, mode=args.mode), job_type='evaluate') as logger:
        run = logger.directory
        print('RUN_DIR='+str(run), flush=True)
        rollout_report.snapshot(run)
        report = dict(official_name=OFFICIAL_NAME, status='RUNNING', mode=args.mode, run_id=logger.run_id,
                      trained_policy_evaluated=False, episodes=[], contracts={},
                      remote_training_accessed=config.get('execution_context',{}).get('remote_training_accessed',False),
                      dependency_installation=config.get('execution_context',{}).get('dependency_installation',False),
                      training_process_modified=False)
        env = vision = None
        failed = False
        try:
            if smoke:
                command = [sys.executable, '-m', 'unittest',
                    'graph.integration_baseline.tests.test_stop_clock',
                    'graph.integration_baseline.tests.test_online_vision',
                    'graph.integration_baseline.tests.test_rollout_checkpoint',
                    'graph.integration_baseline.tests.test_baseline', '-v']
                tests = subprocess.run(command, capture_output=True, text=True)
                (run/'contract_tests.log').write_text(tests.stdout+tests.stderr)
                report['contracts']['fixtures'] = dict(status='PASS' if tests.returncode == 0 else 'FAIL', command=command)
                if tests.returncode:
                    raise RuntimeError('contract tests failed')
            from envs.maniskill import make_peg_insertion_env
            from mani_skill.utils.registration import TimeLimitWrapper
            env = make_peg_insertion_env(obs_mode='rgb+depth', control_mode='pd_ee_delta_pose', device='cpu')
            env = TimeLimitWrapper(env.unwrapped, 250)
            env.reset(seed=config['smoke_seeds'][0])
            schema = state_schema(robot_observation(env)[0])
            task = env.unwrapped
            conditions = dict(control_freq=task.control_freq, physics_freq=task.sim_freq,
                physics_substeps=task._sim_steps_per_control, horizon=env._max_episode_steps,
                physics_timestep=float(task.scene.px.timestep),
                backend=str(task.backend), robot=task.agent.uid,
                controller_config={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()},
                model_device=config['model_device'], vision_device=config['vision_device'],
                cuda='EXECUTED' if any(config[k].startswith('cuda') for k in ('model_device','vision_device')) else 'NOT_EXECUTED',
                mps='NOT_EXECUTED')
            from recurrent.common import versions
            conditions['versions'] = versions()
            if (conditions['control_freq'],conditions['physics_freq'],conditions['physics_substeps'],schema['dimension']) != (20,100,5,76):
                raise ValueError('installed simulator contract mismatch')
            if not np.isclose(conditions['physics_timestep'], .01, atol=1e-9, rtol=0):
                raise ValueError('actual physics timestep must be 10ms')
            atomic(run/'executed_conditions.json', conditions)
            report['executed_conditions'] = conditions
            atomic(run/'robot_state_schema.json', schema)
            from mani_skill.envs.sapien_env import BaseEnv
            from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            from sam2.utils import misc
            installed = run/'installed_source'; installed.mkdir()
            for cls in (BaseEnv, PDEEPoseController, SAM2VideoPredictor, misc):
                path = Path(inspect.getfile(cls)); shutil.copy2(path, installed/path.name)
            if smoke and args.checkpoint_manifest is None:
                model = IntegrationPolicy(schema, **config['model']).to(config['model_device']).eval()
                normalizer = RobotNormalizer(schema, StateNormalizer(torch.zeros(76),torch.ones(76)), [])
                specs = [dict(seed=s, group='smoke') for s in config['smoke_seeds']]
                report['policy'] = 'RANDOM_WEIGHTS_CONTRACT_ONLY; identity normalizer; not trained performance'
                report['contracts']['environment_time'] = time_contract(env, config)
                forbidden = []
            else:
                model, normalizer, provenance = load_bundle(args.checkpoint_manifest,
                    device=config['model_device'], eval_config=config)
                if normalizer.schema != schema:
                    raise ValueError('checkpoint/live robot schema mismatch')
                trained_conditions = provenance['conditions']
                for key in ('control_freq','physics_freq','horizon','robot'):
                    if trained_conditions.get(key) != conditions[key]:
                        raise ValueError('training/live simulator conditions mismatch: '+key)
                if controller_semantics(trained_conditions.get('controller_config',{})) != controller_semantics(conditions['controller_config']):
                    raise ValueError('training/live controller semantics mismatch')
                for key in ('mani_skill','sapien','gymnasium'):
                    if trained_conditions.get('versions',{}).get(key) != conditions['versions'][key]:
                        raise ValueError('training/live simulator version mismatch: '+key)
                atomic(run/'checkpoint_provenance.json', provenance)
                report['source_run_id'] = provenance['manifest']['source_run_id']
                if smoke:
                    specs = [dict(seed=s, group='smoke') for s in config['smoke_seeds']]
                    forbidden = []
                    report['policy'] = 'STRICT_LOADED_TRAINED_CHECKPOINT_CONTRACT_SMOKE; excluded from performance'
                    report['contracts']['environment_time'] = time_contract(env, config)
                else:
                    seed_manifest = read(args.seeds_manifest)
                    all_specs = validate_seeds(seed_manifest, provenance, config)
                    selected_groups = config.get('evaluation_groups', ['frozen_test', 'heldout'])
                    specs = [s for s in all_specs if s['group'] in selected_groups]
                    report['evaluation_groups'] = selected_groups
                    report['excluded_by_evaluation_scope'] = [dict(s,status='NOT_EXECUTED',
                        reason='group excluded by config frozen before performance execution')
                        for s in all_specs if s['group'] not in selected_groups]
                    atomic(run/'seeds_manifest.json', seed_manifest)
                    forbidden = [g for values in provenance['split']['reset_groups'].values() for g in values]
                    report['policy'] = 'STRICT_LOADED_TRAINED_CHECKPOINT'
            report['planned_seeds'] = specs
            report['episodes'] = [dict(s,status='NOT_EXECUTED') for s in specs]
            from visual.runtime.grounding import GDino
            from visual.runtime.tracking import SAM2Tracker
            try:
                runtime_identity = vision_runtime_identity()
                atomic(run/'vision_runtime.json', runtime_identity)
                from graph.integration_baseline.streaming_vision_adapter import make_online_vision
                vision = make_online_vision(config)
                if config['vision'].get('backend', 'legacy_fp32') == 'legacy_fp32':
                    detector, tracker = vision.detector, vision.tracker
            except Exception as exc:
                report['contracts']['live_vision'] = dict(status='BLOCKED_RUNTIME_VISION', error_type=type(exc).__name__)
                raise
            for index, spec in enumerate(specs):
                print(f"Live contract episode {index+1}/{len(specs)} seed={spec['seed']}", flush=True)
                folder = run/'episodes'/f"{spec['group']}_{spec['seed']}"
                try:
                    episode = run_episode(env, model, normalizer, vision, spec, config,
                        folder, smoke=smoke, forbidden_groups=forbidden)
                except Exception as exc:
                    episode = dict(spec,status='ERROR',error_type=type(exc).__name__,error=str(exc))
                    atomic(folder/'episode.json',episode)
                report['episodes'][index] = episode
                atomic(run/'rollout_results.json', report)
                if episode['status'] == 'ERROR':
                    raise RuntimeError('live closed-loop contract failed: '+episode.get('error','unknown'))
                if not smoke:
                    logger.log(dict(success=int(episode['success']),episode_length=episode['episode_length']), step=index)
            report['contracts']['live_vision'] = dict(status='PASS', source='new RGB from reset/each env.step')
            report['contracts']['live_closed_loop'] = dict(status='PASS', random_policy=smoke and args.checkpoint_manifest is None,
                physics_during_inference=0, physics_per_action=5, controller_entry_equal=True)
            if smoke:
                with np.load(run/'episodes'/f"smoke_{config['smoke_seeds'][0]}"/'observations.npz') as saved:
                    rgb = saved['rgb'].copy()
                if config['vision'].get('backend', 'legacy_fp32') == 'visual_stream':
                    from graph.integration_baseline.streaming_vision_checks import compare_stream_sequence
                    report['contracts']['recorded_vision'] = compare_stream_sequence(
                        rgb, vision.initializer.detector.detector, vision.tracker, run)
                else:
                    report['contracts']['recorded_vision'] = compare_saved_sequence(rgb, detector, tracker, run)
                if report['contracts']['recorded_vision']['status'] != 'PASS':
                    raise RuntimeError('recorded online/offline Vision contract failure')
            report.update(status='SMOKE_COMPLETE' if smoke else 'COMPLETE', trained_policy_evaluated=not smoke)
        except Exception as exc:
            failed = True
            report.update(status='BLOCKED_OR_FAILED', error_type=type(exc).__name__, error=str(exc))
        finally:
            if vision is not None:
                vision.close()
            if env is not None:
                env.close()
            report['metrics'] = summarize(report['episodes'], smoke=smoke)
            if failed and not smoke:
                report['metrics']['status'] = 'INCOMPLETE'
            if args.protection_snapshot:
                before = read(args.protection_snapshot/'protection.json')
                report['protection'] = dict(changed_existing_artifacts=[p for p,h in before['artifacts'].items() if sha(p)!=h],
                    changed_existing_source=[p for p,h in before['tracked'].items() if Path(p).is_file() and sha(p)!=h],
                    remote_training='NOT_ACCESSED; no remote process state claim')
                atomic(run/'start_identity.json', read(args.protection_snapshot/'git.json'))
            rollout_report.render(run, report)
            verification = rollout_report.verify(run)
            atomic(run/'verification.json', verification)
            rollout_report.render(run, report)
            rollout_report.verify(run)
            for path in sorted(run.rglob('*')):
                if path.is_file() and 'source_snapshot' not in path.parts and path.suffix in {'.json','.npz','.png','.mp4','.html','.log','.diff'} and not path.name.endswith('.tracking.json'):
                    logger.link_output(path, kind='rollouts' if path.suffix=='.mp4' else 'trajectory' if path.suffix=='.npz' else 'reports')
            logger.record(rollout_status=report['status'], trained_policy_evaluated=report['trained_policy_evaluated'],
                          environment=report.get('executed_conditions',{}),
                          source_run_id=report.get('source_run_id'))
            if failed:
                logger.finish(failed=True)
        print('RESULT='+report['status'], flush=True)
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'evaluate', 'prepare-seeds'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint-manifest', type=Path)
    parser.add_argument('--seeds-manifest', type=Path)
    parser.add_argument('--protection-snapshot', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare-seeds':
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        config = read(args.config); validate_config(config)
        if not args.checkpoint_manifest or not args.seeds_manifest:
            parser.error('prepare-seeds requires checkpoint manifest and seeds output path')
        _, _, provenance = load_bundle(args.checkpoint_manifest, eval_config=config)
        atomic(args.seeds_manifest, prepare_seeds(provenance, config), immutable=True)
        return 0
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
