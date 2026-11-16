"""Branch from pilot trajectories and collect expert recovery after random actions."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from data_collection.collection_batches import collection_defaults, iter_batches
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.mixed_trajectory_collection import (
    MODES,
    camera_sensors,
    initial_fingerprint,
    random_action,
    validate_camera,
)
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.storage import save_frame, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.camera_visibility_search import score_frame
from utils.experiment_logger import ExperimentLogger


MODE = 'expert_random_expert'


def verified_frame(root, row, index):
    """Return the original NPZ bytes after verifying the parent frame identity."""
    path = root / row['path'] / f'frame_{index:06d}.npz'
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != row['frame_sha256'][index]:
        raise ValueError(f'Pilot frame SHA-256 mismatch: {path}')
    return contents


def last_phase_action(root, row, phase):
    """Find the last applied action in a phase whose planned delay did not fit."""
    for index in range(row['steps'] - 1, 0, -1):
        with np.load(io.BytesIO(verified_frame(root, row, index)), allow_pickle=False) as data:
            if str(data['collection/phase']) == phase:
                return index
    return None


def plan_cases(root, manifest, protocol, random_steps, missing_phase_switch_steps=None):
    """Keep the pilot schedule, advancing missed phases and supplying a nonempty expert prefix."""
    if manifest['status'] != 'COMPLETE' or file_sha256(root / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Expected a complete pilot with its original protocol')
    episodes = manifest['episodes']
    by_case = {(row['seed'], row['mode']): row for row in episodes if row['split'] == 'pilot'}
    records = [row for row in protocol['records'] if row['split'] == 'pilot']
    expected_count = len(records)
    if 'hole_motion' in protocol and 'record_range' in manifest['effective_config']:
        start, stop = manifest['effective_config']['record_range']
        records = records[start:stop]
        expected_count = stop - start
    elif expected_count != 100:
        raise ValueError('Expected the original 100-seed fixed-Hole pilot')
    seeds = {row['seed'] for row in records}
    if (len(records) != expected_count or len(seeds) != expected_count
            or set(by_case) != {(s, m) for s in seeds for m in MODES}):
        raise ValueError('The source must contain all three modes for every selected pilot seed')
    if len(episodes) != len(by_case):
        raise ValueError('Expected exactly one source episode per pilot seed and mode')

    cases = []
    for record in records:
        seed = record['seed']
        parent = by_case[seed, 'expert_random']
        switch = parent['switch_step']
        adjustment = None
        if switch is None:
            switch = last_phase_action(root, parent, record['transition_phase'])
            adjustment = 'last_action_in_requested_phase'
            if switch is None:
                reference_step = (missing_phase_switch_steps or {}).get(seed)
                if reference_step is None:
                    raise ValueError(f'No action in the requested phase for {parent["path"]}')
                switch = min(reference_step, parent['steps'] - 1)
                adjustment = 'fixed_hole_switch_step_after_unreached_phase'
        elif switch == 0:
            parent = by_case[seed, 'expert']
            switch = 1
            adjustment = 'one_initial_expert_action'
        recovery = switch + random_steps
        copy_count = recovery if adjustment is None else switch
        if copy_count >= parent['steps'] and switch < parent['steps'] and 'hole_motion' in protocol:
            copy_count = switch
            adjustment = 'replay_expert_prefix_only'
        if not 0 < copy_count < parent['steps']:
            raise ValueError(f'Pilot prefix must end before termination: {parent["path"]}')
        cases.append(dict(record=record, parent=parent, switch_step=switch, recovery_step=recovery,
                          prefix_frames=copy_count, switch_adjustment=adjustment))
    return cases


def replay_prefix(env, observation, root, pending, case, motion=None):
    """Apply exactly the original controls and verify the live branch-point state."""
    parent = case['parent']
    for index in range(case['prefix_frames']):
        contents = verified_frame(root, parent, index)
        with np.load(io.BytesIO(contents), allow_pickle=False) as data:
            if not bool(data['action_valid']) or int(data['collection/frame']) != index:
                raise ValueError(f'Invalid pilot action at {parent["path"]}/{index}')
            action = torch.as_tensor(data['action'], device=env.unwrapped.device)
        (pending / f'frame_{index:06d}.npz').write_bytes(contents)
        observation, _, terminated, truncated, _ = env.step(action)
        if bool(terminated.any() or truncated.any()):
            raise RuntimeError(f'Unexpected pilot prefix termination at {parent["path"]}/{index}')
        if motion and motion.advance(index + 1, None):
            task = env.unwrapped
            observation = task.get_obs(task.get_info())

    boundary = case['prefix_frames']
    state = get_gt_state(env)
    with np.load(io.BytesIO(verified_frame(root, parent, boundary)), allow_pickle=False) as data:
        for key in ('qpos', 'peg_pose', 'hole_pose'):
            if not np.array_equal(data[f'ground_truth/{key}'], getattr(state, key).cpu().numpy()):
                raise ValueError(f'Replay mismatch at {parent["path"]}/{boundary}: {key}')
    return observation, float(action.reshape(-1)[-1])


def save_observation(pending, observation, state, action, valid, status, *, compress=True):
    """Write the same base-camera and raw action schema as the pilot."""
    stored = dict(sensor_data={'base_camera': observation['sensor_data']['base_camera']},
                  sensor_param={'base_camera': observation['sensor_param']['base_camera']},
                  agent=observation['agent'], extra={'tcp_pose': observation['extra']['tcp_pose']})
    path = pending / f'frame_{status["frame"]:06d}.npz'
    save_frame(path, stored, compress=compress, ground_truth=vars(state), action=action,
               action_valid=np.asarray(valid), collection=status)
    return file_sha256(path)


def collect_tail(env, observation, pending, case, protocol, recovery_steps, last_gripper, motion=None, *, compress=True):
    """Apply the perturbation when needed, then restart the scripted expert in place."""
    record, parent = case['record'], case['parent']
    switch, recovery = case['switch_step'], case['recovery_step']
    steps = case['prefix_frames']
    frame_hashes = list(parent['frame_sha256'][:steps])
    samples = [row for row in parent['hole_visibility_samples'] if row['frame'] < steps]
    rng = np.random.default_rng(np.random.SeedSequence([record['action_seed'], MODES.index('expert_random')]))
    task, expert = env.unwrapped, ScriptedPegPolicy()
    grip = last_gripper
    restarted = terminated = truncated = False
    while True:
        state = get_gt_state(env)
        success = bool(state.success[0])
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        valid = not (terminated or truncated or success or steps >= recovery + recovery_steps)
        phase = expert.phase if restarted else 'random'
        if not valid:
            action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=task.device)
        elif steps < recovery:
            if record['random_gripper'] == 'random' and (steps - switch) % 15 == 0:
                grip = float(rng.choice([-1, 1]))
            action = random_action(rng, record['random_limit'], grip, task)
        else:
            if not restarted:
                expert.reset(env, record['seed'])
                restarted = True
            phase = expert.phase
            action = expert.act(observation)
        if steps % protocol['visibility_stride'] == 0 or not valid:
            sample = score_frame(observation, task, 7, 0.003)
            samples.append(dict(sample, frame=steps, phase=phase))
        status = dict(frame=steps, phase=phase, switched=True, grasp=grasp, success=success,
                      terminated=terminated, truncated=truncated)
        frame_hashes.append(save_observation(
            pending, observation, state, action, valid, status, compress=compress
        ))
        if not valid:
            break
        observation, _, term, trunc, _ = env.step(action)
        terminated, truncated = bool(term.any()), bool(trunc.any())
        steps += 1
        if motion and not (terminated or truncated) and motion.advance(steps, None):
            observation = task.get_obs(task.get_info())

    reason = 'terminated' if terminated else 'truncated' if truncated else 'success' if success else 'step_limit'
    return dict(observations=len(frame_hashes), steps=steps, successful=success, stop_reason=reason,
                restarted_expert=restarted, expert_recovery_actions=max(0, steps - recovery),
                random_actions=min(steps, recovery) - switch,
                hole_visibility_samples=samples, frame_sha256=frame_hashes)


def collect_case(root, output, case, protocol, recovery_steps, render_backend='cpu', *, compress=True):
    """Collect one original scene, retaining failed outcomes and terminal observations."""
    record, parent = case['record'], case['parent']
    seed = record['seed']
    name = f'{record["split"]}_{seed}_{MODE}'
    pending = output / f'{name}.partial'
    pending.mkdir()
    started = time.monotonic()
    torch.set_num_threads(1)
    env = make_peg_insertion_env(seed=seed, device='cpu', render_backend=render_backend,
        observation_device='cpu' if render_backend == 'cpu' else None,
        obs_mode='rgb+depth+segmentation', sensor_configs=camera_sensors(protocol['camera']),
        control_mode=protocol['control_mode'], max_episode_steps=protocol['max_steps'] + recovery_steps)
    try:
        observation, _ = env.reset(seed=seed)
        validate_camera(observation, protocol['camera'])
        fingerprint = initial_fingerprint(observation, get_gt_state(env))
        if fingerprint != parent['initial_fingerprint']:
            raise ValueError(f'Reset fingerprint differs from the pilot for seed {seed}')
        settings = protocol.get('hole_motion')
        if settings and settings.get('stop_expert_insert', False):
            raise ValueError('Moving-Hole recovery requires a schedule independent of expert phase')
        motion = EpisodeHoleMotion(env, settings, None) if settings else None
        observation, grip = replay_prefix(env, observation, root, pending, case, motion)
        tail = collect_tail(env, observation, pending, case, protocol, recovery_steps, grip, motion,
                            compress=compress)
    finally:
        env.close()

    receipt = dict(seed=seed, split=record['split'], mode=MODE, path=name,
                   parent_episode=parent['path'], parent_prefix_frames=case['prefix_frames'],
                   switched=True, switch_step=case['switch_step'], recovery_step=case['recovery_step'],
                   switch_adjustment=case['switch_adjustment'], initial_fingerprint=fingerprint,
                   initial_peg_visibility=parent['initial_peg_visibility'],
                   elapsed_seconds=time.monotonic() - started, **tail)
    write_json(pending / 'episode.json', receipt)
    pending.rename(output / name)
    return receipt


def collect_cpu_cases(raw, output, cases, protocol, config, errors):
    """Run independent CPU recovery branches and retain per-case failures."""
    with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        for batch in iter_batches(cases, config['batch_episodes']):
            jobs = [pool.submit(collect_case, raw, output, case, protocol, config['recovery_steps'],
                                config.get('render_backend') or 'cpu',
                                compress=config.get('compress_frames', True)) for case in batch]
            for job in as_completed(jobs):
                try:
                    yield job.result()
                except Exception as error:
                    errors.append(str(error))
                    print(f'Collection error: {error}', flush=True)


def collect(config):
    """Freeze the branch plan and collect each selected case without success filtering."""
    config = collection_defaults(config)
    raw = Path(config['pilot_root'])
    source_manifest = raw / 'manifest.json'
    source_hash = file_sha256(source_manifest)
    if source_hash != config['pilot_manifest_sha256']:
        raise ValueError(f'Pilot manifest SHA-256 mismatch: {source_manifest}')
    components = ('random_steps', 'recovery_steps', 'workers')
    if any(type(config[key]) is not int or config[key] < 1 for key in components):
        raise ValueError('random_steps, recovery_steps and workers must be positive integers')
    manifest = json.loads(source_manifest.read_text())
    protocol = json.loads((raw / 'protocol.json').read_text())
    cases = plan_cases(raw, manifest, protocol, config['random_steps'],
                       config.get('missing_phase_switch_steps'))
    output = Path(config['output_root'])
    reserve = config['minimum_free_gib'] * 1024**3
    if shutil.disk_usage(output.parent).free < reserve:
        raise RuntimeError('Insufficient free disk for recovery collection')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(protocol, modes=[MODE], records=[{
        **case['record'], 'source_episode': case['parent']['path'],
        **{key: case[key] for key in ('switch_step', 'recovery_step', 'prefix_frames', 'switch_adjustment')},
    } for case in cases], parent_manifest_sha256=source_hash, parent_protocol_sha256=manifest['protocol_sha256'],
                random_steps=config['random_steps'], recovery_steps=config['recovery_steps'])
    protocol_path, manifest_path = output / 'protocol.json', output / 'manifest.json'
    write_json(protocol_path, plan)
    order = {row['seed']: index for index, row in enumerate(plan['records'])}
    index = dict(status='RUNNING', protocol_sha256=file_sha256(protocol_path), parent_manifest_sha256=source_hash,
                 effective_config=config, source_sha256=file_sha256(Path(__file__)), episodes=[])
    write_json(manifest_path, index)
    with ExperimentLogger(team='graph', model='mixed_trajectory_recovery', config=config,
                          seed=protocol['generator'], job_type='collection') as logger:
        errors = []
        if config.get('physics_device') == 'cuda':
            from data_collection.batched_mixed_trajectory import collect_batches
            receipts = collect_batches([case['record'] for case in cases], config, protocol, output,
                                       recovery_cases=cases, parent_root=raw)
        else:
            receipts = collect_cpu_cases(raw, output, cases, protocol, config, errors)
        try:
            for receipt in receipts:
                index['episodes'].append(receipt)
                index['episodes'].sort(key=lambda row: order[row['seed']])
                write_json(manifest_path, index)
                print(f'{len(index["episodes"])}/{len(cases)} seed={receipt["seed"]} '
                      f'frames={receipt["observations"]} recovered={receipt["successful"]}', flush=True)
        except Exception as error:
            errors.append(str(error))
            print(f'Collection error: {error}', flush=True)
        index['status'] = 'ERROR' if errors else 'COMPLETE'
        if errors:
            index['errors'] = errors
        write_json(manifest_path, index)
        logger.record(episodes=len(index['episodes']), parent_manifest_sha256=source_hash,
                      protocol_sha256=index['protocol_sha256'])
        logger.link_output(manifest_path)
        logger.link_output(protocol_path)
        if errors:
            raise RuntimeError(f'Recovery collection had {len(errors)} technical errors. See {manifest_path}')
