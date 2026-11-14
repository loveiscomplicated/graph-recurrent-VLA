"""Collect unfiltered, split-safe expert and random ManiSkill trajectories."""

import hashlib
import importlib.metadata
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from mani_skill.utils import sapien_utils

from data_collection.collect_episodes import initial_visibility
from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.storage import save_frame, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.camera_visibility_search import score_frame
from utils.experiment_logger import ExperimentLogger

PHASES = ('reach', 'descend', 'close', 'lift', 'align', 'insert')
MODES = ('expert', 'random', 'expert_random')


def validate_config(config):
    """Reject invalid collection settings before creating any output."""
    if not config['collect_splits'] or not set(config['collect_splits']) <= {'pilot', 'train', 'validation', 'test'}:
        raise ValueError('collect_splits must contain pilot, train, validation or test')
    if type(config['max_steps']) is not int or config['max_steps'] < 1:
        raise ValueError('max_steps must be a positive integer')
    if type(config['visibility_stride']) is not int or config['visibility_stride'] < 1:
        raise ValueError('visibility_stride must be a positive integer')
    if not config['random_limits'] or any(not 0 < value <= 1 for value in config['random_limits']):
        raise ValueError('random_limits must contain action magnitudes in (0, 1]')
    counts = [config['pilot_count'], *config['split_counts'].values()]
    if set(config['split_counts']) != {'train', 'validation', 'test'} or any(
        type(value) is not int or value < 1 for value in counts
    ):
        raise ValueError('pilot_count and train/validation/test split_counts must be positive integers')
    if config['camera']['width'] != 512 or config['camera']['height'] != 512:
        raise ValueError('camera must use 512×512 resolution for the raw image contract')
    eye = np.asarray(config['camera']['eye'], dtype=float)
    target = np.asarray(config['camera']['target'], dtype=float)
    if (eye.shape != (3,) or target.shape != (3,) or not np.isfinite(eye).all()
            or not np.isfinite(target).all() or np.allclose(eye, target)):
        raise ValueError('camera eye and target must be distinct finite 3D positions')
    if not 0 < config['camera']['fov_degrees'] < 180:
        raise ValueError('camera fov_degrees must be between 0 and 180')
    if config['minimum_free_gib'] <= 0:
        raise ValueError('minimum_free_gib must be positive')


def load_exclusions(config):
    """Read and validate the named historical seed inventories."""
    sources = config['exclusion_sources']
    training = json.loads(Path(sources['training']).read_text())
    original = set(training['original_collection_seeds'])
    graph = set(training['graph_development_seeds'])
    if len(original) != 900 or len(graph) != 630 or not graph <= original:
        raise ValueError('Training exclusion inventory must contain 900 original and 630 Graph seeds')
    other = json.loads(Path(sources['additional']).read_text())
    additional = {row['seed'] for row in other['episodes']}
    if len(additional) != other['episode_count']:
        raise ValueError('Additional exclusion inventory has duplicate or missing seeds')
    excluded = original | additional
    for name in ('held_out', 'matched_held_out'):
        excluded.update(json.loads(Path(sources[name]).read_text())['seeds'])
    excluded.update(json.loads(Path(sources['matched_candidates']).read_text())['candidates'])
    for start, stop in config['excluded_ranges']:
        excluded.update(range(start, stop + 1))
    return excluded, {name: file_sha256(path) for name, path in sources.items()}


def build_protocol(config):
    """Freeze all pilot/main seeds, splits, and random schedules before collection."""
    excluded, hashes = load_exclusions(config)
    generator = np.random.default_rng(config['seed_generator'])
    low, high = config['seed_range']
    count = config['pilot_count'] + sum(config['split_counts'].values())
    if low < 0 or high <= low or count <= 0 or high - low < count:
        raise ValueError('seed_range and cohort counts must provide enough nonnegative seeds')
    if high - low - sum(low <= seed < high for seed in excluded) < count:
        raise ValueError('Not enough seeds remain in seed_range after exclusions')
    selected = []
    seen = set(excluded)
    while len(selected) < count:
        seed = int(generator.integers(low, high))
        if seed not in seen:
            seen.add(seed)
            selected.append(seed)
    splits = ['pilot'] * config['pilot_count']
    for name in ('train', 'validation', 'test'):
        splits.extend([name] * config['split_counts'][name])
    # Shuffle factorial settings separately in each split to avoid ordered allocation bias.
    records = []
    offset = 0
    for split in ('pilot', 'train', 'validation', 'test'):
        split_seeds = selected[offset:offset + splits.count(split)]
        offset += len(split_seeds)
        schedule = [(phase, strength, grip) for phase in PHASES
                    for strength in config['random_limits'] for grip in ('hold', 'random')]
        indices = np.resize(np.arange(len(schedule)), len(split_seeds))
        generator.shuffle(indices)
        for seed, index in zip(split_seeds, indices):
            phase, strength, grip = schedule[int(index)]
            records.append(dict(seed=seed, split=split, transition_phase=phase,
                                transition_delay=int(generator.integers(0, 10)),
                                random_limit=strength, random_gripper=grip,
                                action_seed=int(generator.integers(0, 2**32))))
    return dict(schema_version=1, generator=config['seed_generator'],
                exclusion_sha256=hashes, exclusions_count=len(excluded),
                limitation='Only named inventories and ranges are checked.',
                camera=config['camera'], control_mode=config['control_mode'],
                max_steps=config['max_steps'], visibility_stride=config['visibility_stride'],
                modes=list(MODES), records=records)


def camera_sensors(camera):
    """Construct an explicit 512-pixel base-camera pose and native hand camera."""
    pose = sapien_utils.look_at(eye=camera['eye'], target=camera['target'])
    return {'base_camera': {'width': camera['width'], 'height': camera['height'],
                            'fov': np.deg2rad(camera['fov_degrees']), 'pose': pose},
            'hand_camera': {'width': 512, 'height': 512}}


def validate_camera(observation, camera):
    """Verify saved calibration against the explicitly configured pose and field of view."""
    calibration = observation['sensor_param']['base_camera']
    expected_pose = sapien_utils.look_at(eye=camera['eye'], target=camera['target'])
    pose_matrix = expected_pose.to_transformation_matrix()[0].cpu().numpy()
    # SAPIEN's pose basis is forward/left/up; camera output is right/up/back.
    expected = pose_matrix.copy()
    expected[:3, :3] = np.column_stack((-pose_matrix[:3, 1], pose_matrix[:3, 2],
                                        -pose_matrix[:3, 0]))
    actual = calibration['cam2world_gl'][0].cpu().numpy()
    focal = camera['height'] / (2 * np.tan(np.deg2rad(camera['fov_degrees']) / 2))
    intrinsic = calibration['intrinsic_cv'][0].cpu().numpy()
    if not np.allclose(actual, expected, rtol=0, atol=1e-6):
        raise ValueError('Recorded base-camera pose does not match configured eye and target')
    if not np.allclose([intrinsic[0, 0], intrinsic[1, 1]], focal, rtol=0, atol=1e-3):
        raise ValueError('Recorded base-camera intrinsics do not match configured field of view')
    rgb = observation['sensor_data']['base_camera']['rgb']
    depth = observation['sensor_data']['base_camera']['depth']
    if tuple(rgb.shape) != (1, camera['height'], camera['width'], 3) or rgb.dtype != torch.uint8:
        raise ValueError('Expected raw base-camera RGB uint8 [1,H,W,3]')
    if tuple(depth.shape) != (1, camera['height'], camera['width'], 1) or depth.dtype != torch.int16:
        raise ValueError('Expected raw base-camera depth int16 [1,H,W,1]')


def initial_fingerprint(observation, gt):
    """Identify an initial physical state independently of the camera image."""
    digest = hashlib.sha256()
    values = [observation['agent']['qpos'], gt.peg_pose, gt.box_pose, gt.hole_pose,
              gt.peg_half_sizes, gt.hole_radius]
    for value in values:
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def random_action(rng, limit, gripper, task):
    values = np.concatenate((rng.uniform(-limit, limit, 6), [gripper])).astype(np.float32)
    return torch.as_tensor(values, device=task.device).reshape(task.action_space.shape)


def collect_episode(env, record, mode, protocol, root):
    """Write pre-action observations and a terminal observation for one trajectory."""
    seed = record['seed']
    rng = np.random.default_rng(np.random.SeedSequence([record['action_seed'], MODES.index(mode)]))
    name = f'{record["split"]}_{seed}_{mode}'
    pending = root / (name + '.partial')
    pending.mkdir()
    observation, _ = env.reset(seed=seed)
    validate_camera(observation, protocol['camera'])
    task = env.unwrapped
    policy = ScriptedPegPolicy()
    if mode != 'random':
        policy.reset(env, seed)
    state = get_gt_state(env)
    fingerprint = initial_fingerprint(observation, state)
    initial_peg_visibility, _ = initial_visibility(env, observation)
    phase_seen = Counter()
    switched = mode == 'random'
    switch_step = 0 if switched else None
    last_gripper = 1.0
    current_gripper = 1.0
    steps = 0
    frame_hashes = []
    samples = []
    successful = False
    terminated = truncated = False
    start_time = time.monotonic()
    while True:
        state = get_gt_state(env)
        success = bool(state.success[0])
        successful |= success
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        phase = policy.phase if mode != 'random' and not switched else 'random'
        if phase in PHASES:
            phase_seen[phase] += 1
        if (mode == 'expert_random' and not switched and phase == record['transition_phase']
                and phase_seen[phase] > record['transition_delay']):
            switched = True
            switch_step = steps
        if steps % protocol['visibility_stride'] == 0 or terminated or truncated or steps == protocol['max_steps']:
            sample = score_frame(observation, task, 7, 0.003)
            sample.update(frame=steps, phase=phase)
            samples.append(sample)
        action_valid = not (terminated or truncated or steps >= protocol['max_steps'])
        if not action_valid:
            action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=task.device)
        elif switched:
            random_step = steps - switch_step
            grip_mode = 'random' if mode == 'random' else record['random_gripper']
            if grip_mode == 'hold':
                current_gripper = last_gripper
            elif random_step % 15 == 0:
                current_gripper = float(rng.choice([-1, 1]))
            action = random_action(rng, record['random_limit'], current_gripper, task)
        else:
            action = policy.act(observation)
            last_gripper = float(action.reshape(-1)[-1])
        sensor = observation['sensor_data']['base_camera']
        camera = observation['sensor_param']['base_camera']
        stored_observation = dict(sensor_data={'base_camera': sensor},
                                  sensor_param={'base_camera': camera},
                                  agent=observation['agent'], extra={'tcp_pose': observation['extra']['tcp_pose']})
        path = pending / f'frame_{steps:06d}.npz'
        save_frame(path, stored_observation, ground_truth=vars(state), action=action,
                   action_valid=np.asarray(action_valid),
                   collection=dict(frame=steps, phase=phase, switched=switched,
                                   grasp=grasp, success=success,
                                   terminated=terminated, truncated=truncated))
        frame_hashes.append(file_sha256(path))
        if not action_valid:
            break
        observation, _, term, trunc, _ = env.step(action)
        steps += 1
        terminated, truncated = bool(term.any()), bool(trunc.any())
    reason = 'terminated' if terminated else 'truncated' if truncated else 'step_limit'
    receipt = dict(seed=seed, split=record['split'], mode=mode, path=name,
                   observations=len(frame_hashes), steps=steps, stop_reason=reason,
                   successful=successful, switched=switched, switch_step=switch_step,
                   transition_reached=record['transition_phase'] in phase_seen,
                   phase_counts=dict(phase_seen), initial_fingerprint=fingerprint,
                   initial_peg_visibility=initial_peg_visibility,
                   hole_visibility_samples=samples, frame_sha256=frame_hashes,
                   elapsed_seconds=time.monotonic() - start_time)
    write_json(pending / 'episode.json', receipt)
    pending.rename(root / name)
    return receipt


def check_main_preconditions(config, protocol):
    """Require the audited pilot and enough storage before collecting main splits."""
    if set(config['collect_splits']) == {'pilot'}:
        return
    audit_path = Path(config['pilot_audit'])
    pilot_root = Path(config['pilot_root'])
    audit = json.loads(audit_path.read_text())
    pilot_protocol_path = pilot_root / 'protocol.json'
    if audit['protocol_sha256'] != file_sha256(pilot_protocol_path):
        raise ValueError('Pilot audit does not match the saved pilot protocol')
    if json.loads(pilot_protocol_path.read_text()) != protocol:
        raise ValueError('Main collection protocol differs from the audited pilot protocol')
    if audit['initial_states'] != config['pilot_count'] or audit['episodes'] != config['pilot_count'] * 3:
        raise ValueError('Pilot audit must cover every planned pilot state and mode')
    destination = Path(config['output_root']).resolve()
    parent = destination.parent
    while not parent.exists():
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = int(audit['estimated_main_bytes'] * 1.2 + config['minimum_free_gib'] * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(f'Insufficient storage for main collection: {free_bytes} bytes available, '
                           f'{required_bytes} bytes required including reserve')


def collect(config):
    """Freeze the collection protocol, then collect and verify the requested cohort."""
    validate_config(config)
    protocol = build_protocol(config)
    check_main_preconditions(config, protocol)
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'protocol.json', protocol)
    protocol_hash = file_sha256(root / 'protocol.json')
    versions = {}
    for name in ('torch', 'mani_skill', 'sapien', 'gymnasium', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    records = [row for row in protocol['records'] if row['split'] in config['collect_splits']]
    if 'record_range' in config:
        start, stop = config['record_range']
        if not (0 <= start < stop <= len(records)):
            raise ValueError('record_range must be a nonempty half-open range within selected records')
        records = records[start:stop]
    torch.set_num_threads(2)
    episodes = []
    manifest = dict(status='RUNNING', protocol_sha256=protocol_hash, versions=versions,
                    effective_config=config, episodes=episodes)
    with ExperimentLogger(team='graph', model='mixed_trajectory_collection',
                          config=config, seed=config['seed_generator'], job_type='collection') as logger:
        try:
            for record in records:
                env = make_peg_insertion_env(seed=record['seed'], obs_mode='rgb+depth+segmentation',
                    sensor_configs=camera_sensors(config['camera']), control_mode=config['control_mode'],
                    max_episode_steps=config['max_steps'])
                try:
                    fingerprints = set()
                    for mode in MODES:
                        free = shutil.disk_usage(root).free
                        if free < config['minimum_free_gib'] * 1024**3:
                            raise RuntimeError('Available storage fell below minimum_free_gib; collection stopped')
                        episode = collect_episode(env, record, mode, protocol, root)
                        fingerprints.add(episode['initial_fingerprint'])
                        if len(fingerprints) != 1:
                            raise ValueError(f'Initial states differ across modes for seed {record["seed"]}')
                        episodes.append(episode)
                        write_json(root / 'manifest.json', manifest)
                        print(f'{len(episodes)}/{len(records)*3} {record["split"]} '
                              f'seed={record["seed"]} mode={mode} steps={episode["steps"]}', flush=True)
                finally:
                    env.close()
            manifest['status'] = 'COMPLETE'
        except BaseException:
            manifest['status'] = 'ERROR'
            raise
        finally:
            write_json(root / 'manifest.json', manifest)
            logger.record(protocol_sha256=protocol_hash, episodes=len(episodes), status=manifest['status'])
            logger.link_output(root / 'manifest.json')
            logger.link_output(root / 'protocol.json')
    print(f'Collection complete: {root}', flush=True)
