"""Collect mixed trajectories in independent lanes of a CUDA PhysX environment."""

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
from time import monotonic

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

from data_collection.collection_batches import DEFAULT_BATCH_EPISODES, DEFAULT_WORKERS
from data_collection.collect_episodes import initial_visibility_from_state
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import HoleMotion
from data_collection.mixed_trajectory_collection import MODES, PHASES, camera_sensors, validate_camera
from data_collection.scripted_policy import ScriptedPegPolicy
from data_collection.storage import save_frame, write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.camera_visibility_search import score_frame


def cpu_snapshot(tree):
    """Read a batch to CPU before stepping PhysX or reusing camera buffers."""
    if isinstance(tree, dict):
        return {key: cpu_snapshot(value) for key, value in tree.items()}
    return tree.detach().to('cpu', copy=True)


def slice_batch(tree, index):
    """Keep the singleton dimensions of the existing raw NPZ format."""
    if isinstance(tree, dict):
        return {key: slice_batch(value, index) for key, value in tree.items()}
    return tree[index:index + 1]


def save_with_hash(path, observation, state, action, valid, status):
    """Write and verify a raw frame using the common storage path."""
    save_frame(path, observation, ground_truth=state, action=action[None],
               action_valid=np.asarray(valid), collection=status)
    return file_sha256(path)


def reset_batch(env, entries, reference_root):
    """Reconfigure seed-specific geometry and install the paired CPU reset state."""
    seeds = [row['seed'] for row in entries]
    observation, _ = env.reset(seed=seeds, options={'reconfigure': True})
    if reference_root is None:
        return observation
    task = env.unwrapped
    fields = ('peg_pose', 'box_pose', 'qpos', 'qvel', 'peg_half_sizes', 'hole_radius', 'hole_pose')
    root = Path(reference_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    references = {(row['split'], row['seed']): row for row in manifest['episodes']
                  if row['mode'] == 'expert'}
    rows = []
    for row in entries:
        source = references[row['split'], row['seed']]
        path = root / source['path'] / 'frame_000000.npz'
        if file_sha256(path) != source['frame_sha256'][0]:
            raise ValueError(f'Reset reference frame SHA-256 mismatch: {path}')
        with np.load(path, allow_pickle=False) as frame:
            rows.append({key: frame[f'ground_truth/{key}'] for key in fields})
    reference = {key: torch.as_tensor(np.concatenate([row[key] for row in rows]), device=task.device)
                 for key in fields}
    for key, value in (('peg_half_sizes', task.peg_half_sizes), ('hole_radius', task.box_hole_radii)):
        torch.testing.assert_close(value, reference[key], rtol=0, atol=1e-7)
    task.peg.set_pose(reference['peg_pose'])
    task.box.set_pose(reference['box_pose'])
    task.agent.robot.set_qpos(reference['qpos'])
    task.agent.robot.set_qvel(reference['qvel'])
    task.scene._gpu_apply_all()
    task.scene.px.gpu_update_articulation_kinematics()
    task.scene._gpu_fetch_all()
    torch.testing.assert_close(task.box_hole_pose.raw_pose, reference['hole_pose'], rtol=0, atol=1e-6)
    return task.get_obs(task.get_info())


def setup_expert(task, index, state, policy):
    """Use the lane's collision mesh and live GPU pose for the shared grasp rule."""
    entity = task.peg._objs[index]
    component = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    mesh = get_component_mesh(component, to_world_frame=False)
    peg = sapien.Pose(state['peg_pose'][0, :3], state['peg_pose'][0, 3:])
    tcp = sapien.Pose(state['ee_pose'][0, :3], state['ee_pose'][0, 3:])
    mesh.apply_transform(peg.to_transformation_matrix())
    policy.reset_from_geometry(mesh.bounding_box_oriented, tcp.to_transformation_matrix()[:3, 1],
                               float(state['peg_half_sizes'][0, 0]), task.agent.build_grasp_pose)


class Lane:
    """Own one episode's policy state, output and optional recovery prefix."""

    def __init__(self, record, mode, root, settings, case):
        self.record, self.mode, self.case = record, mode, case
        self.name = f'{record["split"]}_{record["seed"]}_{mode}'
        self.pending = root / (self.name + '.partial')
        self.pending.mkdir()
        self.root = root
        self.policy = ScriptedPegPolicy()
        stream = 'expert_random' if case else mode
        self.rng = np.random.default_rng(np.random.SeedSequence([record['action_seed'], MODES.index(stream)]))
        self.motion = HoleMotion(settings) if settings else None
        self.stop_on_insert = settings.get('stop_expert_insert', False) if settings else False
        if self.motion and self.motion.start <= 0:
            raise ValueError('Moving-Hole collection needs a positive start_seconds')
        if not isinstance(self.stop_on_insert, bool):
            raise ValueError('stop_expert_insert must be a boolean')
        self.switch = case['switch_step'] if case else 0 if mode == 'random' else None
        self.restarted = False
        self.grip = self.last_grip = 1.0
        self.phases = Counter()
        self.hashes, self.samples = [], []
        self.successful = self.done = False
        self.started = monotonic()

    def random_action(self, step):
        """Preserve the CPU collector's seed, draw order and gripper cadence."""
        offset = step - self.switch
        mode = 'random' if self.mode == 'random' else self.record['random_gripper']
        if mode == 'hold':
            self.grip = self.last_grip
        elif offset % 15 == 0:
            self.grip = float(self.rng.choice([-1, 1]))
        values = self.rng.uniform(-self.record['random_limit'], self.record['random_limit'], 6)
        return np.r_[values, self.grip].astype(np.float32)

    def action(self, task, index, state, goal, step, valid):
        """Choose one action with the mode's original phase transition rules."""
        if self.case:
            phase = self.policy.phase if self.restarted else 'random'
            if valid and step >= self.case['recovery_step'] and not self.restarted:
                setup_expert(task, index, state, self.policy)
                self.restarted = True
                phase = self.policy.phase
        else:
            phase = 'random' if self.switch is not None else self.policy.phase
            if phase in PHASES:
                self.phases[phase] += 1
            if (self.mode == 'expert_random' and self.switch is None
                    and phase == self.record['transition_phase']
                    and self.phases[phase] > self.record['transition_delay']):
                self.switch = step
        if not valid:
            return np.zeros(7, dtype=np.float32), phase
        if self.switch is not None and not self.restarted:
            return self.random_action(step), phase
        tcp = state['ee_pose'][0].numpy()
        peg = state['peg_pose'][0].numpy()
        action = self.policy.act_from_poses(sapien.Pose(tcp[:3], tcp[3:]),
                                            sapien.Pose(peg[:3], peg[3:]),
                                            sapien.Pose(goal[:3], goal[3:]))
        self.last_grip = float(action[-1])
        return action, phase

    def finish(self, step, terminated, truncated, success, fingerprint, visibility):
        """Publish a complete episode only after all frame writers finish."""
        reason = 'terminated' if terminated else 'truncated' if truncated else (
            'success' if self.case and success else 'step_limit')
        receipt = dict(seed=self.record['seed'], split=self.record['split'], mode=self.mode,
                       path=self.name, observations=step + 1, steps=step, stop_reason=reason,
                       successful=self.successful, switched=self.switch is not None,
                       switch_step=self.switch, initial_fingerprint=fingerprint,
                       initial_peg_visibility=visibility, hole_visibility_samples=self.samples,
                       frame_sha256=[item.result() for item in self.hashes],
                       elapsed_seconds=monotonic()-self.started)
        if self.case:
            recovery = self.case['recovery_step']
            receipt.update(parent_episode=self.case['parent']['path'],
                           parent_prefix_frames=0, replayed_prefix_frames=self.case['prefix_frames'],
                           replay_boundary_max_difference=self.replay_boundary_max_difference,
                           recovery_step=recovery, switch_adjustment=self.case['switch_adjustment'],
                           restarted_expert=self.restarted, expert_recovery_actions=max(0, step-recovery),
                           random_actions=min(step, recovery)-self.switch)
        else:
            receipt.update(transition_reached=self.record['transition_phase'] in self.phases,
                           phase_counts=dict(self.phases))
        write_json(self.pending / 'episode.json', receipt)
        self.pending.rename(self.root / self.name)
        return receipt


def advance_boxes(task, lanes, initial, step, active):
    """Apply guarded world offsets to just the live lanes, then sync PhysX."""
    if not any(lane.motion and live for lane, live in zip(lanes, active, strict=True)):
        return False
    tips = task.peg_head_pose.p.cpu().numpy()
    holes = task.box_hole_pose.p.cpu().numpy()
    changes, offsets = [], []
    for index, (lane, live) in enumerate(zip(lanes, active, strict=True)):
        if not live or lane.motion is None:
            continue
        if lane.stop_on_insert and getattr(lane.policy, 'phase', None) == 'insert':
            lane.motion.stopped = True
            lane.motion.status = 'blocked: expert insert phase'
        previous = lane.motion.offset.copy()
        offset = lane.motion.update(step / task.control_freq, tips[index], holes[index])
        if not np.array_equal(previous, offset):
            changes.append(index)
            offsets.append(offset)
    if not changes:
        return False
    indices = torch.as_tensor(changes, device=task.device)
    mask = task.scene._reset_mask.clone()
    task.scene._reset_mask.fill_(False)
    task.scene._reset_mask[indices] = True
    try:
        poses = initial[indices]
        poses[:, :3] += torch.as_tensor(np.asarray(offsets), device=task.device, dtype=poses.dtype)
        task.box.set_pose(poses)
        task.scene.px.gpu_apply_rigid_dynamic_data()
        task.scene.px.gpu_fetch_rigid_dynamic_data()
    finally:
        task.scene._reset_mask = mask
    return True


def replay_action(lane, root, step):
    """Read the parent's applied action and phase, preserving the random stream."""
    from data_collection.mixed_trajectory_recovery import verified_frame

    contents = verified_frame(root, lane.case['parent'], step)
    with np.load(io.BytesIO(contents), allow_pickle=False) as frame:
        if not bool(frame['action_valid']) or int(frame['collection/frame']) != step:
            raise ValueError(f'Invalid replay frame: {lane.name}/{step}')
        action = frame['action'].reshape(7)
        phase = str(frame['collection/phase'])
        switched = bool(frame['collection/switched'])
    lane.last_grip = lane.grip = float(action[-1])
    return action, phase, switched


def compare_boundary(lane, root, step, state):
    """Record GPU replay drift and ensure the branch has the intended Box pose."""
    from data_collection.mixed_trajectory_recovery import verified_frame

    with np.load(io.BytesIO(verified_frame(root, lane.case['parent'], step)), allow_pickle=False) as frame:
        keys = ('qpos', 'peg_pose', 'hole_pose')
        differences = {key: float(np.max(np.abs(state[key].numpy() - frame[f'ground_truth/{key}'])))
                       for key in keys}
    if differences['hole_pose'] > 1e-6 or not np.isfinite(list(differences.values())).all():
        raise ValueError(f'Invalid recovery branch pose for {lane.name}: {differences}')
    return differences


def run_batch(env, entries, mode, protocol, root, config, cases, parent_root, writers):
    """Step once per vector batch and retain each lane's own terminal frame."""
    from data_collection.mixed_trajectory_collection import initial_fingerprint

    task = env.unwrapped
    settings = protocol.get('hole_motion')
    lanes = [Lane(record, mode, root, settings, case)
             for record, case in zip(entries, cases, strict=True)]
    observation = reset_batch(env, entries, config.get('reset_reference_root'))
    initial = task.box.pose.raw_pose.clone()
    advance_boxes(task, lanes, initial, 0, [True] * len(lanes))
    terminated = truncated = np.zeros(len(lanes), dtype=bool)
    fingerprints, visibilities = [], []
    queued = deque()
    step = 0
    while not all(lane.done for lane in lanes):
        gt = cpu_snapshot(vars(get_gt_state(env)))
        goals = task.goal_pose.raw_pose.cpu().numpy()
        gripped = task.agent.is_grasping(task.peg).cpu().numpy()
        stored = cpu_snapshot(dict(
            sensor_data={'base_camera': observation['sensor_data']['base_camera']},
            sensor_param={'base_camera': observation['sensor_param']['base_camera']},
            agent=observation['agent'], extra={'tcp_pose': observation['extra']['tcp_pose']},
        ))
        actions = np.zeros((len(lanes), 7), dtype=np.float32)
        for index, lane in enumerate(lanes):
            if lane.done:
                continue
            state = slice_batch(gt, index)
            obs = slice_batch(stored, index)
            success = bool(state['success'][0])
            lane.successful |= success
            if step == 0:
                validate_camera(obs, protocol['camera'])
                fingerprint = initial_fingerprint(obs, SimpleNamespace(**state))
                fingerprints.append(fingerprint)
                visibilities.append(initial_visibility_from_state(obs, state))
                if lane.case:
                    if fingerprint != lane.case['parent']['initial_fingerprint']:
                        raise ValueError(f'Recovery reset fingerprint differs: {lane.name}')
                elif mode != 'random':
                    setup_expert(task, index, state, lane.policy)
            replaying = lane.case is not None and step < lane.case['prefix_frames']
            limit = (lane.case['recovery_step'] + config['recovery_steps']
                     if lane.case else protocol['max_steps'])
            valid = not (terminated[index] or truncated[index] or step >= limit or (lane.case and success))
            if replaying:
                if not valid:
                    raise RuntimeError(f'Unexpected replay termination: {lane.name}/{step}')
                action, phase, switched = replay_action(lane, parent_root, step)
            else:
                if lane.case and step == lane.case['prefix_frames']:
                    lane.replay_boundary_max_difference = compare_boundary(lane, parent_root, step, state)
                action, phase = lane.action(task, index, state, goals[index], step, valid)
                switched = lane.switch is not None
            actions[index] = action
            if step % protocol['visibility_stride'] == 0 or not valid:
                sample = score_frame(obs, task, 7, 0.003, env_index=index)
                lane.samples.append(dict(sample, frame=step, phase=phase))
            status = dict(frame=step, phase=phase, switched=switched,
                          grasp=bool(gripped[index]), success=success,
                          terminated=bool(terminated[index]), truncated=bool(truncated[index]))
            saved = writers.submit(save_with_hash, lane.pending / f'frame_{step:06d}.npz',
                                   obs, state, action, valid, status)
            lane.hashes.append(saved)
            queued.append(saved)
            if len(queued) >= 3 * len(lanes):
                queued.popleft().result()
            if not valid:
                lane.done = True
                yield lane.finish(step, terminated[index], truncated[index], success,
                                  fingerprints[index], visibilities[index])
        if all(lane.done for lane in lanes):
            break
        observation, _, term, trunc, _ = env.step(torch.as_tensor(actions, device=task.device))
        step += 1
        terminated, truncated = term.cpu().numpy(), trunc.cpu().numpy()
        active = [not (lane.done or terminated[i] or truncated[i]) for i, lane in enumerate(lanes)]
        if advance_boxes(task, lanes, initial, step, active):
            observation = task.get_obs(task.get_info())


def collect_batches(records, config, protocol, root, *, recovery_cases=None, parent_root=None):
    """Schedule fixed-size GPU batches and keep their layout identical for replay."""
    batch_size = config.get('batch_size', config.get('batch_episodes', DEFAULT_BATCH_EPISODES))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('GPU batch_size must be a positive integer')
    recovery = recovery_cases is not None
    if recovery:
        from data_collection.mixed_trajectory_recovery import MODE
        modes = (MODE,)
    else:
        modes = MODES
    max_steps = protocol['max_steps'] + (config['recovery_steps'] if recovery else 0)
    with ThreadPoolExecutor(max_workers=config.get('writers', config.get('workers', DEFAULT_WORKERS))) as writers:
        for start in range(0, len(records), batch_size):
            if shutil.disk_usage(root).free < config['minimum_free_gib'] * 1024**3:
                raise RuntimeError('Available storage fell below minimum_free_gib')
            entries = records[start:start + batch_size]
            cases = recovery_cases[start:start + batch_size] if recovery else [None] * len(entries)
            env = make_peg_insertion_env(
                device='cuda', num_envs=len(entries), obs_mode='rgb+depth+segmentation',
                sensor_configs=camera_sensors(protocol['camera']), control_mode=protocol['control_mode'],
                max_episode_steps=max_steps, render_backend=config.get('render_backend') or 'cuda',
            )
            try:
                fingerprints = {}
                for mode in modes:
                    for episode in run_batch(env, entries, mode, protocol, root, config,
                                             cases, parent_root, writers):
                        seed = episode['seed']
                        fingerprint = episode['initial_fingerprint']
                        if fingerprints.setdefault(seed, fingerprint) != fingerprint:
                            raise ValueError(f'Initial state differs across GPU modes for seed {seed}')
                        yield episode
            finally:
                env.close()
