"""Pilot-gated sequential expert replay, recording PRE-action observations.

Run with --output a new directory; never writes source demonstrations. Raw RGB-D
does not contain simulator segmentation. GT object state is read only in the
separate numerical replay audit, never saved in the robot/model input arrays.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import torch

from graph.prototype.data import reset_episode, state_at, validate_trajectory
from graph.integration_baseline.io import sha256, write_json, save_npz
from graph.integration_baseline.robot import robot_observation, flatten_state, state_schema

SOURCE = Path('outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5')
INITIAL = SOURCE.with_name('trajectory.h5')
FROZEN = Path('outputs/graph/dataset.json')
RESTORE_ATOL = 2e-5
REPLAY_ATOL = 1e-3


def audit_obs(task):
    """Audit-only reconstruction of source state43; cannot enter model inputs."""
    values = [task.agent.robot.get_qpos(), task.agent.robot.get_qvel(), task.agent.tcp.pose.raw_pose,
              task.peg.pose.raw_pose, task.peg_half_sizes, task.box_hole_pose.raw_pose,
              task.box_hole_radii.reshape(1, 1)]
    return torch.cat(values, -1).detach().cpu().numpy()[0]


def make_env():
    from envs.maniskill import make_peg_insertion_env
    return make_peg_insertion_env(obs_mode='rgb+depth', control_mode='pd_ee_delta_pose', device='cpu')


def capture_episode(env, episode, group, initial, repair_id, dataset_hash, schema=None, record=True,
                    preserve_verified_prefix=False):
    actions, reference = validate_trajectory(group)
    # Validate without regenerating/re-normalizing the stored target.
    if group['actions'].dtype != np.float32 or not np.array_equal(actions, group['actions'][()]):
        raise ValueError('source action dtype/value changed')
    reset_episode(env, episode)
    task = env.unwrapped
    task.set_state_dict(initial)
    initial_error = float(np.abs(audit_obs(task) - reference[0]).max())
    if initial_error > RESTORE_ATOL:
        raise ValueError(f'initial source mismatch {initial_error}')
    n = len(actions)
    rows, geometries, observations, raw, flags = [], [], [], {}, []
    errors, deterministic = [], []
    tick = perf_counter()
    failure = None
    for t in range(n + 1):
        actual = audit_obs(task)
        delta = np.abs(actual.astype(np.float64) - reference[t])
        errors.append([float(delta.max()), float(delta[:9].max()), float(delta[9:18].max()),
                       float(delta[18:25].max())])
        if delta.max() > REPLAY_ATOL:
            failure = dict(timestep=t, state43_error=float(delta.max()),
                           qpos_error=float(delta[:9].max()), qvel_error=float(delta[9:18].max()),
                           tcp_pose_error=float(delta[18:25].max()), largest_component=int(np.argmax(delta)),
                           reference_value=float(reference[t, np.argmax(delta)]),
                           replay_value=float(actual[np.argmax(delta)]))
            if not preserve_verified_prefix:
                raise ValueError(f'episode {episode["episode_id"]} t={t} replay error {delta.max():.9g}')
            errors.pop()  # failing observation is audit-only, never a valid input
            break
        fields, geometry = robot_observation(env)
        current = state_schema(fields)
        if schema is None:
            schema = current
        if current != schema:
            raise ValueError('robot state schema differs between frames')
        state = flatten_state(fields, schema)
        deterministic.append(np.r_[state, geometry.ravel()])
        if record:
            obs = task.get_obs()
            for camera, data in obs['sensor_data'].items():
                params = obs['sensor_param'][camera]
                # Preserve raw int16 mm including invalid/overflow values. The
                # derived floating metre view follows the existing RGB-D API.
                entries = dict(rgb=data['rgb'][0].cpu().numpy().copy(),
                               depth_raw_mm=data['depth'][0, ..., 0].cpu().numpy().copy(),
                               intrinsic=params['intrinsic_cv'][0].cpu().numpy().copy(),
                               extrinsic_cv=params['extrinsic_cv'][0].cpu().numpy().copy(),
                               cam2world_gl=params['cam2world_gl'][0].cpu().numpy().copy())
                world_from_cv = np.linalg.inv(np.vstack((entries['extrinsic_cv'], [0, 0, 0, 1]))).astype(np.float32)
                entries['cam2world'] = world_from_cv
                entries['depth'] = entries['depth_raw_mm'].astype(np.float32) / 1000
                for key, value in entries.items():
                    raw.setdefault(camera + '/' + key, []).append(value)
            rows.append(state)
            geometries.append(geometry)
            observations.append(fields['tcp_pose'])
        if t == n:
            break  # final observation has NO target action
        _, _, terminated, truncated, _ = env.step(torch.from_numpy(actions[t].copy())[None])
        observed_flags = [bool(terminated.item()), bool(truncated.item())]
        source_flags = [bool(group['terminated'][t]), bool(group['truncated'][t])]
        flags.append(observed_flags)
        if observed_flags != source_flags:
            raise ValueError(f'episode {episode["episode_id"]} t={t} termination mismatch: {observed_flags} != {source_flags}')
    error_array = np.asarray(errors)
    observations_count = len(deterministic)
    valid_targets = observations_count - 1
    actions = actions[:valid_targets]
    flags = flags[:valid_targets]
    result = dict(episode_id=episode['episode_id'], reset_identity=episode['reset_kwargs'],
                  reset_seed=episode['episode_seed'], original_episode_id=repair_id,
                  steps=valid_targets, source_steps=n, observations=observations_count,
                  aligned_targets=valid_targets, initial_error=initial_error,
                  max_errors=dict(zip(['state43', 'qpos', 'qvel', 'tcp_pose'], error_array.max(0).tolist())),
                  elapsed_seconds=perf_counter()-tick, status='PARTIAL_REAL_ALIGNED' if failure else 'REAL_ALIGNED',
                  failure=failure, invalid_or_unrecorded_observations=list(range(observations_count, n+1)),
                  timestamp_source='simulation timestep / task.control_freq', control_freq=int(task.control_freq),
                  control_mode='pd_ee_delta_pose', source_sha256=dataset_hash,
                  alignment='observation_t -> stored action_t -> step -> observation_t+1; terminal observation unlabelled',
                  raw_observation_provenance='sequential physx_cpu, repaired original state0; no per-step state teleport')
    arrays = dict(robot_state=np.stack(rows), robot_world_geometry=np.stack(geometries), tcp_pose=np.stack(observations),
                  actions=actions.copy(), timestep=np.arange(observations_count, dtype=np.int64),
                  timestamp=np.arange(observations_count, dtype=np.float64)/task.control_freq,
                  terminated=np.asarray(flags, bool)[:, 0], truncated=np.asarray(flags, bool)[:, 1],
                  episode_id=np.full(observations_count, episode['episode_id'], np.int64),
                  source_sha256=np.array(dataset_hash)) if record else {}
    arrays.update({k: np.stack(v) for k, v in raw.items()})
    return result, arrays, schema, np.stack(deterministic)


def run(output, source=SOURCE, initial_source=INITIAL, frozen=FROZEN, count=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    metadata = json.loads(source.with_suffix('.json').read_text())
    manifest = json.loads(frozen.read_text())
    if metadata['env_info']['env_id'] != 'PegInsertionSide-v1':
        raise ValueError('wrong source task')
    if any(e['control_mode'] != 'pd_ee_delta_pose' for e in metadata['episodes']):
        raise ValueError('mixed control modes')
    repairs = {r['episode_id']: r['source_episode_id'] for r in manifest['snapshot_checks']}
    source_hash = sha256(source)
    inventory = dict(schema_version=1, status='RUNNING', source=str(source.resolve()), source_sha256=source_hash,
                     source_json_sha256=sha256(source.with_suffix('.json')),
                     initial_source=str(initial_source.resolve()), initial_source_sha256=sha256(initial_source),
                     frozen_split_manifest=str(frozen.resolve()), frozen_split_sha256=sha256(frozen),
                     splits=manifest['splits'], episodes_expected=len(metadata['episodes']),
                     actions_expected=sum(e['elapsed_steps'] for e in metadata['episodes']),
                     episodes=[], failures=[], missing_episode_ids=[],
                     pilot_gate=dict(status='NOT_EXECUTED', restore_atol=RESTORE_ATOL, replay_atol=REPLAY_ATOL,
                         basis='existing prepare_dataset.py snapshot 2e-5 / independent replay 1e-3; thresholds not relaxed'),
                     action_schema='pd_ee_delta_pose_regression7', mask_status='NOT_EXECUTED',
                     vision_policy=dict(keep_ratio=.5, intra_object='raw_extended_delaunay', edge_validation='none'))
    write_json(output/'manifest.json', inventory)
    env = make_env()
    schema = None
    episodes = metadata['episodes']
    try:
        inventory['backend'] = str(env.unwrapped.backend)
        with h5py.File(source, 'r') as trajectories, h5py.File(initial_source, 'r') as originals:
            inventory['actions_expected'] = sum(len(trajectories[f'traj_{e["episode_id"]}/actions']) for e in episodes)
            pilot = []
            for ep in episodes[:3]:
                eid = ep['episode_id']
                original = state_at(originals[f'traj_{repairs[eid]}/env_states'], 0)
                runs = []
                for repeat in range(2):
                    result, arrays, schema, signature = capture_episode(env, ep, trajectories[f'traj_{eid}'], original,
                                                                       repairs[eid], source_hash, schema, record=True,
                                                                       preserve_verified_prefix=True)
                    runs.append(signature)
                    print(f'pilot episode={eid} repeat={repeat} max_errors={result["max_errors"]}', flush=True)
                determinism_error = float(np.max(np.abs(runs[0]-runs[1])))
                # Identical executable/reset/state/actions should be deterministic,
                # not merely inside the looser source-replay tolerance.
                if determinism_error > RESTORE_ATOL:
                    raise ValueError(f'repeated replay determinism error {determinism_error}')
                result['repeat_max_robot_error'] = determinism_error
                path = output/'episodes'/f'ep{eid:03d}.npz'
                save_npz(path, **arrays)
                result.update(path=str(path.relative_to(output)), sha256=sha256(path))
                inventory['episodes'].append(result)
                if result['failure']:
                    inventory['failures'].append(dict(episode_id=eid, reason='sequential replay drift', **result['failure']))
                pilot.append(result)
                write_json(output/'manifest.json', inventory)
            inventory['robot_state_schema'] = schema
            gate_pass = all(r['status'] == 'REAL_ALIGNED' for r in pilot)
            inventory['pilot_gate'].update(status='PASS' if gate_pass else 'BLOCKED_DATA', episodes=pilot, repeats=2)
            write_json(output/'schema.json', schema)
            write_json(output/'manifest.json', inventory)
            # A failing pilot forbids expanding collection to the remaining 88.
            for ep in (episodes[3:count] if gate_pass else []):
                eid = ep['episode_id']
                try:
                    original = state_at(originals[f'traj_{repairs[eid]}/env_states'], 0)
                    result, arrays, _, _ = capture_episode(env, ep, trajectories[f'traj_{eid}'], original,
                                                           repairs[eid], source_hash, schema, record=True)
                    path = output/'episodes'/f'ep{eid:03d}.npz'
                    save_npz(path, **arrays)
                    result.update(path=str(path.relative_to(output)), sha256=sha256(path))
                    inventory['episodes'].append(result)
                    print(f'recorded episode={eid} T={result["steps"]} errors={result["max_errors"]}', flush=True)
                except Exception as exc:
                    inventory['failures'].append(dict(episode_id=eid, type=type(exc).__name__, reason=str(exc)))
                    print(f'FAILED episode={eid}: {type(exc).__name__}: {exc}', flush=True)
                write_json(output/'manifest.json', inventory)
    except Exception as exc:
        inventory['pilot_gate'].update(status='BLOCKED_DATA', error_type=type(exc).__name__, reason=str(exc))
        print(f'PILOT BLOCKED: {type(exc).__name__}: {exc}', flush=True)
    finally:
        env.close()
        done = {e['episode_id'] for e in inventory['episodes']}
        inventory['missing_episode_ids'] = [e['episode_id'] for e in episodes if e['episode_id'] not in done]
        inventory['coverage'] = dict(episodes=len(done), complete_episodes=sum(e['status']=='REAL_ALIGNED' for e in inventory['episodes']),
                                     aligned_frames=sum(e['steps'] for e in inventory['episodes']),
                                     observations=sum(e['observations'] for e in inventory['episodes']))
        inventory['status'] = ('REAL_ALIGNED' if len(done) == len(episodes) else 'PARTIAL_REAL_ALIGNED' if done else 'BLOCKED_DATA')
        if inventory['pilot_gate']['status'] != 'PASS':
            inventory['status'] = 'BLOCKED_DATA'
        write_json(output/'manifest.json', inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--initial-source', type=Path, default=INITIAL)
    parser.add_argument('--frozen-split', type=Path, default=FROZEN)
    parser.add_argument('--count', type=int)
    args = parser.parse_args()
    run(args.output, args.source, args.initial_source, args.frozen_split, args.count)


if __name__ == '__main__':
    main()
