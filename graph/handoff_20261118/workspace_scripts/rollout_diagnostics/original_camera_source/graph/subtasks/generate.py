"""Collect native official expert, convert actions, verify, segment and evaluate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import torch

from graph.prototype.common import write_json, versions as prototype_versions
from graph.prototype.data import state_at, sha256, validate_trajectory
from .dataset import STATE_KEYS
from .env import tree_error
from .live import LiveSubtaskAccessor
from .specs import NAMES

ENV_KWARGS = dict(num_envs=1, obs_mode='state', control_mode='pd_joint_pos',
                  sim_backend='physx_cpu', render_backend='gpu', render_mode=None, max_episode_steps=1000,
                  reconfiguration_freq=1)


def versions():
    result = prototype_versions()
    for package in ('numpy','pin','mplib','opencv-python'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def randomization(env):
    task = env.unwrapped
    values = dict(peg_pose=task.peg.pose.raw_pose, box_pose=task.box.pose.raw_pose,
                  hole_pose=task.box_hole_pose.raw_pose, peg_half_sizes=task.peg_half_sizes,
                  hole_radius=task.box_hole_radii, hole_offset=task.box_hole_offsets.p,
                  robot_qpos=task.agent.robot.get_qpos())
    return {k:v.detach().cpu().numpy()[0].tolist() for k,v in values.items()}


def episode_split(seeds, split_seed=0):
    """Group by unique seed/config; tiny smoke runs intentionally have empty splits."""
    if len(seeds) != len(set(seeds)):
        raise ValueError('duplicate seed/configuration')
    order = np.random.default_rng(split_seed).permutation(seeds).tolist()
    n = len(order)
    held = max(1, n//10) if n >= 3 else 0
    return dict(train=order[:n-2*held], validation=order[n-2*held:n-held] if held else [],
                test=order[n-held:] if held else [])


def make_env(control_mode):
    from envs.maniskill.peg_insertion import _configure_macos_vulkan
    _configure_macos_vulkan()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    original_cwd = Path.cwd()
    try:
        return gym.make('PegInsertionSide-v1', **dict(ENV_KWARGS, control_mode=control_mode))
    finally:
        os.chdir(original_cwd)


def seed_planner(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    import mplib
    if hasattr(mplib, 'set_global_seed'):
        mplib.set_global_seed(seed)


def collect_native(seed, directory):
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve

    class Recorder(RecordEpisode):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self.initial_randomization = randomization(self)
            write_json(Path(directory)/'initial.json', dict(seed=seed, environment=ENV_KWARGS,
                       randomization=self.initial_randomization, reset_kwargs=kwargs, versions=versions()))
            return result

    seed_planner(seed)
    env = Recorder(make_env('pd_joint_pos'), str(directory), trajectory_name='trajectory',
                   save_video=False, save_on_reset=False, clean_on_close=False,
                   source_type='motionplanning', source_desc='installed official PegInsertionSide solve()')
    record = dict(seed=seed, native_control_mode='pd_joint_pos', source_success=False,
                  environment=ENV_KWARGS, versions=versions(), planner_version=importlib.metadata.version('mplib'))
    try:
        result = solve(env, seed=seed, debug=False, vis=False)
        record['solver_status'] = 'planning_failed' if isinstance(result, int) and result == -1 else 'returned'
        record['source_success'] = bool(env.unwrapped.evaluate()['success'].item())
        if not record['source_success']:
            record['failure_reason'] = 'planner_returned_minus_one' if record['solver_status']=='planning_failed' else 'official_success_false_at_solver_return'
    except Exception as exc:
        record.update(solver_status='exception', failure_reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        record['randomization'] = getattr(env, 'initial_randomization', None)
        record['backend'] = str(env.unwrapped.backend)
        if env._trajectory_buffer is not None:
            env.flush_trajectory(ignore_empty_transition=False)
        env.close()
    return record


def verify_native(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    episode = meta['episodes'][0]
    env = make_env('pd_joint_pos')
    error = 0.
    try:
        with h5py.File(path) as h5:
            group = h5['traj_0']
            actions = np.asarray(group['actions'])
            if actions.shape != (episode['elapsed_steps'], 8) or not np.isfinite(actions).all():
                raise ValueError('native joint actions must be finite [T,8]')
            lengths = []
            group['env_states'].visititems(lambda _, v: lengths.append(len(v)) if isinstance(v, h5py.Dataset) else None)
            if not lengths or any(n != len(actions)+1 for n in lengths) or len(group['obs']) != len(actions)+1:
                raise ValueError('native state/action alignment failed')
            obs, _ = env.reset(**episode['reset_kwargs'])
            for t in range(len(actions)+1):
                error = max(error, tree_error(obs, group['obs'][t]), tree_error(env.unwrapped.get_state_dict(), state_at(group['env_states'], t)))
                if t < len(actions):
                    if not env.action_space.contains(actions[t]):
                        raise ValueError('native action exceeds joint controller bounds')
                    obs, *_ = env.step(actions[t])
            return dict(native_replay_valid=error <= .001, native_replay_error=error,
                        native_replay_success=bool(env.unwrapped.evaluate()['success'].item()), native_steps=len(actions))
    finally:
        env.close()


def convert(raw_path, directory):
    """Installed official action converter; real steps, correct recorder s0.

    Calling the converter directly avoids 3.0.1 replay_cpu_sim's s1-in-s0
    recording bug. No use_env_states, snapshot-per-step or library source edits.
    """
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos
    meta = json.loads(raw_path.with_suffix('.json').read_text())
    ep = meta['episodes'][0]
    original = make_env('pd_joint_pos')
    target = RecordEpisode(make_env('pd_ee_delta_pose'), str(directory), trajectory_name='trajectory',
                           save_video=False, save_on_reset=False, clean_on_close=False)
    try:
        with h5py.File(raw_path) as h5:
            obs, _ = original.reset(**ep['reset_kwargs'])
            target.reset(**ep['reset_kwargs'])
            initial = state_at(h5['traj_0/env_states'], 0)
            if max(tree_error(original.unwrapped.get_state_dict(), initial),
                   tree_error(target.unwrapped.get_state_dict(), initial)) > 2e-6:
                raise ValueError('seed reset does not reproduce source s0')
            source_geo = randomization(original)
            target_geo = randomization(target)
            if tree_error(source_geo, target_geo) > 2e-6:
                raise ValueError('randomized object geometry changed during conversion')
            info = from_pd_joint_pos('pd_ee_delta_pose', np.asarray(h5['traj_0/actions']), original, target)
            conversion_success = bool(info.get('success', False))
        return dict(conversion_success=conversion_success, converted_control_mode='pd_ee_delta_pose',
                    conversion_method='installed from_pd_joint_pos; uninterrupted action conversion',
                    geometry_match=True)
    finally:
        if target._trajectory_buffer is not None:
            target.flush_trajectory(ignore_empty_transition=False)
        target.close()
        original.close()


def verify_converted(path, raw_path, audit_path, record):
    meta = json.loads(path.with_suffix('.json').read_text())
    ep = meta['episodes'][0]
    env = make_env('pd_ee_delta_pose')
    arrays = {}
    max_obs_error = max_state_error = 0.
    try:
        with h5py.File(path) as h5, h5py.File(raw_path) as raw:
            group = h5['traj_0']
            actions, observations = validate_trajectory(group)
            obs, _ = env.reset(**ep['reset_kwargs'])
            initial = state_at(raw['traj_0/env_states'], 0)
            if tree_error(state_at(group['env_states'], 0), initial) > 2e-6:
                raise ValueError('converted s0 alignment failure')
            accessor = LiveSubtaskAccessor().reset(env)
            for t in range(len(actions)+1):
                row = accessor(env, contact_valid=t > 0)
                for k in STATE_KEYS:
                    arrays.setdefault(k, []).append(row[k])
                max_obs_error = max(max_obs_error, tree_error(obs, observations[t]))
                max_state_error = max(max_state_error, tree_error(env.unwrapped.get_state_dict(), state_at(group['env_states'], t)))
                if t < len(actions):
                    obs, *_ = env.step(actions[t])
            valid = max_obs_error <= .001 and max_state_error <= .001
            # Check actual collision geometry, not only pose state dictionaries.
            from graph.prototype.maniskill_gt import GTAdapter, validate_live_collision
            collision = validate_live_collision(env, GTAdapter()(env))
            replay = dict(episode_id=record['episode_id'], source_episode_id=record['episode_id'],
                          steps=len(actions), max_observation_error=max_obs_error, max_state_error=max_state_error,
                          replay_valid=valid, official_success=bool(np.any(arrays['success'])),
                          final_success=bool(arrays['success'][-1]), collision=collision,
                          initial_obs_error=tree_error(observations[0], raw['traj_0/obs'][0]),
                          final_state_ref=dict(trajectory=str(path), group='traj_0/env_states', state_index=len(actions)))
            replay['conversion_final_native_state_error'] = tree_error(env.unwrapped.get_state_dict(),state_at(raw['traj_0/env_states'],-1))
            replay['conversion_final_native_observation_error'] = tree_error(obs,raw['traj_0/obs'][-1])
            np.savez_compressed(audit_path, **{k:np.asarray(v) for k,v in arrays.items()}, actions=actions,
                                reach_local=accessor.reach_local, grasp_local=accessor.grasp_local,
                                replay_record=json.dumps(replay), source_sha256=sha256(path),
                                initial_state_source_sha256=sha256(raw_path))
            return replay
    finally:
        env.close()


def assemble(output, records, seed_splits):
    """Build new aggregate H5 files; originals and failed per-seed files survive."""
    from .segment import run as segment
    accepted = [r for r in records if r.get('replay', {}).get('replay_valid')]
    raw_path = output/'raw/dataset.h5'
    converted_path = output/'converted/dataset.h5'
    if raw_path.exists() or converted_path.exists():
        raise FileExistsError('aggregate dataset already exists; choose a new output directory')
    raw_eps, eps, checks = [], [], []
    with h5py.File(raw_path, 'w') as raw, h5py.File(converted_path, 'w') as converted:
        for record in records:
            eid = record['episode_id']
            path = Path(record['raw_trajectory'])
            with h5py.File(path) as src:
                src.copy('traj_0', raw, name=f'traj_{eid}')
            native_ep = json.loads(path.with_suffix('.json').read_text())['episodes'][0]
            raw_eps.append(dict(native_ep, episode_id=eid))
        for record in accepted:
            eid = record['episode_id']
            path = Path(record['converted_trajectory'])
            with h5py.File(path) as src:
                src.copy('traj_0', converted, name=f'traj_{eid}')
            ep = json.loads(path.with_suffix('.json').read_text())['episodes'][0]
            if ep['reset_kwargs'].get('options') is None:
                ep['reset_kwargs']['options'] = {}
            eps.append(dict(ep, episode_id=eid, source_uid=record['source_uid']))
            checks.append(dict(episode_id=eid, source_episode_id=eid))
    write_json(raw_path.with_suffix('.json'), dict(env_info=dict(env_id='PegInsertionSide-v1', env_kwargs=ENV_KWARGS), episodes=raw_eps))
    write_json(converted_path.with_suffix('.json'), dict(env_info=dict(env_id='PegInsertionSide-v1', env_kwargs=dict(ENV_KWARGS, control_mode='pd_ee_delta_pose')), episodes=eps))
    splits = {k:[r['episode_id'] for r in accepted if r['seed'] in seeds] for k,seeds in seed_splits.items()}
    manifest = dict(source=str(converted_path), source_sha256=sha256(converted_path),
                    initial_state_source=str(raw_path), initial_state_source_sha256=sha256(raw_path),
                    control_mode='pd_ee_delta_pose', splits=splits, episodes=eps, snapshot_checks=checks,
                    dataset_id=output.name, environment=ENV_KWARGS, versions=versions())
    write_json(output/'manifests/dataset.json', manifest)
    for r in accepted:
        eid = r['episode_id']
        with np.load(r['audit_file']) as z:
            data = {k:z[k] for k in z.files}
        data['source_sha256'] = manifest['source_sha256']
        data['initial_state_source_sha256'] = manifest['initial_state_source_sha256']
        np.savez_compressed(output/f'audit/episode_{eid}.npz', **data)
    index = segment(output/'manifests/dataset.json', output/'audit', output/'segments')
    for ep in index['episodes']:
        record = next(r for r in accepted if r['episode_id'] == ep['episode_id'])
        for s in ep['segments']:
            s['initial_state_ref'].update(method='seed_reset_then_expert_prefix',controller_reset_at_s0=False)
            s.update(source_uid=record['source_uid'], source_seed=record['seed'],
                     source_trajectory=str(converted_path), segmentation_status=s['status'],
                     eligibility_status='candidate' if s['status'] == 'detected' else 'unresolved',
                     bc_eligible=False, expert_phase_provenance='official solve; no waypoint events captured')
    index.update(environment=ENV_KWARGS, dataset_id=output.name)
    write_json(output/'segments/index.json', index)
    return index


def generate_one(eid, seed, output):
    """Worker; parent isolates native planner crashes and preserves their logs."""
    torch.set_num_threads(4)
    start = time.monotonic()
    raw = output/f'raw/seed_{seed}'
    converted = output/f'converted/seed_{seed}'
    record = dict(episode_id=eid, seed=seed)
    uid = hashlib.sha256(json.dumps(dict(seed=seed, env=ENV_KWARGS), sort_keys=True).encode()).hexdigest()[:20]
    record['source_uid'] = f'generated:{uid}'
    record_path = output/f'reports/seed_{seed}.json'
    try:
        record.update(stage='native_collection', raw_trajectory=str(raw/'trajectory.h5'))
        write_json(record_path,record)
        record.update(collect_native(seed, raw))
        record['stage'] = 'native_verification'
        write_json(record_path,record)
        record.update(verify_native(raw/'trajectory.h5'))
        if not record['native_replay_valid']:
            raise ValueError('native replay diverged')
        if record['native_steps'] == 0:
            raise ValueError('official planner recorded no actions')
        record.update(stage='conversion', conversion_attempted=True)
        write_json(record_path,record)
        record.update(convert(raw/'trajectory.h5', converted))
        record.update(stage='converted_verification', converted_trajectory=str(converted/'trajectory.h5'),
                      audit_file=str(output/f'audit/episode_{eid}.npz'))
        write_json(record_path,record)
        record['replay'] = verify_converted(converted/'trajectory.h5', raw/'trajectory.h5', record['audit_file'], record)
        record['status'] = 'verified' if record['replay']['replay_valid'] else 'invalid_replay'
    except Exception as exc:
        record.update(status='failed', pipeline_failure_reason=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    record['seconds'] = time.monotonic()-start
    write_json(record_path,record)
    return record


def generate_expert_data(num_episodes=3, seed_start=10000, output_dir='outputs/subtasks/generated', split_seed=0, seeds=None, worker_timeout=300):
    output = Path(output_dir).resolve()
    seeds = list(range(seed_start, seed_start+num_episodes)) if seeds is None else list(seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError('need nonnegative, unique seeds')
    # Do not replace prior results, even after a failed run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output is nonempty: {output}; use a new directory')
    for name in ('raw','converted','audit','segments','manifests','reports','videos'):
        (output/name).mkdir(parents=True, exist_ok=True)
    splits = episode_split(seeds, split_seed)
    write_json(output/'manifests/seeds.json', dict(seeds=seeds, seed_splits=splits, split_seed=split_seed,
                                                environment=ENV_KWARGS, versions=versions()))
    torch.set_num_threads(4)
    records = []
    for eid, seed in enumerate(seeds):
        start = time.monotonic()
        log_path = output/f'reports/seed_{seed}.log'
        with open(log_path,'w') as log:
            try:
                result = subprocess.run([sys.executable,'-u','-X','faulthandler','-m','graph.subtasks.generate',
                                         '--worker',str(eid),str(seed),'--output',str(output)],
                                        stdout=log,stderr=subprocess.STDOUT,timeout=worker_timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        record_path = output/f'reports/seed_{seed}.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else dict(episode_id=eid,seed=seed)
        if code != 0:
            record.update(status='failed',pipeline_failure_reason=f'worker exit {code}; see {log_path}',worker_exit=code)
            initial = output/f'raw/seed_{seed}/initial.json'
            if initial.exists():
                record.update(json.loads(initial.read_text()))
        record['seconds'] = time.monotonic()-start
        records.append(record)
        write_json(output/'reports/collection.json', dict(records=records))
        print(json.dumps({k:v for k,v in record.items() if k not in ('randomization','traceback','environment','versions')}), flush=True)
    # Exceptions without a recorded source remain in collection.json, not H5.
    recorded = []
    for r in records:
        if r.get('raw_trajectory') and Path(r['raw_trajectory']).exists():
            try:
                with h5py.File(r['raw_trajectory']) as h5:
                    if 'traj_0' in h5:
                        recorded.append(r)
            except OSError:
                pass  # interrupted native H5 survives as a diagnostic artifact
    index = assemble(output, recorded, splits) if recorded else None
    evaluation = None
    if index and index['episodes']:
        from .evaluate import run as evaluate
        evaluation = evaluate(output/'segments/index.json', output/'reports/independent')
        eligible = {(r['episode_id'],r['subtask']):r for r in evaluation['records']}
        for ep in index['episodes']:
            for s in ep['segments']:
                r = eligible[(ep['episode_id'],s['subtask'])]
                s['bc_eligible'] = r['bc_eligible']
                s['eligibility_status'] = r.get('eligibility_status','unresolved')
                s['independent_replay'] = r
        # Keep the geometric index immutable: eligibility is supplied via sidecar.
        write_json(output/'segments/verified_index.json', index)
    summary = dict(attempted=len(seeds), native_recorded=len(recorded),
                   source_success=sum(bool(r.get('source_success')) for r in records),
                   native_replay_valid=sum(bool(r.get('native_replay_valid')) for r in records),
                   conversion_attempted=sum(bool(r.get('conversion_attempted')) for r in records),
                   conversion_success=sum(bool(r.get('conversion_success')) for r in records),
                   converted_replay_valid=sum(bool(r.get('replay',{}).get('replay_valid')) for r in records),
                   converted_replay_success=sum(bool(r.get('replay',{}).get('official_success')) for r in records),
                   segmentation={n:sum(s['status']=='detected' for e in (index or {}).get('episodes',[]) for s in e['segments'] if s['subtask']==n) for n in NAMES},
                   independent=evaluation['summary'] if evaluation else None,
                   seconds=sum(r['seconds'] for r in records), versions=versions())
    write_json(output/'reports/summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-episodes', type=int, default=3)
    p.add_argument('--seed-start', type=int, default=10000)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--output', default='outputs/subtasks/generated')
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--worker', type=int, nargs=2, help=argparse.SUPPRESS)
    p.add_argument('--worker-timeout', type=int, default=300)
    a = p.parse_args()
    if a.num_episodes < 1:
        p.error('--num-episodes must be positive')
    if a.worker:
        generate_one(*a.worker,Path(a.output).resolve())
    else:
        print(generate_expert_data(a.num_episodes, a.seed_start, a.output, a.split_seed, a.seeds,a.worker_timeout))


if __name__ == '__main__':
    main()
