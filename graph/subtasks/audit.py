"""Collect real contact GT by uninterrupted expert prefix replay, never snapshots."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from graph.prototype.common import write_json, versions
from graph.prototype.data import reset_episode, state_at, validate_trajectory, sha256


def collect(manifest_path, output, ids=None):
    from envs.maniskill import make_peg_insertion_env
    from .live import LiveSubtaskAccessor
    from .dataset import STATE_KEYS
    from mani_skill.utils.registration import TimeLimitWrapper
    torch.set_num_threads(4)
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ('source', 'initial_state_source'):
        if sha256(manifest[key]) != manifest[key+'_sha256']:
            raise ValueError(f'{key} checksum differs')
    env = make_peg_insertion_env(control_mode=manifest['control_mode'])
    records = []
    checks = {r['episode_id']: r for r in manifest['snapshot_checks']}
    try:
        with h5py.File(manifest['source']) as converted, h5py.File(manifest['initial_state_source']) as original:
            for ep in manifest['episodes']:
                eid = ep['episode_id']
                if ids is not None and eid not in ids:
                    continue
                path = output / f'episode_{eid}.npz'
                if path.exists():
                    with np.load(path) as z:
                        if str(z['source_sha256']) != manifest['source_sha256'] or str(z['initial_state_source_sha256']) != manifest['initial_state_source_sha256']:
                            raise ValueError('cached audit source differs')
                        records.append(json.loads(str(z['replay_record'])))
                    continue
                group = converted[f'traj_{eid}']
                actions, obs = validate_trajectory(group)
                env = TimeLimitWrapper(env.unwrapped, len(actions)+1)
                reset_episode(env, ep)
                source_id = checks[eid]['source_episode_id']
                env.unwrapped.set_state_dict(state_at(original[f'traj_{source_id}/env_states'], 0))
                env.unwrapped.agent.controller.reset()
                task = env.unwrapped
                initial_error = float(np.max(np.abs(task.get_obs().cpu().numpy()[0]-obs[0])))
                if initial_error > 2e-5:
                    raise ValueError(f'initial state mismatch {eid}: {initial_error}')
                accessor = LiveSubtaskAccessor().reset(env)
                arrays = {}
                max_error = initial_error
                for t in range(len(actions)+1):
                    live_state = accessor(env, contact_valid=t>0)
                    row = {k:live_state[k] for k in STATE_KEYS}
                    for k,v in row.items(): arrays.setdefault(k,[]).append(v)
                    if t < len(actions):
                        live,*_ = env.step(torch.from_numpy(actions[t])[None])
                        max_error = max(max_error,float(np.max(np.abs(live.cpu().numpy()[0]-obs[t+1]))))
                record = dict(episode_id=eid, steps=len(actions), initial_obs_error=initial_error,
                    max_observation_error=max_error, replay_valid=max_error<=.001,
                    official_success=bool(np.any(arrays['success'])), source_episode_id=source_id)
                np.savez_compressed(path, **{k:np.asarray(v) for k,v in arrays.items()}, actions=actions,
                    reach_local=accessor.reach_local, grasp_local=accessor.grasp_local,
                    source_sha256=manifest['source_sha256'], initial_state_source_sha256=manifest['initial_state_source_sha256'],
                    replay_record=json.dumps(record))
                records.append(record)
                print(record,flush=True)
        write_json(output/'replay.json', dict(records=records, versions=versions(),
            backend=str(env.unwrapped.backend), contact_method='live Panda.is_grasping(min_force=0.5,max_angle=85); s0 contact unknown',
            all_replay_valid=all(r['replay_valid'] for r in records)))
    finally: env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='outputs/graph/dataset.json')
    p.add_argument('--output',default='outputs/subtasks/audit')
    p.add_argument('--episodes',type=int,nargs='+')
    a=p.parse_args(); collect(a.manifest,a.output,a.episodes)
if __name__=='__main__': main()
