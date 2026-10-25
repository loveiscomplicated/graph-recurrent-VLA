"""Separate state-only drift diagnostic; never relaxes replay tolerance."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from graph.integration_baseline.replay import SOURCE, INITIAL, FROZEN, audit_obs
from graph.integration_baseline.io import write_json, sha256, save_npz
from graph.prototype.data import reset_episode, state_at, validate_trajectory


def run(output):
    from envs.maniskill import make_peg_insertion_env
    meta = json.loads(SOURCE.with_suffix('.json').read_text())
    frozen = json.loads(FROZEN.read_text())
    env = make_peg_insertion_env(obs_mode='state', control_mode='pd_ee_delta_pose', render_mode='rgb_array')
    summary = dict(source_sha256=sha256(SOURCE), obs_mode='state', episodes=[],
                   interpretation='state-only and RGB-D drift comparison; root cause remains unconfirmed',
                   tolerance_changed=False, controller_state={})
    try:
        summary['backend'] = str(env.unwrapped.backend)
        with h5py.File(SOURCE) as trajectories, h5py.File(INITIAL) as originals:
            for ep in meta['episodes'][:3]:
                eid = ep['episode_id']
                actions, reference = validate_trajectory(trajectories[f'traj_{eid}'])
                reset_episode(env, ep)
                task = env.unwrapped
                source_id = frozen['snapshot_checks'][eid]['source_episode_id']
                task.set_state_dict(state_at(originals[f'traj_{source_id}/env_states'], 0))
                actual = [audit_obs(task)]
                for action in actions:
                    env.step(torch.from_numpy(action.copy())[None])
                    actual.append(audit_obs(task))
                actual = np.stack(actual)
                delta = np.abs(actual.astype(np.float64)-reference)
                bad = np.flatnonzero(delta.max(1) > .001)
                item = dict(episode_id=eid, max_errors=dict(state43=float(delta.max()),
                    qpos=float(delta[:,:9].max()), qvel=float(delta[:,9:18].max()), tcp_pose=float(delta[:,18:25].max())),
                    first_failure_timestep=int(bad[0]) if len(bad) else None,
                    source_steps=len(actions), source_time_limit=100)
                # Independent exact snapshot restoration check, including known
                # first-state repair. This is NOT substituted for action replay.
                restoration_error = 0.
                for t in range(len(actions)+1):
                    snapshot = (state_at(originals[f'traj_{source_id}/env_states'],0) if t == 0
                                else state_at(trajectories[f'traj_{eid}/env_states'],t))
                    task.set_state_dict(snapshot)
                    restoration_error = max(restoration_error, float(np.abs(audit_obs(task)-reference[t]).max()))
                item['snapshot_restore_max_error'] = restoration_error
                summary['episodes'].append(item)
                save_npz(Path(output).with_name(f'diagnostic_ep{eid:03d}.npz'), reference_state43=reference,
                         sequential_replay_state43=actual, absolute_error=delta)
                print(item, flush=True)
        summary['controller_config'] = {k: {name: getattr(v,name) for name in (
            ['pos_lower','pos_upper','rot_lower','rot_upper','frame','use_delta','use_target','normalize_action']
            if k=='arm' else ['lower','upper','use_delta','use_target','normalize_action'])}
            for k,v in env.unwrapped.agent.controller.configs.items()}
        summary['package_versions'] = {name: __import__('importlib.metadata',fromlist=['version']).version(name)
                                        for name in ('mani_skill','sapien','torch','numpy','pin')}
    finally:
        env.close()
    write_json(output, summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
