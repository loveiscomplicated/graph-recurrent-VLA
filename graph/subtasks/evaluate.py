"""Independent expert replay / policy SR from explicit, paired start manifests."""
import argparse
from pathlib import Path
import numpy as np
import torch

from graph.prototype.common import write_json, versions
from graph.prototype.data import state_at, sha256
from .dataset import load_segments
from .env import make_subtask_env, InvalidSubtaskReset, tree_error
from .specs import NAMES


def evaluate_policy(env, policy, starts):
    """Policy receives native observations; current GT is accessible via env.

    Reset failures remain in the denominator. Never sample replacement episodes.
    Call once per subtask with starts from evaluation_starts.json.
    """
    results = []
    for start in starts:
        if start['subtask'] != env.subtask:
            continue
        if start.get('index_sha256') != sha256(env.index_path) or start.get('source_sha256') != env.dataset['source_sha256']:
            raise ValueError('paired evaluation manifest differs from selected dataset')
        try:
            obs, info = env.reset(seed=start['seed'], options={'source_episode_id': start['source_episode_id']})
        except InvalidSubtaskReset as exc:
            results.append(dict(start=start, reset_valid=False, subtask_success=False, steps=0, failure_reason=str(exc)))
            continue
        while True:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated:
                break
        results.append(dict(start=start, reset_valid=True, subtask_success=info['subtask_success'],
                            steps=info['subtask_step'], failure_reason=info['failure_reason'],
                            initialization=info['initialization']))
    return results


def run(index_path, output, ids=None, render=False):
    index = load_segments(index_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    records, starts = [], []
    for name in NAMES:
        env = make_subtask_env(name, index=index_path, render_mode='rgb_array' if render else None)
        try:
            for ep in index['episodes']:
                eid = ep['episode_id']
                if ids is not None and eid not in ids:
                    continue
                segment = next(s for s in ep['segments'] if s['subtask'] == name)
                row = dict(episode_id=eid, subtask=name, split=ep['split'], reset_valid=False,
                           replay_success=False, steps=0, status=segment['status'],
                           bc_eligible=False, unexpected_early_termination=False)
                if segment['status'] != 'detected':
                    row['failure_reason'] = segment['boundary_reason']
                    records.append(row)
                    continue
                i, j = segment['start_t'], segment['end_t']
                try:
                    obs, info = env.reset(options={'source_episode_id': eid})
                    row.update(reset_valid=True, start_state=info['start_state'],
                               prefix_steps=i, prefix_observation_error=env.max_prefix_observation_error,
                               prefix_state_error=env.max_prefix_state_error)
                    starts.append(dict(subtask=name, source_episode_id=eid, source_uid=info['start_state']['source_uid'],
                                       seed=info['start_state']['seed'], split=ep['split'], start_t=i,
                                       index=str(Path(index_path).resolve()), source_sha256=index['source_sha256'],
                                       index_sha256=sha256(index_path), reset_kwargs=info['start_state']['reset_kwargs']))
                    if render:
                        from PIL import Image
                        Image.fromarray(env.render()[0].cpu().numpy()).save(output/f'{eid}_{name}_reset.png')
                    state_error = observation_error = 0.
                    completion_info = None
                    segment_failure = []
                    for t in range(i, j):
                        if completion_info is None:
                            obs, _, terminated, truncated, info = env.step(env.audit['actions'][t])
                            if terminated or truncated:
                                completion_info = info
                                row['unexpected_early_termination'] = bool(t+1 < j and not info['subtask_success'])
                        else:
                            # Diagnostic tail verifies the saved endpoint even
                            # after wrapper SR has terminated at an earlier success.
                            obs, *_ = env.unwrapped.step(env.audit['actions'][t])
                            env._update(t+1, True)
                        state_error = max(state_error, tree_error(env.unwrapped.get_state_dict(), state_at(env.group['env_states'], t+1)))
                        observation_error = max(observation_error, tree_error(obs, env.group['obs'][t+1]))
                        segment_failure.extend(r for r in env.subtask_spec.failure_reasons(env.metrics,env.context) if r != 'timeout')
                    info = completion_info or info
                    endpoint_success = env.subtask_spec.check_success(env.metrics,env.context)
                    aligned = state_error <= env.runtime['state_tolerance'] and observation_error <= env.runtime['observation_tolerance']
                    # Geometric Transport/Align boundaries remain candidates even
                    # when executable: no independent expert-phase provenance.
                    eligible = bool(info['subtask_success'] and endpoint_success and not segment_failure and aligned and name in ('reach', 'grasp', 'insert'))
                    row.update(replay_success=info['subtask_success'], steps=info['subtask_step'],
                               max_state_error=state_error, max_observation_error=observation_error,
                               alignment_valid=aligned, bc_eligible=eligible,
                               eligibility_status='verified' if eligible else 'candidate',
                               failure_reason=info['failure_reason'], full_task_success=info['full_task_success'],
                               completion_source_t=i+info['subtask_step'], segment_end_t=j,
                               segment_endpoint_success=endpoint_success,
                               segment_failure_reasons=sorted(set(segment_failure)),
                               boundary_provenance='geometric milestone; expert phase unavailable')
                    if render:
                        Image.fromarray(env.render()[0].cpu().numpy()).save(output/f'{eid}_{name}_end.png')
                except InvalidSubtaskReset as exc:
                    row['failure_reason'] = str(exc)
                    row['prefix_state_error'] = env.max_prefix_state_error
                    row['prefix_observation_error'] = env.max_prefix_observation_error
                records.append(row)
                print(row, flush=True)
            write_json(output/'replay.json', dict(records=records, versions=versions(), backend=str(env.unwrapped.backend)))
        finally:
            env.close()
    summary = {}
    for name in NAMES:
        rows = [r for r in records if r['subtask'] == name]
        valid = [r for r in rows if r['reset_valid']]
        successful = [r for r in valid if r['replay_success']]
        summary[name] = dict(attempted=len(rows), start_valid=len(valid), replay_success=len(successful),
                             eligible_episodes=sum(r['bc_eligible'] for r in rows),
                             avg_steps=float(np.mean([r['steps'] for r in successful])) if successful else None,
                             replay_success_denominator=len(valid))
    payload = dict(records=records, summary=summary, versions=versions(), index=str(Path(index_path).resolve()),
                   index_sha256=sha256(index_path), backend='physx_cpu',
                   eligibility_note='Transport/Align remain candidates due to geometric boundaries')
    write_json(output/'replay.json', payload)
    write_json(output/'evaluation_starts.json', dict(index=payload['index'], index_sha256=payload['index_sha256'],
                                                    starts=starts, prefix_steps_excluded=True))
    # A sidecar deliberately leaves original 91-segment eligibility untouched.
    write_json(output/'eligibility.json', dict(index_sha256=payload['index_sha256'], records=[
        {k:r.get(k) for k in ('episode_id','subtask','bc_eligible','eligibility_status','failure_reason')} for r in records]))
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', default='outputs/subtasks/segments/index.json')
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', nargs='+', type=int)
    p.add_argument('--render', action='store_true')
    a = p.parse_args()
    print(run(a.index, a.output, a.episodes, a.render)['summary'])


if __name__ == '__main__':
    main()
