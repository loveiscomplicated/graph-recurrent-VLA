"""Export verified segments to the unchanged Graph v1 Flat MLP tensor format."""
import argparse
import json
from pathlib import Path
import torch

from graph.prototype.common import config, write_json
from graph.prototype.data import sha256, state_at
from graph.prototype.geometry import concatenate
from graph.prototype.maniskill_gt import GTAdapter
from .dataset import load_segments
from .env import make_subtask_env, tree_error
from .specs import NAMES


def export(index_path, output, subtask='reach', eligibility=None):
    index = load_segments(index_path)
    cfg = config()
    approved = None
    if eligibility:
        evidence = json.loads(Path(eligibility).read_text())
        if evidence['index_sha256'] != sha256(index_path):
            raise ValueError('eligibility evidence belongs to another index')
        approved = {(r['episode_id'],r['subtask']):r['bc_eligible'] for r in evidence['records']}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    env = make_subtask_env(subtask, index=index_path)
    adapter = GTAdapter(cfg['peg_points'],cfg['hole_points'],cfg['sampling_seed'])
    geometries, actions, ids, times = [], [], [], []
    used = []
    try:
        for ep in index['episodes']:
            s = next(s for s in ep['segments'] if s['subtask'] == subtask)
            eligible = approved.get((ep['episode_id'],subtask),False) if approved is not None else s['bc_eligible']
            if s['status'] != 'detected' or not eligible:
                continue
            env.reset(options={'source_episode_id':ep['episode_id']})
            i,j = s['start_t'],s['end_t']
            # Keep the saved segment's entire [i,j) interval for BC; independent
            # wrapper SR may terminate at an earlier qualifying success state.
            for t in range(i,j):
                geometries.append(adapter(env))
                actions.append(torch.from_numpy(env.audit['actions'][t].copy()))
                ids.append(ep['episode_id'])
                times.append(t)
                obs,*_ = env.unwrapped.step(env.audit['actions'][t])
                if tree_error(obs,env.group['obs'][t+1]) > .001:
                    raise ValueError('export replay diverged')
                if tree_error(env.unwrapped.get_state_dict(),state_at(env.group['env_states'],t+1)) > .001:
                    raise ValueError('export state replay diverged; use collection runtime')
            used.append(ep['episode_id'])
    finally:
        env.close()
    if not actions:
        raise ValueError('no verified eligible segments')
    manifest = dict(config=cfg, control_mode=index['control_mode'],
                    splits={k:[eid for eid in v if eid in used] for k,v in index['splits'].items()},
                    source_index=str(Path(index_path).resolve()), index_sha256=sha256(index_path),
                    eligibility=str(eligibility) if eligibility else 'index bc_eligible flags',
                    subtask=subtask, source_uids={str(ep['episode_id']):next(s for s in ep['segments'] if s['subtask']==subtask).get('source_uid',f"legacy91:{ep['episode_id']}")
                                                for ep in index['episodes'] if ep['episode_id'] in used},
                    scope='existing Graph v1 geometry format; no model or feature changes')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(geometry=concatenate(geometries).as_dict(),actions=torch.stack(actions),
                    episode_id=torch.tensor(ids),t=torch.tensor(times),manifest=manifest),output)
    write_json(output.with_suffix('.json'),dict(manifest=manifest,episodes=len(used),actions=len(actions),sha256=sha256(output)))
    return dict(episodes=len(used),actions=len(actions),splits={k:len(v) for k,v in manifest['splits'].items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',default='outputs/subtasks/segments/index.json')
    p.add_argument('--output',required=True)
    p.add_argument('--subtask',choices=NAMES,default='reach')
    p.add_argument('--eligibility',help='independent replay eligibility.json sidecar')
    a = p.parse_args()
    torch.set_num_threads(4)
    print(export(a.index,a.output,a.subtask,a.eligibility))


if __name__ == '__main__':
    main()
