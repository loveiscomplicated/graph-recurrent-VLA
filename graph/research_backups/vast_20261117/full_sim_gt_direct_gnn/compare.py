"""Validate full and visible GT frame alignment and summarize the oracle comparison."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VISIBLE = ROOT.parent/'visible_gt_direct_gnn/full'
CONDITIONS = ('visible_gt', 'full_front20', 'full_both24')


def read(path):
    return json.loads(path.read_text())


def frames(root):
    return [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]


def stats(values):
    p50,p90 = np.quantile(values,[.5,.9])
    return dict(p50=float(p50),p90=float(p90))


def compare():
    previous = read(VISIBLE/'summary.json')
    visible = frames(VISIBLE)
    direct, runs, terminals = [], [], {}
    for start in range(200,1400,300):
        folder=ROOT/'source_runs'/f'shard_{start}_{start+300}'
        s=read(folder/'summary.json')
        cfg=s['config']
        meta=read(folder/'experiments'/s['run_id']/'metadata.json')
        assert meta['status']=='finished' and meta['git_commit'].startswith('2b6c40e')
        assert cfg['seeds']==list(range(start,start+300)) and cfg['observation']=='state_dict'
        assert cfg['gnn_sha256']==previous['runtime_runs'][0]['config']['gnn_sha256']
        direct.extend(frames(folder))
        terminals.update(s['expert_terminals'])
        runs.append(dict(run_id=s['run_id'],config=cfg,runtime=meta['runtime']))
    assert terminals==previous['expert_terminals']
    assert len({(r['seed'],r['step']) for r in direct})==len(direct)==len(visible)
    cohorts,tails=defaultdict(list),defaultdict(list)
    max_delta=0.
    for a,b in zip(visible,direct,strict=True):
        assert (a['seed'],a['step'],a['stages'])==(b['seed'],b['step'],b['stages'])
        diff=float(np.abs(np.array(a['gt21'])-b['gt21']).max())
        max_delta=max(max_delta,diff)
        assert diff<=1e-6,(a['seed'],a['step'],diff)
        row=dict(visible_gt=a['errors'],**{mode:b[mode]['errors'] for mode in CONDITIONS[1:]})
        for stage in a['stages']:
            cohorts[stage].append(row)
        if 'held_align' in a['stages'] and terminals[str(a['seed'])]=='insert_entry':
            tails[a['seed']].append(row)
    cohorts['last_5_held_align']=[row for seq in tails.values() for row in seq[-5:]]
    out={}
    for stage,rows in cohorts.items():
        out[stage]=dict(frames=len(rows))
        for mode in CONDITIONS:
            errors = {k:stats([r[mode][k] for r in rows]) for k in rows[0][mode]}
            out[stage][mode]=errors
    folder=ROOT/'full'
    folder.mkdir(exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(dict(verified_frames=len(direct),max_gt_delta=max_delta,
        terminals=terminals,runs=runs,stages=out),indent=2,allow_nan=False)+'\n')
    with (folder/'frames.jsonl').open('w') as f:
        for row in direct:
            f.write(json.dumps(row,allow_nan=False)+'\n')


if __name__=='__main__':
    compare()
