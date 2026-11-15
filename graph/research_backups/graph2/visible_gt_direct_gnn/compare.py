"""Summarize direct GT GNN inference and verify paired evaluation frames."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'hole_front_object21_ablation/full'
LATEST = ROOT.parent / 'hole_front_object21_latest_detector/full'
COMMON_METRICS = ('hole_center_mm', 'axis_deg', 'tip_lateral_mm')


def read(path):
    return json.loads(path.read_text())


def frames(root):
    return [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]


def stats(values):
    p50, p90 = np.quantile(values, [.5,.9])
    return dict(p50=float(p50), p90=float(p90))


def summarize():
    old, new = frames(PRIOR), frames(LATEST)
    old_summary = read(PRIOR/'summary.json')
    terminals, direct, runs = {}, [], []
    for start in range(200,1400,300):
        folder = ROOT/'source_runs'/f'shard_{start}_{start+300}'
        s = read(folder/'summary.json')
        cfg=s['config']
        meta=read(folder/'experiments'/s['run_id']/'metadata.json')
        assert meta['status']=='finished' and meta['git_commit'].startswith('2b6c40e')
        assert cfg['gnn_sha256']==old_summary['config']['checkpoint_sha256']['gnn']
        assert cfg['camera']==old_summary['config']['camera']
        assert cfg['seeds']==list(range(start,start+300))
        direct.extend(frames(folder))
        terminals.update(s['expert_terminals'])
        runs.append(dict(run_id=s['run_id'],config=cfg,runtime=meta['runtime']))
    assert terminals==old_summary['expert_terminals']
    assert len({(r['seed'],r['step']) for r in direct})==len(direct)==len(old)==len(new)
    groups, tails=defaultdict(list),defaultdict(list)
    for a,b,c in zip(old,new,direct,strict=True):
        id_key=lambda r:(r['seed'],r['step'],r['stages'])
        assert id_key(a)==id_key(b)==id_key(c)
        assert a['gt21']==b['gt21']==c['gt21']
        assert a['counts']['visible_gt_front']==c['counts']['hole_rim']
        item=(a,b,c)
        for stage in c['stages']:
            groups[stage].append(item)
        if 'held_align' in c['stages'] and terminals[str(c['seed'])]=='insert_entry':
            tails[c['seed']].append(item)
    groups['last_5_held_align']=[x for items in tails.values() for x in items[-5:]]
    result={}
    for name,group in groups.items():
        current=[r for _,_,r in group]
        scores={m:stats([r['errors'][m] for r in current]) for m in current[0]['errors']}
        geo=np.array([r['errors']['geom_tip_lateral_mm'] for r in current])
        corrected=np.array([r['errors']['tip_lateral_mm'] for r in current])
        corrected_gain=dict(improved=int((corrected<geo-1e-5).sum()),worsened=int((corrected>geo+1e-5).sum()),
                            median_delta_mm=float(np.median(corrected-geo)))
        counts={k:stats([r['counts'][k] for r in current]) for k in current[0]['counts']}
        missing={k:sum(r['counts'][k]==0 for r in current) for k in current[0]['counts']}
        comparators={
            label:{m:stats([row[idx][condition]['errors'][m] for row in group]) for m in COMMON_METRICS}
            for label,idx,condition in (
                ('previous_detected',0,'detected'),('latest_grid_detected',1,'detected'),
                ('previous_gt_rim_tracked',0,'visible_gt_front'))}
        with_rim_4=[r for r in current if r['counts']['hole_rim']==4]
        result[name]=dict(frames=len(group),gt_direct=scores,**comparators,visible_counts=counts,
                          no_visible=missing,residual_effect=corrected_gain,
                          gt_direct_all_four_hole_rim=dict(
                              frames=len(with_rim_4),
                              **{m:stats([r['errors'][m] for r in with_rim_4]) for m in COMMON_METRICS}))
    out=ROOT/'full'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(frames=len(direct),runtime_runs=runs,
        expert_terminals=terminals,stages=result),indent=2,allow_nan=False)+'\n')
    with (out/'frames.jsonl').open('w') as f:
        for frame in direct:
            f.write(json.dumps(frame,allow_nan=False)+'\n')


if __name__=='__main__':
    summarize()
