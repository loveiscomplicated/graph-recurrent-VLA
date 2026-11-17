"""Compare the current CUDA detector against the previous paired Object21 run."""
from collections import defaultdict
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'hole_front_object21_ablation'
sys.path.insert(0, str(PREVIOUS))
from analyze import analyze, paired, distribution, METRICS


def read_json(path):
    return json.loads(path.read_text())


def read_frames(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def merge_shards():
    previous_config = read_json(PREVIOUS / 'full/summary.json')['config']
    rows, terminals, runs = [], {}, []
    for start in range(200, 1400, 300):
        stop = start + 300
        shard = ROOT / 'source_runs' / f'shard_{start}_{stop}'
        result = read_json(shard / 'summary.json')
        config = result['config']
        metadata = read_json(shard / 'experiments' / result['run_id'] / 'metadata.json')
        assert metadata['git_commit'].startswith('2b6c40e')
        assert metadata['status'] == 'finished'
        assert config['seeds'] == list(range(start, stop))
        for key in ('camera', 'max_expert_steps', 'device', 'tracker_gate_m', 'checkpoint_sha256'):
            assert config[key] == previous_config[key], key
        assert config['detector'] == 'TorchCornerDetector.main(device=cuda:0), CUDA defaults'
        terminals.update(result['expert_terminals'])
        rows.extend(read_frames(shard / 'frames.jsonl'))
        runs.append(dict(run_id=result['run_id'], range=[start, stop], config=config,
                         runtime=metadata.get('runtime')))
    out = ROOT / 'full'
    out.mkdir(exist_ok=True)
    config = dict(runs[0]['config'], seeds=list(range(200, 1400)),
                  expert_seeds=list(range(200, 1400, 75)))
    summary = dict(config=config, expert_terminals=terminals, source_runs=runs,
                   metrics={stage: {'frames': sum(stage in r['stages'] for r in rows)}
                            for stage in ('initial', 'first_grasp', 'held_align')})
    write_json(out / 'summary.json', summary)
    with (out / 'frames.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    analyze(out)
    return rows, summary


def compare():
    updated, summary = merge_shards()
    previous = read_frames(PREVIOUS / 'full/frames.jsonl')
    previous_status = read_json(PREVIOUS / 'full/summary.json')['expert_terminals']
    assert previous_status == summary['expert_terminals']
    assert len(previous) == len(updated)
    stages, align_tails = defaultdict(list), defaultdict(list)
    max_gt_delta = 0.
    for old, new in zip(previous, updated, strict=True):
        assert (old['seed'], old['step'], old['stages']) == (new['seed'], new['step'], new['stages'])
        np.testing.assert_allclose(old['gt21'], new['gt21'], atol=1e-6, rtol=0)
        max_gt_delta = max(max_gt_delta, float(np.abs(np.asarray(old['gt21']) - new['gt21']).max()))
        assert old['counts']['visible_gt_front'] == new['counts']['visible_gt_front']
        row = dict(seed=old['seed'], old=old['detected'], new=new['detected'])
        for stage in old['stages']:
            stages[stage].append(row)
        if 'held_align' in old['stages'] and previous_status[str(old['seed'])] == 'insert_entry':
            align_tails[old['seed']].append(row)
    stages['last_5_held_align'] = [row for values in align_tails.values() for row in values[-5:]]
    result = {}
    for stage, rows in stages.items():
        result[stage] = dict(
            frames=len(rows), seeds=len({row['seed'] for row in rows}),
            **{condition: {metric: distribution([r[condition]['errors'][metric] for r in rows])
                           for metric in METRICS} for condition in ('old', 'new')},
            paired={metric: paired(rows, 'old', 'new', metric) for metric in METRICS})
    write_json(ROOT / 'full' / 'comparison.json',
               dict(frame_match_count=len(updated), max_gt_component_delta=max_gt_delta,
                    stages=result))


if __name__ == '__main__':
    compare()
