"""Audit the fixed-Box filter ablation and compare it with the moving-Box result."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_runs'
OLD = ROOT.parent / 'closed_loop_hybrid_mamba'
MOVING = ROOT.parent / 'closed_loop_moving_box_no_filter/source_runs/summary.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def point_errors(rows):
    result = {}
    for name, key in [('raw', 'object21'), ('filtered', 'filtered')]:
        output = np.array([row[key] for row in rows], dtype=np.float64)
        gt = np.array([row['object21_gt'] for row in rows], dtype=np.float64)
        hole_mm = np.linalg.norm(output[:, 10:13] - gt[:, 10:13], axis=1) * 1000
        peg_mm = np.linalg.norm(output[:, :3] - gt[:, :3], axis=1) * 1000
        lateral = np.array([row[f'{name}_lateral_mm'] for row in rows])
        result[name] = dict(hole_center_p90_mm=float(np.percentile(hole_mm, 90)),
                            peg_center_p90_mm=float(np.percentile(peg_mm, 90)),
                            lateral_p90_mm=float(np.percentile(lateral, 90)))
    return result


def metrics(runs):
    frames = [frame for run in runs for frame in run['trace']]
    grasped = [row for row in frames if row['grasp']]
    successes = [r for r in runs if r['success']]
    return dict(success=len(successes), total=len(runs), total_frames=len(frames),
                grasp_frames=len(grasped), error=point_errors(frames),
                grasp_error=point_errors(grasped) if grasped else None,
                median_success_steps=float(np.median([r['steps'] for r in successes])) if successes else None,
                grasped=sum(r['ever_grasped'] for r in runs))


def main():
    off, on, runs = [], [], []
    for start in range(0, 32, 4):
        shard = f'shard_{start}_{start+4}'
        path = SOURCE / shard
        original_path = OLD / 'source_runs' / shard
        report = read(path / 'summary.json')
        config = report['config']
        prev_path = original_path / 'rollouts.jsonl'
        assert config['prior_rollouts_sha256'] == sha(prev_path)
        assert config['script_sha256'] == sha(ROOT / 'run_no_filter.py')
        assert config['base_script_sha256'] == sha(OLD / 'run_rollouts.py')
        assert config['filter_mode'] == 'all KinFilter outputs bypassed' and config['hole_motion'] == 'none'
        old_config = read(original_path / 'summary.json')['config']
        for field in ('hybrid_hashes', 'recurrent_hashes', 'camera', 'hole_motion', 'horizon'):
            assert config[field] == old_config[field]
        metadata = read(path / 'experiments' / report['run_id'] / 'metadata.json')
        assert metadata['status'] == 'finished' and metadata['git_commit'].startswith('b2e8e4c')
        assert not metadata['git_dirty']
        a, b = jsonl(path / 'rollouts.jsonl'), jsonl(prev_path)
        assert config['seeds'] == [r['seed'] for r in a] == [r['seed'] for r in b]
        assert report['success'] == sum(r['success'] for r in a)
        for c, old in zip(a, b, strict=True):
            d = old['hybrid']
            assert c['steps'] == len(c['trace']) <= 250
            assert c['success'] == (c['end_reason'] == 'success')
            assert all(x['object21'] == x['filtered'] for x in c['trace'])
            np.testing.assert_allclose(c['trace'][0]['object21'], d['trace'][0]['object21'], rtol=0, atol=5e-5)
            np.testing.assert_allclose(c['trace'][0]['object21_gt'], d['trace'][0]['object21_gt'], rtol=0, atol=1e-6)
            off.append(c)
            on.append(d)
        runs.append(report['run_id'])
    assert len(off) == len(on) == 32 and len({r['seed'] for r in off}) == 32
    paired = Counter((a['success'],b['success']) for a,b in zip(on,off,strict=True))
    down, up = paired[True,False], paired[False,True]
    n = down + up
    moving = read(MOVING)
    result = dict(filter_on=metrics(on), filter_off=metrics(off),
                  paired=dict(filter_on_only=down, filter_off_only=up, both=paired[True,True],
                              neither=paired[False,False],
                              exact_p=min(1.,2*sum(math.comb(n,i) for i in range(min(down,up)+1))/2**n)),
                  moving_box_comparison={k: {'success': moving[k]['success'], 'total': moving[k]['total']}
                                         for k in ('filter_on','filter_off')}, execution_ids=runs)
    (SOURCE / 'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    for row, prior in zip(off,on,strict=True):
        print(row['seed'], prior['success'], row['success'], row['steps'])


if __name__ == '__main__':
    main()
