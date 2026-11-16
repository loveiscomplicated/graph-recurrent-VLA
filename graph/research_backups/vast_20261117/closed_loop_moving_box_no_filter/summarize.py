"""Verify the filter ablation and compare against the saved filtered rollouts."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_runs'
OLD = ROOT.parent / 'closed_loop_moving_box_mamba'
CLIPS = {700488, 700627, 700866, 700901}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(rows):
    active = [frame for row in rows for frame in row['trace'] if frame['box_moving']]
    observations = {}
    for mode, key in [('before_filter', 'object21'), ('mamba_input', 'filtered')]:
        error = [np.linalg.norm(np.asarray(r[key][10:13]) - r['object21_gt'][10:13])*1000
                 for r in active]
        field = 'raw_lateral_mm' if mode == 'before_filter' else 'filtered_lateral_mm'
        observations[mode] = dict(hole_center_p90_mm=float(np.percentile(error,90)),
                                  lateral_p90_mm=float(np.percentile([r[field] for r in active],90)))
    successes = [r for r in rows if r['success']]
    return dict(success=len(successes), total=len(rows), actual_motion=sum(r['box_max_shift_mm'] > 1 for r in rows),
                moving_frames=len(active), moving_error=observations,
                moved_mm_range=[min(r['box_max_shift_mm'] for r in rows),
                                max(r['box_max_shift_mm'] for r in rows)],
                median_success_steps=float(np.median([r['steps'] for r in successes])) if successes else None,
                before_stop_success=sum(r['trace'][-1]['box_moving'] for r in successes),
                grasped=sum(r['ever_grasped'] for r in rows))


def main():
    off, on, video_meta, runs = [], [], [], []
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
        assert config['filter_mode'] == 'all KinFilter outputs bypassed'
        old_config = read(original_path / 'summary.json')['config']
        for field in ('hybrid_hashes', 'recurrent_hashes', 'camera', 'motion_start_step', 'horizon'):
            assert config[field] == old_config[field]
        metadata = read(path / 'experiments' / report['run_id'] / 'metadata.json')
        assert metadata['status'] == 'finished' and metadata['git_commit'].startswith('b2e8e4c')
        assert not metadata['git_dirty']
        a, b = jsonl(path / 'rollouts.jsonl'), jsonl(prev_path)
        assert config['seeds'] == [r['seed'] for r in a] == [r['seed'] for r in b]
        assert report['success'] == sum(r['success'] for r in a)
        for c, old in zip(a,b,strict=True):
            d = old['hybrid']
            assert c['motion_plan'] == d['motion_plan']
            assert c['steps'] == len(c['trace']) and c['steps'] <= 250
            assert all(x['object21'] == x['filtered'] for x in c['trace'])
            np.testing.assert_allclose(c['trace'][0]['object21'], d['trace'][0]['object21'], rtol=0, atol=5e-5)
            np.testing.assert_allclose(c['trace'][0]['object21_gt'], d['trace'][0]['object21_gt'], rtol=0, atol=1e-6)
            xyz = np.asarray([x['box_offset_mm'] for x in c['trace']])
            assert (xyz[:61] == 0).all()
            np.testing.assert_array_equal((np.diff(xyz,axis=0,prepend=[[0,0,0]]) != 0).any(1),
                                          [x['box_moving'] for x in c['trace']])
            assert bool(c['movie']) == (c['seed'] in CLIPS)
            if c['movie']:
                movie = SOURCE / 'videos' / c['movie']
                assert movie.exists()
                result = json.loads(subprocess.check_output(
                    ['ffprobe','-v','error','-show_entries','stream=codec_name,nb_frames,width,height,avg_frame_rate',
                     '-select_streams','v:0','-of','json',str(movie)], text=True))['streams'][0]
                assert (result['codec_name'],result['width'],result['height']) == ('h264',512,512)
                assert int(result['nb_frames']) == c['video_frames']
                n, den = map(int, result['avg_frame_rate'].split('/'))
                assert n/den == c['video_fps']
                video_meta.append(dict(seed=c['seed'], success=c['success'], file=str(movie.relative_to(SOURCE)),
                                       box_mm=c['box_max_shift_mm'], steps=c['steps']))
            off.append(c)
            on.append(d)
        runs.append(report['run_id'])
    assert len(off) == len(on) == 32 and len({r['seed'] for r in off}) == 32
    assert len(video_meta) == len(CLIPS)
    paired = Counter((a['success'],b['success']) for a,b in zip(on,off,strict=True))
    down, up = paired[True,False], paired[False,True]
    n = down + up
    result = dict(filter_on=metrics(on), filter_off=metrics(off),
                  paired=dict(filter_on_only=down, filter_off_only=up, both=paired[True,True],
                              neither=paired[False,False],
                              exact_p=min(1.,2*sum(math.comb(n,i) for i in range(min(down,up)+1))/2**n)),
                  videos=video_meta, execution_ids=runs)
    (SOURCE / 'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    for row, prior in zip(off,on,strict=True):
        print(row['seed'], prior['success'], row['success'], row['steps'], round(row['box_max_shift_mm'],1))


if __name__ == '__main__':
    main()
