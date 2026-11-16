"""Audit paired rollout records and summarize the success comparison."""
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np
from representatives import select

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
SOURCE = ROOT / 'source_runs'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z / n
    center = (p + z*z / (2*n)) / denominator
    half = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / denominator
    return [center - half, center + half]


def action_errors(runs):
    rows = [row for run in runs for row in run['trace']]
    times = np.array([row['elapsed_ms'] for row in rows])
    held = [row for row in rows if row['grasp']]
    metrics = {'frames': len(rows), 'held_frames': len(held)}
    for name, selected in [('all', rows), ('held', held)]:
        for kind in ('raw', 'filtered'):
            e = [row[f'{kind}_lateral_mm'] for row in selected]
            if e:
                a,b = np.percentile(e, [50,90])
                metrics[f'{name}_{kind}'] = dict(p50=float(a), p90=float(b))
    motion = [row for row in rows if row['box_moving']]
    for name, selected in [('moving', motion)]:
        assert selected
        for kind, arr_key in [('raw', 'object21'), ('filtered', 'filtered')]:
            e = [row[f'{kind}_lateral_mm'] for row in selected]
            hole = [np.linalg.norm(np.array(row[arr_key][10:13]) - row['object21_gt'][10:13]) * 1000
                    for row in selected]
            metrics[f'{name}_{kind}'] = dict(p50=float(np.percentile(e,50)),
                                             p90=float(np.percentile(e,90)),
                                             hole_center_p90=float(np.percentile(hole,90)))
    metrics['moving_frames'] = len(motion)
    metrics.update(mean_inference_ms=float(times.sum(axis=1).mean()),
                   mean_decoder_ms=float(times[:, 3].mean()))
    return metrics


def probe_video(item):
    path, count, fps = item
    text = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=codec_name,width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)],
        text=True,
    )
    s, = json.loads(text)['streams']
    assert (s['codec_name'], s['width'], s['height']) == ('h264', 512, 512)
    assert int(s['nb_frames']) == count
    n, d = map(int, s['avg_frame_rate'].split('/'))
    assert n / d == fps


def main():
    records, provenance, clips = [], [], []
    fixed = load(OUT / 'closed_loop_hybrid_mamba/source_runs/summary.json')
    fixed_meta = fixed['runs'][0]['config']
    master_seeds = None
    rng = np.random.default_rng(20261117)
    new_tracker = OUT / 'spatiotemporal_corner_tracker_convergence/source_runs/train/spatial_temporal/best.pt'
    new_gnn = OUT / 'hybrid_input_gnn_finetune/source_runs/train/best.pt'
    for start in range(0, 32, 4):
        directory = SOURCE / f'shard_{start}_{start+4}'
        report = load(directory / 'summary.json')
        cfg = report['config']
        meta = load(directory / 'experiments' / report['run_id'] / 'metadata.json')
        assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c') and not meta['git_dirty']
        assert cfg['frontend_sha256'] == sha(ROOT / 'hybrid_frontend.py')
        assert cfg['script_sha256'] == sha(ROOT / 'run_rollouts.py')
        assert cfg['video_script_sha256'] == sha(ROOT / 'video.py')
        assert cfg['hole_motion'] == 'continuous horizontal translation' and cfg['motion_start_step'] == 60
        assert cfg['motion_guard_m'] == .08
        for key in ('graph_hashes','hybrid_hashes','recurrent_hashes','detectors','camera','horizon'):
            assert cfg[key] == fixed_meta[key], key
        assert cfg['hybrid_hashes'] == {'tracker': sha(new_tracker), 'gnn': sha(new_gnn)}
        assert cfg['rigid_sha256'] == sha(OUT / 'rigid_box_object21_eval/rigid_box.py')
        assert len(cfg['recurrent_hashes']) == len(set(cfg['recurrent_hashes'])) == 5
        if master_seeds is None:
            master_seeds = cfg['all_seeds']
        assert cfg['all_seeds'] == master_seeds and cfg['seeds'] == master_seeds[start:start+4]
        shard = [json.loads(line) for line in (directory / 'rollouts.jsonl').read_text().splitlines()]
        assert [row['seed'] for row in shard] == cfg['seeds']
        assert report['success'] == {m: sum(int(row[m]['success']) for row in shard) for m in ('old','hybrid')}
        for i, row in enumerate(shard, start):
            assert row['order'] == (['old','hybrid'] if i % 2 == 0 else ['hybrid','old'])
            assert row['old']['motion_plan'] == row['hybrid']['motion_plan']
            for mode in ('old','hybrid'):
                result = row[mode]
                assert result['steps'] == len(result['trace']) <= cfg['horizon']
                assert result['success'] == (result['end_reason'] == 'success')
                off = np.array([x['box_offset_mm'] for x in result['trace']])
                assert (off[:61] == 0).all()
                moved = (np.diff(off, axis=0, prepend=[[0,0,0]]) != 0).any(axis=1)
                np.testing.assert_array_equal(moved, [x['box_moving'] for x in result['trace']])
                assert result['box_moved_frames'] == moved.sum()
                if moved.any():
                    assert result['first_move_step'] == np.flatnonzero(moved)[0]
                np.testing.assert_allclose(result['box_max_shift_mm'], np.linalg.norm(off,axis=1).max(),
                                           atol=1e-9, rtol=0)
                plan = result['motion_plan']
                rng_motion = np.random.default_rng(row['seed'] + 7351)
                angle, distance = rng_motion.uniform(0,2*np.pi), rng_motion.uniform(.06,.12)
                np.testing.assert_allclose(plan['translation_m'], distance*np.array([np.cos(angle),np.sin(angle),0]),
                                           atol=1e-9,rtol=0)
                assert plan['duration_seconds'] == float(rng_motion.uniform(.8,1.4))
                assert result['box_max_shift_mm'] <= distance * 1000 + 1e-6
                assert result['video_frames'] == result['steps'] + 1 + round(result['video_fps']) // 2
                clip = directory / 'videos' / result['movie']
                clips.append((clip,result['video_frames'],result['video_fps'],mode,row['seed']))
                assert all(x['step'] == t and len(x['action']) == 7 for t,x in enumerate(result['trace']))
                assert all(len(x['object21_gt']) == len(x['object21']) == len(x['filtered']) == 21
                           for x in result['trace'])
        records.extend(dict(shard=directory.name, **r) for r in shard)
        provenance.append(dict(run_id=report['run_id'], config=cfg, runtime=meta['runtime']))
    assert [r['seed'] for r in records] == master_seeds == sorted(set(master_seeds)) == fixed_meta['all_seeds']
    assert len(clips) == 64 and len({c[0].name for c in clips}) == 64
    chosen = {(r['seed'],'hybrid') for r in select(records)}
    selected_clips = [c for c in clips if (c[4],c[3]) in chosen]
    assert len(selected_clips) == len(chosen) and all(c[0].stat().st_size > 1000 for c in selected_clips)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(probe_video, [c[:3] for c in selected_clips]))
    for row in provenance[1:]:
        for k in ('recurrent_hashes','graph_hashes','hybrid_hashes','detectors','camera','horizon','historical_eval_hashes'):
            assert row['config'][k] == provenance[0]['config'][k]
    pairs = Counter((r['old']['success'],r['hybrid']['success']) for r in records)
    old_only, new_only = pairs[(True,False)], pairs[(False,True)]
    discordant = old_only + new_only
    p = min(1.,2*sum(math.comb(discordant,k) for k in range(min(old_only,new_only)+1))/2**discordant)
    rates = {}
    for mode in ('old','hybrid'):
        runs = [r[mode] for r in records]
        success = sum(int(r['success']) for r in runs)
        moved = [r for r in runs if r['box_max_shift_mm'] > 1]
        rates[mode] = dict(moved_count=len(moved), moved_success=sum(int(r['success']) for r in moved),
                           moved_mm_min=float(min(r['box_max_shift_mm'] for r in moved)),
                           moved_mm_median=float(np.median([r['box_max_shift_mm'] for r in moved])),
                           moved_mm_max=float(max(r['box_max_shift_mm'] for r in moved)),
                           success=success, total=len(records), rate=success/len(records),
                           wilson95=wilson(success,len(records)),
                           any_grasp=sum(int(r['ever_grasped']) for r in runs),
                           median_success_steps=float(np.median([r['steps'] for r in runs if r['success']]))
                           if success else None, error=action_errors(runs))
    out = dict(rates=rates, comparison=dict(old_only=old_only, hybrid_only=new_only,
                                             both=pairs[(True,True)], neither=pairs[(False,False)],
                                             mcnemar_exact_p=p),
               same_seeds_fixed_success={k:v['success'] for k,v in fixed['rates'].items()},
               common_actual_moved=sum(r['old']['box_max_shift_mm'] > 1 and r['hybrid']['box_max_shift_mm'] > 1 for r in records),
               historical_subset_success=sum(int(r['archival_success']) for r in records), runs=provenance,
               bootstrap_note='paired empirical percentile interval, exploratory')
    differences = np.array([int(r['hybrid']['success'])-int(r['old']['success']) for r in records])
    out['difference_bootstrap95'] = np.percentile(rng.choice(differences, (10000,len(records))).mean(1),
                                                  [2.5,97.5]).tolist()
    out['downloaded_examples'] = [
        dict(seed=r['seed'], success=r['hybrid']['success'],
             box_shift_mm=r['hybrid']['box_max_shift_mm'], steps=r['hybrid']['steps'],
             path=f"{r['shard']}/videos/{r['hybrid']['movie']}")
        for r in select(records)
    ]
    (SOURCE / 'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    abbreviated = {k:v for k,v in out.items() if k != 'runs'}
    print(json.dumps(abbreviated, indent=2))
    for r in records:
        print(r['seed'], [(k,int(r[k]['success']),r[k]['steps']) for k in ('old','hybrid')])


if __name__ == '__main__':
    main()
