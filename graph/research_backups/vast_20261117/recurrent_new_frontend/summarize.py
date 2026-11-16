"""Audit synchronized datasets, five training runs and paired closed-loop results."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
SOURCE = ROOT / 'source_runs'
CONDITIONS = ('fixed', 'moving')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_run(path, summary):
    m = read(path / 'experiments' / summary['run_id'] / 'metadata.json')
    assert m['status'] == 'finished' and m['git_commit'].startswith('b2e8e4c')
    assert not m['git_dirty']


def verify_collection():
    records, sources = [], []
    cohorts = {f'{a}_{b}': Counter() for a in ('train','validation') for b in CONDITIONS}
    for start in range(3100000,3100144,18):
        shard = SOURCE / f'shard_{start}_{start+18}'
        s=read(shard/'summary.json')
        verify_run(shard,s)
        assert s['config']['script_sha256'] == sha(ROOT/'collect_expert.py')
        assert s['sha256'] == sha(shard/'demos.npz')
        sources.append(dict(run_id=s['run_id'],sha256=s['sha256']))
        assert s['config']['seed_range'] == [start,start+18]
        assert [e['seed'] for e in s['episodes']]==list(range(start,start+18))
        records.extend(s['episodes'])
        successes=[e for e in s['episodes'] if e['success']]
        with np.load(shard/'demos.npz',allow_pickle=False) as d:
            assert np.array_equal(d['reset_seed'],[e['seed'] for e in successes])
            assert d['state36_gt'].shape==d['state36_transformer'].shape==(s['frames'],36)
            assert d['expert_action'].shape==(s['frames'],7)
            np.testing.assert_array_equal(d['state36_gt'][:,:15],d['state36_transformer'][:,:15])
            offsets=d['episode_offsets']
            assert offsets[0]==0 and offsets[-1]==s['frames']
            for a,b,e in zip(offsets[:-1],offsets[1:],successes,strict=True):
                assert b-a==e['steps'] and e['insert_steps']==(d['phase'][a:b]=='insert').sum()>0
                split='train' if e['seed']<3100120 else 'validation'
                condition='moving' if e['moved_mm']>1 else 'fixed'
                counts=cohorts[f'{split}_{condition}']
                counts.update(episodes=1,frames=int(b-a),cached=e['cached'],insert_frames=e['insert_steps'])
    return records,cohorts,sources


def verify_training(sources):
    summaries, hashes = [], []
    for seed in range(5):
        run = SOURCE / 'train' / f'mamba_s{seed}'
        s=read(run/'summary.json')
        verify_run(run,s)
        assert s['config']['script_sha256']==sha(ROOT/'train_mamba.py')
        assert s['config']['sources']==sources and not s['config']['filter']
        best = min(s['history'],key=lambda item:item['validation']['balanced'])
        assert s['best_step']==best['step'] and s['best_loss']==best['validation']['balanced']
        assert s['checkpoint_sha256']==sha(run/'last.pt')
        hashes.append(s['checkpoint_sha256'])
        summaries.append(dict(seed=seed, selected=s['best_step'],end=s['final_step'],stop=s['stop'],
                              initial=s['history'][0]['validation'], best=best['validation'],seconds=s['seconds']))
    return summaries,hashes


def cohort_metrics(rows):
    success=sum(r['success'] for r in rows)
    result=dict(success=success,total=len(rows),any_grasp=sum(r['ever_grasped'] for r in rows))
    if success:
        result['median_success_steps']=float(np.median([r['steps'] for r in rows if r['success']]))
    observed=[f for r in rows for f in r['trace'] if f['grasp']]
    if observed:
        result['lat_p90_while_grasped_mm']=float(np.percentile([f['raw_lateral_mm'] for f in observed],90))
    if 'box_max_shift_mm' in rows[0]:
        moved = [row for row in rows if row['box_max_shift_mm'] > 1]
        result['actually_moved'] = len(moved)
        result['success_after_motion_setup'] = sum(row['success'] for row in moved)
    return result


def paired(old,new):
    tab=Counter(zip([r['success'] for r in old],[r['success'] for r in new],strict=True))
    lost,gained=tab[True,False],tab[False,True]
    total=lost+gained
    return dict(both=tab[True,True],old_only=lost,new_only=gained,neither=tab[False,False],
                exact_p=min(1.,2*sum(math.comb(total,i) for i in range(min(lost,gained)+1))/2**total))


def verify_evaluation(checkpoint_hashes, demo_seeds):
    data={k:[] for k in CONDITIONS}
    baselines={k:[] for k in CONDITIONS}
    run_ids=[]
    for start in range(0,32,4):
        name=f'shard_{start}_{start+4}'
        shard=SOURCE/'eval'/name
        s=read(shard/'summary.json')
        verify_run(shard,s)
        cfg=s['config']
        assert cfg['no_filter'] and cfg['recurrent_sha256']==checkpoint_hashes
        assert cfg['script_sha256']==sha(ROOT/'evaluate_mamba.py')
        assert cfg['fixed_source_sha256']==sha(OUT/'closed_loop_hybrid_mamba/run_rollouts.py')
        assert cfg['moving_source_sha256']==sha(OUT/'closed_loop_moving_box_mamba/run_rollouts.py')
        source={k: OUT/f'closed_loop_{k}_box_no_filter/source_runs'/name for k in CONDITIONS}
        previous={k:jsonl(folder/'rollouts.jsonl') for k, folder in source.items()}
        for condition,path in source.items():
            orig = read(path/'summary.json')['config']
            assert orig['filter_mode']=='all KinFilter outputs bypassed'
            assert orig['horizon']==cfg['fixed_horizon'] and orig['seeds']==cfg['seeds']
            if condition=='moving':
                assert orig['motion_start_step']==cfg['moving_start']
            assert orig['recurrent_hashes']==cfg['old_recurrent_sha256']
            assert orig['camera']==cfg['camera']
            assert orig['hybrid_hashes']=={'gnn':cfg['gnn_sha256'],'tracker':cfg['tracker_sha256']}
        rows=jsonl(shard/'rollouts.jsonl')
        assert [r['seed'] for r in rows]==cfg['seeds']
        assert not demo_seeds.intersection(cfg['seeds'])
        for k in CONDITIONS:
            assert sum(row[k]['success'] for row in rows)==s['success'][k]
            for row,prev in zip(rows,previous[k],strict=True):
                assert row['seed']==prev['seed']
                a=row[k]
                assert a['steps']==len(a['trace'])<=250
                assert all(f['filtered']==f['object21'] for f in a['trace'])
                assert a['success']==(a['end_reason']=='success')
                for field in ('object21','object21_gt'):
                    np.testing.assert_allclose(a['trace'][0][field],prev['trace'][0][field],atol=5e-5,rtol=0)
                if k=='moving':
                    assert a['motion_plan']==prev['motion_plan']
                    offsets = np.asarray([f['box_offset_mm'] for f in a['trace']])
                    np.testing.assert_allclose(a['box_max_shift_mm'], np.linalg.norm(offsets, axis=1).max(),
                                               atol=1e-9, rtol=0)
                data[k].append(a)
                baselines[k].append(prev)
        run_ids.append(s['run_id'])
    assert len(data['fixed'])==len(data['moving'])==32
    return {k:dict(old=cohort_metrics(baselines[k]),new=cohort_metrics(data[k]),
                   pairs=paired(baselines[k],data[k])) for k in CONDITIONS},run_ids


def main():
    records,cohorts,sources=verify_collection()
    training,hashes=verify_training(sources)
    scores,run_ids=verify_evaluation(hashes,{r['seed'] for r in records})
    result=dict(collection=cohorts,attempts=len(records), training=training,closed_loop=scores,eval_runs=run_ids)
    (SOURCE/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
