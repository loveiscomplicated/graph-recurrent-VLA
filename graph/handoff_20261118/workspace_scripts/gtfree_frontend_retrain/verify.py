"""Independently recompute same-slot and same-Graph held-out errors."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'source_runs'/'finetuned'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root,summary):
    meta=read(root/'experiments'/summary['run_id']/'metadata.json')
    assert meta['status']=='finished' and not meta['git_dirty']
    assert meta['git_commit'].startswith('b2e8e4c')


def lateral(pred,truth):
    q=truth[:,13:17].astype(float)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    v=pred[:,7:10].astype(float)-truth[:,7:10]
    w=2*np.cross(-q[:,1:],v)
    return np.linalg.norm((v+q[:,:1]*w+np.cross(-q[:,1:],w))[:,1:],axis=1)*1000


def main():
    dataset=read(RUNS/'dataset.json')
    tk=read(RUNS/'tracker/summary.json')
    gn=read(RUNS/'gnn/summary.json')
    for root,s in ((RUNS/'tracker',tk),(RUNS/'gnn',gn)):
        audit(root,s)
    assert sha(RUNS/'tracker/best.pt')==tk['sha256']==gn['config']['selected_tracker_sha256']
    assert sha(RUNS/'gnn/best.pt')==gn['checkpoint_sha256']
    assert gn['config']['script_sha256']==sha(ROOT/'train_gtfree.py')
    assert gn['config']['matcher_sha256']==sha(ROOT.parent/'gtfree_corner_ids/match.py')
    for p in (ROOT.parent/'full_insertion_frontend/source_runs/trained_until_plateau',):
        assert gn['config']['warm_start_sha256']==sha(p/'gnn/best.pt')
        assert tk['config']['initial_sha256']==gn['config']['baseline_tracker_sha256']==sha(p/'tracker/best.pt')
    assert tk['best_epoch']==min(tk['history'],key=lambda r:r['score'])['epoch']
    history=read(RUNS/'gnn/history.json')
    assert gn['stopping']['best_epoch']==min(history,key=lambda r:r['validation']['loss'])['epoch']
    p=RUNS/'gnn/heldout.npz'
    assert sha(p)==gn['heldout_sha256']
    with np.load(p,allow_pickle=False) as d:
        ids=np.unique(d['seed'])
        assert ids.tolist()==dataset['splits'][2]['seeds']==list(range(3100208,3100232))
        assert set(ids).isdisjoint(dataset['splits'][0]['seeds'])
        assert set(ids).isdisjoint(dataset['splits'][1]['seeds'])
        assert len(d['seed'])==dataset['splits'][2]['frames']
        obs=d['observed']
        for name in ('old_raw','new_raw','old_corrected','shared_new_gnn_input'):
            np.testing.assert_array_equal(d[name][obs],d['detected_xyz'][obs])
        depth=d['depth_mm']
        masks=dict(all=np.ones(len(depth),bool),pre=depth<=0,entry=(depth>0)&(depth<=20),deep=depth>20)
        for name,rows in masks.items():
            score=gn['geometry'][name]
            assert score['frames']==rows.sum()
            missing=~obs[rows]
            assert score['all']['n_missing']==missing.sum()
            for tag in ('old','new'):
                delta=d[tag+'_raw'][rows]-d['xyz'][rows]
                err=np.linalg.norm(delta,axis=-1)[missing]*1000
                np.testing.assert_allclose(np.percentile(err,90),score['all'][tag]['p90'],atol=1e-5,rtol=0)
            for tag in ('baseline','new_tracker','new_front'):
                e=lateral(d[tag][rows],d['target'][rows])
                np.testing.assert_allclose(np.percentile(e,90),gn['object21'][name][tag]['lateral_mm']['p90'],
                                           atol=1e-4,rtol=0)
    successes=Counter()
    for e in dataset['episodes']:
        if e['success']:
            successes['test' if e['seed']>=3100208 else 'validation' if e['seed']>=3100120 else 'train']+=1
    print('Successful demos',dict(successes))
    for phase,row in gn['geometry'].items():
        print(phase,row['frames'],row['all']['n_missing'],
              [round(row['all'][mode]['p90'],2) for mode in ('old','new')],
              [round(gn['object21'][phase][mode]['lateral_mm']['p90'],2) for mode in
               ('baseline','new_tracker','new_front')])
    print('Verified checkpoints, input mask preservation and reprojection statistics')


if __name__=='__main__':
    main()
