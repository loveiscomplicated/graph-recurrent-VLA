"""Audit that the BC input mixture is the experimental variable."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
TRAIN=ROOT.parent/'recurrent_new_frontend/train_mamba.py'
EVAL=ROOT.parent/'recurrent_new_frontend/evaluate_mamba.py'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_run(root,report):
    meta=read(root/'experiments'/report['run_id']/'metadata.json')
    assert meta['status']=='finished' and not meta['git_dirty']
    assert meta['git_commit'].startswith('b2e8e4c')


def train_stats(mixed,pure):
    scores,hashes=[],[]
    for i in range(5):
        a,b=(path/'train'/f'mamba_s{i}' for path in (mixed,pure))
        left,right=[read(p/'summary.json') for p in (a,b)]
        verify_run(b,right)
        ca,cb=(r['config'] for r in (left,right))
        assert ca['alpha']==[0.,.25,.5,.75,1.]
        assert cb['alpha']==[1.]*5
        for key in ca:
            if key!='alpha':
                assert ca[key]==cb[key],key
        assert cb['script_sha256']==sha(TRAIN)
        baseline=left['history'][0]['validation']
        for key,value in baseline.items():
            np.testing.assert_allclose(value,right['history'][0]['validation'][key],atol=1e-7,rtol=0)
        best=min(right['history'],key=lambda item:item['validation']['balanced'])
        assert best['step']==right['best_step'] and best['validation']['balanced']==right['best_loss']
        assert sha(b/'last.pt')==right['checkpoint_sha256']
        hashes.append(right['checkpoint_sha256'])
        scores.append(dict(seed=i,mixed=dict(loss=left['best_loss'],step=left['best_step'],
                                             end=left['final_step'],stop=left['stop']),
                           graph_only=dict(loss=right['best_loss'],step=right['best_step'],
                                           end=right['final_step'],stop=right['stop'])))
    return scores,hashes


def evaluate_stats(mixed,pure,hashes):
    totals={k:Counter() for k in ('fixed','moving')}
    per_seed={}
    for i in range(0,32,4):
        label=f'shard_{i}_{i+4}'
        a,b=(path/'eval'/label for path in (mixed,pure))
        left,right=[read(p/'summary.json') for p in (a,b)]
        ca,cb=(r['config'] for r in (left,right))
        verify_run(b,right)
        assert {k:v for k,v in ca.items() if k!='recurrent_sha256'}=={k:v for k,v in cb.items() if k!='recurrent_sha256'}
        assert cb['no_filter'] and cb['fixed_horizon']==250
        assert cb['recurrent_sha256']==hashes and cb['script_sha256']==sha(EVAL)
        rows_a,rows_b=[jsonl(p/'rollouts.jsonl') for p in (a,b)]
        assert [r['seed'] for r in rows_a]==[r['seed'] for r in rows_b]==cb['seeds']
        for condition in totals:
            assert sum(row[condition]['success'] for row in rows_b)==right['success'][condition]
        for x,y in zip(rows_a,rows_b,strict=True):
            seed=x['seed']
            assert seed not in per_seed
            per_seed[seed]={}
            for condition in totals:
                r,s=x[condition],y[condition]
                assert s['success']==(s['end_reason']=='success') and s['steps']==len(s['trace'])
                np.testing.assert_allclose(r['trace'][0]['object21_gt'],s['trace'][0]['object21_gt'],
                                           atol=5e-5,rtol=0)
                np.testing.assert_allclose(r['trace'][0]['object21'],s['trace'][0]['object21'],
                                           atol=5e-5,rtol=0)
                assert all(f['object21']==f['filtered'] for f in s['trace'])
                old,new=bool(r['success']),bool(s['success'])
                totals[condition].update(mixed=int(old),graph_only=int(new),
                                         both=int(old and new),mixed_only=int(old and not new),
                                         graph_only_only=int(new and not old),total=1)
                item=dict(mixed=old,graph_only=new,mixed_steps=r['steps'],graph_only_steps=s['steps'])
                if condition=='moving':
                    assert r['motion_plan']==s['motion_plan']
                    totals[condition].update(mixed_actually_moved=int(r['box_max_shift_mm']>1),
                                             graph_only_actually_moved=int(s['box_max_shift_mm']>1))
                    item.update(mixed_shift_mm=r['box_max_shift_mm'],graph_only_shift_mm=s['box_max_shift_mm'])
                per_seed[seed][condition]=item
    assert len(per_seed)==32
    return totals,per_seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remote',action='store_true')
    parser.add_argument('--training-only',action='store_true')
    args=parser.parse_args()
    mixed=ROOT.parent/'recurrent_full_insertion'
    pure=ROOT
    if not args.remote:
        mixed,pure=mixed/'source_runs',pure/'source_runs'
    training,hashes=train_stats(mixed,pure)
    if args.training_only:
        print(json.dumps(training,indent=2))
        return
    totals,paired=evaluate_stats(mixed,pure,hashes)
    report=dict(training=training,closed_loop=totals,paired_seeds=paired)
    (pure/'verified.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(totals,indent=2))
    for row in training:
        print(row)
    print("700674",paired[700674],"700882",paired[700882])


if __name__=='__main__':
    main()
