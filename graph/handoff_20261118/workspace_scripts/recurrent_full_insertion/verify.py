"""Verify the action-aligned dataset, checkpoint selection and paired closed-loop tests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT=Path(__file__).resolve().parent
REMOTE='--remote' in sys.argv
RUN=ROOT if REMOTE else ROOT/'source_runs'
BASE=ROOT.parent/'recurrent_new_frontend'
OUT=BASE if REMOTE else BASE/'source_runs'
FRONT=ROOT.parent/'full_insertion_frontend'
if not REMOTE:
    FRONT=FRONT/'source_runs'
FRONT=FRONT/'trained_until_plateau'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_run(root,s):
    meta=read(root/'experiments'/s['run_id']/'metadata.json')
    assert meta['status']=='finished' and not meta['git_dirty']
    assert meta['git_commit'].startswith('b2e8e4c')


def demos():
    sources,tally=[],Counter()
    drift=np.zeros(2)
    for folder in sorted((RUN/'demos').glob('shard_*')):
        s=read(folder/'summary.json')
        check_run(folder,s)
        path=folder/'demos.npz'
        initial=OUT/folder.name/'demos.npz'
        assert sha(path)==s['sha256'] and sha(initial)==s['config']['original_sha256']
        assert s['config']['tracker_sha256']==sha(FRONT/'tracker/best.pt')
        assert s['config']['gnn_sha256']==sha(FRONT/'gnn/best.pt')
        assert s['config']['script_sha256']==sha(ROOT/'prepare_demos.py')
        drift=np.maximum(drift,[s['config']['max_motion_drift_mm'],s['config']['max_pose_drift_deg']])
        sources.append(dict(run_id=s['run_id'],sha256=s['sha256']))
        with np.load(path,allow_pickle=False) as a,np.load(initial,allow_pickle=False) as b:
            for key in b.files:
                if key!='state36_transformer':
                    np.testing.assert_array_equal(a[key],b[key])
            assert not np.array_equal(a['state36_transformer'][:,15:],b['state36_transformer'][:,15:])
            np.testing.assert_array_equal(a['state36_transformer'][:,:15],b['state36_transformer'][:,:15])
            assert np.isfinite(a['state36_transformer']).all()
            tally['frames']+=len(a['expert_action'])
            tally['train']+=int((a['reset_seed']<3100120).sum())
            tally['validation']+=int((a['reset_seed']>=3100120).sum())
    assert len(sources)==8 and tally==Counter(train=116,validation=21,frames=16693)
    assert (drift<=.05).all()
    return sources,tally,drift.tolist()


def models(sources):
    hashes,training=[],[]
    for seed in range(5):
        part=RUN/'train'/f'mamba_s{seed}'
        s=read(part/'summary.json')
        old=read(OUT/'train'/f'mamba_s{seed}'/'summary.json')
        check_run(part,s)
        cfg=s['config']
        assert cfg['sources']==sources
        assert cfg['script_sha256']==sha(BASE/'train_mamba.py')
        comparable=('prior_sha256','alpha','batch','steps','interval','patience_checks','minimum_gain',
                    'init_lr','final_lr','normalization','filter','counts','selection')
        assert all(cfg[key]==old['config'][key] for key in comparable)
        best=min(s['history'],key=lambda v:v['validation']['balanced'])
        assert best['step']==s['best_step'] and best['validation']['balanced']==s['best_loss']
        assert s['checkpoint_sha256']==sha(part/'last.pt')
        hashes.append(s['checkpoint_sha256'])
        training.append(dict(seed=seed,start=s['history'][0]['validation'],best=best['validation'],
                             selected=best['step'],end=s['final_step'],stop=s['stop'],
                             old_best=old['best_loss']))
    return hashes,training


def paired(hashes):
    summary={k:Counter() for k in ('fixed','moving')}
    details={}
    for low in range(0,32,4):
        part=f'shard_{low}_{low+4}'
        old_dir,new_dir=OUT/'eval'/part,RUN/'eval'/part
        old_summary,new_summary=[read(path/'summary.json') for path in (old_dir,new_dir)]
        check_run(new_dir,new_summary)
        a,b=(r['config'] for r in (old_summary,new_summary))
        for key in ('camera','no_filter','fixed_horizon','moving_start','seeds',
                    'old_recurrent_sha256','fixed_source_sha256','moving_source_sha256'):
            assert a[key]==b[key]
        assert b['recurrent_sha256']==hashes
        assert b['tracker_sha256']==sha(FRONT/'tracker/best.pt') and b['gnn_sha256']==sha(FRONT/'gnn/best.pt')
        assert b['script_sha256']==sha(BASE/'evaluate_mamba.py')
        old,new=(jsonl(p/'rollouts.jsonl') for p in (old_dir,new_dir))
        assert [r['seed'] for r in new]==b['seeds']
        for x,y in zip(old,new,strict=True):
            assert x['seed']==y['seed']
            seed=x['seed']
            details[seed]={}
            for k in ('fixed','moving'):
                r,s=x[k],y[k]
                assert s['success']==(s['end_reason']=='success')
                assert all(f['object21']==f['filtered'] for f in s['trace'])
                np.testing.assert_allclose(r['trace'][0]['object21_gt'],s['trace'][0]['object21_gt'],atol=5e-5,rtol=0)
                summary[k].update(old_success=int(r['success']),new_success=int(s['success']),
                                  both=int(r['success'] and s['success']),
                                  old_only=int(r['success'] and not s['success']),
                                  new_only=int(s['success'] and not r['success']),total=1)
                item=dict(old=bool(r['success']),new=bool(s['success']),old_steps=r['steps'],new_steps=s['steps'])
                if k=='moving':
                    assert r['motion_plan']==s['motion_plan']
                    summary[k].update(actually_moved=int(s['box_max_shift_mm']>1))
                    item.update(old_shift_mm=r['box_max_shift_mm'],new_shift_mm=s['box_max_shift_mm'])
                details[seed][k]=item
    assert summary['fixed']['total']==summary['moving']['total']==32
    return summary,details


def main():
    sources,count,drift=demos()
    hashes,training=models(sources)
    result,details=paired(hashes)
    payload=dict(dataset=count,max_state_drift=dict(position_mm=drift[0],orientation_deg=drift[1]),
                 training=training,closed_loop=result,paired_seeds=details)
    (RUN/'verified.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    for m in training:
        print(m['seed'],m['start']['balanced'],m['best']['balanced'],m['old_best'],m['selected'],m['end'],m['stop'])
    print('Notable seeds',{s:details[s] for s in (700674,700882)})


if __name__=='__main__':
    main()
