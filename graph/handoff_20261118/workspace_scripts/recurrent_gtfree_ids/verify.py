"""Audit same-weight closed-loop rollouts, changing only corner assignment."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'recurrent_full_insertion'
SLOTS=ROOT.parent/'gtfree_corner_ids'
SOURCE=ROOT.parent/'recurrent_new_frontend'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(rows):
    return [json.loads(x) for x in rows.read_text().splitlines()]


def stats(row):
    t=row['trace']
    truth=np.asarray([f['object21_gt'] for f in t])
    tip=np.asarray([f['actual_relative_mm'] for f in t])
    depth=tip[:,0]+truth[:,17]*1000
    aperture=(np.abs(tip[:,1:])<=truth[:,20,None]*1000).all(axis=1)
    in_opening=depth[aperture]
    error=np.array([f['raw_lateral_mm'] for f in t])
    bad=np.asarray([f['observed'][0]==0 for f in t])
    bands=dict(pre=depth<=0,shallow=aperture&(depth>0)&(depth<=20),deep=aperture&(depth>20))
    details=dict(success=bool(row['success']),steps=row['steps'],
                 max_depth_mm=float(in_opening.max()) if in_opening.size else None,
                 target_depth_mm=float(truth[0,17]*1000-15),
                 final_depth_mm=float(depth[-1]),zero_peg_observed_frames=int(bad.sum()),
                 box_max_shift_mm=row.get('box_max_shift_mm',0.))
    return details,{key:error[choice] for key,choice in bands.items()}


def main():
    totals={k:Counter() for k in ('fixed','moving')}
    distributions={(cond,tag):{} for cond in totals for tag in ('gt_match','gtfree')}
    detail={}
    new_hashes={module:sha(path) for module,path in (
        ('gtfree_frontend',SLOTS/'gtfree_frontend.py'),
        ('match',SLOTS/'match.py'),
        ('box_matcher',ROOT.parent/'box_corner_id_matching/box_matcher.py'))}
    for i in range(0,32,4):
        shard=f'shard_{i}_{i+4}'
        old,new=OLD/'eval'/shard,ROOT/'eval'/shard
        s_old,s_new=(read(d/'summary.json') for d in (old,new))
        metadata=read(new/'experiments'/s_new['run_id']/'metadata.json')
        assert metadata['status']=='finished' and not metadata['git_dirty']
        a,b=s_old['config'],s_new['config']
        assert all(a[k]==b[k] for k in a)
        assert b['wrapper_sha256']==sha(ROOT/'evaluate_gtfree_ids.py')
        assert b['id_code_sha256']==new_hashes
        assert b['script_sha256']==sha(SOURCE/'evaluate_mamba.py')
        assert b['no_filter'] and b['fixed_horizon']==250
        rows=[frames(path/'rollouts.jsonl') for path in (old,new)]
        for condition in totals:
            assert sum(r[condition]['success'] for r in rows[1])==s_new['success'][condition]
        assert [x['seed'] for x in rows[0]]==[y['seed'] for y in rows[1]]==b['seeds']
        for x,y in zip(*rows,strict=True):
            seed=x['seed']
            assert seed not in detail
            detail[seed]={}
            for condition in totals:
                p,q=x[condition],y[condition]
                np.testing.assert_allclose(p['trace'][0]['object21_gt'],q['trace'][0]['object21_gt'],atol=5e-5,rtol=0)
                assert all(f['object21']==f['filtered'] for f in q['trace'])
                assert q['success']==(q['end_reason']=='success')
                assert q['steps']==len(q['trace'])<=250
                success_p,success_q=bool(p['success']),bool(q['success'])
                totals[condition].update(gt_match=int(success_p),gtfree=int(success_q),
                                         gt_match_only=int(success_p and not success_q),
                                         gtfree_only=int(success_q and not success_p),
                                         both=int(success_p and success_q),total=1)
                if condition=='moving':
                    assert p['motion_plan']==q['motion_plan']
                    totals[condition].update(old_actually_moved=int(p['box_max_shift_mm']>1),
                                             new_actually_moved=int(q['box_max_shift_mm']>1))
                entry={}
                for tag,row in [('gt_match',p),('gtfree',q)]:
                    current,bands=stats(row)
                    entry[tag]=current
                    phase_data=distributions[condition,tag]
                    for phase,values in bands.items():
                        phase_data.setdefault(phase,[]).append(values)
                detail[seed][condition]=entry
    assert len(detail)==32
    metrics={}
    for (cond,tag),parts in distributions.items():
        metrics[f'{cond}_{tag}']={}
        for k,values in parts.items():
            a=np.concatenate(values)
            metrics[f'{cond}_{tag}'][k]=dict(frames=len(a),p90_mm=float(np.percentile(a,90)) if len(a) else None)
    result=dict(closed_loop=totals,paired_seeds=detail,lateral_error=metrics)
    (ROOT/'verified.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(totals,indent=2))
    for cond in totals:
        for seed,pair in detail.items():
            a,b=pair[cond]['gt_match'],pair[cond]['gtfree']
            if a['success']!=b['success'] or seed in (700674,700806,700882):
                print(cond,seed,a['success'],b['success'],a['max_depth_mm'],b['max_depth_mm'])


if __name__=='__main__':
    main()
