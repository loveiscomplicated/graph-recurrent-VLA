"""Independently aggregate rigid corrections and original track errors."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'moving_box_detector_tracking_eval'))
from summarize import REFERENCE, stats

OLD=ROOT.parent/'moving_box_detector_tracking_eval/full/frames.jsonl'
MODES=('shared_hold','shared_cv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def scores(rows):
    result={}
    for family in ('box_outer','hole_front'):
        observations=sum(r['visible'][family] for r in rows)
        stack=lambda key,path:np.array([v for r in rows for v in
                                        (r[key][family] if path is None else r[key][path][family])])
        windows={}
        for subgroup in ('all_predicted','hidden'):
            baseline=stack('predicted_only' if subgroup=='all_predicted' else 'hidden_prediction',None)
            methods={'transformer':stats(baseline)}
            for mode in MODES:
                arr=stack('rigid_predicted' if subgroup=='all_predicted' else 'rigid_hidden',mode)
                assert arr.shape==baseline.shape
                methods[mode]=dict(**stats(arr),improved=int((arr<baseline).sum()))
            windows[subgroup]=methods
        coverage={}
        for mode in (*MODES,'transformer','detector'):
            block={}
            for gate in ('3','10'):
                part=[r['rigid_visible'][mode][gate][family] if mode in MODES
                      else r['tracker' if mode=='transformer' else 'detector'][gate][family] for r in rows]
                matched=sum(map(len,part))
                block[gate]=dict(matched=matched,recall=matched/observations if observations else None,
                                 zero=sum(bool(r['visible'][family]) and not p for r,p in zip(rows,part,strict=True)))
            coverage[mode]=block
        result[family]=dict(visible=observations,coverage=coverage,**windows)
    result['fits']={}
    for mode in MODES:
        pos=np.array([r['rigid_fit'][mode]['pose_m'] for r in rows])
        gt=np.array([r['offset_m'] for r in rows])
        result['fits'][mode]=dict(updated=sum(r['rigid_fit'][mode]['support']>=3 for r in rows),
                                 past_window=sum(r['rigid_fit'][mode]['age']>5 for r in rows),
                                 gt_offset_mm=stats(np.linalg.norm(pos-gt,axis=1)*1000))
    return dict(frames=len(rows),**result)


def main():
    rows,runs=[],[]
    for start in range(3100144,3100176,8):
        d=ROOT/'source_runs'/f'shard_{start}_{start+8}'
        s=json.loads((d/'summary.json').read_text())
        m=json.loads((d/'experiments'/s['run_id']/'metadata.json').read_text())
        assert m['status']=='finished' and m['git_commit'][:7]=='b2e8e4c'
        c=s['config']
        assert c['seeds']==list(range(start,start+8))
        assert c['reference_sha256']==sha(REFERENCE)
        assert c['script_sha256']==sha(ROOT/'evaluate_rigid.py')
        assert c['rigid_sha256']==sha(ROOT/'rigid_box.py')
        batch=load_rows(d/'frames.jsonl')
        assert len(batch)==sum(s['counts'].values())
        rows.extend(batch)
        runs.append(s)
    old=load_rows(OLD)
    assert len(old)==len(rows)==3352
    for a,b in zip(old,rows,strict=True):
        assert (a['seed'],a['step'],a['phase'],a['moving'],a['visible'])==(
            b['seed'],b['step'],b['phase'],b['moving'],b['visible'])
        assert a['offset_m']==b['offset_m']
    cohorts={w:[r for r in rows if r['window']==w] for w in ('before','during','after')}
    cohorts['all']=rows
    result={k:scores(v) for k,v in cohorts.items()}
    previous_rerun={k:int(sum(len(a['tracker']['3']['hole_front'])!=len(b['tracker']['3']['hole_front'])
                              for a,b in zip(old,rows,strict=True) if b['window']==k))
                    for k in ('before','during','after')}
    out=ROOT/'full'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(stages=result,runs=runs,
        baseline_3mm_match_count_difference_frames=previous_rerun),indent=2,allow_nan=False)+'\n')
    with (out/'frames.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
