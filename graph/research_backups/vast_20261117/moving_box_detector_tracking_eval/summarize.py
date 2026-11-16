"""Aggregate physical-corner errors by motion phase and verify replay alignment."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'moving_box_gt20_training/remote/train/test_moving.npz'
FAMILIES=('peg','box_outer','hole_front','hole_rear')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    if not len(values):
        return dict(n=0,p50=None,p90=None,rmse=None,within_3mm=0,within_10mm=0)
    a=np.asarray(values)
    p50,p90=np.quantile(a,[.5,.9])
    return dict(n=len(a),p50=float(p50),p90=float(p90),
                rmse=float(np.sqrt(np.mean(a**2))),
                within_3mm=int((a<=3).sum()),within_10mm=int((a<=10).sum()))


def summary(rows):
    output=dict(frames=len(rows),episodes=len({r['seed'] for r in rows}))
    for group in FAMILIES:
        visible=np.asarray([r['visible'][group] for r in rows])
        values=dict(visible_points=int(visible.sum()),visible_frames=int((visible>0).sum()),
                    id_conflicts=sum(r['id_conflicts'][group] for r in rows))
        for key in ('detector','tracker'):
            block={}
            for gate in ('3','5','10'):
                per_frame=[r[key][gate][group] for r in rows]
                lengths=np.array([len(a) for a in per_frame])
                matched=int(lengths.sum())
                block[gate]=dict(matches=matched,recall=matched/int(visible.sum()) if visible.sum() else None,
                                 zero_match_frames=int(((visible>0)&(lengths==0)).sum()),
                                 errors=stats([v for f in per_frame for v in f]))
            values[key]=block
        for key in ('predicted_only','previous_detection','hidden_prediction','observed_known'):
            values[key]=stats([error for row in rows for error in row[key][group]])
        output[group]=values
    output['counts']={}
    for k in rows[0]['counts']:
        vals=np.array([row['counts'][k] for row in rows])
        output['counts'][k]=dict(sum=int(vals.sum()),p50=float(np.median(vals)),p90=float(np.percentile(vals,90)))
    return output


def main():
    rows,runs=[],[]
    for start in range(3100144,3100176,8):
        folder=ROOT/'source_runs'/f'shard_{start}_{start+8}'
        run=json.loads((folder/'summary.json').read_text())
        m=json.loads((folder/'experiments'/run['run_id']/'metadata.json').read_text())
        assert m['status']=='finished' and m['git_commit'][:7]=='b2e8e4c'
        cfg=run['config']
        assert cfg['seeds']==list(range(start,start+8))
        assert cfg['reference_sha256']==sha(REFERENCE)
        assert cfg['script_sha256']==sha(ROOT/'evaluate_points.py')
        batch=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        assert len(batch)==sum(run['counts'].values())
        rows.extend(batch)
        runs.append(run)
    with np.load(REFERENCE,allow_pickle=False) as ref:
        assert len(rows)==len(ref['seed'])==3352
        for key in ('seed','step','phase'):
            np.testing.assert_array_equal(np.array([r[key] for r in rows]),ref[key])
        np.testing.assert_array_equal(np.array([r['moving'] for r in rows]),ref['moving_now'])
        np.testing.assert_allclose(np.array([r['offset_m'] for r in rows]),ref['offset_m'],atol=1e-7,rtol=0)
    stages={'all':summary(rows)}
    for window in ('before','during','after'):
        stages[window]=summary([r for r in rows if r['window']==window])
    out=ROOT/'full'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(stages=stages,runs=runs),indent=2,allow_nan=False)+'\n')
    with (out/'frames.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
