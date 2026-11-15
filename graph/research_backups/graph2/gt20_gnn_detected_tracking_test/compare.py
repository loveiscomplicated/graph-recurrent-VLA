"""Check matched frames and compare actual detector/track inputs against GT20."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
GT20=ROOT.parent/'gt20_object21_retraining/remote/train'


def read(path):
    return json.loads(path.read_text())


def rot(q,v):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    t=2*np.cross(q[:,1:],v)
    return v+q[:,:1]*t+np.cross(q[:,1:],t)


def evaluate(pred,gt):
    delta=pred-gt
    axes=np.repeat([[1.,0.,0.]],len(gt),axis=0)
    def angle(a,b):
        return np.degrees(np.arccos(np.clip((rot(a,axes)*rot(b,axes)).sum(axis=1),-1,1)))
    inv=gt[:,13:17].copy()
    inv[:,1:]*=-1
    local=rot(inv,delta[:,7:10])
    return dict(peg_center_mm=np.linalg.norm(delta[:,:3],axis=1)*1000,
                hole_center_mm=np.linalg.norm(delta[:,10:13],axis=1)*1000,
                lateral_mm=np.linalg.norm(local[:,1:],axis=1)*1000,
                relative_3d_mm=np.linalg.norm(delta[:,7:10],axis=1)*1000,
                peg_tip_mm=np.linalg.norm(delta[:,7:10]+delta[:,10:13],axis=1)*1000,
                peg_axis_deg=angle(pred[:,3:7],gt[:,3:7]),
                hole_axis_deg=angle(pred[:,13:17],gt[:,13:17]))


def statistics(data):
    p50,p90=np.quantile(data,[.50,.90])
    return dict(p50=float(p50),p90=float(p90),mean=float(data.mean()))


def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source=ROOT/'source_runs'
    rows,runs,terminals=[],[],{}
    for start in range(3100144,3100176,8):
        path=source/f'shard_{start}_{start+8}'
        run=read(path/'summary.json')
        metadata=read(path/'experiments'/run['run_id']/'metadata.json')
        assert metadata['status']=='finished' and metadata['git_commit'][:7]=='2b6c40e'
        assert run['config']['seeds']==list(range(start,start+8))
        assert run['config']['reference_sha256']==hash_file(GT20/'test_predictions.npz')
        assert run['config']['retrained_sha256']==hash_file(GT20/'best.pt')
        assert run['config']['script_sha256']==hash_file(ROOT/'evaluate.py')
        shard=[json.loads(l) for l in (path/'frames.jsonl').read_text().splitlines()]
        assert len(shard)==sum(run['frames_by_seed'].values())
        rows.extend(shard)
        run['runtime']=metadata['runtime']
        runs.append(run)
        terminals.update(run['terminals'])
    assert terminals==read(GT20/'summary.json')['test_terminals']
    arrays={k:np.asarray([r[k] for r in rows]) for k in ('seed','step','phase','grasp','gt21','original','retrained')}
    with np.load(GT20/'test_predictions.npz',allow_pickle=False) as ref:
        assert all(np.array_equal(arrays[k],ref[k]) for k in ('seed','step','phase','grasp'))
        np.testing.assert_allclose(arrays['gt21'],ref['target'],atol=1e-6,rtol=0)
        max_gt_delta=float(np.abs(arrays['gt21']-ref['target']).max())
        predictions=dict(actual_original=arrays['original'],actual_retrained=arrays['retrained'],
                         gt_original=ref['original'],gt_retrained=ref['retrained'])
    gt=arrays['gt21']
    errors={k:evaluate(v,gt) for k,v in predictions.items()}
    assert all(v.shape==(len(rows),21) and np.isfinite(v).all() for v in predictions.values())
    assert all(np.isfinite(x).all() for v in errors.values() for x in v.values())
    steps,seeds,phase,grasp=(arrays[k] for k in ('step','seed','phase','grasp'))
    align=grasp & (phase=='align')
    late=np.zeros(len(rows),bool)
    first=np.zeros(len(rows),bool)
    for seed in np.unique(seeds):
        same=seeds==seed
        g=np.flatnonzero(same&grasp)
        if len(g):
            first[g[0]]=True
        if terminals[str(seed)]=='insert_entry':
            late[np.flatnonzero(same&align)[-5:]]=True
    result={}
    for tag,mask in dict(all=np.ones(len(rows),bool),initial=steps==0,first_grasp=first,
                         held_align=align,last_5_held_align=late).items():
        actual_delta=(errors['actual_retrained']['lateral_mm']-errors['actual_original']['lateral_mm'])[mask]
        seed_means=np.array([actual_delta[seeds[mask]==s].mean() for s in np.unique(seeds[mask])])
        weights=np.random.default_rng(0).choice(seed_means,(5000,len(seed_means)),replace=True).mean(1)
        counts=[r['counts'] for r,take in zip(rows,mask,strict=True) if take]
        result[tag]=dict(
            frames=int(mask.sum()),seeds=len(seed_means),
            **{model:{m:statistics(v[mask]) for m,v in metrics.items()} for model,metrics in errors.items()},
            actual_lateral_paired=dict(frames_improved=int((actual_delta<0).sum()),
                                       seed_means_improved=int((seed_means<0).sum()),
                                       mean_seed_delta=statistics(seed_means),
                                       mean_seed_delta_ci95=np.quantile(weights,[.025,.975]).tolist()),
            input_counts={k:statistics(np.array([r[k] for r in counts])) for k in counts[0]})
    out=ROOT/'full'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(verified_frames=len(rows),runs=runs,terminals=terminals,
        max_gt_delta=max_gt_delta,stages=result),indent=2,allow_nan=False)+'\n')
    with (out/'frames.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
