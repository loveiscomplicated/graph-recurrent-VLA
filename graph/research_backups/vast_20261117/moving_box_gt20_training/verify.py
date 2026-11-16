"""Verify exact saved GT datasets and reproduce the moving test's lateral P90."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'remote'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lateral(pred,gt):
    q=gt[:,13:17].astype(float)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    xyz=-q[:,1:]
    v=pred[:,7:10]-gt[:,7:10]
    t=2*np.cross(xyz,v)
    return np.linalg.norm((v+q[:,:1]*t+np.cross(xyz,t))[:,1:],axis=1)*1000


def main():
    summary=json.loads((ROOT/'train/summary.json').read_text())
    history=json.loads((ROOT/'train/history.json').read_text())
    assert min(history,key=lambda row:row['validation']['loss'])['epoch']==summary['best_epoch']
    assert summary['config']['script_sha256']==sha(ROOT/'train_mixed.py')
    seeds=set()
    for folder in sorted(ROOT.glob('data_*')):
        if not folder.is_dir():
            continue
        s=json.loads((folder/'summary.json').read_text())
        assert s['script_sha256']==sha(ROOT/'collect_moving.py')
        data=folder/'dataset.npz'
        assert sha(data)==s['dataset_sha256']
        with np.load(data,allow_pickle=False) as a:
            local=set(a['seed'].tolist())
            assert seeds.isdisjoint(local)
            seeds|=local
            assert a['xyz'].shape==(len(a['seed']),20,3)
            for seed in local:
                group=a['seed']==seed
                start=(a['step']==0)&group
                hole=a['target'][group,10:13]+a['tcp'][group]
                initial=(a['target'][start,10:13]+a['tcp'][start])[0]
                np.testing.assert_allclose(hole-initial,a['offset_m'][group],atol=2e-6,rtol=0)
    assert seeds==set(range(3100000,3100176))
    with np.load(ROOT/'train/test_moving.npz',allow_pickle=False) as a:
        gt=a['target']
        mask=a['moving_now']
        assert set(a['seed'])==set(range(3100144,3100176))
        for model in ('fixed_model','mixed_model'):
            assert np.isfinite(a[model]).all() and a[model].shape==gt.shape
            actual=np.percentile(lateral(a[model],gt)[mask],90)
            expected=summary['test']['moving']['while_box_moving'][model]['lateral_mm']['p90']
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=0)
    print('Verified the 176 motion trajectories and held-out moving-frame P90.')


if __name__=='__main__':
    main()
