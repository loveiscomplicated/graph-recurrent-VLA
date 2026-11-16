"""Independently audit the saved held-out Box ID assignments."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_runs'
REFERENCE = ROOT.parent / 'moving_box_gt20_training/remote/train/test_moving.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    folder = SOURCE / 'eval'
    summary = read(folder / 'summary.json')
    config = summary['config']
    assert config['matcher_sha256'] == sha(ROOT / 'box_matcher.py')
    assert config['runner_sha256'] == sha(ROOT / 'evaluate_ids.py')
    meta = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
    assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c')
    for capture in sorted(SOURCE.glob('shard_*')):
        if not capture.is_dir():
            continue
        r = read(capture / 'summary.json')
        assert r['config']['script_sha256'] == sha(ROOT / 'collect_ransac.py')
        assert r['sha256'] == sha(capture / 'detections.npz')
        m = read(capture / 'experiments' / r['run_id'] / 'metadata.json')
        assert m['plane_fit_method'] == 'ransac' and m['status'] == 'finished'
    with np.load(folder / 'assignments.npz', allow_pickle=False) as a, \
            np.load(REFERENCE, allow_pickle=False) as t:
        test = a['test']
        assert test.sum() == 3352 and len(np.unique(a['seed'][test])) == 32
        for k in ('seed','step'):
            np.testing.assert_array_equal(a[k][test], t[k])
        ends = np.cumsum(a['raw_count'])
        starts = ends - a['raw_count']
        idx, assigned = a['assigned_candidate_index'], a['assigned']
        assert (idx[assigned] >= np.broadcast_to(starts[:, None], idx.shape)[assigned]).all()
        assert (idx[assigned] < np.broadcast_to(ends[:, None], idx.shape)[assigned]).all()
        for row in idx:
            used = row[row >= 0]
            assert len(set(used)) == len(used)
        np.testing.assert_array_equal(a['raw_xyz'][idx[assigned]], a['assigned_xyz'][assigned])
        delta = a['assigned_xyz'].astype(float)[:, :, None] - a['gt_xyz'].astype(float)[:, None]
        dist = np.linalg.norm(delta, axis=-1)
        near = dist.min(axis=-1) <= .005
        right = near & assigned & (dist.argmin(axis=-1) == np.arange(8))
        wrong = near & assigned & ~right
        for name, frames in [('all', test), ('motion', test & a['moving']), ('last5', a['last5']),
                             ('first', test & (a['step'] == 0))]:
            r = summary['test'][name]
            assert r['assigned'] == assigned[frames].sum()
            assert r['correct'] == right[frames].sum()
            assert r['wrong_id'] == wrong[frames].sum()
            assert r['could_match'] == a['oracle_available'][frames].sum()
            assert r['correct_of_could_match'] == (right & a['oracle_available'])[frames].sum()
    print('Verified all 3352 held-out frames, one-to-one assignments and accuracy counts.')


if __name__ == '__main__':
    main()
