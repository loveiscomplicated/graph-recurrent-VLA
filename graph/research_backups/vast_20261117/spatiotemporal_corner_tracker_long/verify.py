"""Verify the longer run against the original 80 epochs and held-out predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'spatiotemporal_corner_tracker_extended/source_runs'
LONG = ROOT / 'source_runs'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_finished(folder, report):
    log = read(folder / 'experiments' / report['run_id'] / 'metadata.json')
    assert log['status'] == 'finished'
    assert log['git_commit'].startswith('b2e8e4c') and not log['git_dirty']


def lateral(pred, gt):
    q = gt[:, 13:17].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xyz = -q[:, 1:]
    vec = pred[:, 7:10].astype(float) - gt[:, 7:10]
    cross = 2 * np.cross(xyz, vec)
    vec_local = vec + q[:, :1] * cross + np.cross(xyz, cross)
    return np.linalg.norm(vec_local[:, 1:], axis=1) * 1000


def main():
    run_dir = LONG / 'train/spatial_temporal'
    comparison = read(run_dir / 'summary.json')
    assert_finished(run_dir, comparison)
    original = read(BASE / 'train/spatial_temporal/summary.json')
    c0, c1 = original['config'], comparison['config']
    for field in c0.keys() - {'epochs', 'patience', 'train_script_sha256'}:
        assert c0[field] == c1[field], field
    assert c1['epochs'] == 240 and c1['patience'] == 25
    assert c1['tracker_sha256'] == sha(ROOT / 'tracker.py')
    assert c1['train_script_sha256'] == sha(ROOT / 'train_tracker.py')
    assert comparison['checkpoint_sha256'] == sha(run_dir / 'best.pt')
    assert comparison['resume_sha256'] == sha(run_dir / 'resume.pt')
    best = min(comparison['history'], key=lambda ep: ep['val_missing_mean_mm'])
    assert best['epoch'] == comparison['best_epoch']
    progress = [key for key in original['history'][0] if key != 'epoch']
    prefix = comparison['history'][:len(original['history'])]
    delta = {}
    for key in progress:
        a = np.asarray([epoch[key] for epoch in original['history']])
        b = np.asarray([epoch[key] for epoch in prefix])
        delta[key] = float(np.max(np.abs(a - b)))

    older = read(BASE / 'eval/summary.json')
    report = read(LONG / 'eval/summary.json')
    assert_finished(LONG / 'eval', report)
    assert report['config']['gnn_sha256'] == older['config']['gnn_sha256']
    assert report['config']['script_sha256'] == sha(ROOT / 'evaluate_tracker.py')
    assert report['config']['predictions_sha256'] == sha(LONG / 'train/test_predictions.npz')
    name = 'spatial_temporal__real_flags'
    with np.load(BASE / 'train/test_predictions.npz') as a, np.load(LONG / 'train/test_predictions.npz') as b:
        for key in ('seed', 'step', 'xyz', 'visible', 'observed', 'detected_xyz', 'moving_now', 'target', 'tcp'):
            np.testing.assert_array_equal(a[key], b[key])
        obs = b['observed']
        assert int((~obs).sum()) == 20961
        pred = b['spatial_temporal']
        np.testing.assert_array_equal(pred[obs], b['detected_xyz'][obs])
        errors = np.linalg.norm(pred.astype(float) - b['xyz'], axis=-1)[~obs] * 1000
        p90 = report['corner']['spatial_temporal']['missing_all']['all']['all']['p90_mm']
        np.testing.assert_allclose(np.percentile(errors, 90), p90, atol=1e-4, rtol=0)
        late = np.zeros(len(obs), dtype=bool)
        for seed in np.unique(b['seed']):
            valid = (b['seed'] == seed) & (b['phase'] == 'align') & b['grasp']
            late[np.flatnonzero(valid)[-5:]] = True
        assert late.sum() == 160
    with np.load(BASE / 'eval/object21_predictions.npz') as a, np.load(LONG / 'eval/object21_predictions.npz') as b:
        for key in ('seed', 'step', 'target'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_allclose(a['clean__all_observed'], b['clean__all_observed'],
                                   atol=1e-6, rtol=0)
        p90 = report['object21'][name]['align_last5']['lateral_mm']['p90']
        np.testing.assert_allclose(np.percentile(lateral(b[name], b['target'])[late], 90),
                                   p90, atol=2e-5, rtol=0)
    print('Max absolute differences vs original history for first 80 epochs:', delta)
    print('Verified identical inputs and GNN checkpoint on 3352 held-out frames.')


if __name__ == '__main__':
    main()
