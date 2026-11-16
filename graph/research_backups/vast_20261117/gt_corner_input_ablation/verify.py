"""Verify replay identities, stored predictions, and the reported transverse P90."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'source_runs'
OUT = ROOT.parent
FAMILIES = {'peg': slice(0, 8), 'box_outer': slice(8, 16), 'hole_front': slice(16, 20)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def lateral(pred, gt):
    q = gt[:, 13:17].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xyz = -q[:, 1:]
    v = pred[:, 7:10].astype(float) - gt[:, 7:10]
    cross = 2 * np.cross(xyz, v)
    return np.linalg.norm((v + q[:, :1] * cross + np.cross(xyz, cross))[:, 1:], axis=1) * 1000


def main():
    data_ref = OUT / 'moving_box_gt20_training/remote/data_3100132_3100176/dataset.npz'
    model_root = OUT / 'moving_box_gt20_training/remote/train'
    ref_path = model_root / 'test_moving.npz'
    ref = load_npz(ref_path)
    for name in ('jitter', 'full'):
        folder = SOURCES / name
        summary = read(folder / 'summary.json')
        run = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
        assert run['status'] == 'finished' and run['git_commit'].startswith('b2e8e4c') and not run['git_dirty']
        config = summary['config']
        for field, path in [('dataset', data_ref), ('checkpoint', model_root / 'best.pt'),
                            ('reference', ref_path), ('script', ROOT / 'evaluate_ablation.py')]:
            assert config[field + '_sha256'] == digest(path), field
        pred = load_npz(folder / 'predictions.npz')
        for field in ('seed', 'step', 'target'):
            np.testing.assert_array_equal(pred[field], ref[field])
        np.testing.assert_allclose(pred['clean'], ref['mixed_model'], atol=2e-5, rtol=0)
        last = np.zeros(len(ref['seed']), dtype=bool)
        for seed in np.unique(ref['seed']):
            indices = np.flatnonzero((ref['seed'] == seed) & (ref['phase'] == 'align') & ref['grasp'])
            last[indices[-5:]] = True
        assert last.sum() == 160
        for condition, entries in summary['conditions'].items():
            for group, mask in [('moving', ref['moving_now']), ('align_last5', last)]:
                local = np.percentile(lateral(pred[condition], ref['target'])[mask], [50, 90])
                recorded = entries[group]['lateral_mm']
                np.testing.assert_allclose(local, [recorded['p50'], recorded['p90']], atol=2e-5, rtol=0)
    matches = []
    for folder in sorted(SOURCES.glob('shard_*_*')):
        if not folder.is_dir():
            continue
        summary = read(folder / 'summary.json')
        assert summary['config']['script_sha256'] == digest(ROOT / 'collect_matches.py')
        assert summary['config']['rigid_sha256'] == digest(ROOT / 'rigid_box.py')
        assert summary['matches_sha256'] == digest(folder / 'matches.npz')
        matches.append(load_npz(folder / 'matches.npz'))
    assert len(matches) == 4
    combined = {key: np.concatenate([a[key] for a in matches]) for key in matches[0]}
    prev = OUT / 'moving_box_detector_tracking_eval/full/frames.jsonl'
    rigid = OUT / 'rigid_box_tracking_eval/full/frames.jsonl'
    for i, (detection, correction) in enumerate(zip(jsonl(prev), jsonl(rigid), strict=True)):
        assert detection['seed'] == correction['seed'] == combined['seed'][i]
        assert detection['step'] == correction['step'] == combined['step'][i]
        for family, slc in FAMILIES.items():
            assert combined['visible'][i, slc].sum() == detection['visible'][family]
            assert combined['detected'][i, slc].sum() == len(detection['detector']['5'][family])
        np.testing.assert_allclose(combined['fit_m'][i], correction['rigid_fit']['shared_cv']['pose_m'],
                                   atol=1e-6, rtol=0)
    assert i + 1 == len(combined['seed']) == len(ref['seed'])
    print('Verified all 3352 saved frames, per-family visibility/detection counts and transverse P50/P90.')


if __name__ == '__main__':
    main()
