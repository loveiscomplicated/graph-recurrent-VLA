"""Verify matched inputs, the detector audit and fixed-ID test predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_runs'
MODEL = ROOT.parent / 'spatiotemporal_corner_tracker_convergence/source_runs/train'
FAMILIES = {'all': slice(None), 'peg': slice(0, 8), 'box_exterior': slice(8, 16), 'hole': slice(16, 20)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def run_metadata(folder, summary):
    meta = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
    assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c')
    assert not meta['git_dirty']
    return meta


def main():
    eval_folder = SOURCE / 'eval'
    summary = read(eval_folder / 'summary.json')
    run_metadata(eval_folder, summary)
    assert summary['config']['checkpoint_sha256'] == sha(MODEL / 'spatial_temporal/best.pt')
    assert summary['config']['checkpoint_epoch'] == 954
    assert summary['config']['script_sha256'] == sha(ROOT / 'evaluate_final.py')
    for start in range(3100144, 3100176, 8):
        folder = SOURCE / f'shard_{start}_{start+8}'
        capture = read(folder / 'summary.json')
        meta = run_metadata(folder, capture)
        assert meta['plane_fit_method'] == 'grid'
        assert capture['config']['script_sha256'] == sha(ROOT / 'collect_final.py')
        assert capture['sha256'] == sha(folder / 'detections.npz')
    with np.load(eval_folder / 'predictions.npz', allow_pickle=False) as data, \
            np.load(MODEL / 'test_predictions.npz', allow_pickle=False) as baseline:
        for key in ('seed', 'step', 'visible'):
            np.testing.assert_array_equal(data[key], baseline[key])
        np.testing.assert_array_equal(data['xyz_gt'], baseline['xyz'])
        assert data['xyz_final'].shape == (3352, 20, 3) and np.isfinite(data['xyz_final']).all()
        observed = data['observed']
        np.testing.assert_array_equal(data['xyz_final'][observed], data['detected_xyz'][observed])
        assert np.max(np.linalg.norm(data['xyz_gt'][observed] - data['detected_xyz'][observed], axis=-1)) <= .005002
        error = np.linalg.norm(data['xyz_final'].astype(float) - data['xyz_gt'], axis=-1) * 1000
        for name, frames in [('moving', data['moving']), ('align_last5', data['insertion_last5']),
                             ('all', np.ones(len(observed), dtype=bool))]:
            totals = summary['detector'][name]
            assert totals['frames'] == frames.sum()
            for family, sl in FAMILIES.items():
                for key, selection in [('observed', observed), ('missing', ~observed)]:
                    values = error[frames, sl][selection[frames, sl]]
                    if not values.size:
                        continue
                    expected = summary['point'][key][name][family]
                    assert expected['n'] == values.size
                    np.testing.assert_allclose(np.percentile(values, [50, 90]),
                                               [expected['p50_mm'], expected['p90_mm']], atol=1e-4, rtol=0)
                n_visible = int(data['visible'][frames, sl].sum())
                recall = data['recall'][frames, :, sl].sum(axis=(0, 2)) / n_visible
                np.testing.assert_allclose(recall, totals[family]['recall'], atol=1e-12, rtol=0)
    print('Verified 3352 frames, detector recall, retained coordinates, and point error P50/P90.')


if __name__ == '__main__':
    main()
