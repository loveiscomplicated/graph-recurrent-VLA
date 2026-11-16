"""Verify the held-out split, source checksums, matched observations, and summary."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source_runs'
REF = ROOT.parent / 'moving_box_gt20_training/remote/train'
MODES = ('temporal', 'spatial_temporal', 'rigid_loss')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_run(folder, summary):
    meta = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
    assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c')
    assert not meta['git_dirty']


def main():
    folder = SOURCE / 'train'
    config_hashes = {sha(ROOT / 'collect_detections.py'), sha(ROOT / 'collect_detections_initial.py')}
    counts = []
    for start in range(3100000, 3100144, 24):
        shard = SOURCE / f'shard_{start}_{start + 24}'
        summary = read(shard / 'summary.json')
        check_run(shard, summary)
        assert summary['config']['script_sha256'] in config_hashes
        data_path = shard / 'detections.npz'
        assert summary['sha256'] == sha(data_path)
        with np.load(data_path, allow_pickle=False) as data:
            assert np.unique(data['seed']).tolist() == list(range(start, start + 24))
            assert (data['observed'] <= data['visible']).all()
            assert (data['detected_xyz'][~data['observed']] == 0).all()
            counts.append(len(data['seed']))
    for mode in MODES:
        run_dir = folder / mode
        summary = read(run_dir / 'summary.json')
        check_run(run_dir, summary)
        config = summary['config']
        assert config['tracker_sha256'] == sha(ROOT / 'tracker.py')
        assert config['train_script_sha256'] == sha(ROOT / 'train_tracker.py')
        assert summary['checkpoint_sha256'] == sha(run_dir / 'best.pt')
        assert config['splits'] == [list(range(a, b)) for a, b in
                                    [(3100000, 3100120), (3100120, 3100144), (3100144, 3100176)]]
        best = min(summary['history'], key=lambda ep: ep['val_missing_mean_mm'])
        assert best['epoch'] == summary['best_epoch']
    out = read(SOURCE / 'eval/summary.json')
    check_run(SOURCE / 'eval', out)
    assert out['config']['gnn_sha256'] == sha(REF / 'best.pt')
    assert out['config']['script_sha256'] == sha(ROOT / 'evaluate_tracker.py')
    assert out['config']['predictions_sha256'] == sha(folder / 'test_predictions.npz')
    with np.load(REF / 'test_moving.npz') as ref, np.load(folder / 'test_predictions.npz') as heldout:
        assert heldout['xyz'].shape == (3352, 20, 3)
        for key in ('step', 'seed', 'target'):
            np.testing.assert_array_equal(heldout[key], ref[key])
        obs = heldout['observed']
        for mode in MODES:
            xyz = heldout[mode]
            assert np.isfinite(xyz).all()
            np.testing.assert_array_equal(xyz[obs], heldout['detected_xyz'][obs])
            result = out['corner'][mode]['missing_all']['all']['all']
            error = np.linalg.norm(xyz.astype(float) - heldout['xyz'], axis=-1) * 1000
            assert int((~obs).sum()) == result['n']
            np.testing.assert_allclose(np.percentile(error[~obs], 90), result['p90_mm'], atol=1e-4, rtol=0)
    print(f'Verified {sum(counts[:5])} train, {counts[5]} validation, and 3352 test frames.')


if __name__ == '__main__':
    main()
