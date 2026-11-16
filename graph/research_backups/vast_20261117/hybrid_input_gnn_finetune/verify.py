"""Independently verify split, frozen inputs, stopping schedule and test error."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
RUN = ROOT / 'source_runs'
MIXED_GNN = OUT / 'moving_box_gt20_training/remote/train/best.pt'
TRACKER = OUT / 'spatiotemporal_corner_tracker_convergence/source_runs/train/spatial_temporal/best.pt'
HYBRID = OUT / 'spatiotemporal_rigid_box_eval/source_runs/eval/predictions.npz'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(folder, summary):
    meta = read(folder / 'experiments' / summary['run_id'] / 'metadata.json')
    assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c')
    assert not meta['git_dirty']


def lateral(pred, gt):
    q = gt[:, 13:17].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xyz = -q[:, 1:]
    vec = pred[:, 7:10].astype(float) - gt[:, 7:10]
    cross = 2 * np.cross(xyz, vec)
    return np.linalg.norm((vec + q[:, :1] * cross + np.cross(xyz, cross))[:, 1:], axis=1) * 1000


def verify_schedule(run, history):
    cfg = run['config']
    rates = cfg['lr_schedule']
    min_gain, patience = cfg['min_relative_val_gain'], cfg['patience']
    best = threshold = history[0]['validation']['loss']
    assert best == run['initial_validation_loss']
    last_gain = lr_index = 0
    events, stopped = [], False
    for epoch, row in enumerate(history[1:], 1):
        assert row['epoch'] == epoch and row['lr'] == rates[lr_index] and not stopped
        loss = row['validation']['loss']
        best = min(best, loss)
        if loss <= threshold * (1 - min_gain):
            threshold, last_gain = loss, epoch
        if epoch - last_gain >= patience:
            if lr_index + 1 == len(rates):
                stopped = True
            else:
                lr_index += 1
                events.append(dict(epoch=epoch, next_lr=rates[lr_index], best_val_loss=best))
                threshold, last_gain = best, epoch
    assert run['lr_events'] == events and run['final_epoch'] == epoch
    assert run['best_validation_loss'] == best
    assert run['best_epoch'] == min(history, key=lambda e: e['validation']['loss'])['epoch']
    if stopped:
        assert run['stop_reason'] == 'plateau_at_lowest_lr'
    else:
        assert run['stop_reason'] == 'epoch_limit' and epoch == cfg['training']['epochs']


def main():
    cache = RUN / 'cache'
    train = RUN / 'train'
    collection = read(cache / 'summary.json')
    report = read(train / 'summary.json')
    check_run(cache, collection)
    check_run(train, report)
    assert collection['config']['tracker_sha256'] == sha(TRACKER)
    assert collection['config']['prior_hybrid_sha256'] == sha(HYBRID)
    assert collection['config']['cache_script_sha256'] == sha(ROOT / 'cache_inputs.py')
    assert report['config']['initial_sha256'] == sha(MIXED_GNN)
    assert report['config']['script_sha256'] == sha(ROOT / 'train_finetune.py')
    assert report['best_sha256'] == sha(train / 'best.pt')
    assert report['resume_sha256'] == sha(train / 'resume.pt')
    assert report['config']['inputs_sha256'] == collection['inputs_sha256'] == sha(cache / 'input.npz')
    verify_schedule(report, read(train / 'history.json'))
    with np.load(cache / 'input.npz', allow_pickle=False) as inputs, \
            np.load(train / 'test_predictions.npz', allow_pickle=False) as test, \
            np.load(HYBRID, allow_pickle=False) as prior:
        seed = inputs['seed']
        masks = [seed < 3100120, (3100120 <= seed) & (seed < 3100144), seed >= 3100144]
        assert [len(np.unique(seed[mask])) for mask in masks] == [120, 24, 32]
        assert [int(mask.sum()) for mask in masks] == [13448, 2706, 3352]
        heldout = masks[-1]
        for k in ('seed', 'step', 'target'):
            np.testing.assert_array_equal(test[k], inputs[k][heldout])
            np.testing.assert_array_equal(test[k], prior[k])
        np.testing.assert_allclose(inputs['xyz'][heldout], prior['xyz_hybrid'], atol=3e-6, rtol=0)
        np.testing.assert_allclose(test['gt_trained'], prior['hybrid'], atol=4e-5, rtol=0)
        last = np.zeros(int(heldout.sum()), dtype=bool)
        for scene in np.unique(test['seed']):
            frames = (test['seed'] == scene) & (inputs['phase'][heldout] == 'align') & inputs['grasp'][heldout]
            last[np.flatnonzero(frames)[-5:]] = True
        assert last.sum() == 160
        for name in ('gt_trained', 'hybrid_finetuned'):
            error = lateral(test[name], test['target'])
            for key, frames in [('all', np.ones_like(last)), ('moving', inputs['moving_now'][heldout]),
                                ('align_last5', last)]:
                expected = report['test'][name][key]['lateral_mm']['p90']
                np.testing.assert_allclose(np.percentile(error[frames], 90), expected, atol=3e-5, rtol=0)
    print('Verified 120/24/32-scene split, checkpoint and cache hashes, LR schedule, and held-out P90.')


if __name__ == '__main__':
    main()
