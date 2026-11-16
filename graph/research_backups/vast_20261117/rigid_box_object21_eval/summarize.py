"""Verify and summarize a paired detector-to-Object21 evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
TRAIN_MOVING = OUT / 'moving_box_gt20_training/remote/train'
MODELS = ('original', 'fixed', 'mixed')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate(q, vec):
    q = q.copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = 2 * np.cross(q[:, 1:], vec)
    return vec + q[:, :1] * v + np.cross(q[:, 1:], v)


def errors(pred, gt):
    gt_q = gt[:, 13:17]
    basis_x = np.array([1., 0., 0.])
    gt_inverse = gt_q * np.array([1., -1., -1., -1.])
    lateral = rotate(gt_inverse, pred[:, 7:10] - gt[:, 7:10])[:, 1:]
    delta = pred - gt

    def axis_error(pred_q, true_q):
        pred_axis = rotate(pred_q, basis_x)
        true_axis = rotate(true_q, basis_x)
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(pred_axis, true_axis), axis=-1),
                                    (pred_axis * true_axis).sum(axis=-1)))

    return {
        'transverse_mm': np.linalg.norm(lateral, axis=-1) * 1000,
        'hole_center_mm': np.linalg.norm(delta[:, 10:13], axis=-1) * 1000,
        'hole_axis_deg': axis_error(pred[:, 13:17], gt_q),
        'peg_center_mm': np.linalg.norm(delta[:, :3], axis=-1) * 1000,
        'peg_axis_deg': axis_error(pred[:, 3:7], gt[:, 3:7]),
        'relative_3d_mm': np.linalg.norm(delta[:, 7:10], axis=-1) * 1000,
        'peg_half_size_mm': np.linalg.norm(delta[:, 17:20], axis=-1) * 1000,
        'hole_radius_mm': np.abs(delta[:, 20]) * 1000,
    }


def stats(values):
    assert np.isfinite(values).all() and len(values) > 0
    p50, p90 = np.percentile(values, [50, 90])
    return dict(p50=float(p50), p90=float(p90))


def main():
    reference_path = TRAIN_MOVING / 'test_moving.npz'
    fixed = OUT / 'gt20_object21_retraining/remote/train/best.pt'
    tracking = OUT / 'rigid_box_tracking_eval/full/frames.jsonl'
    expected_hashes = dict(mixed=sha(TRAIN_MOVING / 'best.pt'), fixed=sha(fixed))
    rows, runs = [], []
    for start in range(3100144, 3100176, 8):
        folder = ROOT / 'source_runs' / f'shard_{start}_{start + 8}'
        run = json.loads((folder / 'summary.json').read_text())
        config = run['config']
        metadata = json.loads((folder / 'experiments' / run['run_id'] / 'metadata.json').read_text())
        assert metadata['status'] == 'finished' and not metadata['git_dirty']
        assert metadata['git_commit'] == 'b2e8e4c1061be53209a75513bb45f901f36bb36c'
        assert config['gnn_sha256']['original'].startswith('3c642e')
        for model, digest in expected_hashes.items():
            assert config['gnn_sha256'][model] == digest
        assert config['script_sha256'] == sha(ROOT / 'evaluate_object21.py')
        assert config['rigid_sha256'] == sha(ROOT / 'rigid_box.py')
        assert config['reference_sha256'] == sha(reference_path)
        assert config['seeds'] == list(range(start, start + 8))
        shard_rows = [json.loads(line) for line in (folder / 'frames.jsonl').read_text().splitlines()]
        assert {str(s): sum(row['seed'] == s for row in shard_rows) for s in config['seeds']} == run['counts']
        rows.extend(shard_rows)
        runs.append(dict(run_id=run['run_id'], seconds=run['seconds'], runtime=metadata['runtime'],
                         hashes=config['gnn_sha256'], tracker_sha256=config['tracker_sha256']))
    assert all(run['hashes'] == runs[0]['hashes'] for run in runs)
    ref = np.load(reference_path, allow_pickle=False)
    for key, ref_key in [('seed', 'seed'), ('step', 'step'), ('phase', 'phase'), ('grasp', 'grasp'),
                         ('moving', 'moving_now'), ('gt21', 'target')]:
        np.testing.assert_array_equal([r[key] for r in rows], ref[ref_key])
    np.testing.assert_array_equal(np.array([r['offset_m'] for r in rows], dtype=ref['offset_m'].dtype),
                                  ref['offset_m'])
    prior = [json.loads(line) for line in tracking.read_text().splitlines()]
    for row, old in zip(rows, prior, strict=True):
        assert all(row[key] == old[key] for key in ('seed', 'step', 'window'))
        assert row["fit"] == old["rigid_fit"]["shared_cv"]
    corrected = np.array([r['corrected_box_points'] > 0 for r in rows])
    cohorts = {'all': np.ones(len(rows), dtype=bool), 'corrected': corrected}
    for window in ('before', 'during', 'after'):
        cohorts[window] = np.array([row['window'] == window for row in rows])
    align = np.zeros(len(rows), dtype=bool)
    for seed in np.unique(ref['seed']):
        tail = np.flatnonzero((ref['seed'] == seed) & (ref['phase'] == 'align') & ref['grasp'])[-5:]
        assert len(tail) == 5
        align[tail] = True
    cohorts['align_last5'] = align
    outputs = {f'{model}_{mode}': np.array([row[f'{model}_{mode}'] for row in rows])
               for model in MODELS for mode in ('transformer', 'rigid')}
    for model in MODELS:
        np.testing.assert_allclose(outputs[f'{model}_transformer'][~corrected],
                                   outputs[f'{model}_rigid'][~corrected], atol=1e-6, rtol=0)
    for model in ('fixed', 'mixed'):
        outputs[f'{model}_gt'] = ref[f'{model}_model']
    summary = dict(runs=runs, cohorts={key: dict(frames=int(mask.sum()),
                                              corrected_frames=int((mask & corrected).sum()))
                                      for key, mask in cohorts.items()}, metrics={}, paired={})
    per_frame = {key: errors(pred.astype(float), ref['target'].astype(float))
                 for key, pred in outputs.items()}
    for key, metrics in per_frame.items():
        summary['metrics'][key] = {group: {metric: stats(values[mask]) for metric, values in metrics.items()}
                                   for group, mask in cohorts.items()}
    expected_gt_p90 = {'fixed': 2.41, 'mixed': 2.86}
    for model, expected in expected_gt_p90.items():
        actual = summary['metrics'][f'{model}_gt']['during']['transverse_mm']['p90']
        np.testing.assert_allclose(actual, expected, atol=.0051, rtol=0)
    for model in MODELS:
        baseline, rigid = (per_frame[f'{model}_{mode}'] for mode in ('transformer', 'rigid'))
        summary['paired'][model] = {}
        for group, mask in cohorts.items():
            summary['paired'][model][group] = {}
            for metric, base in baseline.items():
                change = rigid[metric][mask] - base[mask]
                summary['paired'][model][group][metric] = dict(median_delta=float(np.median(change)),
                                                               better=int((change < -0.001).sum()),
                                                               worse=int((change > 0.001).sum()))
    full = ROOT / 'full'
    full.mkdir(exist_ok=True)
    (full / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (full / 'frames.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    for group, info in summary['cohorts'].items():
        print(group, info)
        for key, entry in summary['metrics'].items():
            print(key, *(f"{stats['p50']:.2f}/{stats['p90']:.2f}" for stats in entry[group].values()))


if __name__ == '__main__':
    main()
