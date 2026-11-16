"""Evaluate raw detector and fixed-ID Transformer 3D output on matched test frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_tracker import describe_xyz, FAMILIES, get_masks
from tracker import CornerTransformer
from train_tracker import TEST, END, prepare, evaluate, read_npz, read_run
from utils.experiment_logger import ExperimentLogger, write


def read_capture(root, reference):
    parts, links = [], []
    for start in range(TEST, END, 8):
        path = root / f'shard_{start}_{start + 8}'
        data, link = read_run(path, 'detections.npz', 'sha256')
        cfg = json.loads((path / 'summary.json').read_text())['config']
        assert cfg['script_sha256'] == file_sha256(root / 'collect_final.py')
        assert cfg['detector_sha256'] == file_sha256('visual/corner_detection/detector.py')
        assert cfg['reference_hashes'] == {reference.parent.name: file_sha256(reference)}
        assert np.unique(data['seed']).tolist() == list(range(start, start + 8))
        for key in ('raw_xyz', 'raw_kind', 'raw_family_hint'):
            assert len(data[key]) == data['raw_count'].sum()
        parts.append(data)
        links.append(link)
    rows = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    np.testing.assert_array_equal(rows['recall'][:, 1], rows['observed'])
    assert (rows['recall'] <= rows['visible'][:, None, :]).all()
    return rows, links


def input_counts(data, groups):
    raw_counts = data['raw_count']
    obs, visible, recall = (data[k] for k in ('observed', 'visible', 'recall'))
    result = {}
    for group, selected in groups.items():
        n = int(selected.sum())
        coords = {}
        for family, sl in FAMILIES.items():
            v = visible[selected, sl]
            observed = obs[selected, sl]
            hits = recall[selected, :, sl].sum(axis=(0, 2))
            n_visible = int(v.sum())
            coords[family] = dict(n_visible=n_visible, recall=(hits / n_visible).tolist(),
                                  mean_ids=float(observed.sum(axis=1).mean()),
                                  zero_ids=int((~observed.any(axis=1)).sum()),
                                  complete_frames=int(observed.all(axis=1).sum()))
        result[group] = dict(raw_candidates=int(raw_counts[selected].sum()),
                             visible_pairs_10mm=int(recall[selected, 2].sum()),
                             frames=n, **coords)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'capture', 'old-test', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    raw, links = read_capture(args.capture, args.reference)
    ref = read_npz(args.reference)
    data = {k: v[ref['seed'] >= TEST] for k, v in ref.items()}
    for key in ('seed', 'step'):
        np.testing.assert_array_equal(data[key], raw[key])
    data.update({k: raw[k] for k in ('observed', 'visible', 'detected_xyz')})
    assert len(data['seed']) == 3352
    assert np.all(np.linalg.norm(data['detected_xyz'][data['observed']] -
                                 data['xyz'][data['observed']], axis=-1) <= .005002)
    assert (data['detected_xyz'][~data['observed']] == 0).all()
    baseline = read_npz(args.old_test)
    for key in ('seed', 'step', 'target', 'tcp', 'xyz', 'visible'):
        np.testing.assert_array_equal(data[key], baseline[key])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state['epoch'] == 954
    model = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    tracker_path = Path(__import__('tracker').__file__)
    assert state['config']['tracker_sha256'] == file_sha256(tracker_path)
    prepared = prepare(data, state['fallback'].numpy(), range(TEST, END), device)
    pred = evaluate(model, prepared, state['config']['batch_size'])
    mask = data['observed']
    np.testing.assert_array_equal(pred[mask], data['detected_xyz'][mask])
    groups = get_masks(data)
    all_slots = np.ones_like(mask)
    ever = prepared['ever'][prepared['valid']].cpu().numpy()
    raw_error = np.linalg.norm(pred - data['xyz'], axis=-1) * 1000
    baseline_error = np.linalg.norm(baseline['spatial_temporal'] - data['xyz'], axis=-1) * 1000
    common_missing = ~mask & ~baseline['observed']
    modes = dict(all=all_slots, observed=mask, missing=~mask, missing_seen=~mask & ever,
                 missing_never=~mask & ~ever, genuinely_hidden=~data['visible'], common_missing=common_missing)
    statistics = {name: describe_xyz(raw_error, selection, groups) for name, selection in modes.items()}
    comparison = {'all': describe_xyz(baseline_error, all_slots, groups),
                  'missing': describe_xyz(baseline_error, ~baseline['observed'], groups),
                  'common_missing': describe_xyz(baseline_error, common_missing, groups)}
    cfg = dict(checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=state['epoch'],
               checkpoint_run_id=state['run_id'], source_shards=links,
               reference_sha256=file_sha256(args.reference), script_sha256=file_sha256(__file__),
               old_test_sha256=file_sha256(args.old_test), tracker_sha256=file_sha256(tracker_path))
    args.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='final_detector_transformer_eval', job_type='evaluate',
                          seed=0, config=cfg, root=args.output / 'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        summary, predictions = args.output / 'summary.json', args.output / 'predictions.npz'
        write(summary, dict(run_id=logger.run_id, config=cfg, point=statistics, previous_ransac=comparison,
                            detector=input_counts(raw, groups)))
        np.savez_compressed(predictions, **raw, xyz_gt=data['xyz'], xyz_final=pred, previously_seen=ever,
                            moving=data['moving_now'], insertion_last5=groups['align_last5'])
        logger.log(statistics['all'])
        logger.link_output(summary)
        logger.link_output(predictions)
    print('Completed final detector / Transformer evaluation', flush=True)


if __name__ == '__main__':
    main()
