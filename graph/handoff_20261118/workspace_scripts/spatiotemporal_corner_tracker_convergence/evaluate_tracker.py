"""Evaluate held-out per-corner error and the downstream frozen Object21 probe."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train_gt20 import describe, error_metrics, predict
from train_tracker import MODES, TEST, END, read_npz, read_run
from utils.experiment_logger import ExperimentLogger, write

FAMILIES = {'all': slice(None), 'peg': slice(0, 8), 'box_exterior': slice(8, 16), 'hole': slice(16, 20)}


def get_masks(data):
    tail = np.zeros(len(data['seed']), dtype=bool)
    for seed in np.unique(data['seed']):
        ix = np.flatnonzero((data['seed'] == seed) & (data['phase'] == 'align') & data['grasp'])[-5:]
        assert len(ix) == 5
        tail[ix] = True
    return {'all': np.ones_like(tail), 'moving': data['moving_now'], 'align_last5': tail}


def describe_xyz(error, selected, groups):
    result = {}
    for region, frames in groups.items():
        result[region] = {}
        for name, scope in FAMILIES.items():
            values = error[frames, scope][selected[frames, scope]]
            if values.size:
                p50, p90 = np.percentile(values, [50, 90])
                result[region][name] = dict(n=int(values.size), mean_mm=float(values.mean()),
                                            p50_mm=float(p50), p90_mm=float(p90))
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ('train', 'previous', 'output', 'gnn', 'reference'):
        p.add_argument(f'--{key}', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(3)
    device = torch.device('cuda:0')
    path = a.train / 'test_predictions.npz'
    data = read_npz(path)
    modes = tuple(name for name in MODES if name in data)
    assert modes, 'No tracker predictions found'
    prev = []
    for start in range(TEST, END, 8):
        shard, _ = read_run(a.previous / f'shard_{start}_{start+8}', 'matches.npz', 'matches_sha256')
        prev.append(shard)
    prior = {key: np.concatenate([d[key] for d in prev]) for key in
             ('seed','step','tracked','tracked_xyz','corrected_xyz')}
    ref = read_npz(a.reference)
    for key in ('seed', 'step'):
        np.testing.assert_array_equal(data[key], prior[key])
        np.testing.assert_array_equal(data[key], ref[key])
    assert len(data['seed']) == 3352
    np.testing.assert_array_equal(data['target'], ref['target'])
    obs = data['observed']
    groups = get_masks(data)
    error_tests = {'missing_all': ~obs, 'missing_previously_detected': ~obs & data['ever'],
                   'truly_hidden_previously_detected': ~data['visible'] & data['ever'],
                   'legacy_matched_missing': ~obs & prior['tracked']}
    methods = {'last_or_train_mean': np.where(obs[..., None], data['detected_xyz'], data['anchor']),
               **{name: data[name] for name in modes},
               **{name: prior[key] for name, key in [('old_transformer', 'tracked_xyz'),
                                                     ('old_rigid', 'corrected_xyz')]}}
    point = {}
    for name, value in methods.items():
        error = np.linalg.norm(value - data['xyz'], axis=-1) * 1000
        masks = ('legacy_matched_missing',) if name.startswith('old_') else error_tests
        point[name] = {key: describe_xyz(error, error_tests[key], groups) for key in masks}
        if name in MODES:
            np.testing.assert_array_equal(value[obs], data['detected_xyz'][obs])
    state = torch.load(a.gnn, weights_only=True, map_location='cpu')
    assert file_sha256(a.gnn).startswith('e2650277aafa189')
    gnn = VertexEdgeGatedMPNN(**state['config']['gnn']).to(device).eval().requires_grad_(False)
    gnn.load_state_dict(state['model'], strict=True)
    target = torch.tensor(data['target'])
    kinds = torch.tensor([0]*8 + [1]*12, device=device).expand(len(obs), 20)
    valid = torch.ones_like(kinds, dtype=torch.bool)
    indices = torch.arange(len(kinds), device=device)
    tcp = torch.as_tensor(data['tcp'], device=device)
    gnn_inputs = dict(clean=data['xyz'], **{k: methods[k] for k in ('last_or_train_mean', *modes)})
    objects, predictions = {}, {}
    for name, coords in gnn_inputs.items():
        flags = ('all_observed',) if name == 'clean' else ('real_flags', 'all_observed')
        xyz = torch.as_tensor(coords, device=device)
        for flag in flags:
            inputs = dict(xyz=xyz, kinds=kinds, valid=valid, tcp=tcp,
                          observed=valid if flag == 'all_observed' else torch.as_tensor(obs, device=device))
            pred = predict(gnn, inputs, indices, state['config']['training']['batch_size'])
            if name == 'clean':
                np.testing.assert_allclose(pred.numpy(), ref['mixed_model'], rtol=0, atol=2e-5)
            key = name + '__' + flag
            predictions[key] = pred.numpy()
            metrics = error_metrics(pred, target)
            objects[key] = {region: describe(metrics, mask) for region, mask in groups.items()}
    summaries = {name: json.loads((a.train / name / 'summary.json').read_text()) for name in modes}
    config = dict(gnn_sha256=file_sha256(a.gnn), script_sha256=file_sha256(__file__),
                  predictions_sha256=file_sha256(path), runs={m: s['run_id'] for m, s in summaries.items()})
    with ExperimentLogger(team='graph', model='spatiotemporal_corner_tracker_eval', job_type='evaluate',
                          seed=0, config=config, root=a.output / 'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        out, npz = a.output / 'summary.json', a.output / 'object21_predictions.npz'
        write(out, dict(config=config, run_id=logger.run_id, corner=point, object21=objects,
                        frames={key: int(mask.sum()) for key, mask in groups.items()},
                        observed_fraction=float(obs.mean()), never_seen_fraction=float((~data['ever']).mean())))
        np.savez_compressed(npz, seed=data['seed'], step=data['step'], target=data['target'], **predictions)
        logger.log({m: s.get('missing_all', {}).get('all', {}).get('all', {}) for m,s in point.items()})
        logger.link_output(out)
        logger.link_output(npz)
    print('Completed tracker/GNN evaluation', flush=True)


if __name__ == '__main__':
    main()
