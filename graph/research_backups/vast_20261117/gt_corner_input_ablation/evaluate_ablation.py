"""Compare coordinate perturbation and missing-node effects on a frozen Object21 GNN."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train_gt20 import describe, error_metrics, predict
from utils.experiment_logger import ExperimentLogger, write

JITTER_MM = (1, 3, 5, 10, 20, 40)
RANDOM_SEED = 3100144
FAMILIES = {'peg': slice(0, 8), 'box': slice(8, 16), 'hole': slice(16, 20)}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def read_matches(root, reference):
    shards, sources = [], []
    for start in range(3100144, 3100176, 8):
        folder = root / f'shard_{start}_{start + 8}'
        summary = json.loads((folder / 'summary.json').read_text())
        metadata = json.loads((folder / 'experiments' / summary['run_id'] / 'metadata.json').read_text())
        assert metadata['status'] == 'finished' and metadata['git_commit'].startswith('b2e8e4c')
        assert not metadata['git_dirty']
        path = folder / 'matches.npz'
        assert file_sha256(path) == summary['matches_sha256']
        assert summary['config']['tracker_sha256'].startswith('fb555e')
        assert summary['config']['reference_sha256'] == file_sha256(reference)
        assert summary['config']['script_sha256'] == file_sha256(root / 'collect_matches.py')
        assert summary['config']['rigid_sha256'] == file_sha256(root / 'rigid_box.py')
        shards.append(load_npz(path))
        sources.append({'run_id': summary['run_id'], 'sha256': file_sha256(path)})
    return {key: np.concatenate([shard[key] for shard in shards]) for key in shards[0]}, sources


def cohorts(data):
    late = np.zeros(len(data['seed']), dtype=bool)
    for seed in np.unique(data['seed']):
        ix = np.flatnonzero((data['seed'] == seed) & (data['phase'] == 'align') & data['grasp'])[-5:]
        assert len(ix) == 5
        late[ix] = True
    return dict(all=np.ones_like(late), moving=data['moving_now'],
                after=(~data['moving_now']) & (np.linalg.norm(data['offset_m'], axis=1) > 1e-6),
                align_last5=late)


def full_conditions(data):
    xyz = data['xyz']
    all_valid = np.ones_like(data['kinds'], dtype=bool)
    yield 'clean', xyz, all_valid, all_valid
    rng = np.random.default_rng(RANDOM_SEED)
    directions = rng.standard_normal(size=xyz.shape).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    for mm in JITTER_MM:
        shifted = xyz + directions * (mm / 1000)
        np.testing.assert_allclose(np.linalg.norm(shifted - xyz, axis=-1), mm / 1000, atol=1e-7, rtol=0)
        yield f'jitter_{mm}mm', shifted, all_valid, all_valid


def matched_conditions(data, matched):
    xyz = data['xyz']
    all_valid = np.ones_like(data['kinds'], dtype=bool)
    for key in ('visible', 'detected', 'tracked'):
        mask = matched[key]
        yield f'{key}_drop_only', xyz, mask, mask
    for body, corner_slice in FAMILIES.items():
        mask = all_valid.copy()
        mask[:, corner_slice] = matched['tracked'][:, corner_slice]
        yield f'tracked_drop_{body}_only', xyz, mask, mask
    tracked, observed = matched['tracked'], matched['detected']
    for prefix, coord_key in [('raw', 'tracked_xyz'), ('rigid', 'corrected_xyz')]:
        offsets = matched[coord_key] - matched['xyz']
        changed = xyz + offsets
        # Real offsets are known only for identified points. Fill unmatched slots with exact GT.
        np.testing.assert_array_equal(changed[~tracked], xyz[~tracked])
        yield f'{prefix}_coord_only', changed, all_valid, all_valid
        yield f'{prefix}_coord_and_drop', changed, tracked, tracked
        yield f'{prefix}_coord_drop_and_flags', changed, tracked, observed
    yield 'tracked_drop_and_flags_gt', xyz, tracked, observed


def describe_inputs(matched, data, groups):
    result = {}
    for name, frames in groups.items():
        result[name] = {}
        for family, slc in [('all', slice(None)), *FAMILIES.items()]:
            part = {}
            for status in ('visible', 'detected', 'tracked'):
                mask = matched[status][frames, slc]
                part[status] = dict(mean_count=float(mask.sum(axis=1).mean()),
                                    no_points=int((~mask.any(axis=1)).sum()))
            for status, coords in [('raw', 'tracked_xyz'), ('rigid', 'corrected_xyz')]:
                mask = matched['tracked'][frames, slc]
                distance = np.linalg.norm((matched[coords] - data['xyz'])[frames, slc], axis=-1)[mask] * 1000
                p50, p90 = np.percentile(distance, [50, 90])
                part[status] = dict(p50_mm=float(p50), p90_mm=float(p90), points=len(distance))
            result[name][family] = part
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'dataset', 'reference', 'output'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--matches', type=Path)
    args = p.parse_args()
    start = monotonic()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    test = load_npz(args.dataset)
    test = {key: val[test['seed'] >= 3100144] for key, val in test.items()}
    reference = load_npz(args.reference)
    for key in ('seed', 'step', 'target', 'phase', 'grasp', 'offset_m', 'moving_now'):
        np.testing.assert_array_equal(test[key], reference[key])
    assert np.unique(test['seed']).tolist() == list(range(3100144, 3100176))
    assert len(test['seed']) == 3352
    matched, sources = (read_matches(args.matches, args.reference) if args.matches else (None, []))
    if matched is not None:
        for key in ('seed', 'step', 'kinds'):
            np.testing.assert_array_equal(test[key], matched[key])
        for key in ('xyz', 'tcp'):
            np.testing.assert_allclose(test[key], matched[key], atol=1e-6, rtol=0)
        assert np.all(~matched['detected'] | (matched['visible'] & matched['tracked']))
        assert np.all(np.linalg.norm(matched['corrected_xyz'] - matched['xyz'], axis=-1)[matched['detected']] <= .005001)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert file_sha256(args.checkpoint) == 'e2650277aafa18994b3efc2b0aaac61949f3e4daf5f654b96282c884064cc64d'
    model = VertexEdgeGatedMPNN(**state['config']['gnn']).to(device).eval().requires_grad_(False)

    model.load_state_dict(state['model'], strict=True)
    tensors = {key: torch.as_tensor(test[key], device=device) for key in ('kinds', 'tcp')}
    indices = torch.arange(len(test['seed']), device=device)
    gt = torch.from_numpy(test['target'])
    groups = cohorts(test)
    config = dict(checkpoint_sha256=file_sha256(args.checkpoint), dataset_sha256=file_sha256(args.dataset),
                  reference_sha256=file_sha256(args.reference), source_run_id=state['run_id'],
                  jitter_mm=JITTER_MM, source_matches=sources, script_sha256=file_sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='gt_corner_input_ablation', job_type='evaluate',
                          seed=RANDOM_SEED, root=args.output / 'experiments', config=config) as logger:
        logger.record(runtime=dict(device=str(device), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        predictions = {}
        summary = {}
        conditions = list(full_conditions(test))
        if matched is not None:
            conditions.extend(matched_conditions(test, matched))
        for name, coords, valid, observed in conditions:
            assert np.all(~observed | valid)
            inputs = dict(tensors, xyz=torch.as_tensor(coords, device=device),
                          valid=torch.as_tensor(valid, device=device),
                          observed=torch.as_tensor(observed, device=device))
            pred = predict(model, inputs, indices, state['config']['training']['batch_size'])
            if name == 'clean':
                np.testing.assert_allclose(pred.numpy(), reference['mixed_model'], atol=2e-5, rtol=0)
            predictions[name] = pred.numpy()
            err = error_metrics(pred, gt)
            summary[name] = {group: describe(err, mask) for group, mask in groups.items()}
            print(name, {g: round(data['lateral_mm']['p90'], 2) for g, data in summary[name].items()}, flush=True)
        summary_path = args.output / 'summary.json'
        prediction_path = args.output / 'predictions.npz'
        write(summary_path, dict(config=config, run_id=logger.run_id, conditions=summary,
                                 input_stats=describe_inputs(matched, test, groups) if matched is not None else {},
                                 frames={g: int(mask.sum()) for g, mask in groups.items()},
                                 seconds=monotonic() - start))
        np.savez_compressed(prediction_path, seed=test['seed'], step=test['step'], target=test['target'],
                            **predictions)
        logger.log({name: {g: d['lateral_mm']['p90'] for g, d in conditions.items()}
                    for name, conditions in summary.items()})
        logger.link_output(summary_path)
        logger.link_output(prediction_path)
        print('Completed', logger.run_id, flush=True)


if __name__ == '__main__':
    main()
