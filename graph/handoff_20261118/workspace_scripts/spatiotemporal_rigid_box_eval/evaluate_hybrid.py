"""Apply the existing Box translation correction to fixed-slot 954-epoch predictions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_tracker import describe_xyz, get_masks
from graph.gated_mpnn import VertexEdgeGatedMPNN
from rigid_box import RigidBoxTranslation
from train_gt20 import describe, error_metrics, predict
from train_tracker import read_npz
from utils.experiment_logger import ExperimentLogger, write

KINDS = np.array([0] * 8 + [1] * 12, dtype=np.int64)


def apply_existing_correction(data):
    original = data['spatial_temporal']
    observed = data['observed']
    output = original.copy()
    shifts = np.zeros((len(original), 3))
    support = np.zeros(len(original), dtype=np.int32)
    age = np.zeros_like(support)
    for seed in np.unique(data['seed']):
        rigid = RigidBoxTranslation(use_velocity=True)
        indices = np.flatnonzero(data['seed'] == seed)
        np.testing.assert_array_equal(data['step'][indices], np.arange(len(indices)))
        for i in indices:
            exterior = set(np.flatnonzero(observed[i, 8:16]) + 8)
            output[i], fit = rigid.update(original[i], observed[i], KINDS, exterior)
            shifts[i], support[i], age[i] = fit['pose_m'], fit['support'], fit['age']
    untouched = observed | ~data['ever'] | np.broadcast_to(KINDS == 0, observed.shape)
    np.testing.assert_array_equal(output[untouched], original[untouched])
    return output, shifts, support, age


def run_gnn(coords, data, device, checkpoint, reference):
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert file_sha256(checkpoint).startswith('e2650277aafa189')
    model = VertexEdgeGatedMPNN(**state['config']['gnn']).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    kinds = torch.as_tensor(KINDS, device=device).expand(len(data['xyz']), 20)
    valid = torch.ones_like(kinds, dtype=torch.bool)
    observed = torch.as_tensor(data['observed'], device=device)
    tcp = torch.as_tensor(data['tcp'], device=device)
    ix = torch.arange(len(data['xyz']), device=device)
    results = {}
    for name, xyz in coords.items():
        tensors = dict(xyz=torch.as_tensor(xyz, device=device), kinds=kinds, valid=valid, tcp=tcp,
                       observed=valid if name == 'clean' else observed)
        results[name] = predict(model, tensors, ix, state['config']['training']['batch_size']).numpy()
    np.testing.assert_allclose(results['clean'], reference['mixed_model'], rtol=0, atol=2e-5)
    return results


def main():
    p = argparse.ArgumentParser()
    for name in ('tracker-output', 'prior-eval', 'gnn', 'reference', 'output'):
        p.add_argument(f'--{name}', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(3)
    device = torch.device('cuda:0')
    data, ref, prior = map(read_npz, (a.tracker_output, a.reference, a.prior_eval))
    for k in ('seed', 'step', 'target'):
        np.testing.assert_array_equal(data[k], ref[k])
        np.testing.assert_array_equal(data[k], prior[k])
    assert len(data['seed']) == 3352
    assert data['spatial_temporal'].shape == (3352, 20, 3)
    np.testing.assert_array_equal(data['spatial_temporal'][data['observed']],
                                  data['detected_xyz'][data['observed']])
    corrected, shifts, support, age = apply_existing_correction(data)
    coords = dict(transformer=data['spatial_temporal'], hybrid=corrected)
    groups = get_masks(data)
    groups['after_motion'] = (~data['moving_now']) & (np.linalg.norm(ref['offset_m'], axis=1) > 1e-6)
    obs, seen = data['observed'], data['ever']
    mask = dict(all=np.ones_like(obs), missing=~obs, missing_seen=~obs & seen,
                missing_never=~obs & ~seen, hidden=~data['visible'])
    point = {}
    for name, xyz in coords.items():
        err = np.linalg.norm(xyz - data['xyz'], axis=-1) * 1000
        point[name] = {label: describe_xyz(err, selection, groups) for label, selection in mask.items()}
    preds = run_gnn(dict(clean=data['xyz'], **coords), data, device, a.gnn, ref)
    np.testing.assert_allclose(preds['transformer'], prior['spatial_temporal__real_flags'],
                               rtol=0, atol=2e-5)
    object21 = {}
    for name, pred in preds.items():
        error = error_metrics(torch.as_tensor(pred), torch.as_tensor(data['target']))
        object21[name] = {group: describe(error, selection) for group, selection in groups.items()}
    changed = (corrected != coords['transformer']).any(axis=-1)
    statistics = {name: dict(frames=int(selection.sum()), valid_fit=int((support[selection] >= 3).sum()),
                             box_corrected=int(changed[selection, 8:16].sum()),
                             hole_corrected=int(changed[selection, 16:20].sum()))
                  for name, selection in groups.items()}
    config = dict(gnn_sha256=file_sha256(a.gnn), reference_sha256=file_sha256(a.reference),
                  tracker_output_sha256=file_sha256(a.tracker_output), prior_eval_sha256=file_sha256(a.prior_eval),
                  script_sha256=file_sha256(__file__), rigid_sha256=file_sha256(__import__('rigid_box').__file__),
                  inlier_m=.008, min_support=3, max_speed_m_per_frame=.012, max_prediction_frames=5)
    a.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='spatiotemporal_rigid_box_eval', job_type='evaluate',
                          seed=0, config=config, root=a.output / 'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        summary, npz = a.output / 'summary.json', a.output / 'predictions.npz'
        write(summary, dict(run_id=logger.run_id, config=config, point=point, object21=object21, fit=statistics))
        np.savez_compressed(npz, seed=data['seed'], step=data['step'], target=data['target'],
                            observed=obs, previously_seen=seen, xyz_gt=data['xyz'],
                            xyz_transformer=coords['transformer'], xyz_hybrid=corrected,
                            box_shift_m=shifts, fit_support=support, fit_age=age, **preds)
        logger.log({'corner': point['hybrid']['missing']['all'], 'object21': object21['hybrid']['all']})
        logger.link_output(summary)
        logger.link_output(npz)
    print('Completed fixed-ID Transformer + rigid Box evaluation', flush=True)


if __name__ == '__main__':
    main()
