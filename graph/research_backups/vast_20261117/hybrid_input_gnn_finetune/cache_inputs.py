"""Build frozen detector → tracker → rigid Box inputs for Object21 fine-tuning."""
import argparse
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_hybrid import apply_existing_correction
from tracker import CornerTransformer
from train_tracker import FIRST, VAL, TEST, END, evaluate, prepare, read_data, read_npz
from utils.experiment_logger import ExperimentLogger, write


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'capture', 'test', 'checkpoint', 'prior-hybrid', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    data, sources = read_data(a.reference, a.capture, a.test)
    checkpoint = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    assert checkpoint['epoch'] == 954
    assert checkpoint['config']['tracker_sha256'] == file_sha256(__import__('tracker').__file__)
    model = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    split = [range(FIRST, VAL), range(VAL, TEST), range(TEST, END)]
    fallback = checkpoint['fallback'].numpy()
    config = dict(sources=sources, tracker_sha256=file_sha256(a.checkpoint),
                  tracker_run_id=checkpoint['run_id'], tracker_epoch=checkpoint['epoch'],
                  gt_assignment_m=.005, gt_visible_selection=True, rigid_box_policy='shared translation',
                  rigid_sha256=file_sha256(__import__('rigid_box').__file__),
                  correction_runner_sha256=file_sha256(__import__('evaluate_hybrid').__file__),
                  cache_script_sha256=file_sha256(__file__), prior_hybrid_sha256=file_sha256(a.prior_hybrid))
    a.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='hybrid_corner_input_cache', job_type='collect',
                          seed=0, config=config, root=a.output / 'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        parts, histories = [], []
        for ids in split:
            packed = prepare(data, fallback, ids, device)
            parts.append(evaluate(model, packed, checkpoint['config']['batch_size']))
            histories.append(packed['ever'][packed['valid']].cpu().numpy())
        data['spatial_temporal'] = np.concatenate(parts)
        data['ever'] = np.concatenate(histories)
        coords, _, support, _ = apply_existing_correction(data)
        observed = data['observed']
        np.testing.assert_array_equal(coords[observed], data['detected_xyz'][observed])
        heldout = data['seed'] >= TEST
        prior = read_npz(a.prior_hybrid)
        for key in ('seed', 'step', 'target'):
            np.testing.assert_array_equal(data[key][heldout], prior[key])
        np.testing.assert_allclose(coords[heldout], prior['xyz_hybrid'], atol=3e-6, rtol=0)
        assert np.isfinite(coords).all() and not any(p.requires_grad for p in model.parameters())
        keys = ('seed', 'step', 'target', 'tcp', 'phase', 'grasp', 'moving_now', 'observed')
        path = a.output / 'input.npz'
        np.savez_compressed(path, xyz=coords, fit_support=support, **{key: data[key] for key in keys})
        counts = {name: int(((ids.start <= data['seed']) & (data['seed'] < ids.stop)).sum())
                  for name, ids in zip(('train', 'validation', 'test'), split, strict=True)}
        assert counts == {'train': 13448, 'validation': 2706, 'test': 3352}
        report = a.output / 'summary.json'
        write(report, dict(config=config, run_id=logger.run_id, frames=counts, inputs_sha256=file_sha256(path)))
        logger.log(counts)
        logger.link_output(path)
        logger.link_output(report)
    print('Completed frozen input cache', counts, flush=True)


if __name__ == '__main__':
    main()
