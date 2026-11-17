"""Verify that fresh held-out sequences agree with batched causal inference."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch

from hybrid_frontend import StreamingTracker
from tracker import CornerTransformer
from train import load, TEST
from utils.experiment_logger import ExperimentLogger, write
from data_collection.episode_dataset import file_sha256


def main():
    parser = ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trained-directory', default='trained')
    args = parser.parse_args()
    torch.set_num_threads(1)
    data, _, _ = load(args.root)
    trained = args.root/args.trained_directory
    model_path = trained/'tracker/best.pt'
    evaluation = trained/'gnn/heldout.npz'
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    device = torch.device('cuda:0')
    model = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    stream = StreamingTracker(model, state['fallback'].numpy(), device)
    selected = data['seed'] >= TEST
    data = {key: values[selected] for key, values in data.items()}
    with np.load(evaluation, allow_pickle=False) as f:
        np.testing.assert_array_equal(f['seed'], data['seed'])
        np.testing.assert_array_equal(f['step'], data['step'])
        batch = f['new_raw']
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(tracker_sha256=file_sha256(model_path), heldout_sha256=file_sha256(evaluation),
                  script_sha256=file_sha256(__file__))
    with ExperimentLogger(team='graph', model='full_insertion_streaming_check', job_type='evaluate',
                          seed=0, config=config, root=args.output/'experiments') as log:
        checks = []
        for parity in (0, 1):
            seed = next(seed for seed in np.unique(data['seed']) if seed % 2 == parity)
            frames = np.flatnonzero(data['seed'] == seed)
            stream.reset()
            predicted = np.stack([stream.step(data['detected_xyz'][i], data['observed'][i], data['tcp'][i])
                                  for i in frames])
            seen = data['observed'][frames]
            np.testing.assert_array_equal(predicted[seen], data['detected_xyz'][frames][seen])
            delta = np.linalg.norm(predicted - batch[frames], axis=-1)*1000
            assert delta.max() < .1
            checks.append(dict(seed=int(seed), frames=len(frames),
                               max_depth_mm=float(data['depth_mm'][frames].max()),
                               max_difference_mm=float(delta.max())))
        result = args.output/'summary.json'
        write(result, dict(run_id=log.run_id, config=config, checks=checks))
        log.link_output(result)
        log.log({'max_difference_mm':max(row['max_difference_mm'] for row in checks)})
    print('Verified batch to streaming', checks, flush=True)


if __name__ == '__main__':
    main()
