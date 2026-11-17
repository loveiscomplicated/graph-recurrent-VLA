"""Join final-front Object21 estimates to the same Expert actions and robot state."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from final_frontend import FRONT, ROOT, FinalFrontend
from hybrid_frontend import HybridFrontend
from prepare_demos import align, read_npz
from train import predict_coordinates
from train_finetune import prepare_tensors
from train_gt20 import predict
from train_gtfree import TEST, load
from utils.experiment_logger import ExperimentLogger, write

EXPERT = Path('/workspace/recurrent_full_insertion/demos')
TRACKER = FRONT / 'tracker/best.pt'
GNN = FRONT / 'gnn/best.pt'


def verify_streaming(front, data, points, predictions):
    for seed in np.unique(data['seed'])[:2]:
        front.reset()
        for i in np.flatnonzero(data['seed'] == seed):
            front.detect = lambda *_: (data['detected_xyz'][i], data['observed'][i])
            streaming, *_ = HybridFrontend.step(front, None, data['tcp'][i], None)
            np.testing.assert_allclose(streaming, predictions[i], atol=2e-5, rtol=0)
    np.testing.assert_array_equal(points[data['observed']], data['detected_xyz'][data['observed']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    data, _, _, sources = load(ROOT)
    selected = (data['seed'] < TEST) & (data['phase'] != 'success')
    data = {key: values[selected] for key, values in data.items()}
    front = FinalFrontend(TRACKER, GNN, device)
    _, points = predict_coordinates(front.tracker.model, data, device, front.tracker.fallback)
    tensors = prepare_tensors(dict(data, xyz=points), device)
    predictions = predict(front.gnn, tensors, torch.arange(len(points), device=device), 96).numpy()
    verify_streaming(front, data, points, predictions)
    args.output.mkdir()
    for folder in sorted(EXPERT.glob('shard_*')):
        source = folder / 'demos.npz'
        summary = json.loads((folder / 'summary.json').read_text())
        digest = file_sha256(source)
        assert digest == summary['sha256']
        demos = read_npz(source)
        estimated, drift = align(demos, data, predictions)
        demos['state36_transformer'] = estimated
        dest = args.output / folder.name
        dest.mkdir()
        cfg = dict(original_sha256=digest, original_run_id=summary['run_id'],
                   raw_sources=sources, tracker_sha256=file_sha256(TRACKER),
                   gnn_sha256=file_sha256(GNN), frontend_code_sha256=front.code_hashes,
                   script_sha256=file_sha256(__file__), max_motion_drift_mm=float(drift[0]),
                   max_pose_drift_deg=float(drift[1]), streaming_parity=True)
        with ExperimentLogger(team='recurrent', model='mamba_final_gtfree_demos',
                              job_type='preprocess', seed=int(demos['reset_seed'][0]),
                              config=cfg, root=dest/'experiments') as log:
            out = dest / 'demos.npz'
            np.savez_compressed(out, **demos)
            info = dest / 'summary.json'
            episodes = len(demos['reset_seed'])
            write(info, dict(run_id=log.run_id, config=cfg, frames=len(estimated),
                             episodes=episodes, sha256=file_sha256(out)))
            log.link_output(out)
            log.link_output(info)
            log.log({'frames':len(estimated), 'episodes':episodes})
        print(folder.name, episodes, len(estimated), drift.tolist(), flush=True)


if __name__ == '__main__':
    main()
