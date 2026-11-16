"""Record a selected moving-Box replay with the evaluated Mamba/Graph artifacts."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from evaluate_mamba import exp2, fixed, import_moving
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write

BASE = Path('/workspace/recurrent_new_frontend')
TRACKER = Path('/workspace/spatiotemporal_corner_tracker_convergence/train/spatial_temporal/best.pt')
GNN = Path('/workspace/hybrid_input_gnn_finetune/train/best.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    source = next(path.parent for path in sorted((BASE / 'eval').glob('shard_*/summary.json'))
                  if args.seed in json.loads(path.read_text())['config']['seeds'])
    summary = json.loads((source / 'summary.json').read_text())
    config = summary['config']
    paths = [BASE / 'train' / f'mamba_s{i}' / 'last.pt' for i in range(5)]
    assert [file_sha256(p) for p in paths] == config['recurrent_sha256']
    assert file_sha256(TRACKER) == config['tracker_sha256']
    assert file_sha256(GNN) == config['gnn_sha256']
    moving = import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    assert file_sha256(moving.__file__) == config['moving_source_sha256']
    assert config['camera'] == fixed.CAMERA and config['no_filter']
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    assert all(not ck['kin'] for ck in checkpoints)
    front = fixed.HybridFrontend(TRACKER, GNN, torch.device('cuda:0'))
    decoder = exp2.Ensemble([exp2.Decoder(ck, 'step') for ck in checkpoints])
    config = dict(config, replay_seed=args.seed, source_run_id=summary['run_id'],
                  script_sha256=file_sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='recurrent', model='mamba_moving_failure_replay', job_type='evaluate',
                          seed=args.seed, config=config, root=args.output / 'experiments') as log:
        env = fixed.make_env()
        try:
            _, episode = moving.episode(env, front, decoder, args.seed, 'hybrid', args.output)
        finally:
            env.close()
        assert all(frame['object21'] == frame['filtered'] for frame in episode['trace'])
        movie = args.output / episode['movie']
        final_name = movie.with_name(movie.name.replace('_hybrid_', '_retrained_mamba_'))
        movie.rename(final_name)
        episode['movie'] = final_name.name
        frame_max = max(frame['raw_lateral_mm'] for frame in episode['trace'])
        write(args.output / 'rollout.json', dict(seed=args.seed, run_id=log.run_id, **episode))
        log.link_output(args.output / 'rollout.json')
        log.link_output(final_name)
        log.log(dict(success=int(episode['success']), steps=episode['steps'], max_lateral_error_mm=frame_max))
        print(args.seed, episode['success'], episode['steps'], frame_max, flush=True)


if __name__ == '__main__':
    main()
