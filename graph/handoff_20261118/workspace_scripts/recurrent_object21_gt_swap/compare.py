"""Replace only the Mamba Object21 input from first shallow entry onward."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_mamba import exp2, fixed, import_moving
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write

BASE = Path('/workspace/recurrent_new_frontend')
TRACKER = Path('/workspace/spatiotemporal_corner_tracker_convergence/train/spatial_temporal/best.pt')
GNN = Path('/workspace/hybrid_input_gnn_finetune/train/best.pt')
THRESHOLD_MM = 0.1


class InputSwitch:
    def __init__(self, decoder, env, enabled):
        self.decoder = decoder
        self.env = env
        self.enabled = enabled

    @property
    def decs(self):
        return self.decoder.decs

    @property
    def last(self):
        return self.decoder.last

    def reset(self):
        self.decoder.reset()
        self.step = 0
        self.entry_step = None
        self.entry_depth_mm = None

    def act(self, state, counts):
        truth = exp2.gt_object21(self.env)
        local = exp2.q_rot(exp2.q_conj(truth[13:17]), truth[7:10])
        depth_mm = float((local[0] + truth[17]) * 1000)
        in_opening = np.all(np.abs(local[1:]) <= truth[20])
        if self.entry_step is None and depth_mm > THRESHOLD_MM and in_opening:
            self.entry_step = self.step
            self.entry_depth_mm = depth_mm
        self.step += 1
        if self.enabled and self.entry_step is not None:
            state = np.r_[state[:15], truth]
        return self.decoder.act(state, counts)


def summarize(swap, rollout):
    frames = rollout['trace']
    states = np.array([f['object21_gt'] for f in frames])
    position = np.array([f['actual_relative_mm'] for f in frames])
    depth = position[:, 0] + states[:, 17] * 1000
    inside = (np.abs(position[:, 1:]) <= states[:, 20, None] * 1000).all(axis=1)
    expected = np.flatnonzero(inside & (depth > THRESHOLD_MM))
    assert len(expected) and int(expected[0]) == swap.entry_step
    for step, frame in enumerate(frames):
        consumed = frame['filtered']
        expected_value = frame['object21_gt'] if swap.enabled and step >= swap.entry_step else frame['object21']
        assert consumed == expected_value
    return dict(success=rollout['success'], steps=rollout['steps'], entry_step=swap.entry_step,
                entry_depth_mm=swap.entry_depth_mm, max_depth_mm=float(depth[inside].max()),
                target_depth_mm=float(states[0, 17] * 1000 - 15),
                final_depth_mm=float(depth[-1]), final_lateral_mm=float(np.linalg.norm(position[-1, 1:])),
                max_box_shift_mm=rollout['box_max_shift_mm'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, choices=(700674, 700882))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    source = next(s for p in sorted((BASE/'eval').glob('shard_*/summary.json'))
                  if args.seed in (s := json.loads(p.read_text()))['config']['seeds'])
    config = source['config']
    paths = [BASE/'train'/f'mamba_s{i}'/'last.pt' for i in range(5)]
    assert [file_sha256(p) for p in paths] == config['recurrent_sha256']
    assert file_sha256(TRACKER) == config['tracker_sha256']
    assert file_sha256(GNN) == config['gnn_sha256']
    moving = import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    assert file_sha256(moving.__file__) == config['moving_source_sha256']
    assert config['camera'] == fixed.CAMERA and config['no_filter']
    checkpoints = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    assert all(not ck['kin'] for ck in checkpoints)
    config = dict(config, source_run_id=source['run_id'], intervention='always GT from first aperture entry',
                  entry_threshold_mm=THRESHOLD_MM, script_sha256=file_sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='recurrent', model='mamba_object21_gt_swap',
                          job_type='evaluate', seed=args.seed, config=config,
                          root=args.output/'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        front = fixed.HybridFrontend(TRACKER, GNN, torch.device('cuda:0'))
        decoder = exp2.Ensemble([exp2.Decoder(ck, 'step') for ck in checkpoints])
        outcomes, resets = {}, {}
        for label, use_gt in [('graph', False), ('gt_after_entry', True)]:
            folder = args.output/label
            folder.mkdir()
            env = fixed.make_env()
            try:
                swap = InputSwitch(decoder, env, use_gt)
                reset, rollout = moving.episode(env, front, swap, args.seed, 'hybrid', folder)
            finally:
                env.close()
            resets[label] = reset
            outcomes[label] = summarize(swap, rollout)
            trace = folder/'rollout.json'
            write(trace, rollout)
            logger.link_output(trace)
            logger.link_output(folder/rollout['movie'])
            print(args.seed, label, outcomes[label], flush=True)
        np.testing.assert_allclose(resets['graph'], resets['gt_after_entry'], atol=1e-6, rtol=0)
        summary = args.output/'summary.json'
        write(summary, dict(seed=args.seed, run_id=logger.run_id, config=config, outcomes=outcomes))
        logger.link_output(summary)
        logger.log({label: int(outcome['success']) for label, outcome in outcomes.items()})


if __name__ == '__main__':
    main()
