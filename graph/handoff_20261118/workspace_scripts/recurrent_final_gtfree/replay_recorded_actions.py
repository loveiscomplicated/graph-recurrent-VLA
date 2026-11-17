"""Replay saved action commands and score the frozen frontend without a controller model."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_final import exp2, fixed, import_moving
from final_frontend import FRONT, FinalFrontend
from utils.experiment_logger import ExperimentLogger, write

ROOT = Path('/workspace/recurrent_final_gtfree')
SEED = 700882


def find_record(arm):
    for path in sorted((ROOT / arm).glob('shard_*/rollouts.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['seed'] == SEED:
                    meta = json.loads(path.with_name('summary.json').read_text())
                    return row['moving'], meta['config'], path
    raise ValueError(f'Seed {SEED} absent in {arm}')


def distance(left, right):
    return float(np.linalg.norm(left - right))


@torch.inference_mode()
def replay(env, front, original, moving):
    obs, _ = env.reset(seed=SEED)
    fixed.reset_state(obs, env)
    task = env.unwrapped
    front.reset()
    plan = original['motion_plan']
    motion = moving.EpisodeHoleMotion(env, plan, None)
    frames = []
    for step, archived in enumerate(original['trace']):
        assert archived['step'] == step
        moved = motion.advance(step, None)
        if moved:
            obs = task.get_obs(task.get_info())
        offset = motion.frame_data()['offset_m'] * 1000
        tcp = exp2.robot15(env)[:3]
        predicted, counts, _, _ = front.step(obs, tcp)
        target = exp2.gt_object21(env)
        inverse = exp2.q_conj(target[13:17])
        actual = exp2.q_rot(inverse, target[7:10]) * 1000
        estimate = exp2.q_rot(inverse, predicted[7:10]) * 1000
        error = distance(actual[1:], estimate[1:])
        depth = float(actual[0] + target[17] * 1000)
        shallow = 0 < depth <= 20 and bool((np.abs(actual[1:]) <= target[20] * 1000).all())
        old_target = np.asarray(archived['object21_gt'])
        old_prediction = np.asarray(archived['object21'])
        frames.append(dict(step=step, depth_mm=depth, shallow=shallow, lateral_mm=error,
                           original_lateral_mm=archived['raw_lateral_mm'], observed=list(counts),
                           original_observed=archived['observed'], object21=predicted.tolist(),
                           object21_gt=target.tolist(),
                           peg_center_gt_drift_mm=1000 * distance(target[:3], old_target[:3]),
                           hole_center_gt_drift_mm=1000 * distance(target[10:13], old_target[10:13]),
                           peg_relative_gt_drift_mm=1000 * distance(target[7:10], old_target[7:10]),
                           relative_prediction_drift_mm=1000 * distance(predicted[7:10], old_prediction[7:10]),
                           box_offset_diff_mm=distance(offset, np.asarray(archived['box_offset_mm']))))
        # Recorded commands are the only actions passed to the simulator.
        action = torch.tensor([archived['action']], dtype=torch.float32, device=task.device)
        obs, _, term, trunc, info = env.step(action)
        success = exp2.is_success(info)
        if success or bool(term[0]) or bool(trunc[0]):
            break
    return frames, bool(success)


def summarize(frames, success, original):
    large = [f['step'] for f in frames if f['shallow'] and f['lateral_mm'] >= 50]
    old_large = []
    for row in original['trace']:
        actual = np.asarray(row['actual_relative_mm'])
        target = row['object21_gt']
        depth = actual[0] + target[17] * 1000
        if 0 < depth <= 20 and (np.abs(actual[1:]) <= target[20] * 1000).all() and row['raw_lateral_mm'] >= 50:
            old_large.append(row['step'])
    shallow = [row['lateral_mm'] for row in frames if row['shallow']]
    drifts = ('peg_center_gt_drift_mm', 'hole_center_gt_drift_mm', 'peg_relative_gt_drift_mm',
              'relative_prediction_drift_mm', 'box_offset_diff_mm')
    return dict(original_steps=original['steps'], replay_steps=len(frames),
                original_success=original['success'], replay_success=success,
                old_large_shallow_steps=old_large, replay_large_shallow_steps=large,
                matched_large_shallow_steps=sorted(set(large) & set(old_large)),
                detector_count_matches=sum(f['observed'] == f['original_observed'] for f in frames),
                replay_shallow_p90_mm=float(np.percentile(shallow, 90)) if shallow else None,
                lateral_max_diff_mm=max(abs(f['lateral_mm'] - f['original_lateral_mm']) for f in frames),
                max_drift={key:max(f[key] for f in frames) for key in drifts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    tracker = FRONT / 'tracker/best.pt'
    gnn = FRONT / 'gnn/best.pt'
    front = FinalFrontend(tracker, gnn, torch.device('cuda:0'))
    moving = import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    args.output.mkdir(exist_ok=False)
    for arm in ('baseline', 'finetuned'):
        archived, config, path = find_record(arm)
        assert file_sha256(tracker) == config['tracker_sha256']
        assert file_sha256(gnn) == config['gnn_sha256']
        assert front.code_hashes == config['frontend_code_sha256']
        assert file_sha256(fixed.__file__) == config['fixed_source_sha256']
        assert file_sha256(moving.__file__) == config['moving_source_sha256']
        assert archived['motion_plan']['stop_expert_insert'] is False
        run = args.output / arm
        run.mkdir()
        cfg = dict(seed=SEED, arm=arm, policy='archived Action7', trajectory_sha256=file_sha256(path),
                   original_evaluation=config, script_sha256=file_sha256(__file__))
        with ExperimentLogger(team='integration', model='object21_action_replay', job_type='evaluate',
                              seed=SEED, config=cfg, root=run/'experiments') as log:
            env = fixed.make_env()
            try:
                log.record(runtime=dict(gpu=torch.cuda.get_device_name(), backend=str(env.unwrapped.backend)))
                frames, success = replay(env, front, archived, moving)
            finally:
                env.close()
            summary = summarize(frames, success, archived)
            write(run/'summary.json', dict(run_id=log.run_id, **summary))
            write(run/'frames.json', frames)
            log.link_output(run/'summary.json')
            log.link_output(run/'frames.json')
            log.log(dict(replayed_frames=len(frames), large_replay=len(summary['replay_large_shallow_steps']),
                         large_reproduced=len(summary['matched_large_shallow_steps'])))
        print(arm, {k:(len(v) if k.endswith('shallow_steps') else v) for k,v in summary.items()}, flush=True)


if __name__ == '__main__':
    main()
