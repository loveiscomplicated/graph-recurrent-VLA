"""Paired closed-loop moving-Box replay with the same frozen Mamba ensemble and MP4 recording."""
import argparse
import json
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np
import torch

import exp2_eval as exp2
from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from hybrid_frontend import HybridFrontend
from oracle_reference import CAMERA
from utils.experiment_logger import ExperimentLogger, write
from video import RolloutVideo

TOTAL_SEEDS = 32
SAMPLE_SEED = 20261117
MOTION_START_STEP = 60


def source_seeds(folder):
    paths = sorted((folder / 'recurrent/exp2/eval/final_ens').glob('results_*.jsonl'))
    assert len(paths) == 8
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    selected = [row for row in rows if row['cand'] == 'ens_mamba5@step']
    assert len(selected) == 200 and len({r['env_seed'] for r in selected}) == 200
    archival = {row['env_seed']: row['none']['success'] for row in selected}
    seeds = sorted(np.random.default_rng(SAMPLE_SEED).choice(sorted(archival), TOTAL_SEEDS, replace=False).tolist())
    return seeds, archival, {path.name: file_sha256(path) for path in paths}


def make_env():
    return make_peg_insertion_env(device='cpu', render_backend='cuda', obs_mode='rgb+depth+segmentation',
                                  control_mode='pd_ee_delta_pose', max_episode_steps=1000,
                                  sensor_configs={'base_camera': dict(width=512, height=512),
                                                  'hand_camera': dict(width=16, height=16)})


def reset_state(obs, env):
    images = {key: val[0].cpu().numpy() for key, val in obs['sensor_data']['base_camera'].items()}
    camera = {key: val[0].cpu().numpy() for key, val in obs['sensor_param']['base_camera'].items()}
    verify_camera(images, camera, CAMERA)
    return np.r_[exp2.robot15(env), exp2.gt_object21(env)]


@torch.inference_mode()
def episode(env, front, decoder, seed, mode, videos):
    obs, _ = env.reset(seed=seed)
    reset = reset_state(obs, env)
    decoder.reset()
    task = env.unwrapped
    if mode == 'hybrid':
        front.reset({0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])})
    else:
        front.reset()
    rows = []
    success = False
    end_reason = 'horizon'
    initial_peg_z = float(task.peg.pose.p[0, 2])
    freq = float(task.control_freq)
    plan = schedule(seed, MOTION_START_STEP / freq)
    plan['stop_expert_insert'] = False
    motion = EpisodeHoleMotion(env, plan, None)
    video = RolloutVideo(videos, seed, mode, freq)
    try:
        return moving_loop(env, front, decoder, seed, mode, obs, reset, task, rows,
                           success, end_reason, initial_peg_z, plan, motion, video, freq)
    finally:
        video.close()


def moving_loop(env, front, decoder, seed, mode, obs, reset, task, rows,
                success, end_reason, initial_peg_z, plan, motion, video, freq):
    for step in range(exp2.HORIZON):
        moving = motion.advance(step, None)
        if moving:
            obs = task.get_obs(task.get_info())
        state = motion.frame_data()
        delta = state['offset_m']
        shift_mm = float(np.linalg.norm(delta) * 1000)
        video.add(obs, step, 'BOX MOVING' if moving else ('Box stopped' if shift_mm else 'Box stationary'), shift_mm)
        robot = exp2.robot15(env)
        if mode == 'hybrid':
            gt_visibility = {k: v[0].cpu().numpy() for k, v in vars(get_gt_state(env)).items()
                             if k in ('peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes', 'hole_radius')}
            obj, counts, timings, votes = front.step(obs, robot[:3], gt_visibility)
        else:
            obj, _, timings, _ = front.step(obs)
            counts, votes = front.counts, None
        assert obj.shape == (21,) and np.isfinite(obj).all()
        started = perf_counter()
        action = decoder.act(np.r_[robot, obj], counts)
        decoder_seconds = perf_counter() - started
        if not np.isfinite(action).all():
            raise FloatingPointError(f'{mode} produced a nonfinite action at {seed}:{step}')
        filtered = decoder.last[15:].copy()
        assert all(np.array_equal(d.last, decoder.last) for d in decoder.decs)
        target = exp2.gt_object21(env)
        inverse = exp2.q_conj(target[13:17])

        def tip_in_hole(x):
            return exp2.q_rot(inverse, x[7:10]) * 1000

        actual = tip_in_hole(target)
        raw_error = float(np.linalg.norm((tip_in_hole(obj) - actual)[1:]))
        filtered_error = float(np.linalg.norm((tip_in_hole(filtered) - actual)[1:]))
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        rows.append(dict(step=step, raw_lateral_mm=raw_error, filtered_lateral_mm=filtered_error,
                         actual_relative_mm=actual.tolist(), object21=obj.tolist(), filtered=filtered.tolist(),
                         object21_gt=target.tolist(), grasp=grasp, observed=list(counts), rigid_votes=votes,
                         lift_m=float(task.peg.pose.p[0, 2]) - initial_peg_z, action=action.tolist(),
                         elapsed_ms=[v * 1000 for v in (*timings, decoder_seconds)],
                         box_moving=bool(moving), box_offset_mm=(delta * 1000).tolist(),
                         box_status=state['status']))
        obs, _, term, trunc, info = env.step(torch.as_tensor(action[None], device=task.device))
        if exp2.is_success(info):
            success, end_reason = True, 'success'
            break
        if bool(term[0]) or bool(trunc[0]):
            end_reason = 'terminated' if bool(term[0]) else 'truncated'
            break
    path, nframes = video.finish(obs, len(rows), success, end_reason, shift_mm)
    moves = [r['step'] for r in rows if r['box_moving']]
    return reset, dict(success=success, steps=len(rows), end_reason=end_reason,
                       ever_grasped=any(r['grasp'] for r in rows), trace=rows, motion_plan=plan,
                       box_max_shift_mm=max(np.linalg.norm(r['box_offset_mm']) for r in rows).item(),
                       box_moved_frames=len(moves), first_move_step=moves[0] if moves else None,
                       movie=str(path.name), video_frames=nframes, video_fps=freq)


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'tracker', 'gnn', 'output'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.start < a.stop <= TOTAL_SEEDS
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    device = torch.device('cuda:0')
    seeds, archive, archival_hashes = source_seeds(a.source)
    previous = json.loads((a.source / 'graph_candidate_paths.json').read_text())
    paths = [a.source / f'recurrent/exp2/checkpoints/runs_r6_mix_filter/mamba_step_s{i}/last.pt' for i in range(5)]
    dec_ckpts = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    assert all(c['mix'] == '0,0.25,0.5,0.75,1' and c['chunk'] == 1 and c['rot6'] and c['obs_dim'] == 44
               and c['kin'] == {'min_box': 2, 'min_peg': 5, 'robust': False} for c in dec_ckpts)
    decoder = exp2.Ensemble([exp2.Decoder(c, 'step') for c in dec_ckpts])
    old, hybrid = (exp2.Graph(previous['gnn'], previous['corner'], device),
                   HybridFrontend(a.tracker, a.gnn, device))
    # Use the same detector, camera, policy and filter for both comparison arms.
    old.detector = hybrid.detector
    cfg = dict(seeds=seeds[a.start:a.stop], all_seeds=seeds, camera=CAMERA, horizon=exp2.HORIZON,
               sample_seed=SAMPLE_SEED, historical_eval_hashes=archival_hashes,
               hole_motion='continuous horizontal translation', motion_start_step=MOTION_START_STEP,
               motion_guard_m=.08, expected_motion_distance_m=[.06, .12],
               video_script_sha256=file_sha256(Path(__file__).with_name('video.py')),
               detectors='same improved CUDA RANSAC, NumPy RGB-D', hybrid_ids='GT assignment only',
               graph_hashes={k: file_sha256(v) for k, v in previous.items()},
               hybrid_hashes=dict(tracker=file_sha256(a.tracker), gnn=file_sha256(a.gnn)),
               recurrent_hashes=[file_sha256(path) for path in paths],
               script_sha256=file_sha256(__file__),
               frontend_sha256=file_sha256(Path(__file__).with_name('hybrid_frontend.py')),
               rigid_sha256=file_sha256(__import__('rigid_box').__file__))
    a.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='closed_loop_moving_box_mamba', job_type='evaluate',
                          seed=SAMPLE_SEED, config=cfg, root=a.output / 'experiments') as logger:
        env = make_env()
        logger.record(runtime=dict(simulation=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name()))
        totals = {'old': 0, 'hybrid': 0}
        rows_path = a.output / 'rollouts.jsonl'
        videos = a.output / 'videos'
        videos.mkdir()
        began = monotonic()
        try:
            with rows_path.open('w') as stream:
                for pos, seed in enumerate(cfg['seeds'], a.start):
                    order = ('old', 'hybrid') if pos % 2 == 0 else ('hybrid', 'old')
                    states, runs = {}, {}
                    for mode in order:
                        states[mode], runs[mode] = episode(env, old if mode == 'old' else hybrid, decoder, seed, mode, videos)
                        totals[mode] += int(runs[mode]['success'])
                    np.testing.assert_allclose(states['old'], states['hybrid'], rtol=0, atol=1e-6)
                    assert runs['old']['motion_plan'] == runs['hybrid']['motion_plan']
                    stream.write(json.dumps(dict(seed=seed, archival_success=archive[seed], order=order,
                                                  **runs), allow_nan=False) + '\n')
                    stream.flush()
                    print(seed, {k: (v['success'],v['steps'], round(v['box_max_shift_mm'],1)) for k,v in runs.items()},
                          f'{monotonic()-began:.1f}s', flush=True)
        finally:
            env.close()
        summary = a.output / 'summary.json'
        write(summary, dict(run_id=logger.run_id, config=cfg, success=totals,
                            episodes=len(cfg['seeds']), seconds=monotonic()-began))
        logger.log(totals)
        logger.link_output(rows_path)
        logger.link_output(summary)
        for video in sorted(videos.glob('*.mp4')):
            logger.link_output(video)
    print('Completed matched moving-Box rollouts', flush=True)


if __name__ == '__main__':
    main()
