"""Collect synchronized full Expert episodes with the validated new frontend."""
import argparse
import json
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import cv2
import numpy as np
import torch

import exp2_eval as exp2
import run_rollouts as base
from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state
from utils.experiment_logger import ExperimentLogger, write

FIRST, VAL, STOP = 3100000, 3100120, 3100144


def read_capture(root, reference, seed):
    capture_start = FIRST + (seed - FIRST) // 24 * 24
    capture = root / f'shard_{capture_start}_{capture_start + 24}' / 'detections.npz'
    ref_start = FIRST + (seed - FIRST) // 44 * 44
    ref = reference / f'data_{ref_start}_{ref_start + 44}' / 'dataset.npz'
    with np.load(capture, allow_pickle=False) as a, np.load(ref, allow_pickle=False) as b:
        mask_a, mask_b = a['seed'] == seed, b['seed'] == seed
        np.testing.assert_array_equal(a['step'][mask_a], b['step'][mask_b])
        return dict(detected=a['detected_xyz'][mask_a], observed=a['observed'][mask_a],
                    target=b['target'][mask_b], camera_step=b['step'][mask_b])


@torch.inference_mode()
def episode(env, frontend, seed, cached):
    obs, _ = env.reset(seed=seed)
    base.reset_state(obs, env)
    policy = make_policy()
    policy.reset(env, seed)
    task = env.unwrapped
    frontend.reset({0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])})
    moving = seed % 2 == 0
    motion = None
    cache_valid = True
    rows = []
    success = False
    end = 'timeout'
    for t in range(exp2.HORIZON):
        phase = policy.phase
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        if moving and motion is None and phase == 'align' and grasp:
            motion = EpisodeHoleMotion(env, schedule(seed, t/task.control_freq), policy)
        if motion is not None and motion.advance(t, policy):
            obs = task.get_obs(task.get_info())
        offset = np.zeros(3) if motion is None else motion.frame_data()['offset_m']
        robot = exp2.robot15(env)
        target = exp2.gt_object21(env)
        cache_valid = (cache_valid and t < len(cached['target']) and
                       np.allclose(target, cached['target'][t], rtol=0, atol=2e-6))
        if cache_valid:
            with patch.object(frontend, 'detect', return_value=(cached['detected'][t], cached['observed'][t])):
                estimated, counts, _, _ = frontend.step(obs, robot[:3], None)
        else:
            visibility = {k: v[0].cpu().numpy() for k, v in vars(get_gt_state(env)).items()
                          if k in ('peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes', 'hole_radius')}
            estimated, counts, _, _ = frontend.step(obs, robot[:3], visibility)
        action = np.asarray(exp2.to_np(policy.act(obs)), dtype=np.float32).reshape(7)
        rows.append(dict(state36_gt=np.r_[robot, target], state36_transformer=np.r_[robot, estimated],
                         expert_action=action.copy(), phase=phase, counts=counts, box_offset=offset.copy(),
                         cached=cache_valid))
        obs, _, term, trunc, info = env.step(torch.as_tensor(action[None], device=task.device))
        if exp2.is_success(info):
            success, end = True, 'success'
            break
        if bool(term[0]) or bool(trunc[0]):
            end = 'terminated' if bool(term[0]) else 'truncated'
            break
    return rows, dict(seed=seed, success=success, end=end, steps=len(rows), moving_requested=moving,
                      moved_mm=max(np.linalg.norm(r['box_offset']) for r in rows)*1000,
                      cached=int(sum(r['cached'] for r in rows)),
                      insert_steps=int(sum(r['phase'] == 'insert' for r in rows)))


def main():
    p = argparse.ArgumentParser()
    for name in ('tracker', 'gnn', 'captures', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    a = p.parse_args()
    assert FIRST <= a.start < a.stop <= STOP
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    front = base.HybridFrontend(a.tracker, a.gnn, torch.device('cuda:0'))
    env = base.make_env()
    identities = dict(seed_range=[a.start,a.stop], guide='ScriptedPegPolicy, successful full trajectories',
                      motion='even seeds move on grasp and align; odd seeds fixed',
                      tracker_sha256=file_sha256(a.tracker), gnn_sha256=file_sha256(a.gnn),
                      script_sha256=file_sha256(__file__), expert_noise=0., camera=base.CAMERA)
    a.output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    rows, histories, offsets = [], [], [0]
    with ExperimentLogger(team='graph', model='recurrent_new_frontend_demos', job_type='collect',
                          seed=0, config=identities, root=a.output / 'experiments') as log:
        try:
            for seed in range(a.start,a.stop):
                parts, info = episode(env, front, seed, read_capture(a.captures, a.reference, seed))
                histories.append(info)
                if info['success']:
                    assert info['insert_steps'] > 0
                    rows.extend(parts)
                    offsets.append(len(rows))
                print(seed, info['success'], info['steps'], info['cached'], info['insert_steps'],
                      f'{monotonic()-started:.1f}s', flush=True)
        finally:
            env.close()
        assert rows, 'No successful Expert episodes in this shard'
        data = a.output / 'demos.npz'
        arrays = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
        np.testing.assert_array_equal(arrays['state36_gt'][:,:15],arrays['state36_transformer'][:,:15])
        assert np.isfinite(arrays['state36_transformer']).all()
        np.savez_compressed(data, **arrays, episode_offsets=np.array(offsets,dtype=np.int64),
                            reset_seed=np.array([e['seed'] for e in histories if e['success']], dtype=np.int64))
        summary = a.output / 'summary.json'
        write(summary, dict(run_id=log.run_id, config=identities, episodes=histories,
                            frames=len(rows), sha256=file_sha256(data)))
        log.log({'successes': len(offsets)-1, 'frames':len(rows)})
        log.link_output(data)
        log.link_output(summary)
    print('Completed full Expert collection', flush=True)


if __name__=='__main__':
    main()
