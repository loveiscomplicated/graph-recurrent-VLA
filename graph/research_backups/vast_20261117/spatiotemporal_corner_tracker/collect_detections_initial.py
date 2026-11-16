"""Label existing RGB-D detections with simulator vertex IDs for an offline tracker probe."""
import argparse
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluate_points import identity
from evaluation.initial_corner_detection import verify_camera
from graph.temporal.detected_corner_tracking import match_positions
from oracle_reference import CAMERA, target
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, visible_reference


@torch.inference_mode()
def episode(env, detector, seed, reference):
    obs, _ = env.reset(seed=seed)
    policy = make_policy()
    policy.reset(env, seed)
    task = env.unwrapped
    actor_ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    motion = None
    rows = []
    for step in range(len(reference['step'])):
        phase = policy.phase
        grasp = bool(task.agent.is_grasping(task.peg)[0])
        if motion is None and phase == 'align' and grasp:
            motion = EpisodeHoleMotion(env, schedule(seed, step / task.control_freq), policy)
        if motion is not None and motion.advance(step, policy):
            obs = task.get_obs(task.get_info())
        assert step == reference['step'][step]
        assert phase == reference['phase'][step] and grasp == reference['grasp'][step]
        gt = {key: value[0].cpu().numpy() for key, value in vars(get_gt_state(env)).items()}
        tcp = obs['extra']['tcp_pose'][0, :3].cpu().numpy()
        np.testing.assert_allclose(target(gt, tcp), reference['target'][step], atol=1e-6, rtol=0)
        all_corners = corners(gt)
        assert [identity(c) for c in all_corners] == list(range(24))
        xyz = np.asarray([c['xyz'] for c in all_corners[:20]], dtype=np.float32)
        np.testing.assert_allclose(xyz, reference['xyz'][step], atol=1e-6, rtol=0)
        images = {key: value[0].cpu().numpy() for key, value in obs['sensor_data']['base_camera'].items()}
        camera = {key: value[0].cpu().numpy() for key, value in obs['sensor_param']['base_camera'].items()}
        verify_camera(images, camera, CAMERA)
        raw = detector(images['rgb'], images['depth'], camera['intrinsic_cv'], camera['extrinsic_cv'])
        points, labels = raw.xyz.cpu().numpy(), raw.object_id.cpu().numpy()
        visible = visible_reference(gt, {**images, **camera}, actor_ids)
        visible_xyz = np.asarray([r['xyz'] for r in visible]).reshape(-1, 3)
        visible_labels = np.asarray([r['object_id'] for r in visible], dtype=np.int64)
        visible_ids = [identity(r) for r in visible]
        observed = np.zeros(20, dtype=bool)
        detected_xyz = np.zeros((20, 3), dtype=np.float32)
        for i, j in match_positions(points, labels, visible_xyz, visible_labels, .005):
            slot = visible_ids[j]
            if slot < 20:
                observed[slot] = True
                detected_xyz[slot] = points[i]
        assert np.all(np.linalg.norm(detected_xyz[observed] - xyz[observed], axis=-1) <= .005001)
        rows.append(dict(seed=seed, step=step, observed=observed, detected_xyz=detected_xyz,
                         visible=np.isin(np.arange(20), visible_ids)))
        obs, _, term, trunc, _ = env.step(policy.act(obs))
        assert not bool(term[0]) and not bool(trunc[0]), (seed, step)
    assert policy.phase == 'insert', (seed, policy.phase)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    assert 3100000 <= args.start < args.stop <= 3100144
    torch.set_num_threads(2)
    root = args.data_root
    parts, hashes = [], {}
    for start in range(3100000, 3100176, 44):
        if args.start >= start + 44 or args.stop <= start:
            continue
        folder = root / f'data_{start}_{start + 44}'
        path = folder / 'dataset.npz'
        import json
        digest = file_sha256(path)
        assert digest == json.loads((folder / 'summary.json').read_text())['dataset_sha256']
        hashes[path.parent.name] = digest
        with np.load(path, allow_pickle=False) as source:
            keep = (args.start <= source['seed']) & (source['seed'] < args.stop)
            parts.append({key: source[key][keep] for key in ('seed', 'step', 'xyz', 'target', 'phase', 'grasp')})
    refs = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    seeds = list(range(args.start, args.stop))
    assert np.unique(refs['seed']).tolist() == seeds
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(seeds=seeds, reference_hashes=hashes, camera=CAMERA, match_m=.005,
                  detector='RANSAC compiled CUDA, NumPy RGB-D', script_sha256=file_sha256(__file__))
    start = monotonic()
    with ExperimentLogger(team='graph', model='spatiotemporal_corner_collection', job_type='evaluate',
                          seed=0, config=config, root=args.output / 'experiments') as logger:
        env = make_peg_insertion_env(device='cpu', render_backend='cuda', obs_mode='rgb+depth+segmentation',
                                     control_mode='pd_ee_delta_pose', max_episode_steps=200)
        detector = TorchCornerDetector.main(device='cuda:0', plane_fit_method='ransac')
        logger.record(runtime=dict(backend=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name()))
        rows = []
        try:
            for seed in seeds:
                index = refs['seed'] == seed
                rows.extend(episode(env, detector, seed, {key: value[index] for key, value in refs.items()}))
                print(seed, len(rows), f'{monotonic() - start:.1f}s', flush=True)
        finally:
            env.close()
        path = args.output / 'detections.npz'
        np.savez_compressed(path, **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        pet = args.output / 'summary.json'
        write(pet, dict(run_id=logger.run_id, config=config, frames=len(rows),
                        seconds=monotonic() - start, sha256=file_sha256(path)))
        logger.log({'frames': len(rows)})
        logger.link_output(path)
        logger.link_output(pet)
    print('Completed', logger.run_id, flush=True)


if __name__ == '__main__':
    main()
