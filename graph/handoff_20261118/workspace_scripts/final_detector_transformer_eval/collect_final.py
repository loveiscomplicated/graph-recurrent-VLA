"""Label existing RGB-D detections with simulator vertex IDs for an offline tracker probe."""
import argparse
import json
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
def episode(env, detector, seed, reference, expected_terminal):
    obs, _ = env.reset(seed=seed)
    policy = make_policy()
    policy.reset(env, seed)
    task = env.unwrapped
    actor_ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    motion = None
    rows = []
    terminal = None
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
        recall = np.zeros((3, 20), dtype=bool)
        for k, gate in enumerate((.003, .005, .010)):
            for i, j in match_positions(points, labels, visible_xyz, visible_labels, gate):
                slot = visible_ids[j]
                if slot < 20:
                    recall[k, slot] = True
        np.testing.assert_array_equal(recall[1], observed)
        hinted_hole = np.array([kind == 1 and (meta.get('contour_hole', False) or
                                                meta.get('method') == 'box_opening')
                                for kind, meta in zip(labels, raw.metadata, strict=True)])
        family_hint = np.where(labels == 0, 0, np.where(hinted_hole, 2, 1))
        rows.append(dict(seed=seed, step=step, observed=observed, detected_xyz=detected_xyz,
                         visible=np.isin(np.arange(20), visible_ids), recall=recall,
                         raw_xyz=points, raw_kind=labels, raw_family_hint=family_hint,
                         raw_count=len(points), detector_ms=raw.elapsed_ms))
        obs, _, term, trunc, _ = env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            assert step + 1 == len(reference['step']), (seed, step)
            terminal = 'terminated' if bool(term[0]) else 'truncated'
            break
    if terminal is None:
        terminal = 'insert_entry' if policy.phase == 'insert' else 'step_limit'
    assert terminal == expected_terminal, (seed, terminal, expected_terminal)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    assert 3100144 <= args.start < args.stop <= 3100176
    torch.set_num_threads(2)
    root = args.data_root
    parts, hashes, terminals = [], {}, {}
    for start in range(3100000, 3100176, 44):
        if args.start >= start + 44 or args.stop <= start:
            continue
        folder = root / f'data_{start}_{start + 44}'
        path = folder / 'dataset.npz'
        digest = file_sha256(path)
        summary = json.loads((folder / 'summary.json').read_text())
        assert digest == summary['dataset_sha256']
        terminals.update(summary['terminals'])
        hashes[path.parent.name] = digest
        with np.load(path, allow_pickle=False) as source:
            keep = (args.start <= source['seed']) & (source['seed'] < args.stop)
            parts.append({key: source[key][keep] for key in ('seed', 'step', 'xyz', 'target', 'phase', 'grasp')})
    refs = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    seeds = list(range(args.start, args.stop))
    assert np.unique(refs['seed']).tolist() == seeds
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(seeds=seeds, reference_hashes=hashes, camera=CAMERA, match_m=.005,
                  detector='TorchCornerDetector.main default CUDA, NumPy RGB-D',
                  detector_sha256=file_sha256('visual/corner_detection/detector.py'),
                  script_sha256=file_sha256(__file__))
    start = monotonic()
    with ExperimentLogger(team='graph', model='final_detector_transformer_collection', job_type='evaluate',
                          seed=0, config=config, root=args.output / 'experiments') as logger:
        env = make_peg_insertion_env(device='cpu', render_backend='cuda', obs_mode='rgb+depth+segmentation',
                                     control_mode='pd_ee_delta_pose', max_episode_steps=200)
        detector = TorchCornerDetector.main(device='cuda:0')
        assert detector.plane_fit_method == 'grid'
        logger.record(plane_fit_method=detector.plane_fit_method)
        logger.record(runtime=dict(backend=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name()))
        rows = []
        try:
            for seed in seeds:
                index = refs['seed'] == seed
                rows.extend(episode(env, detector, seed, {key: value[index] for key, value in refs.items()}, terminals[str(seed)]))
                print(seed, len(rows), f'{monotonic() - start:.1f}s', flush=True)
        finally:
            env.close()
        path = args.output / 'detections.npz'
        ragged = ('raw_xyz', 'raw_kind', 'raw_family_hint')
        values = {key: np.stack([row[key] for row in rows]) for key in rows[0] if key not in ragged}
        values.update({key: np.concatenate([row[key] for row in rows]) for key in ragged})
        np.savez_compressed(path, **values)
        pet = args.output / 'summary.json'
        write(pet, dict(run_id=logger.run_id, config=config, frames=len(rows),
                        seconds=monotonic() - start, sha256=file_sha256(path)))
        logger.log({'frames': len(rows)})
        logger.link_output(path)
        logger.link_output(pet)
    print('Completed', logger.run_id, flush=True)


if __name__ == '__main__':
    main()
