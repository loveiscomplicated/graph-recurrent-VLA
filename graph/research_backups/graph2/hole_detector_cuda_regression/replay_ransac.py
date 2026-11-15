"""Paired, frozen Object21 ablation at the current Main camera."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from data_collection.episode_dataset import file_sha256
from data_collection.object21_bc import predict_graphs
from data_collection.scripted_policy import make_policy
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from evaluation.gnn.vertex_metrics import statistics
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, visible_reference

CONDITIONS = ('detected', 'remove_front', 'visible_gt_front')
METRICS = ('hole_center_mm', 'axis_deg', 'tip_lateral_mm')
CAMERA = dict(camera_eye_m=[0.65, -0.30, 0.65], camera_target_m=[0.065, 0.297, 0.11],
              resolution=[512, 512], vertical_fov_deg=90)


def load_models(paths, device):
    for path in paths.values():
        if file_sha256(path) != path.parent.name:
            raise ValueError(f'Checkpoint content differs from its artifact digest: {path}')
    graph = torch.load(paths['gnn'], map_location='cpu', weights_only=True)
    temporal = torch.load(paths['corner'], map_location='cpu', weights_only=True)
    config = graph['config']
    temporal_config = temporal['config']
    model_config = temporal_config['model']
    if (config['temporal']['sha256'] != file_sha256(paths['corner'])
            or temporal['dataset_sha256'] != config['data']['dataset_sha256']
            or model_config['architecture'] != config['temporal']['architecture']
            or temporal_config['seed'] != config['seed']):
        raise ValueError('GNN and temporal checkpoint identities differ')
    hashes = temporal_config['implementation']['files_sha256']
    for file in ('graph/temporal/fixed_id_corner.py', 'graph/temporal/causal_transformer.py',
                 'data_collection/temporal/fixed_id_corner_sequences.py'):
        if file_sha256(file) != hashes[file]:
            raise ValueError(f'Temporal source differs from checkpoint: {file}')
    gnn = VertexEdgeGatedMPNN(**config['gnn'])
    tracker = FixedIdCornerPredictor(**model_config)
    gnn.load_state_dict(graph['model'], strict=True)
    tracker.load_state_dict(temporal['model'], strict=True)
    return [m.to(device).eval().requires_grad_(False) for m in (gnn, tracker)]


def rotation(quat):
    return Rotation.from_quat(np.roll(quat, -1))


def target(gt, tcp):
    peg, hole = gt['peg_pose'], gt['hole_pose']
    half = gt['peg_half_sizes'].reshape(3)
    tip = peg[:3] + rotation(peg[3:]).apply([half[0], 0., 0.])
    return np.concatenate((peg[:3] - tcp, peg[3:], tip - hole[:3], hole[:3] - tcp,
                           hole[3:], half, gt['hole_radius'].reshape(1)))


def error(prediction, truth):
    hole_rot = rotation(truth[13:17])
    axis = hole_rot.apply([1., 0., 0.])
    pred_axis = rotation(prediction[13:17]).apply([1., 0., 0.])
    lateral = hole_rot.inv().apply(prediction[7:10] - truth[7:10])[1:]
    return dict(hole_center_mm=float(np.linalg.norm(prediction[10:13] - truth[10:13]) * 1000),
                axis_deg=float(np.degrees(np.arccos(np.clip(axis @ pred_axis, -1., 1.)))),
                tip_lateral_mm=float(np.linalg.norm(lateral) * 1000))


def prepare_frame(env, obs, detector, trackers):
    sensor = {key: t[0].cpu().numpy()
              for key, t in obs['sensor_data']['base_camera'].items()}
    calibration = {key: t[0].cpu().numpy()
                   for key, t in obs['sensor_param']['base_camera'].items()}
    verify_camera(sensor, calibration, CAMERA)
    sensor.update(calibration)
    result = detector(sensor['rgb'], sensor['depth'], sensor['intrinsic_cv'], sensor['extrinsic_cv'])
    xyz, kinds = result.xyz.cpu().numpy(), result.object_id.cpu().numpy()
    assert len(xyz) == len(result.metadata)
    flagged = np.array([kind == 1 and (meta.get('contour_hole', False)
                                      or meta.get('method') == 'box_opening')
                        for kind, meta in zip(kinds, result.metadata, strict=True)], dtype=bool)
    gt = {key: t[0].cpu().numpy() for key, t in vars(get_gt_state(env)).items()}
    task = env.unwrapped
    ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    visible = visible_reference(gt, sensor, ids)
    fronts = [r for r in visible if r['kind'] == 'hole_rim' and r['corner_id'] < 4]
    oracle = np.asarray([r['xyz'] for r in fronts], dtype=np.float32).reshape(-1, 3)

    keep = ~flagged
    frames = ((xyz, kinds),
              (xyz[keep], kinds[keep]),
              (np.concatenate((xyz[keep], oracle)), np.concatenate((kinds[keep], np.ones(len(oracle), np.int64)))))
    robot = np.concatenate((obs['extra']['tcp_pose'][0].cpu().numpy(),
                            obs['agent']['qpos'][0, -2:].cpu().numpy())).astype(np.float32)
    assert np.allclose(robot[:3], gt['ee_pose'][:3], atol=1e-6, rtol=0)
    tracked_frames = []
    for (points, labels), tracker in zip(frames, trackers, strict=True):
        tracked = tracker.step(points, labels, robot)
        tracked_frames.append((tracked['positions'], [t.kind for t in tracker.tracks], tracked['observed']))

    front = np.array([r['xyz'] for r in corners(gt) if r['kind'] == 'hole_rim' and r['corner_id'] < 4])
    near_front = (np.linalg.norm(xyz[:, None] - front[None], axis=2).min(axis=1) <= .005) & (kinds == 1)
    counts = dict(peg=int((kinds == 0).sum()), box_retained=int((keep & (kinds == 1)).sum()),
                  rim_flagged=int(flagged.sum()), visible_gt_front=len(oracle),
                  untagged_within_5mm_front=int((near_front & keep).sum()),
                  flagged_within_5mm_front=int((near_front & flagged).sum()))
    return robot[:3], target(gt, robot[:3]), tracked_frames, counts


@torch.inference_mode()
def run_episode(env, seed, trajectory, max_steps, detector, temporal, gnn, device):
    obs, _ = env.reset(seed=seed)
    if trajectory:
        policy = make_policy()
        policy.reset(env, seed)
    trackers = [DetectedCornerTracker(temporal, .030, device) for _ in CONDITIONS]
    rows, frames, tcp, targets = [], [], [], []
    grasp_seen = False
    terminal = 'initial_only'
    for step in range(max_steps if trajectory else 1):
        phase = policy.phase if trajectory else None
        if phase == 'insert':
            terminal = 'insert_entry'
            break
        task = env.unwrapped
        grasp = trajectory and bool(task.agent.is_grasping(task.peg)[0])
        stages = ['initial'] if step == 0 else []
        if grasp and not grasp_seen:
            stages.append('first_grasp')
            grasp_seen = True
        if grasp and phase == 'align':
            stages.append('held_align')
        tcp_now, truth, tracked, counts = prepare_frame(env, obs, detector, trackers)
        if stages:
            rows.append(dict(seed=seed, step=step, stages=stages, counts=counts,
                             track_counts=dict(zip(CONDITIONS, [len(t[0]) for t in tracked], strict=True))))
            frames.append(tracked)
            targets.append(truth)
            tcp.append(tcp_now)
        if not trajectory:
            break
        obs, _, terminated, truncated, _ = env.step(policy.act(obs))
        if bool(terminated[0]) or bool(truncated[0]):
            terminal = 'terminated' if bool(terminated[0]) else 'truncated'
            break
    else:
        terminal = 'step_limit'

    if not np.isfinite(targets).all():
        raise ValueError(f'Non-finite GT for seed {seed}')
    tcp = np.asarray(tcp)
    for name, inputs in zip(CONDITIONS, zip(*frames, strict=True), strict=True):
        predictions = predict_graphs(gnn, inputs, tcp, 32, device)
        if predictions.shape != (len(rows), 21) or not np.isfinite(predictions).all():
            raise ValueError(f'Invalid 21D for seed {seed}: {name}')
        for row, prediction, truth in zip(rows, predictions, targets, strict=True):
            row[name] = dict(output21=prediction.tolist(), errors=error(prediction, truth))
            row['gt21'] = truth.tolist()
    return rows, terminal


def summary(rows):
    def metrics(group):
        result = {}
        for condition in CONDITIONS:
            result[condition] = {
                key: statistics([row[condition]['errors'][key] for row in group]) for key in METRICS}
        return result

    stages = {}
    for stage in ('initial', 'first_grasp', 'held_align'):
        group = [row for row in rows if stage in row['stages']]
        stages[stage] = dict(frames=len(group), seeds=len({row['seed'] for row in group}),
                             with_visible_gt=sum(r['counts']['visible_gt_front'] > 0 for r in group),
                             with_rim_flagged=sum(r['counts']['rim_flagged'] > 0 for r in group),
                             flagged_points=sum(r['counts']['rim_flagged'] for r in group),
                             untagged_near_front=sum(r['counts']['untagged_within_5mm_front'] for r in group),
                             all=metrics(group),
                             with_gt=metrics([r for r in group if r['counts']['visible_gt_front']]))
    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--select-seeds', type=int, nargs='+')
    parser.add_argument('--start', type=int, default=200)
    parser.add_argument('--stop', type=int, default=1400)
    args = parser.parse_args()
    paths = {name: Path(path) for name, path in json.loads(args.source.read_text()).items()}
    if not 200 <= args.start < args.stop <= 1400:
        parser.error('Expected a non-empty subrange of seeds 200–1399')
    seeds = args.select_seeds or list(range(args.start, args.stop))
    if args.smoke:
        seeds = seeds[:3]
    expert_seeds = [seed for seed in seeds if (seed - 200) % 75 == 0]
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(camera=CAMERA, seeds=seeds, expert_seeds=expert_seeds, max_expert_steps=200,
                  device=str(device), tracker_gate_m=.030,
                  detector='TorchCornerDetector.main(device=cuda:0, plane_fit_method=ransac), CUDA compilation defaults',
                  checkpoint_sha256={name: file_sha256(path) for name, path in paths.items()},
                  script_sha256=file_sha256(__file__), source_files={name: str(path) for name, path in paths.items()})
    with ExperimentLogger(team='graph', config=config, seed=0, model='hole_front_object21_ablation',
                          job_type='evaluate', root=args.output / 'experiments') as logger:
        gnn, temporal = load_models(paths, device)
        detector = TorchCornerDetector.main(device=device, plane_fit_method="ransac")
        env = make_peg_insertion_env(device='cpu', render_backend='cuda', obs_mode='rgb+depth+segmentation',
                                     control_mode='pd_ee_delta_pose', max_episode_steps=200,
                                     sensor_configs={'hand_camera': dict(width=16, height=16)})
        try:
            task = env.unwrapped
            logger.record(runtime=dict(torch=str(torch.__version__), maniskill=version('mani-skill'),
                                       gpu=torch.cuda.get_device_name(device),
                                       simulator=str(task.backend)))
            rows, terminals = [], {}
            with (args.output / 'frames.jsonl').open('w') as trace:
                for index, seed in enumerate(seeds, start=1):
                    batch, terminal = run_episode(env, seed, seed in expert_seeds, 200,
                                                  detector, temporal, gnn, device)
                    rows.extend(batch)
                    if seed in expert_seeds:
                        terminals[str(seed)] = terminal
                    for row in batch:
                        trace.write(json.dumps(row, allow_nan=False) + '\n')
                    trace.flush()
                    if index == 1 or index % 100 == 0 or seed in expert_seeds:
                        print(f'{index}/{len(seeds)} seeds, {len(rows)} scored frames; seed {seed}: {terminal}', flush=True)
            result = dict(run_id=logger.run_id, config=config, expert_terminals=terminals, metrics=summary(rows))
            write_json(args.output / 'summary.json', result)
            logger.log(result['metrics'], prefix='eval')
            logger.link_output(args.output / 'frames.jsonl')
            logger.link_output(args.output / 'summary.json')
            print(f'Completed {logger.run_id}', flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
