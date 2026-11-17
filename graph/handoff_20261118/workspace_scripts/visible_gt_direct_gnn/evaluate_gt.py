"""Run the frozen GNN directly on GT vertices supported by the camera."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import make_policy
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn import graph_batch
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection.evaluation import visible_reference

from evaluate import CAMERA, error, rotation, target


def collect(env, obs):
    sensor = {key: value[0].cpu().numpy()
              for key, value in obs['sensor_data']['base_camera'].items()}
    calibration = {key: value[0].cpu().numpy()
                   for key, value in obs['sensor_param']['base_camera'].items()}
    verify_camera(sensor, calibration, CAMERA)
    sensor.update(calibration)
    gt = {key: value[0].cpu().numpy() for key, value in vars(get_gt_state(env)).items()}
    task = env.unwrapped
    ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    refs = visible_reference(gt, sensor, ids)
    refs = [r for r in refs if r['kind'] != 'hole_rim' or r['corner_id'] < 4]
    tcp = obs['extra']['tcp_pose'][0, :3].cpu().numpy()
    np.testing.assert_allclose(tcp, gt['ee_pose'][:3], atol=1e-6, rtol=0)
    points = np.array([r['xyz'] for r in refs], np.float32).reshape(-1, 3)
    kinds = np.array([r['object_id'] for r in refs], np.int64)
    counts = {name: sum(r['kind'] == name for r in refs) for name in ('peg', 'box_outer', 'hole_rim')}
    return points, kinds, tcp, target(gt, tcp), counts


def direct_predictions(gnn, samples, device):
    for start in range(0, len(samples), 32):
        batch = samples[start:start+32]
        width = max(1, max(len(s[0]) for s in batch))
        xyz = np.zeros((len(batch),width,3), np.float32)
        kinds = np.full((len(batch),width), -1, np.int64)
        valid = np.zeros((len(batch),width), bool)
        for index, (points, labels, *_) in enumerate(batch):
            xyz[index,:len(points)] = points
            kinds[index,:len(points)] = labels
            valid[index,:len(points)] = True
        arrays = (xyz, kinds, valid, valid, np.stack([sample[2] for sample in batch]))
        graph, flags = graph_batch(*(torch.as_tensor(array, device=device) for array in arrays))
        out = gnn(graph, flags)
        values = [out[key].cpu().numpy() for key in ('object21','endpoint_geom','endpoint_residual')]
        yield from zip(*values, strict=True)


def point_norm(value):
    return float(np.linalg.norm(value)*1000)


def errors(pred, geom, residual, gt):
    result = error(pred, gt)
    peg_rot = rotation(pred[3:7])
    gt_peg_rot = rotation(gt[3:7])
    gt_hole_rot = rotation(gt[13:17])
    peg_axis = peg_rot.apply([1.,0.,0.])
    gt_peg_axis = gt_peg_rot.apply([1.,0.,0.])
    xyz_tip_error = pred[7:10] + pred[10:13] - gt[7:10] - gt[10:13]
    geom_lateral = gt_hole_rot.inv().apply(geom - gt[7:10])[1:]
    result.update(peg_center_mm=point_norm(pred[:3]-gt[:3]),
                  peg_axis_deg=float(np.degrees(np.arccos(np.clip(peg_axis @ gt_peg_axis, -1, 1)))),
                  peg_tip_mm=point_norm(xyz_tip_error),
                  geom_tip_lateral_mm=point_norm(geom_lateral),
                  residual_lateral_mm=point_norm(gt_hole_rot.inv().apply(residual)[1:]))
    return result


@torch.inference_mode()
def run_episode(env, seed, trajectory, gnn, device):
    obs,_ = env.reset(seed=seed)
    if trajectory:
        policy = make_policy()
        policy.reset(env, seed)
    seen_grasp = False
    terminal = 'initial_only'
    rows, samples = [], []
    for step in range(200 if trajectory else 1):
        phase = policy.phase if trajectory else None
        if phase == 'insert':
            terminal = 'insert_entry'
            break
        task = env.unwrapped
        grasp = trajectory and bool(task.agent.is_grasping(task.peg)[0])
        stages = ['initial'] if step == 0 else []
        if grasp and not seen_grasp:
            stages.append('first_grasp')
            seen_grasp = True
        if grasp and phase == 'align':
            stages.append('held_align')
        if stages:
            row = collect(env, obs)
            rows.append(dict(seed=seed, step=step, stages=stages, counts=row[4]))
            samples.append(row)
        if not trajectory:
            break
        obs,_,terminated,truncated,_ = env.step(policy.act(obs))
        if bool(terminated[0]) or bool(truncated[0]):
            terminal = 'terminated' if bool(terminated[0]) else 'truncated'
            break
    else:
        terminal = 'step_limit'
    for row, inp, (pred, geom, residual) in zip(rows, samples, direct_predictions(gnn,samples,device), strict=True):
        gt = inp[3]
        assert pred.shape == (21,) and np.isfinite(pred).all()
        np.testing.assert_allclose(pred[7:10], geom+residual, atol=1e-7, rtol=0)
        row.update(gt21=gt.tolist(), output21=pred.tolist(), endpoint_geom=geom.tolist(),
                   endpoint_residual=residual.tolist(), errors=errors(pred,geom,residual,gt))
    return rows, terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=200)
    parser.add_argument('--stop', type=int, default=1400)
    args = parser.parse_args()
    path = Path(json.loads(args.source.read_text())['gnn'])
    digest = file_sha256(path)
    if digest != path.parent.name:
        raise ValueError('GNN checksum mismatch')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    gnn = VertexEdgeGatedMPNN(**ck['config']['gnn'])
    gnn.load_state_dict(ck['model'], strict=True)
    gnn.to(device).eval().requires_grad_(False)
    seeds = list(range(args.start, args.stop))
    expert = [s for s in seeds if (s-200)%75 == 0]
    args.output.mkdir(parents=True,exist_ok=False)
    cfg = dict(camera=CAMERA,seeds=seeds,expert_seeds=expert,device=str(device),
               gnn_sha256=digest, max_expert_steps=200, script_sha256=file_sha256(__file__))
    with ExperimentLogger(team='graph',model='visible_gt_direct_gnn',config=cfg,seed=200,job_type='evaluate',
                          root=args.output/'experiments') as logger:
        env = make_peg_insertion_env(device='cpu',render_backend='cuda',
            obs_mode='rgb+depth+segmentation',control_mode='pd_ee_delta_pose',max_episode_steps=200,
            sensor_configs={'hand_camera':dict(width=16,height=16)})
        try:
            logger.record(runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
                                       maniskill=version('mani-skill'),backend=str(env.unwrapped.backend)))
            terminals = {}
            collected = []
            with (args.output/'frames.jsonl').open('w') as f:
                for i,seed in enumerate(seeds,1):
                    rows,terminal = run_episode(env,seed,seed in expert,gnn,device)
                    collected.extend(rows)
                    if seed in expert:
                        terminals[str(seed)] = terminal
                    for row in rows:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    f.flush()
                    if i == 1 or i % 100 == 0 or seed in expert:
                        print(i,len(seeds),seed,terminal,flush=True)
            summary = {}
            for stage in ('initial','first_grasp','held_align'):
                group = [r for r in collected if stage in r['stages']]
                if group:
                    summary[stage] = {k:float(np.median([r['errors'][k] for r in group]))
                                      for k in group[0]['errors']}
            logger.log(summary,prefix='eval/median')
            output = args.output/'summary.json'
            write_json(output,dict(run_id=logger.run_id,config=cfg,expert_terminals=terminals,
                                   frames=len(collected),summary=summary))
            logger.link_output(output)
            logger.link_output(args.output/'frames.jsonl')
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
