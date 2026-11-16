"""Compare two frozen GNNs on identical detector and tracker outputs."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.object21_bc import predict_graphs
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector

from oracle_reference import CAMERA, load_models, target


def sensor_inputs(obs):
    rgbd = {k:v[0].cpu().numpy() for k,v in obs['sensor_data']['base_camera'].items()}
    camera = {k:v[0].cpu().numpy() for k,v in obs['sensor_param']['base_camera'].items()}
    verify_camera(rgbd,camera,CAMERA)
    return rgbd['rgb'],rgbd['depth'],camera['intrinsic_cv'],camera['extrinsic_cv']


@torch.inference_mode()
def episode(env, seed, detector, temporal, models, device, reference):
    obs,_=env.reset(seed=seed)
    policy=make_policy()
    policy.reset(env,seed)
    tracker=DetectedCornerTracker(temporal,.030,device)
    rows, track_frames, tcp = [], [], []
    terminal='step_limit'
    for step in range(200):
        phase=policy.phase
        if phase=='insert':
            terminal='insert_entry'
            break
        ref=reference[step]
        if phase != ref['phase']:
            raise ValueError(f'Policy phase differs at {seed}/{step}')
        task=env.unwrapped
        grasp=bool(task.agent.is_grasping(task.peg)[0])
        if grasp!=ref['grasp']:
            raise ValueError(f'Grasp state differs at {seed}/{step}')
        gt={k:v[0].cpu().numpy() for k,v in vars(get_gt_state(env)).items()}
        robot=np.concatenate((obs['extra']['tcp_pose'][0].cpu().numpy(),
                              obs['agent']['qpos'][0,-2:].cpu().numpy())).astype(np.float32)
        truth=target(gt,robot[:3])
        np.testing.assert_allclose(truth,ref['target'],atol=1e-6,rtol=0)
        det=detector(*sensor_inputs(obs))
        xyz,kinds=det.xyz.cpu().numpy(),det.object_id.cpu().numpy()
        track=tracker.step(xyz,kinds,robot)
        labels=np.array([t.kind for t in tracker.tracks],dtype=np.int64)
        track_frames.append((track['positions'],labels,track['observed']))
        tcp.append(robot[:3])
        raw_rim=sum(kind==1 and (meta.get('contour_hole',False) or meta.get('method')=='box_opening')
                    for kind,meta in zip(kinds,det.metadata,strict=True))
        rows.append(dict(seed=seed,step=step,phase=phase,grasp=grasp,gt21=truth.tolist(),
                         counts=dict(peg=int((kinds==0).sum()),box=int((kinds==1).sum()),
                                     hole_tagged=int(raw_rim),tracked=len(labels),
                                     observed=int(track['observed'].sum()))))
        obs,_,term,trunc,_=env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            terminal='terminated' if bool(term[0]) else 'truncated'
            break
    if len(rows)!=len(reference):
        raise ValueError(f'Episode length differs for seed {seed}: {len(rows)} vs {len(reference)}')
    for name,model in models.items():
        values=predict_graphs(model,track_frames,np.asarray(tcp),32,device)
        if values.shape!=(len(rows),21) or not np.isfinite(values).all():
            raise ValueError(f'Invalid {name} prediction for seed {seed}')
        for row,pred in zip(rows,values,strict=True):
            row[name]=pred.tolist()
    return rows,terminal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    paths={k:Path(v) for k,v in json.loads(args.source.read_text()).items()}
    original,temporal=load_models(paths,device)
    trained=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    new=VertexEdgeGatedMPNN(**trained['config']['gnn']).to(device)
    new.load_state_dict(trained['model'],strict=True)
    new.eval().requires_grad_(False)
    seeds=list(range(args.start,args.stop))
    with np.load(args.reference,allow_pickle=False) as ref:
        episodes={seed:[dict(step=int(ref['step'][i]),phase=str(ref['phase'][i]),
                             grasp=bool(ref['grasp'][i]),target=ref['target'][i])
                        for i in np.flatnonzero(ref['seed']==seed)] for seed in seeds}
    for seed,rows in episodes.items():
        if not rows or [row['step'] for row in rows]!=list(range(len(rows))):
            raise ValueError(f'Reference has a missing trajectory step: {seed}')
    config=dict(seeds=seeds,camera=CAMERA,detector='TorchCornerDetector.main(cuda:0, plane_fit_method=ransac)',
                tracker='Transformer, 30mm matching',input='NumPy RGB-D',device=str(device),
                original_sha256=file_sha256(paths['gnn']),tracker_sha256=file_sha256(paths['corner']),
                retrained_sha256=file_sha256(args.checkpoint),reference_sha256=file_sha256(args.reference),
                script_sha256=file_sha256(__file__))
    args.output.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='graph',model='gt20_gnn_detected_tracking_test',job_type='evaluate',
                          config=config,seed=0,root=args.output/'experiments') as logger:
        env=make_peg_insertion_env(device='cpu',render_backend='cuda',obs_mode='rgb+depth',
            control_mode='pd_ee_delta_pose',max_episode_steps=200,
            sensor_configs={'hand_camera':dict(width=16,height=16)})
        detector=TorchCornerDetector.main(device=device,plane_fit_method='ransac')
        models=dict(original=original,retrained=new)
        started=monotonic()
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
                                   maniskill=version('mani-skill'),backend=str(env.unwrapped.backend)))
        try:
            terminals,counts={},{}
            frames=args.output/'frames.jsonl'
            with frames.open('w') as f:
                for index,seed in enumerate(seeds,1):
                    rows,terminal=episode(env,seed,detector,temporal,models,device,episodes[seed])
                    terminals[str(seed)]=terminal
                    counts[str(seed)]=len(rows)
                    for row in rows:
                        f.write(json.dumps(row,allow_nan=False)+'\n')
                    f.flush()
                    print(f'{index}/{len(seeds)} seed {seed}, {len(rows)} frames, {terminal}, {monotonic()-started:.1f}s',
                          flush=True)
            result=args.output/'summary.json'
            write(result,dict(config=config,run_id=logger.run_id,terminals=terminals,frames_by_seed=counts,
                              seconds=monotonic()-started))
            logger.link_output(result)
            logger.link_output(frames)
            logger.log({'frames':sum(counts.values()),'episodes':len(seeds)})
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__=='__main__':
    main()
