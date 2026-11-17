"""Compare GT-free versus GT-assisted IDs on the same RGB-D throughout insertion."""
import argparse
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import cv2
import numpy as np
import torch

from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state
from evaluate_full_gt import corners
import exp2_eval as task_api
from frontend import FullInsertionFrontend, TRACKER, GNN
from gtfree_frontend import GTFreeFrontend, detect_raw
from run_rollouts import make_env, reset_state, CAMERA
from utils.experiment_logger import ExperimentLogger, write


def pack(obs,env,free,oracle,detector,seed,step,phase):
    gt={k:v[0].cpu().numpy() for k,v in vars(get_gt_state(env)).items()}
    tcp=obs['extra']['tcp_pose'][0,:3].cpu().numpy()
    raw=detect_raw(detector,obs)
    # The GT-free path's inputs contain RGB-D, calibration and TCP only.
    data=obs['sensor_data']['base_camera']
    stripped={'sensor_data':{'base_camera':{k:data[k] for k in ('rgb','depth')}},
              'sensor_param':obs['sensor_param']}
    with patch.object(free,'detector',new=lambda *_:raw):
        free_out,*_=free.step(stripped,tcp)
    with patch.object(oracle,'detector',new=lambda *_:raw):
        matched,matched_mask=oracle.detect(obs,gt)
        with patch.object(oracle,'detect',new=lambda *_:(matched,matched_mask)):
            oracle_out,*_=oracle.step(obs,tcp,gt)
    task=task_api.gt_object21(env)
    local=task_api.q_rot(task_api.q_conj(task[13:17]),task[7:10])
    depth=(local[0]+task[17])*1000
    xyz=np.asarray([r['xyz'] for r in corners(gt) if r['kind']!='hole_rim' or r['corner_id']<4],
                   dtype=np.float32)
    return dict(seed=seed,step=step,phase=phase,depth_mm=depth,target=task,gt_xyz=xyz,
                free_object21=free_out,oracle_object21=oracle_out,
                free_xyz=free.last_coordinates,free_observed=free.last_observed,
                oracle_xyz=matched,oracle_observed=matched_mask)


@torch.inference_mode()
def episode(env,free,oracle,seed):
    obs,_=env.reset(seed=seed)
    reset_state(obs,env)
    task=env.unwrapped
    free.reset()
    oracle.reset({0:int(task.peg.per_scene_id[0]),1:int(task.box.per_scene_id[0])})
    detector=free.detector
    ref=make_policy()
    ref.reset(env,seed)
    rows=[]
    move=None
    success=False
    for step in range(250):
        phase=ref.phase
        if seed%2==0 and move is None and phase=='align' and bool(task.agent.is_grasping(task.peg)[0]):
            move=EpisodeHoleMotion(env,schedule(seed,step/task.control_freq),ref)
        if move is not None and move.advance(step,ref):
            obs=task.get_obs(task.get_info())
        rows.append(pack(obs,env,free,oracle,detector,seed,step,phase))
        obs,_,term,trunc,info=env.step(ref.act(obs))
        if task_api.is_success(info):
            rows.append(pack(obs,env,free,oracle,detector,seed,step+1,'success'))
            success=True
            break
        if bool(term[0]) or bool(trunc[0]):
            break
    return rows,success


def main():
    p=argparse.ArgumentParser()
    p.add_argument('start',type=int)
    p.add_argument('stop',type=int)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    device=torch.device('cuda:0')
    free=GTFreeFrontend(TRACKER,GNN,device)
    oracle=FullInsertionFrontend(TRACKER,GNN,device)
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(tracker_sha256=file_sha256(TRACKER),gnn_sha256=file_sha256(GNN),
                script_sha256=file_sha256(__file__),camera=CAMERA,seed_range=[args.start,args.stop])
    with ExperimentLogger(team='graph',model='gtfree_id_live_frontend',job_type='evaluate',
                          seed=args.start,config=config,root=args.output/'experiments') as log:
        start=monotonic()
        output,results=[],[]
        env=make_env()
        try:
            for seed in range(args.start,args.stop):
                rows,success=episode(env,free,oracle,seed)
                output.extend(rows)
                case=dict(seed=seed,success=success,frames=len(rows),deep=sum(r['depth_mm']>20 for r in rows))
                results.append(case)
                print(case,f'{monotonic()-start:.1f}s',flush=True)
        finally:
            env.close()
        path=args.output/'predictions.npz'
        np.savez_compressed(path,**{k:np.asarray([r[k] for r in output]) for k in output[0]})
        summary=args.output/'summary.json'
        write(summary,dict(run_id=log.run_id,config=config,sha256=file_sha256(path),episodes=results))
        log.link_output(path)
        log.link_output(summary)
        log.log({'frames':len(output),'deep':sum(x['deep'] for x in results)})


if __name__=='__main__':
    main()
