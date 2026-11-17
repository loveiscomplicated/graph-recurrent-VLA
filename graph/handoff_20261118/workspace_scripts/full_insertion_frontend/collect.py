"""Capture actual RANSAC observations and GT through the final insertion frame."""
import argparse
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch

import run_rollouts as base
from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state
from evaluate_full_gt import corners, target
from exp2_eval import q_conj, q_rot, is_success, HORIZON
from recurrent_capture import read_capture
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector

FIRST, VAL, TEST, END = 3100000, 3100120, 3100176, 3100208


class Detector:
    detect = base.HybridFrontend.detect

    def __init__(self):
        self.detector = TorchCornerDetector.main(device=torch.device('cuda:0'), plane_fit_method='ransac')


def capture(env, obs, det, step, phase, offset, cached):
    gt = {k: v[0].cpu().numpy() for k,v in vars(get_gt_state(env)).items()}
    tcp = gt['ee_pose'][:3].astype(np.float32)
    obj = np.asarray(target(gt,tcp),dtype=np.float32)
    matches = [r for r in corners(gt) if r['kind'] != 'hole_rim' or r['corner_id'] < 4]
    xyz = np.empty((20,3),dtype=np.float32)
    for row in matches:
        xyz[{'peg':0,'box_outer':8,'hole_rim':16}[row['kind']]+row['corner_id']] = row['xyz']
    assert len(matches) == 20
    cached = cached is not None and step < len(cached['target']) and np.allclose(
        obj, cached['target'][step], atol=2e-6, rtol=0)
    if cached:
        detected, observed = det.cache['detected'][step], det.cache['observed'][step]
    else:
        detected, observed = det.detect(obs, gt)
    assert np.all(np.linalg.norm(detected[observed]-xyz[observed],axis=-1)<=.00501)
    local = q_rot(q_conj(obj[13:17]),obj[7:10])
    depth = float((local[0]+obj[17])*1000)
    return dict(step=step, phase=phase, xyz=xyz, detected_xyz=detected, observed=observed,
                target=obj, tcp=tcp, depth_mm=depth, offset_m=np.asarray(offset).copy(), cached=cached)


@torch.inference_mode()
def episode(env, det, seed, cached):
    obs,_=env.reset(seed=seed)
    base.reset_state(obs,env)
    task=env.unwrapped
    det.actor_ids={0:int(task.peg.per_scene_id[0]),1:int(task.box.per_scene_id[0])}
    det.cache=cached
    policy=make_policy()
    policy.reset(env, seed)
    rows=[]
    motion=None
    offset=np.zeros(3)
    success=False
    should_move=seed%2==0
    for step in range(HORIZON):
        phase=policy.phase
        if should_move and motion is None and phase=='align' and bool(task.agent.is_grasping(task.peg)[0]):
            motion=EpisodeHoleMotion(env,schedule(seed,step/task.control_freq),policy)
        if motion is not None:
            if motion.advance(step,policy):
                obs=task.get_obs(task.get_info())
            offset=motion.frame_data()['offset_m']
        row=capture(env,obs,det,step,phase,offset,cached)
        rows.append(row)
        if not row['cached']:
            cached=None
        obs,_,term,trunc,info=env.step(policy.act(obs))
        if is_success(info):
            rows.append(capture(env,obs,det,step+1,'success',offset,None))
            success=True
            break
        if bool(term[0]) or bool(trunc[0]):
            break
    for row in rows:
        row['seed']=seed
    info=dict(seed=seed,success=success,frames=len(rows),cached=sum(r['cached'] for r in rows),
              moved_mm=float(max(np.linalg.norm(r['offset_m']) for r in rows)*1000),
              max_depth_mm=max(r['depth_mm'] for r in rows))
    if success:
        assert rows[-1]['phase']=='success' and info['max_depth_mm']>20
    return rows,info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('start',type=int)
    parser.add_argument('stop',type=int)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert FIRST<=args.start<args.stop<=END
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    capture_root=Path('/workspace/spatiotemporal_corner_tracker')
    reference=Path('/workspace/moving_box_gt20_training')
    det=Detector()
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(seed_range=[args.start,args.stop], camera=base.CAMERA, detector='CUDA RANSAC',
                gt_matching_m=.005,script_sha256=file_sha256(__file__),
                rule='even seeds move at grasp and align; odd seeds fixed', includes_terminal_rgbd=True)
    with ExperimentLogger(team='graph',model='full_insertion_detector_capture',job_type='collect',
                          seed=args.start,config=config,root=args.output/'experiments') as logger:
        started=monotonic()
        env=base.make_env()
        rows,records=[],[]
        try:
            for seed in range(args.start,args.stop):
                previous=read_capture(capture_root,reference,seed) if seed<3100144 else None
                frames,record=episode(env,det,seed,previous)
                if record['success']:
                    rows.extend(frames)
                records.append(record)
                print(record,f'{monotonic()-started:.1f}s',flush=True)
        finally:
            env.close()
        assert rows
        data=args.output/'data.npz'
        np.savez_compressed(data,**{key:np.asarray([row[key] for row in rows]) for key in rows[0]})
        result=args.output/'summary.json'
        write(result,dict(run_id=logger.run_id,config=config,episodes=records,frames=len(rows),
                          sha256=file_sha256(data)))
        for path in (data,result):
            logger.link_output(path)
        logger.log({'successes':sum(r['success'] for r in records),'frames':len(rows)})
    print('Completed full insertion capture',flush=True)


if __name__=='__main__':
    main()
