"""Capture raw detector candidates, with GT stored separately for scoring."""
import argparse
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import cv2
import numpy as np
import torch

import live_compare
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector

RAGGED=('raw_xyz','raw_family')


class Detector:
    def __init__(self):
        self.detector=TorchCornerDetector.main(device=torch.device('cuda:0'),plane_fit_method='ransac')

    def reset(self):
        pass


class EvaluationReference:
    def reset(self,actor_ids):
        pass


def capture_frame(obs,env,free,oracle,detector,seed,step,phase):
    raw=live_compare.detect_raw(detector,obs)
    xyz=raw.xyz.cpu().numpy()
    kind=raw.object_id.cpu().numpy()
    hints=np.array([bool(meta.get('contour_hole',False) or meta.get('method')=='box_opening')
                    for meta in raw.metadata],dtype=bool)
    family=np.where(kind==0,0,np.where(hints,2,1))
    gt={k:v[0].cpu().numpy() for k,v in vars(live_compare.get_gt_state(env)).items()}
    tcp=obs['extra']['tcp_pose'][0,:3].cpu().numpy()
    truth=live_compare.task_api.gt_object21(env)
    hole=live_compare.task_api.q_rot(live_compare.task_api.q_conj(truth[13:17]),truth[7:10])
    depth=float((hole[0]+truth[17])*1000)
    all_xyz=np.asarray([r['xyz'] for r in live_compare.corners(gt)
                        if r['kind']!='hole_rim' or r['corner_id']<4],dtype=np.float32)
    return dict(seed=seed,step=step,phase=phase,tcp=tcp,xyz=all_xyz,depth_mm=depth,
                target=truth,raw_count=len(xyz),raw_xyz=xyz,raw_family=family)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('start',type=int)
    p.add_argument('stop',type=int)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=dict(script_sha256=file_sha256(__file__),camera=live_compare.CAMERA,
             seed_range=[args.start,args.stop],detector='CUDA RANSAC',
             fixed='odd seed',moving='even seed')
    with ExperimentLogger(team='graph',model='gtfree_frontend_raw_training',job_type='collect',
                          seed=args.start,config=cfg,root=args.output/'experiments') as logger:
        env=live_compare.make_env()
        free,oracle=Detector(),EvaluationReference()
        rows,episodes=[],[]
        start=monotonic()
        try:
            with patch.object(live_compare,'pack',new=capture_frame):
                for seed in range(args.start,args.stop):
                    data,success=live_compare.episode(env,free,oracle,seed)
                    record=dict(seed=seed,success=bool(success),frames=len(data),
                                deep=sum(row['depth_mm']>20 for row in data))
                    rows.extend(data)
                    episodes.append(record)
                    print(record,f'{monotonic()-start:.1f}s',flush=True)
        finally:
            env.close()
        path=args.output/'raw.npz'
        arrays={key:(np.concatenate if key in RAGGED else np.asarray)([row[key] for row in rows])
                for key in rows[0]}
        np.savez_compressed(path,**arrays)
        summary=args.output/'summary.json'
        write(summary,dict(run_id=logger.run_id,config=cfg,episodes=episodes,sha256=file_sha256(path)))
        logger.link_output(path)
        logger.link_output(summary)
        logger.log({'frames':len(rows),'successes':sum(r['success'] for r in episodes)})
    print('Completed raw insertion capture',flush=True)


if __name__=='__main__':
    main()
