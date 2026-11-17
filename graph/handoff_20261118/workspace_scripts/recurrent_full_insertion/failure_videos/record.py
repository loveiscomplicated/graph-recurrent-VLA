"""Record selected fixed and moving failure replays with geometry overlaid."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_mamba import exp2, fixed, import_moving, eval_episode
from frontend import FullInsertionFrontend, GNN, TRACKER
from utils.experiment_logger import ExperimentLogger, write
from video import RolloutVideo

BASE = Path('/workspace/recurrent_full_insertion')


def geometry(obj):
    rel = exp2.q_rot(exp2.q_conj(obj[13:17]), obj[7:10]) * 1000
    depth = float(rel[0] + obj[17]*1000)
    lateral = float(np.linalg.norm(rel[1:]))
    in_opening = bool(np.all(np.abs(rel[1:])<=obj[20]*1000))
    return depth,lateral,in_opening


def stats(episode):
    frames=episode['trace']
    depths=[geometry(np.array(frame['object21_gt'])) for frame in frames]
    in_opening=[d for d,_,ok in depths if ok]
    return dict(success=episode['success'],steps=episode['steps'],
                max_depth_mm=max(in_opening,default=None),last_depth_mm=depths[-1][0],
                last_lateral_mm=depths[-1][1],steps_with_entry=sum(d>.1 for d in in_opening),
                target_depth_mm=frames[0]['object21_gt'][17]*1000-15,
                max_box_shift_mm=episode.get('box_max_shift_mm',0.))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('condition',choices=('fixed','moving'))
    p.add_argument('seed',type=int)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    source=next(p.parent for p in sorted((BASE/'eval').glob('shard_*/summary.json'))
                if a.seed in json.loads(p.read_text())['config']['seeds'])
    summary=json.loads((source/'summary.json').read_text())
    expected=next(r[a.condition] for line in (source/'rollouts.jsonl').read_text().splitlines()
                  if (r:=json.loads(line))['seed']==a.seed)
    config=summary['config']
    paths=[BASE/'train'/f'mamba_s{i}'/'last.pt' for i in range(5)]
    assert [file_sha256(p) for p in paths]==config['recurrent_sha256']
    assert file_sha256(TRACKER)==config['tracker_sha256'] and file_sha256(GNN)==config['gnn_sha256']
    moving=import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    assert file_sha256(moving.__file__)==config['moving_source_sha256']
    assert config['camera']==fixed.CAMERA and config['no_filter'] and not expected['success']
    ckpts=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    assert all(not ck['kin'] for ck in ckpts)
    frontend=FullInsertionFrontend(TRACKER,GNN,torch.device('cuda:0'))
    decoder=exp2.Ensemble([exp2.Decoder(ck,'step') for ck in ckpts])
    a.output.mkdir(parents=True,exist_ok=False)
    config=dict(config, seed_for_video=a.seed, condition=a.condition,
                source_run_id=summary['run_id'],script_sha256=file_sha256(__file__))
    with ExperimentLogger(team='recurrent',model='full_insertion_failure_video',job_type='evaluate',
                          seed=a.seed,config=config,root=a.output/'experiments') as logger:
        env=fixed.make_env()
        task=env.unwrapped
        video=RolloutVideo(a.output,a.seed,'hybrid',float(task.control_freq))
        actual_step=0
        origin=None
        original_step=frontend.step

        def record_step(obs,*args,**kwargs):
            nonlocal actual_step,origin
            hole=task.box_hole_pose.p[0].cpu().numpy()
            if origin is None:
                origin=hole.copy()
            shift=float(np.linalg.norm(hole-origin)*1000)
            depth,lateral,_=geometry(exp2.gt_object21(env))
            note=f'{a.condition.upper()}  d={depth:.1f}mm  lateral={lateral:.1f}mm'
            video.add(obs,actual_step,note,shift)
            actual_step+=1
            return original_step(obs,*args,**kwargs)

        try:
            with patch.object(frontend,'step',new=record_step):
                episode=eval_episode(env,frontend,decoder,a.seed,a.condition,moving)
            final=task.get_obs(task.get_info())
            final_shift=float(np.linalg.norm(task.box_hole_pose.p[0].cpu().numpy()-origin)*1000)
            path,nframes=video.finish(final,episode['steps'],episode['success'],episode['end_reason'],final_shift)
        finally:
            video.close()
            env.close()
        assert actual_step==episode['steps']
        verdict='success' if episode['success'] else 'fail'
        movie=a.output/f'{a.condition}_{a.seed}_{verdict}.mp4'
        path.rename(movie)
        if a.condition=='moving':
            assert episode['motion_plan']==expected['motion_plan']
        np.testing.assert_allclose(episode['trace'][0]['object21_gt'],expected['trace'][0]['object21_gt'],
                                   atol=1e-5,rtol=0)
        result=dict(seed=a.seed,condition=a.condition,run_id=logger.run_id,original=stats(expected),
                    replay=stats(episode),movie=movie.name,frames=nframes,episode=episode)
        out=a.output/'result.json'
        write(out,result)
        logger.link_output(out)
        logger.link_output(movie)
        logger.log({'success':int(episode['success']),'frames':nframes})
        print(a.condition,a.seed,stats(episode),flush=True)


if __name__=='__main__':
    main()
