"""Render archived final-policy actions and verify GT geometry against each recorded frame."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_final import exp2, fixed, import_moving
from utils.experiment_logger import ExperimentLogger, write
from video import RolloutVideo

ROOT = Path('/workspace/recurrent_final_gtfree')
OUT = ROOT/'case_videos'


def find(seed, condition):
    for path in sorted((ROOT/'finetuned').glob('shard_*/rollouts.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['seed'] == seed:
                    return row[condition], path
    raise ValueError(seed)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('index',type=int)
    args = p.parse_args()
    case = json.loads((OUT/'selection.json').read_text())[args.index]
    seed, condition = case['seed'], case['condition']
    original, source = find(seed, condition)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    moving = import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    folder = OUT/f"{condition}_{case['group']}_{seed}"
    folder.mkdir()
    cfg = dict(case=case, replay='recorded final-policy Action7',
               original_rollouts_sha256=file_sha256(source), script_sha256=file_sha256(__file__),
               source_summary=json.loads(source.with_name('summary.json').read_text()))
    with ExperimentLogger(team='integration',model='final_policy_representative_video',
                          job_type='evaluate',seed=seed,config=cfg,root=folder/'experiments') as log:
        env = fixed.make_env()
        video = RolloutVideo(folder,seed,'hybrid',float(env.unwrapped.control_freq))
        drift = 0.
        motion_diff = 0.
        n = 0
        success = False
        reason = 'horizon'
        try:
            obs,_ = env.reset(seed=seed)
            fixed.reset_state(obs,env)
            task = env.unwrapped
            motion = moving.EpisodeHoleMotion(env,original['motion_plan'],None) if condition=='moving' else None
            for step, archived in enumerate(original['trace']):
                assert archived['step']==step
                if motion is not None and motion.advance(step,None):
                    obs=task.get_obs(task.get_info())
                target = exp2.gt_object21(env)
                delta = target-np.array(archived['object21_gt'])
                drift=max(drift,float(np.max(np.abs(delta))))
                np.testing.assert_allclose(target,archived['object21_gt'],atol=1e-6,rtol=0)
                moved = float(np.linalg.norm(motion.frame_data()['offset_m'])*1000) if motion else 0.
                if motion:
                    motion_diff=max(motion_diff,float(np.linalg.norm(motion.frame_data()['offset_m']*1000
                                               -np.array(archived['box_offset_mm']))))
                actual=np.asarray(archived['actual_relative_mm'])
                depth=actual[0]+target[17]*1000
                note=f"{condition} d={depth:.1f}mm err={archived['raw_lateral_mm']:.1f}mm"
                video.add(obs,step,note,moved)
                action=torch.tensor([archived['action']],device=task.device,dtype=torch.float32)
                obs,_,term,trunc,info=env.step(action)
                n+=1
                success=bool(exp2.is_success(info))
                if success or bool(term[0]) or bool(trunc[0]):
                    reason='success' if success else ('terminated' if bool(term[0]) else 'truncated')
                    break
            assert n==original['steps'] and success==original['success'] and reason==original['end_reason']
            path,frames=video.finish(obs,n,success,reason,moved)
        finally:
            video.close()
            env.close()
        movie=folder/f"{condition}_{case['group']}_{seed}.mp4"
        path.replace(movie)
        report=folder/'summary.json'
        result=dict(run_id=log.run_id,case=case,steps=n,success=success,end_reason=reason,
                    video=movie.name,frames=frames,max_gt_component_drift=drift,
                    box_motion_drift_mm=motion_diff,movie_sha256=file_sha256(movie),
                    source_sha256=cfg['original_rollouts_sha256'])
        write(report,result)
        log.link_output(report)
        log.link_output(movie)
        log.log({'frames':frames,'replayed_steps':n,'success':int(success)})
    print(args.index,condition,case['group'],seed,n,drift,flush=True)


if __name__=='__main__':
    main()
