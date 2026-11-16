"""Collect moving Box counterparts to the existing fixed Box GT20 dataset."""
import argparse
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from collect_gt20 import FIRST_SEED, END
from evaluate_full_gt import collect
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger, write


def schedule(seed, seconds):
    rng = np.random.default_rng(seed+7351)
    angle = rng.uniform(0,2*np.pi)
    length = rng.uniform(.06,.12)
    return dict(mode='continuous', start_seconds=seconds, end_seconds=seconds+2,
                duration_seconds=float(rng.uniform(.8,1.4)),minimum_distance_m=.08,
                translation_m=(length*np.array([np.cos(angle),np.sin(angle),0.])).tolist(),
                stop_expert_insert=True)


def episode(env,seed):
    obs,_=env.reset(seed=seed)
    policy=make_policy()
    policy.reset(env,seed)
    task=env.unwrapped
    initial=task.box_hole_pose.p[0].cpu().numpy().copy()
    rows, motion, plan = [], None, None
    offset=np.zeros(3)
    for step in range(200):
        phase=policy.phase
        if phase=='insert':
            return rows,'insert_entry',plan
        grasp=bool(task.agent.is_grasping(task.peg)[0])
        if motion is None and phase=='align' and grasp:
            plan=schedule(seed,step/task.control_freq)
            motion=EpisodeHoleMotion(env,plan,policy)
        last=offset
        if motion is not None:
            if motion.advance(step,policy):
                obs=task.get_obs(task.get_info())
            offset=motion.frame_data()['offset_m']
        target,inputs=collect(env,obs)
        xyz,kinds,tcp=inputs['full_front20']
        np.testing.assert_allclose(target[10:13]+tcp-initial,offset,atol=2e-6,rtol=0)
        rows.append(dict(xyz=xyz,kinds=kinds,tcp=tcp,target=target,seed=seed,step=step,phase=phase,grasp=grasp,
                         offset_m=offset.copy(),moving_now=np.linalg.norm(offset-last)>1e-6))
        obs,_,term,trunc,_=env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            return rows,'terminated' if bool(term[0]) else 'truncated',plan
    return rows,'step_limit',plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not FIRST_SEED<=args.start<args.stop<=END:
        parser.error('Invalid seed range')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(seeds=list(range(args.start,args.stop)), input='simulator GT20',
                motion=dict(mode='continuous',start='first held align frame',requested_distance_m=[.06,.12],
                            duration_s=[.8,1.4],proximity_guard_m=.08,direction='uniform horizontal'),
                script_sha256=file_sha256(__file__))
    started=monotonic()
    with ExperimentLogger(team='graph',model='moving_box_gt20_collection',job_type='collect',
                          config=config,seed=FIRST_SEED,root=args.output/'experiments') as logger:
        env=make_peg_insertion_env(device='cpu',render_backend='cuda',obs_mode='state_dict',
                                   control_mode='pd_ee_delta_pose',max_episode_steps=200)
        logger.record(runtime=str(env.unwrapped.backend))
        rows,terminals,motions=[],{},{}
        try:
            for i,seed in enumerate(config['seeds'],1):
                part,status,plan=episode(env,seed)
                if not part:
                    raise ValueError(f'Empty seed {seed}')
                rows.extend(part)
                terminals[str(seed)]=status
                motions[str(seed)]=dict(plan=plan,achieved_m=float(np.linalg.norm(part[-1]['offset_m'])),
                                        motion_frames=int(sum(r['moving_now'] for r in part)))
                if i==1 or i%10==0 or i==len(config['seeds']):
                    print(f'{i}/{len(config["seeds"])} episodes {len(rows)} frames {monotonic()-started:.1f}s',flush=True)
            arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
            for k in ('xyz','target','tcp','offset_m'):
                arrays[k]=arrays[k].astype(np.float32)
                if not np.isfinite(arrays[k]).all():
                    raise ValueError(f'Nonfinite {k}')
            path=args.output/'dataset.npz'
            np.savez_compressed(path,**arrays)
            summary=args.output/'summary.json'
            write(summary,dict(run_id=logger.run_id,**config,frames=len(rows),terminals=terminals,
                               motions=motions,dataset_sha256=file_sha256(path),seconds=monotonic()-started))
            logger.log({'episodes':len(terminals),'frames':len(rows),'motion_frames':int(arrays['moving_now'].sum())})
            for result in (path,summary):
                logger.link_output(result)
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__=='__main__':
    main()
