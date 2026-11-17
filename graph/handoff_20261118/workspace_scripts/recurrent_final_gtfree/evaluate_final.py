"""Test the retrained Mamba ensemble in matched fixed/moving closed-loop rollouts."""
import argparse
import importlib.util
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch

import exp2_eval as exp2
import run_rollouts as fixed
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write
from final_frontend import FinalFrontend


class NoVideo:
    def add(self, *_):
        pass

    def finish(self, *_):
        return Path('unrecorded'), 0


def import_moving(folder):
    spec = importlib.util.spec_from_file_location('moving_box_baseline', folder/'run_rollouts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_episode(env, front, dec, seed, condition, moving_base):
    if condition == 'fixed':
        _, result = fixed.episode(env, front, dec, seed, 'hybrid')
    else:
        obs, _ = env.reset(seed=seed)
        reset = fixed.reset_state(obs, env)
        dec.reset()
        task = env.unwrapped
        front.reset({0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])})
        freq = float(task.control_freq)
        plan = moving_base.schedule(seed, moving_base.MOTION_START_STEP / freq)
        plan['stop_expert_insert'] = False
        motion = moving_base.EpisodeHoleMotion(env, plan, None)
        _, result = moving_base.moving_loop(env, front, dec, seed, 'hybrid', obs, reset, task, [], False,
                                            'horizon', float(task.peg.pose.p[0,2]), plan, motion, NoVideo(), freq)
        result['movie'] = None
    assert all(row['object21'] == row['filtered'] for row in result['trace'])
    return result


def main():
    p=argparse.ArgumentParser()
    for key in ('prior','train','tracker','gnn','moving-source','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--stop',type=int,required=True)
    a=p.parse_args()
    assert 0 <= a.start < a.stop <= 32
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    device=torch.device('cuda:0')
    moving = import_moving(a.moving_source)
    seeds, _, _ = fixed.source_seeds(a.prior)
    paths = [a.train / f'mamba_s{i}' / 'last.pt' for i in range(5)]
    ckpts = [torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    old_paths = [a.prior / f'recurrent/exp2/checkpoints/runs_r6_mix_filter/mamba_step_s{i}/last.pt' for i in range(5)]
    if a.warm_start is not None:
        old_paths = [a.warm_start / f'mamba_s{i}' / 'last.pt' for i in range(5)]
    for i, ck in enumerate(ckpts):
        assert not ck['kin'] and ck['obs_dim']==44 and ck['seed']==i
        assert ck['training_config']['prior_sha256'] == file_sha256(old_paths[i])
        assert ck['stats'] == torch.load(old_paths[i],map_location='cpu',weights_only=True)['stats']
    decoder = exp2.Ensemble([exp2.Decoder(ck,'step') for ck in ckpts])
    front = FinalFrontend(a.tracker,a.gnn,device)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=dict(seeds=seeds[a.start:a.stop], camera=fixed.CAMERA, no_filter=True,
             fixed_horizon=exp2.HORIZON,moving_start= moving.MOTION_START_STEP,
             recurrent_sha256=[file_sha256(p) for p in paths],
             old_recurrent_sha256=[file_sha256(p) for p in old_paths],
             tracker_sha256=file_sha256(a.tracker),gnn_sha256=file_sha256(a.gnn),
             frontend_code_sha256=front.code_hashes,
             script_sha256=file_sha256(__file__),fixed_source_sha256=file_sha256(fixed.__file__),
             moving_source_sha256=file_sha256(moving.__file__))
    started=monotonic()
    with ExperimentLogger(team='recurrent',model='mamba_final_gtfree_eval',job_type='evaluate',
                          seed=fixed.SAMPLE_SEED,config=cfg,root=a.output/'experiments') as log:
        env=fixed.make_env()
        log.record(runtime=dict(gpu=torch.cuda.get_device_name(),backend=str(env.unwrapped.backend)))
        trace=a.output/'rollouts.jsonl'
        count=dict(fixed=0,moving=0)
        try:
            with trace.open('w') as stream:
                for i,seed in enumerate(cfg['seeds'],a.start):
                    order=('fixed','moving') if i%2==0 else ('moving','fixed')
                    results={}
                    for condition in order:
                        results[condition]=eval_episode(env,front,decoder,seed,condition,moving)
                        count[condition]+=int(results[condition]['success'])
                    np.testing.assert_allclose(results['fixed']['trace'][0]['object21_gt'],
                                               results['moving']['trace'][0]['object21_gt'],rtol=0,atol=1e-6)
                    stream.write(json.dumps(dict(seed=seed,**results),allow_nan=False)+'\n')
                    stream.flush()
                    print(seed,{k:(r['success'],r['steps']) for k,r in results.items()},
                          f'{monotonic()-started:.1f}s',flush=True)
        finally:
            env.close()
        report=a.output/'summary.json'
        write(report,dict(run_id=log.run_id,config=cfg,success=count,seconds=monotonic()-started))
        log.log(count)
        log.link_output(report)
        log.link_output(trace)
    print('Completed retrained Mamba evaluation',flush=True)


if __name__=='__main__':
    main()
