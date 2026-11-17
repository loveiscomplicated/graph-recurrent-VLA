"""Extend previously failing trials to 500 actions with unchanged model weights."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_mamba import exp2, fixed, import_moving, eval_episode
from frontend import FullInsertionFrontend, TRACKER, GNN
from utils.experiment_logger import ExperimentLogger, write

BASE=Path('/workspace/recurrent_full_insertion')
BEFORE=250
HORIZON=500


def load():
    trials=[]
    for path in sorted((BASE/'eval').glob('shard_*/summary.json')):
        source=json.loads(path.read_text())
        for line in path.with_name('rollouts.jsonl').read_text().splitlines():
            row=json.loads(line)
            for condition in ('fixed','moving'):
                episode=row[condition]
                if not episode['success']:
                    trials.append((condition,row['seed'],episode,source))
    assert len(trials)==29
    focused=[t for t in trials if t[:2]==('moving',700806)]
    assert len(focused)==1
    return focused,[t for t in trials if t[:2]!=('moving',700806)]


def metrics(episode):
    trace=episode['trace']
    pose=np.array([f['actual_relative_mm'] for f in trace])
    truth=np.array([f['object21_gt'] for f in trace])
    depth=pose[:,0]+truth[:,17]*1000
    inside=(np.abs(pose[:,1:])<=truth[:,20,None]*1000).all(axis=1)
    first=np.arange(len(depth))<BEFORE
    earlier=depth[first&inside]
    later=depth[~first&inside]
    return dict(success=episode['success'],steps=episode['steps'],end_reason=episode['end_reason'],
                max_before_mm=float(earlier.max()) if len(earlier) else None,
                max_after_mm=float(later.max()) if len(later) else None,
                final_depth_mm=float(depth[-1]),final_lateral_mm=float(np.linalg.norm(pose[-1,1:])),
                target_depth_mm=float(truth[0,17]*1000-15),
                shift_mm=episode.get('box_max_shift_mm',0.))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('worker',type=int)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    focus,others=load()
    exp2.HORIZON=HORIZON
    trials=focus if args.worker<0 else others[args.worker::7]
    paths=[BASE/'train'/f'mamba_s{i}'/'last.pt' for i in range(5)]
    hashes=[file_sha256(p) for p in paths]
    tracker_sha,gnn_sha=file_sha256(TRACKER),file_sha256(GNN)
    ckpts=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    assert all(not ck['kin'] for ck in ckpts)
    moving=import_moving(Path('/workspace/closed_loop_moving_box_mamba'))
    front=FullInsertionFrontend(TRACKER,GNN,torch.device('cuda:0'))
    dec=exp2.Ensemble([exp2.Decoder(ck,'step') for ck in ckpts])
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(worker=args.worker,horizon=HORIZON,prior_horizon=BEFORE,models=hashes,
                tracker_sha256=tracker_sha,gnn_sha256=gnn_sha,script_sha256=file_sha256(__file__),
                cases=[[a,b] for a,b,*_ in trials])
    with ExperimentLogger(team='recurrent',model='full_insertion_extend_horizon',job_type='evaluate',
                          seed=trials[0][1],config=config,root=args.output/'experiments') as log:
        results=[]
        env=fixed.make_env()
        try:
            for condition,seed,old,source in trials:
                cfg=source['config']
                assert cfg['fixed_horizon']==BEFORE and cfg['no_filter'] and cfg['camera']==fixed.CAMERA
                assert cfg['recurrent_sha256']==hashes and cfg['tracker_sha256']==tracker_sha and cfg['gnn_sha256']==gnn_sha
                task=env.unwrapped
                contact=[]
                original_step=front.step

                def record(obs,*a,**kw):
                    force=task.scene.get_pairwise_contact_forces(task.peg,task.box)[0]
                    contact.append(dict(net_force_n=force.cpu().tolist(),
                                        tcp_xyz_m=task.agent.tcp.pose.p[0].cpu().tolist()))
                    return original_step(obs,*a,**kw)

                if args.worker<0:
                    with patch.object(front,'step',new=record):
                        new=eval_episode(env,front,dec,seed,condition,moving)
                    assert len(contact)==new['steps']
                    trace=args.output/'trace_700806.json'
                    write(trace,dict(episode=new,diagnostics=contact))
                    log.link_output(trace)
                else:
                    new=eval_episode(env,front,dec,seed,condition,moving)
                if condition=='moving':
                    assert old['motion_plan']==new['motion_plan']
                np.testing.assert_allclose(old['trace'][0]['object21_gt'],new['trace'][0]['object21_gt'],atol=5e-5,rtol=0)
                result=dict(condition=condition,seed=seed,original=metrics(old),extended=metrics(new),
                            source_run_id=source['run_id'])
                results.append(result)
                print(condition,seed,result['extended'],flush=True)
                line_path=args.output/'completed.jsonl'
                with line_path.open('a') as f:
                    f.write(json.dumps(result,allow_nan=False)+'\n')
        finally:
            env.close()
        out=args.output/'summary.json'
        write(out,dict(run_id=log.run_id,config=config,results=results))
        log.link_output(out)
        log.link_output(args.output/'completed.jsonl')
        log.log({'success':sum(r['extended']['success'] for r in results),'trials':len(results)})


if __name__=='__main__':
    main()
