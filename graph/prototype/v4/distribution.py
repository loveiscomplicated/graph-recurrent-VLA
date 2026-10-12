"""Descriptive expert-vs-failed-learner distributions, no policy inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..common import write_json
from ..data import load_dataset, split_indices
from ..v2.diagnostics import qvel_dataset
from ..v2.goals import goal_geometry


def summary(array, expert):
    x=np.asarray(array,dtype=float);reference=np.asarray(expert,dtype=float)
    if not len(x):return {'samples':0}
    lo,hi=np.percentile(reference,[1,99],axis=0)
    sd=np.std(reference,axis=0).clip(.001)
    return dict(samples=len(x),mean=x.mean(0).tolist(),std=x.std(0).tolist(),
        expert_standardized_mean_shift=((x.mean(0)-reference.mean(0))/sd).tolist(),
        outside_expert_1_99_percentile_fraction=((x<lo)|(x>hi)).mean(0).tolist(),
        note='Descriptive marginal statistics; dependent rollout timesteps, not a hypothesis test.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v4'));args=p.parse_args()
    data,g=load_dataset('outputs/graph/dataset.pt');ids=split_indices(data,'train');q=qvel_dataset(data);goal=goal_geometry(g.select(ids))
    expert={'qvel':q[ids].numpy(),'gripper_state':g.gripper[ids].numpy(),'head_in_hole_m':goal['head_in_hole'].numpy(),'applied_action':data['actions'][ids].numpy()}
    result={'expert_reference':'TRAIN episodes only, all expert timesteps','features':{k:dict(samples=len(v),mean=v.mean(0).tolist(),std=v.std(0).tolist()) for k,v in expert.items()},'conditions':{}}
    for path in sorted((args.root/'evaluation').glob('*_heldout.json')):
        report=json.loads(path.read_text())
        for stage,setting in report['settings'].items():
            failed=[r for r in setting['episodes'] if not r['success']]
            traces=[t for r in failed for t in r['trace']]
            result['conditions'][f"{report['model']}_{report['history']}_{stage}"]=dict(failed_episodes=len(failed),
                metrics={k:summary([t[k] if k=='applied_action' else t['before'][k] for t in traces],v) for k,v in expert.items()})
    write_json(args.root/'comparison'/'distribution.json',result)


if __name__=='__main__':main()
