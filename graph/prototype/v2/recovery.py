"""Train-support distance diagnostics and honest correction-oracle feasibility."""
import argparse
import ast
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from ..common import seed_all,write_json
from ..data import load_dataset,split_indices
from .diagnostics import qvel_dataset
from .state import shared_features


def features(g,qvel):return torch.cat((shared_features(g,qvel,'goal'),g.gripper),-1)


def nearest(query,reference):
    return torch.cat([torch.cdist(rows,reference).min(1).values for rows in query.split(128)])


def distribution(paths,data,g,qvel):
    train=split_indices(data,'train');test=split_indices(data,'test')
    raw=features(g.select(train),qvel[train]);mean=raw.mean(0);std=raw.std(0,correction=0).clamp_min(.001)
    reference=(raw-mean)/std
    expert=nearest((features(g.select(test),qvel[test])-mean)/std,reference)
    threshold=float(torch.quantile(expert,.95));report={'definition':'Euclidean nearest TRAIN timestep in train-standardized 33D (qvel9 + geometric residual22 + finger qpos2); not likelihood or oracle action','expert_test':{'samples':len(expert),'median':float(expert.median()),'p95':threshold},'models':{}}
    for name,path in paths.items():
        r=json.loads(Path(path).read_text());queries=[];meta=[]
        for ep in r['settings']['full']['episodes']:
            row=torch.where(data['episode_id']==ep['episode_id'])[0][:1];base=g.select(row)
            for step in ep['trace'][::5]:
                s=step['before'];current=replace(base,ee_pose=torch.tensor([s['ee_pose']]),peg_pose=torch.tensor([s['peg_pose']]),hole_pose=torch.tensor([s['hole_pose']]),gripper=torch.tensor([s['gripper_state']]),qpos=torch.tensor([s['qpos']]))
                queries.append(features(current,torch.tensor([s['qvel']])));meta.append(dict(episode_id=ep['episode_id'],t=step['t']))
        distances=nearest((torch.cat(queries)-mean)/std,reference)
        report['models'][name]=dict(samples=len(distances),sampling_stride_steps=5,median=float(distances.median()),p95=float(torch.quantile(distances,.95)),fraction_above_expert_test_p95=float((distances>threshold).float().mean()),per_state=[dict(**m,distance=float(d)) for m,d in zip(meta,distances)])
    return report


def oracle_feasibility():
    import mani_skill
    path=Path(mani_skill.__file__).parent/'examples/motionplanning/panda/solutions/peg_insertion_side.py'
    source=path.read_text();tree=ast.parse(source)
    calls=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reset']
    result=dict(installed_solver_source=str(path),reset_calls=calls,mplib_installed=importlib.util.find_spec('mplib') is not None,
                control_mode_requirement='pd_joint_pos or pd_joint_pos_vel; not learner 7D EE action',arbitrary_state_oracle_verified=False,coverage=None,correction_success_rate=None,dagger_run=False)
    try:
        from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve
        result['import_succeeded']=True
    except Exception as e:
        result['import_succeeded']=False;result['import_error']=repr(e)
    result['conclusion']='Installed solve resets before planning. Its stock entry point does not label the current learner state. Additionally import fails without mplib on this host. No arbitrary-state correction was executed and no nearest-demonstration action was used as a label.'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--oracle',action='store_true');p.add_argument('--reports',type=Path,nargs='+')
    args=p.parse_args();seed_all(0,4)
    if args.oracle:result=oracle_feasibility()
    else:
        data,g=load_dataset('outputs/graph/dataset.pt');q=qvel_dataset(data)
        paths={path.stem:path for path in args.reports};result=distribution(paths,data,g,q)
    write_json(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
    if 'models' in result:print({k:{m:v for m,v in x.items() if m!='per_state'} for k,x in result['models'].items()})

if __name__=='__main__':main()
