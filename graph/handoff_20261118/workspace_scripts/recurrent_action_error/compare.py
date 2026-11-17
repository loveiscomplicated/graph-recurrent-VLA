"""Measure held-out Expert action error with Graph and GT observation histories."""
import argparse
from itertools import chain
import json
from pathlib import Path

import numpy as np
import torch

import exp2_eval as running
import exp2_train as original
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write

ROOT = Path('/workspace/recurrent_new_frontend')
VALIDATION = 3100120
MODES = ('graph','gt')


def load():
    rows, sources = [], []
    for folder in sorted(ROOT.glob('shard_*')):
        if not folder.is_dir():
            continue
        meta=json.loads((folder/'summary.json').read_text())
        path=folder/'demos.npz'
        assert file_sha256(path)==meta['sha256']
        sources.append(meta['sha256'])
        with np.load(path,allow_pickle=False) as d:
            off=d['episode_offsets']
            for seed,a,b in zip(d['reset_seed'],off[:-1],off[1:],strict=True):
                if seed<VALIDATION:
                    continue
                gt,graph=d['state36_gt'][a:b],d['state36_transformer'][a:b]
                np.testing.assert_array_equal(gt[:,:15],graph[:,:15])
                moved=(np.linalg.norm(d['box_offset'][a:b],axis=-1)>.001).any()
                rows.append(dict(seed=int(seed), moved=bool(moved), gt=gt, graph=graph,
                                 action=d['expert_action'][a:b],phase=d['phase'][a:b]))
    assert len(rows)==21
    assert sum(row['moved'] for row in rows)==10
    return rows,sources


def rot(q,v):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    uv=2*np.cross(q[:,1:],v)
    return v+q[:,:1]*uv+np.cross(q[:,1:],uv)


def to_hole(v,state):
    q=state[:,28:32].copy()
    q[:,1:]*=-1
    return rot(q,v)


@torch.inference_mode()
def predict(rows, checkpoints):
    device=torch.device('cuda:0')
    models=[]
    for ck in checkpoints:
        model=original.build(ck['kind'],ck['obs_dim'],7).to(device).eval()
        model.load_state_dict(ck['state_dict'])
        models.append(model)
    sequence_check=True
    for row in rows:
        axes=row['gt']
        truth=row['action']
        for mode in MODES:
            state=row[mode]
            features=original.features(state,hole_frame=True,r6=True)
            actions=[]
            for model,ck in zip(models,checkpoints,strict=True):
                stats={k:np.asarray(v,dtype=np.float32) for k,v in ck['stats'].items()}
                norm=((features-stats['obs_mean'])/stats['obs_std']*stats['obs_keep']).astype(np.float32)
                out,_=model(torch.from_numpy(norm)[None].to(device))
                values=out[0].cpu().numpy()
                move=values[:,:6]*stats['act_std']+stats['act_mean']
                actions.append(np.c_[move,np.where(values[:,6]>0,1.,-1.)])
            ensemble=np.mean(actions,axis=0)
            ensemble[:,6]=np.where(ensemble[:,6]>0,1.,-1.)
            if sequence_check:
                policy=running.Ensemble([running.Decoder(ck,'step') for ck in checkpoints])
                actual=np.stack([policy.act(obs) for obs in state])
                np.testing.assert_allclose(ensemble,actual,atol=2e-4,rtol=0)
            err=ensemble-truth
            # Root axes and world axes coincide for the installed task. Controller translation is 0.1 m per unit.
            trans=(np.clip(ensemble[:,:3],-1,1)-np.clip(truth[:,:3],-1,1))*100
            local=to_hole(trans,axes)
            row[f'{mode}_action']=ensemble
            row[f'{mode}_translation_mm']=np.linalg.norm(local,axis=-1)
            row[f'{mode}_axial_mm']=np.abs(local[:,0])
            row[f'{mode}_lateral_mm']=np.linalg.norm(local[:,1:],axis=-1)
            row[f'{mode}_rotation_command']=np.linalg.norm(err[:,3:6],axis=-1)
            row[f'{mode}_gripper_wrong']=(ensemble[:,6]>0)!=(truth[:,6]>0)
        sequence_check=False
        row['depth_mm']=(to_hole(axes[:,22:25],axes)[:,0]+axes[:,32])*1000


def aggregate(rows):
    groups={}
    for condition in ('all','fixed','moving'):
        selected=[r for r in rows if condition=='all' or r['moved']==(condition=='moving')]
        for phase in ('all','align','insert','deep'):
            masks=[np.ones(len(r['action']),bool) if phase=='all' else
                   r['depth_mm']>20 if phase=='deep' else r['phase']==phase for r in selected]
            result=dict(episodes=len(selected),frames=int(sum(m.sum() for m in masks)))
            for mode in MODES:
                result[mode]={}
                for key in ('translation_mm','axial_mm','lateral_mm','rotation_command','gripper_wrong'):
                    x=np.concatenate([r[f'{mode}_{key}'][m] for r,m in zip(selected,masks,strict=True)]).astype(float)
                    result[mode][key]=dict(mean=float(x.mean()),p50=float(np.median(x)),p90=float(np.percentile(x,90)))
            groups[f'{condition}_{phase}']=result
    return groups


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    rows,sources=load()
    files=[ROOT/'train'/f'mamba_s{i}'/'last.pt' for i in range(5)]
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in files]
    assert all(not ck['kin'] and ck['chunk']==1 and ck['hole_frame'] and ck['rot6'] for ck in states)
    config=dict(models=[file_sha256(p) for p in files],data=sources,script_sha256=file_sha256(__file__),
                evaluation='validation teacher-forced Expert state histories, filter OFF')
    with ExperimentLogger(team='recurrent',model='mamba_graph_gt_action_error',job_type='evaluate',
                          seed=0,config=config,root=args.output/'experiments') as log:
        predict(rows,states)
        scores=aggregate(rows)
        summary=args.output/'summary.json'
        write(summary,dict(run_id=log.run_id,config=config,groups=scores))
        out=args.output/'predictions.npz'
        keys=('action','gt','graph','phase','depth_mm','graph_action','gt_action')
        np.savez_compressed(out,**{k:np.concatenate([r[k] for r in rows]) for k in keys},
                            seed=np.asarray(list(chain.from_iterable([r['seed']]*len(r['action']) for r in rows))))
        log.link_output(summary)
        log.link_output(out)
    print('Computed recurrent action errors',flush=True)


if __name__=='__main__':
    main()
