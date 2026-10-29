"""Opt-in09-D runner:09-C probe loop unchanged; new reserved seeds/task identity."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn

from graph.integration_baseline.geometry09b_probe import (standardize,linear,mlp,absolute,delta,csv_write)
from graph.integration_baseline.io import save_npz,write_json,sha256
from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.audit import protection_snapshot,verify_protection
from graph.persistent_geometry.collect import inventory_sources


def read_features(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def decode_saved(path,features):
    with np.load(path,allow_pickle=False) as a:
        x=(features-a['feature_mean'])/a['feature_std']
        if str(a['kind'])=='ridge':
            y=x@a['weight']+a['bias']
        elif str(a['kind'])=='mlp':
            model=nn.Sequential(nn.Linear(x.shape[1],a['0.weight'].shape[0]),nn.ReLU(),nn.Linear(a['2.weight'].shape[1],3))
            model.load_state_dict({k:torch.from_numpy(a[k]) for k in ('0.weight','0.bias','2.weight','2.bias')})
            model.eval().requires_grad_(False)
            with torch.inference_mode():
                y=model(torch.from_numpy(x.astype(np.float32))).numpy().astype(float)
        else:
            raise ValueError('unknown decoder checkpoint')
        return y*a['target_std']+a['target_mean']


def probe_features(data,name,oid):
    offset=oid*15
    if name=='P0_robot76':return data['robot76']
    if name=='P4_centroid':return data['raw_geometry30'][:,offset:offset+3]
    if name=='P5_geometry_summary':return data['raw_geometry30'][:,offset:offset+15]
    if name=='P3_object_slots':return data['object_slots']
    if name=='P1_readout':return data['readout']
    if name=='P2_readout_robot76':return np.c_[data['readout'],data['robot76']]
    raise ValueError('unknown probe')


def metric_rows(data,y,pred,identity):
    absolute_rows=[];episode_rows=[];pair_rows=[]
    condition=data.get('condition',np.full(len(y),'baseline'))
    physical=data.get('physical_valid',np.ones(len(y),bool))
    visible=data.get('target_valid',np.ones((len(y),36),bool))[:,6+identity['object_id']*15]
    for split in ('train','validation','evaluation'):
        for group in ('baseline','counterfactual','ALL_VALID','GT_VISIBLE_ANY_CAMERA','GT_HIDDEN_BOTH_CAMERAS'):
            selected=(data['split']==split)&physical
            if group=='baseline':selected &= condition=='baseline'
            elif group=='counterfactual':selected &= (condition!='baseline')&(condition!='baseline_repeat')
            elif group=='GT_VISIBLE_ANY_CAMERA':selected &= visible&(condition!='baseline_repeat')
            elif group=='GT_HIDDEN_BOTH_CAMERAS':selected &= ~visible&(condition!='baseline_repeat')
            elif group=='ALL_VALID':selected &= condition!='baseline_repeat'
            absolute_rows.append(dict(**identity,split=split,condition_group=group,**absolute(y[selected],pred[selected])))
            for seed in sorted(set(data['seed'][selected])):
                rows=selected & (data['seed']==seed)
                episode_rows.append(dict(**identity,split=split,condition_group=group,seed=int(seed),**absolute(y[rows],pred[rows])))
    index={(int(s),int(t),str(c)):i for i,(s,t,c) in enumerate(zip(data['seed'],data['timestep'],condition))}
    for i,name in enumerate(condition):
        if name=='baseline':continue
        key=(int(data['seed'][i]),int(data['timestep'][i]),'baseline')
        if key not in index:raise ValueError('paired baseline missing')
        base=index[key]
        np.testing.assert_array_equal(data['robot76'][i],data['robot76'][base])
        pair_rows.append(dict(**identity,seed=int(data['seed'][i]),timestep=int(data['timestep'][i]),
            split=str(data['split'][i]),condition=str(name),physical_valid=bool(physical[i] and physical[base]),
            target_visible=bool(visible[i]),baseline_target_visible=bool(visible[base]),
            visibility_stratum='BOTH_VISIBLE' if visible[i] and visible[base] else 'BOTH_HIDDEN' if not visible[i] and not visible[base] else 'VISIBILITY_CHANGED',
            target_moved=bool((str(name).startswith('peg') and identity['object_id']==0)
                              or (str(name).startswith('box') and identity['object_id']==1)),
            **delta(y[i]-y[base],pred[i]-pred[base])))
        if identity['probe']=='P0_robot76':
            np.testing.assert_array_equal(pred[i],pred[base])
    return absolute_rows,episode_rows,pair_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.training/'model_manifest.json').read_text())
    metadata=json.loads((args.training/'metadata.json').read_text())
    if metadata['status']!='finished':raise ValueError('GNN selection must be finalized before probes')
    config=dict(task='09-D',stage='frozen_geometry_probes',source_training=str(args.training),seed=42,
        linear=dict(alphas=[0.,.01,.1,1.,10.,100.]),
        mlp=dict(hidden=64,lr=.001,weight_decay=[.0001,.01],max_epochs=2000,patience=150,min_delta=1e-8),
        normalization='train only; feature std floor1e-6, target std floor1mm; existing09-B implementation',
        fit_rows='same train/validation group rows as GNN; all input-missing strata retained, no per-variant GT support exclusion',
        targets='TCP-relative actor position3 per object; no grasp/hole/phase/action/reward/success targets',
        encoders_frozen=True,evaluation_used_for_selection=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protected=protection_snapshot(p for p in args.training.rglob('*') if p.is_file())
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09d_frozen_geometry_probes',
                          job_type='train',root=args.output) as logger:
        root=logger.directory;print('TASK09C_PROBE_DIRECTORY='+str(root),flush=True)
        logger.record(source_run_id=args.training.name)
        write_json(root/'protection_before.json',protected)
        absolute_rows=[];episodes=[];pairs=[];probes=[];shared={}
        for model in manifest['models']:
            variant,encoder=model['variant'],model['encoder']
            folder=args.training/model['directory']
            if sha256(folder/'selected.pt')!=model['checkpoint_sha256']:
                raise ValueError('frozen diagnostic GNN checkpoint changed')
            data=read_features(folder/'frozen_features.npz')
            train=data['train_fit_mask'];val=data['validation_selection_mask']
            for oid in (0,1):
                y=data['target6'][:,oid*3:oid*3+3].astype(float)
                ym=y[train].mean(0);ys=np.maximum(y[train].std(0),.001);yn=(y-ym)/ys
                # Unlearned controls preserve the partial-surface centroid bias.
                for control in ('Constant','RawCentroidDirect'):
                    control_model='Shared' if control=='Constant' else variant
                    key=(control_model,control,oid)
                    if key in shared:continue
                    shared[key]=True
                    prediction=(np.broadcast_to(ym,y.shape).copy() if control=='Constant' else
                                data['raw_geometry30'][:,oid*15:oid*15+3].astype(float))
                    target=root/'probes'/f'{control_model}_{control}_object{oid}'
                    target.mkdir(parents=True,exist_ok=False)
                    save_npz(target/'decoder.npz',kind=np.array(control),constant=ym)
                    save_npz(target/'predictions.npz',prediction=prediction,target=y,seed=data['seed'],
                        timestep=data['timestep'],split=data['split'],condition=data['condition'])
                    identity=dict(model=control_model,variant=variant,encoder='input_only',
                                  probe=control,decoder='unlearned',object_id=oid)
                    aa,ee,pp=metric_rows(data,y,prediction,identity)
                    absolute_rows.extend(aa);episodes.extend(ee);pairs.extend(pp)
                    probes.append(dict(**identity,directory=str(target.relative_to(root)),
                        source_model_directory=str(folder),decoder_sha256=sha256(target/'decoder.npz'),
                        feature_dimension=3,train_rows=int(train.sum()),validation_rows=int(val.sum())))
                for probe in ('P0_robot76','P4_centroid','P5_geometry_summary','P3_object_slots','P1_readout','P2_readout_robot76'):
                    baseline=probe in ('P0_robot76','P4_centroid','P5_geometry_summary')
                    source_model='Shared' if probe=='P0_robot76' else variant if baseline else variant+'_'+encoder
                    key=(source_model,probe,oid)
                    if key in shared:continue
                    shared[key]=True
                    x=probe_features(data,probe,oid).astype(float)
                    xx,xm,xs=standardize(x,train)
                    for decoder,fn in (('linear',linear),('mlp',mlp)):
                        pn,weights,selection=fn(xx,yn,train,val,config[decoder])
                        prediction=pn*ys+ym
                        name=f'{source_model}_{probe}_object{oid}_{decoder}'
                        target=root/'probes'/name;target.mkdir(parents=True,exist_ok=False)
                        save_npz(target/'decoder.npz',**weights,feature_mean=xm,feature_std=xs,target_mean=ym,target_std=ys)
                        restored=decode_saved(target/'decoder.npz',x)
                        np.testing.assert_allclose(restored,prediction,atol=1e-10,rtol=1e-10)
                        save_npz(target/'predictions.npz',prediction=prediction,target=y,seed=data['seed'],timestep=data['timestep'],
                            split=data['split'],condition=data.get('condition',np.full(len(y),'baseline')))
                        identity=dict(model=source_model,variant=variant,encoder='input_only' if baseline else encoder,
                                      probe=probe,decoder=decoder,object_id=oid)
                        aa,ee,pp=metric_rows(data,y,prediction,identity)
                        absolute_rows.extend(aa);episodes.extend(ee);pairs.extend(pp)
                        write_json(target/'selection.json',selection)
                        probes.append(dict(**identity,directory=str(target.relative_to(root)),source_model_directory=str(folder),
                            feature_dimension=x.shape[1],decoder_sha256=sha256(target/'decoder.npz'),
                            train_rows=int(train.sum()),validation_rows=int(val.sum())))
                    print('probe '+source_model+'/'+probe+f'/object{oid}',flush=True)
            write_json(root/'probe_manifest.json',dict(config=config,probes=probes))
            csv_write(root/'absolute_metrics.csv',absolute_rows);csv_write(root/'per_episode_metrics.csv',episodes)
            csv_write(root/'paired_delta_metrics.csv',pairs)
        inventory_sources(root)
        audit=verify_protection(protected);write_json(root/'frozen_gnn_protection_audit.json',audit)
        if audit['status']!='PASS':raise ValueError('frozen GNN/input weights changed during probes')
        write_json(root/'frozen_selection.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
            independent_evaluation_seeds=[73141,73142],final_data_used=False,
            model_manifest_sha256=sha256(args.training/'model_manifest.json'),
            probe_manifest_sha256=sha256(root/'probe_manifest.json'),
            checkpoints={str(args.training/m['directory']/'selected.pt'):m['checkpoint_sha256'] for m in manifest['models']},
            decoders={str(root/p['directory']/'decoder.npz'):p['decoder_sha256'] for p in probes}))
        for name,kind in (('probe_manifest.json','metrics'),('absolute_metrics.csv','reports'),
                          ('per_episode_metrics.csv','reports'),('paired_delta_metrics.csv','reports'),
                          ('frozen_gnn_protection_audit.json','metrics')):
            logger.link_output(root/name,kind=kind)


if __name__=='__main__':main()
