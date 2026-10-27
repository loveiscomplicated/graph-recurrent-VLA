"""Equal-budget geometry representation training; no BC/action/recurrent loss."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from graph.integration_baseline.graph import Graph, pack_graphs
from graph.integration_baseline.io import save_npz, write_json, sha256
from graph.integration_baseline.geometry09b_probe import csv_write, absolute
from utils.experiment_logger import ExperimentLogger

from .encoder import GeometryRegressor, geometry_summary
from .graph import FEATURE_DIMENSION
from .audit import protection_snapshot, verify_protection
from .collect import inventory_sources


def load_graph(path):
    with np.load(path,allow_pickle=False) as a:
        g=Graph(**{k:torch.from_numpy(a[k].copy()) for k in
            ('node_input','node_type','edge_index','edge_attr','edge_type')},packet={},timings_ms={})
    if g.node_input.shape[1]!=FEATURE_DIMENSION or not torch.isfinite(g.node_input).all():
        raise ValueError('common graph input schema/finite contract')
    return g


class Dataset:
    def __init__(self,geometry,variant,frames,*,target_statistics=None):
        self.frames=frames
        self.graphs=[]; self.targets=[]; self.valid=[]; self.robot=[]
        self.identities=[]
        for frame in frames:
            folder=geometry/frame['folder']
            path=folder/(variant+'_graph.npz')
            self.graphs.append(load_graph(path))
            with np.load(folder/'evaluation_targets.npz',allow_pickle=False) as a:
                if int(a['timestep'])!=frame['timestep']:
                    raise ValueError('target/graph timestep join')
                self.targets.append(a['target36']); self.valid.append(a['target_valid']); self.robot.append(a['robot76'])
            self.identities.append(dict(seed=frame['seed'],timestep=frame['timestep'],split=frame['split'],
                condition=frame.get('condition','baseline'),
                graph_sha256=sha256(path),target_sha256=sha256(folder/'evaluation_targets.npz')))
        self.targets=np.asarray(self.targets)
        self.valid=np.asarray(self.valid)
        self.robot=np.asarray(self.robot)
        self.train=np.array([f['split']=='train' and f['timestep']%5==0 and f.get('physical_valid',True)
                             and f.get('condition','baseline')!='baseline_repeat' for f in frames])
        self.validation=np.array([f['split']=='validation' and f['timestep']%5==0 and f.get('physical_valid',True)
                                  and f.get('condition','baseline')!='baseline_repeat' for f in frames])
        if target_statistics is None and (not self.train.any() or not self.validation.any()):
            raise ValueError('nonempty independent train/validation episodes required')
        mean=[]; std=[]
        if target_statistics is not None:
            mean,std=target_statistics
            if np.shape(mean)!=(36,) or np.shape(std)!=(36,) or not np.all(np.asarray(std)>0):
                raise ValueError('frozen training target statistics contract')
        else:
            for dim in range(self.targets.shape[1]):
                rows=self.train & self.valid[:,dim]
                if not rows.any():
                    raise ValueError('diagnostic train target has no support')
                mean.append(self.targets[rows,dim].mean())
                std.append(max(self.targets[rows,dim].std(),.001))
        self.mean,self.std=np.array(mean),np.array(std)
        self.normalized=torch.from_numpy(np.where(self.valid,(self.targets-self.mean)/self.std,0).astype(np.float32))
        self.mask=torch.from_numpy(self.valid)

    def batch(self,indices,device='cuda'):
        return (pack_graphs([self.graphs[int(i)] for i in indices]).to(device),
                self.normalized[indices].to(device),self.mask[indices].to(device))


def geometry_loss(prediction,target,valid):
    # Equal weight for actor relative position and observable surface geometry;
    # missing current visible surfaces contribute no invented shape target.
    position=((prediction[:,:6]-target[:,:6])**2).mean()
    shape=((prediction[:,6:]-target[:,6:])**2*valid[:,6:]).sum()/valid[:,6:].sum().clamp_min(1)
    return .5*(position+shape)


def parameter_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def permutation_audit(encoder,graph,device='cuda'):
    """Actual loaded graph, remapped endpoints and frozen encoder on the device."""
    count=len(graph.node_input)
    permutation=torch.randperm(count,generator=torch.Generator().manual_seed(43))
    inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(count)
    permuted=Graph(graph.node_input[permutation],graph.node_type[permutation],
        inverse[graph.edge_index],graph.edge_attr,graph.edge_type,{}, {})
    original,_=encoder(pack_graphs([graph]).to(device))
    changed,_=encoder(pack_graphs([permuted]).to(device))
    torch.testing.assert_close(original,changed,atol=2e-6,rtol=2e-5)
    return dict(status='PASS',device=device,nodes=count,
        maximum_absolute_error=float((original-changed).abs().max()),atol=2e-6,rtol=2e-5)


@torch.inference_mode()
def validation_loss(model,data,batch_size=16):
    model.eval(); total=0.;count=0
    for rows in np.array_split(np.flatnonzero(data.validation),max(1,int(np.ceil(data.validation.sum()/batch_size)))):
        graph,target,valid=data.batch(rows)
        total+=float(geometry_loss(model(graph),target,valid))*len(rows)
        count+=len(rows)
    return total/count


def train_one(root,data,variant,encoder,config,logger):
    folder=root/'models'/(variant+'_'+encoder);folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    model=GeometryRegressor(encoder).cuda()
    initial=parameter_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_rows=np.flatnonzero(data.train)
    rng=np.random.default_rng(config['seed'])
    curve=[]; best_score=float('inf'); best=None; best_step=None
    start=time.perf_counter()
    for step in range(1,config['updates']+1):
        indices=rng.choice(train_rows,config['batch'],replace=len(train_rows)<config['batch'])
        graph,target,valid=data.batch(indices)
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=geometry_loss(model(graph),target,valid)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite diagnostic loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1 or step%config['validation_every']==0 or step==config['updates']:
            score=validation_loss(model,data,config['batch'])
            if score<best_score:
                best_score=score;best=deepcopy(model.state_dict());best_step=step
            row=dict(step=step,train_loss=float(loss.detach()),validation_loss=score)
            curve.append(row)
            logger.log({variant+'/'+encoder:row},step=step)
            print(f'geometry train {variant}/{encoder} {step}/{config["updates"]} val={score:.6g}',flush=True)
    model.load_state_dict(best);model.eval().requires_grad_(False)
    checkpoint=dict(schema='task09c_geometry_training_v1',variant=variant,encoder=encoder,
        config=config,state_dict={k:v.cpu() for k,v in model.state_dict().items()},
        target_mean=data.mean,target_std=data.std,best_step=best_step,best_validation_loss=best_score,
        initial_parameter_hash=initial,selected_parameter_hash=parameter_hash(model),
        parameter_count=sum(p.numel() for p in model.parameters()),
        encoder_parameter_count=sum(p.numel() for p in model.encoder.parameters()),
        readout_dimension=model.encoder.output_dimension,
        train_identity=[data.identities[i] for i in np.flatnonzero(data.train)],
        validation_identity=[data.identities[i] for i in np.flatnonzero(data.validation)],
        actor_position_dimensions=6,surface_summary_dimensions=30,
        training_seconds=time.perf_counter()-start,
        cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        no_policy_or_gru=True)
    torch.save(checkpoint,folder/'selected.pt')
    restored=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    fresh=GeometryRegressor(encoder).cuda().eval().requires_grad_(False)
    fresh.load_state_dict(restored['state_dict'],strict=True)
    write_json(folder/'permutation_audit.json',permutation_audit(fresh.encoder,data.graphs[0]))
    if parameter_hash(fresh)!=checkpoint['selected_parameter_hash']:
        raise ValueError('GNN checkpoint readback identity')
    history_hash=parameter_hash(fresh)
    outputs={k:[] for k in ('readout','object_slots','raw_geometry30','prediction36')}
    with torch.inference_mode():
        for begin in range(0,len(data.graphs),config['batch']):
            indices=np.arange(begin,min(begin+config['batch'],len(data.graphs)))
            graph,_,_=data.batch(indices)
            z,presence=fresh.encoder(graph)
            z_repeat,_=fresh.encoder(graph)
            torch.testing.assert_close(z,z_repeat,rtol=0,atol=0)
            outputs['readout'].append(z.cpu().numpy())
            outputs['object_slots'].append(z[:,192:320].cpu().numpy())
            outputs['raw_geometry30'].append(geometry_summary(graph).cpu().numpy())
            outputs['prediction36'].append(fresh.decoder(z).cpu().numpy()*data.std+data.mean)
    if parameter_hash(fresh)!=history_hash:
        raise ValueError('encoder mutated during frozen extraction')
    arrays={k:np.concatenate(v) for k,v in outputs.items()}
    arrays.update(target6=data.targets[:,:6],target36=data.targets,target_valid=data.valid,robot76=data.robot,
        seed=np.array([f['seed'] for f in data.frames]),timestep=np.array([f['timestep'] for f in data.frames]),
        split=np.array([f['split'] for f in data.frames]),train_fit_mask=data.train,
        condition=np.array([f.get('condition','baseline') for f in data.frames]),
        physical_valid=np.array([f.get('physical_valid',True) for f in data.frames],bool),
        validation_selection_mask=data.validation)
    save_npz(folder/'frozen_features.npz',**arrays)
    write_json(folder/'training_curve.json',curve)
    write_json(folder/'model_identity.json',{k:v for k,v in checkpoint.items() if k not in
        ('state_dict','target_mean','target_std','train_identity','validation_identity')})
    write_json(folder/'sample_manifest.json',data.identities)
    metrics=[]
    for split in ('train','validation','evaluation'):
        selected=arrays['split']==split
        for oid in (0,1):
            metric=absolute(arrays['target6'][selected,oid*3:oid*3+3],arrays['prediction36'][selected,oid*3:oid*3+3])
            metrics.append(dict(variant=variant,encoder=encoder,split=split,object_id=oid,**metric))
    csv_write(folder/'geometry_regressor_metrics.csv',metrics)
    logger.link_output(folder/'selected.pt',kind='checkpoint')
    logger.link_output(folder/'frozen_features.npz',kind='trajectory')
    return dict(variant=variant,encoder=encoder,directory=str(folder.relative_to(root)),
        checkpoint_sha256=sha256(folder/'selected.pt'),features_sha256=sha256(folder/'frozen_features.npz'),
        parameter_count=checkpoint['parameter_count'],readout_dimension=checkpoint['readout_dimension'],
        best_step=best_step,best_validation_loss=best_score,metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.geometry/'geometry_manifest.json').read_text())
    metadata=json.loads((args.geometry/'metadata.json').read_text())
    if metadata['status']!='finished':
        raise ValueError('geometry input evaluation must be finalized first')
    config=dict(task='09-C',seed=42,updates=args.updates,batch=16,lr=.001,weight_decay=1e-5,
        model=dict(d=64,L=2,node_type_embedding=4,edge_type_embedding=4,diagnostic_decoder_hidden=64,output=36),
        optimizer_betas=[.9,.999],optimizer_eps=1e-8,amsgrad=False,scheduler=None,
        dtype='float32',AMP=False,TF32=False,deterministic_algorithms=True,
        validation_every=50,optimizer='AdamW',gradient_clip=1.,target_std_floor_m=.001,
        train_sample_interval=5,loss='0.5 actor-position6 normalized MSE + 0.5 visible-surface30 masked normalized MSE',
        geometry=str(args.geometry),input_schema='task09c_persistent_surface_v1',
        selection='lowest independent validation loss at fixed checkpoints; first tie; fixed total updates',
        evaluation_used_for_selection=False,policy_training=False,gru_executed=False,
        variants=['G0','G1','G2','G3','G4'],encoders=['E0','E1','E2'],
        oracle_encoders=['O1_E0','O2_E0','O3_E0'],status_ablation='G4_status_off_E0')
    if args.smoke:
        config['variants']=['G0'];config['encoders']=['E0'];config['oracle_encoders']=[];config['status_ablation']=None
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    paths=[p for p in args.geometry.rglob('*') if p.is_file()]
    protected=protection_snapshot(paths)
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09c_geometry_encoders',
                          job_type='train',root=args.output) as logger:
        root=logger.directory;print('TASK09C_TRAIN_DIRECTORY='+str(root),flush=True)
        logger.record(source_run_id=args.geometry.name)
        write_json(root/'protection_before.json',protected)
        models=[]
        candidates={variant:config['encoders'] for variant in config['variants']}
        candidates.update({name.split('_')[0]:['E0'] for name in config['oracle_encoders']})
        if config['status_ablation']:
            candidates['G4_status_off']=['E0']
        for variant,encoders in candidates.items():
            data=Dataset(args.geometry,variant,manifest['frames'])
            for encoder in encoders:
                models.append(train_one(root,data,variant,encoder,config,logger))
                write_json(root/'model_manifest.json',dict(config=config,models=models))
            del data
        inventory_sources(root)
        audit=verify_protection(protected);write_json(root/'protection_audit.json',audit)
        if audit['status']!='PASS':
            raise ValueError('geometry input artifacts modified')
        logger.link_output(root/'model_manifest.json',kind='metrics')
        logger.link_output(root/'protection_audit.json',kind='metrics')


if __name__=='__main__':
    main()
