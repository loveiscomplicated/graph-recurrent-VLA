"""Independent training adapter: original equal-budget loop, input factory gap only.

train_one is copied verbatim from the pinned09-C source except constructor
GeometryRegressor -> CanonicalRegressor. No original source is patched.
See readout_adapter.py for identity-on-canonical-input proof/tests.
"""
from copy import deepcopy
import time,resource
import numpy as np
import torch
from graph.integration_baseline.io import write_json,save_npz,sha256
from graph.integration_baseline.geometry09b_probe import absolute,csv_write
from graph.persistent_geometry.train import geometry_loss,validation_loss,parameter_hash,permutation_audit
from graph.persistent_geometry.encoder import geometry_summary
from .readout_adapter import CanonicalRegressor

def train_one(root,data,variant,encoder,config,logger):
    folder=root/'models'/(variant+'_'+encoder);folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    model=CanonicalRegressor(encoder).cuda()
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
    fresh=CanonicalRegressor(encoder).cuda().eval().requires_grad_(False)
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
