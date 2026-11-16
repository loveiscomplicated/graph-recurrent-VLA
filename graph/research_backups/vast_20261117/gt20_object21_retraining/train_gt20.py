"""Train the existing GNN and compare its test predictions to the frozen checkpoint."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN, rotate_local_x
from graph.pose_geometry import hole_frame_vector
from training.detected_corner_gnn import graph_batch, run_epoch
from utils.experiment_logger import ExperimentLogger, write

from collect_gt20 import FIRST_SEED, SPLIT_SIZES, END


def read_collection(root):
    arrays, sources, terminals = [], [], {}
    for start in range(FIRST_SEED, END, 44):
        folder = root/f'data_{start}_{start+44}'
        path = folder/'dataset.npz'
        config = json.loads((folder/'summary.json').read_text())
        metadata = json.loads((folder/'experiments'/config['run_id']/'metadata.json').read_text())
        if (metadata['status'] != 'finished' or metadata['git_commit'][:7] != '2b6c40e'
                or file_sha256(path) != config['dataset_sha256']):
            raise ValueError(f'Collection identity mismatch: {folder}')
        with np.load(path, allow_pickle=False) as data:
            batch = {key:data[key] for key in data.files}
        if np.unique(batch['seed']).tolist() != list(range(start,start+44)):
            raise ValueError(f'Seed range mismatch: {folder}')
        if batch['xyz'].shape != (len(batch['step']),20,3) or batch['target'].shape != (len(batch['step']),21):
            raise ValueError(f'Data shape mismatch: {folder}')
        arrays.append(batch)
        sources.append(config)
        terminals.update(config['terminals'])
    full = {k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    if len(set(zip(full['seed'],full['step'],strict=True))) != len(full['seed']):
        raise ValueError('Duplicate frame')
    expected_kinds = np.array([0]*8+[1]*12)
    if not np.all(full['kinds']==expected_kinds):
        raise ValueError('Unexpected GT20 node kinds')
    return full, sources, terminals


def write_checkpoint(model, epoch, logger, config, output):
    temp = output.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),epoch=epoch,run_id=logger.run_id,config=config),temp)
    temp.replace(output)


@torch.inference_mode()
def predict(model, tensors, indices, batch_size):
    parts = []
    for ix in indices.split(batch_size):
        graph, flags = graph_batch(*(tensors[k][ix] for k in ('xyz','kinds','valid','observed','tcp')))
        result = model(graph,flags)['object21']
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite held-out prediction')
        parts.append(result.cpu())
    return torch.cat(parts)


def error_metrics(pred, target):
    delta = pred-target
    tip_error = delta[:,7:10]
    gt_hole = target[:,13:17]
    norms = lambda x: torch.linalg.vector_norm(x,dim=1)*1000
    angle = lambda a,b: torch.rad2deg(torch.acos((rotate_local_x(a)*rotate_local_x(b)).sum(1).clamp(-1,1)))
    return dict(peg_center_mm=norms(delta[:,:3]),hole_center_mm=norms(delta[:,10:13]),
                peg_axis_deg=angle(pred[:,3:7],target[:,3:7]),hole_axis_deg=angle(pred[:,13:17],gt_hole),
                lateral_mm=norms(hole_frame_vector(tip_error,gt_hole)[:,1:]),
                relative_3d_mm=norms(tip_error),peg_tip_mm=norms(tip_error+delta[:,10:13]))


def describe(errors,mask):
    result={}
    for key,values in errors.items():
        points=values[mask].numpy()
        if len(points):
            p50,p90=np.quantile(points,[.5,.9])
            result[key]=dict(mean=float(points.mean()),p50=float(p50),p90=float(p90))
    return result


def groups(test, terminals):
    phase, step, seed, grasp=(test[k] for k in ('phase','step','seed','grasp'))
    late=np.zeros(len(seed),bool)
    first=np.zeros(len(seed),bool)
    held=grasp & (phase=='align')
    for val in np.unique(seed):
        has_grasp=np.flatnonzero((seed==val)&grasp)
        if len(has_grasp):
            first[has_grasp[0]]=True
        if terminals[str(val)]=='insert_entry':
            late[np.flatnonzero((seed==val)&held)[-5:]]=True
    return dict(all=np.ones(len(seed),bool),initial=step==0,first_grasp=first,held_align=held,last_5_held_align=late)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args()
    source=Path(json.loads(args.source.read_text())['gnn'])
    digest=file_sha256(source)
    if digest!=source.parent.name:
        raise ValueError('Original checkpoint digest mismatch')
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    previous=checkpoint['config']
    full, sources, terminals=read_collection(args.root)
    splits=FIRST_SEED+np.cumsum((0,*SPLIT_SIZES))
    split_rows={name:np.flatnonzero((full['seed']>=a)&(full['seed']<b)) for name,a,b in
                zip(('train','validation','test'),splits[:-1],splits[1:],strict=True)}
    settings=dict(previous['training'],epochs=args.epochs,log_every_batches=100000)
    config=dict(gnn=previous['gnn'],loss=previous['loss'],training=settings,seed=0,
                device='cuda:0',split_seed_ranges=splits.tolist(),split_frames={k:len(v) for k,v in split_rows.items()},
                original_checkpoint_sha256=digest, collection=sources, script_sha256=file_sha256(__file__))
    torch.set_num_threads(settings['cpu_threads'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    rng=np.random.default_rng(config['seed'])
    device=torch.device(config['device'])
    args.output.mkdir(parents=True,exist_ok=False)
    tensors={k:torch.as_tensor(full[k],device=device)
             for k in ('xyz','kinds','tcp','target')}
    tensors['valid']=torch.ones_like(tensors['kinds'],dtype=torch.bool)
    tensors['observed']=tensors['valid']
    model=VertexEdgeGatedMPNN(**config['gnn']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    history=[]
    best=(float('inf'),0)
    started=monotonic()
    with ExperimentLogger(team='graph',model='gt20_object21_retraining',config=config,seed=config['seed'],
                          root=args.output/'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
        val_rows=torch.as_tensor(split_rows['validation'],device=device)
        for epoch in range(1,args.epochs+1):
            before=monotonic()
            indices=torch.as_tensor(rng.permutation(split_rows['train']),device=device)
            train=run_epoch(model,tensors,indices,config,optimizer,epoch,logger,args.output)
            val=run_epoch(model,tensors,val_rows,config,None,epoch,logger,args.output)
            elapsed=monotonic()-before
            if val['loss']<best[0]:
                best=val['loss'],epoch
                write_checkpoint(model,epoch,logger,config,args.output/'best.pt')
            print(f'epoch {epoch}/{args.epochs} train={train["loss"]:.6f} val={val["loss"]:.6f} best={best[1]} epoch_s={elapsed:.1f}',
                  flush=True)
            history.append(dict(epoch=epoch,train=train,validation=val,seconds=elapsed))
            write(args.output/'history.json',history)
        test_rows=split_rows['test']
        test_indices=torch.as_tensor(test_rows,device=device)
        test={k:v[test_rows] for k,v in full.items()}
        best_state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(best_state['model'],strict=True)
        model.eval()
        original=VertexEdgeGatedMPNN(**config['gnn']).to(device)
        original.load_state_dict(checkpoint['model'],strict=True)
        original.eval()
        models=dict(original=original,retrained=model)
        preds={k:predict(m,tensors,test_indices,settings['batch_size']) for k,m in models.items()}
        gt=torch.from_numpy(test['target'])
        errors={k:error_metrics(v,gt) for k,v in preds.items()}
        metrics={}
        for group,mask in groups(test,terminals).items():
            metrics[group]=dict(frames=int(mask.sum()),episodes=len(set(test['seed'][mask])),
                                **{k:describe(v,mask) for k,v in errors.items()})
        test_path=args.output/'test_predictions.npz'
        np.savez_compressed(test_path,**{k:test[k] for k in ('target','seed','step','phase','grasp')},
                            **{k:v.numpy() for k,v in preds.items()})
        summary=dict(run_id=logger.run_id,best_epoch=best[1],validation_loss=best[0],test=metrics,
                     epochs=args.epochs,train_and_test_seconds=monotonic()-started,config=config,
                     test_terminals={str(seed):terminals[str(seed)] for seed in np.unique(test['seed'])})
        result=args.output/'summary.json'
        write(result,summary)
        logger.log(metrics,prefix='eval/test')
        for name in ('best.pt','summary.json','history.json','test_predictions.npz'):
            logger.link_output(args.output/name)
        print('Completed',logger.run_id,flush=True)


if __name__=='__main__':
    main()
