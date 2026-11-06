"""Equal-budget BC ablations with train-only common-state normalization."""
from utils.experiment_logger import ExperimentLogger
import argparse
import os
from pathlib import Path
from time import perf_counter
import torch
from ..common import config,DEFAULT_CONFIG,seed_all,versions,write_json
from ..data import load_dataset,split_indices,sha256
from ..graph_builder import GraphBuilder
from ..models import parameter_count
from ..normalization import FeatureNormalizer
from ..metrics import action_metrics
from .diagnostics import qvel_dataset
from .state import StateNormalizer
from .models import CommonPolicy,bc_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['flat','center','surface'],required=True)
    p.add_argument('--state',choices=['none','proprio','goal'],default='goal')
    p.add_argument('--gripper',choices=['regression','binary'],default='regression')
    p.add_argument('--dataset',type=Path,default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--config',type=Path,default=Path(__file__).with_name("config.toml"))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-batches',type=int)
    args=p.parse_args()
    if args.epochs<1 or (args.max_batches is not None and args.max_batches<1):p.error('positive training budget required')
    if (args.output/'best.pt').exists():p.error('checkpoint already exists; choose a fresh output directory')
    cfg=config(args.config);cfg.update(epochs=args.epochs,seed=args.seed);cfg.setdefault('head_width',128)
    with ExperimentLogger(team='graph', config=dict(parameters=cfg, command=vars(args)), seed=cfg['seed'], model=getattr(args, 'model', getattr(args, 'variant', None))) as experiment:
        experiment.record(dataset=str(Path(args.dataset).resolve()))
        seed_all(cfg['seed'],cfg['threads'])
        data,g=load_dataset(args.dataset);qvel=qvel_dataset(data)
        for k in ('peg_points','hole_points','sampling_seed','control_mode'):
            if cfg[k]!=data['manifest']['config'][k]:raise ValueError(f'dataset configuration mismatch {k}')
        train,val=split_indices(data,'train'),split_indices(data,'validation')
        norm=FeatureNormalizer.fit(g.select(train));sn=StateNormalizer.fit(g.select(train),qvel[train],args.state)
        model=CommonPolicy(args.model,cfg,args.state,args.gripper)
        experiment.record(model_class=type(model).__module__ + '.' + type(model).__qualname__)
        builder=GraphBuilder(cfg['surface_k'],cfg['interaction_k'],center=args.model=='center')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
        generator=torch.Generator().manual_seed(cfg['seed'])
        args.output.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');steps=0;start=perf_counter()
        dataset_hash=sha256(args.dataset)
        experiment.record(dataset_sha256=dataset_hash, environment=data['manifest']['config'])
        for epoch in range(1,cfg['epochs']+1):
            model.train();predictions=[];targets=[];losses=[];es=perf_counter()
            order=train[torch.randperm(len(train),generator=generator)]
            for bi,ids in enumerate(order.split(cfg['batch_size'])):
                if args.max_batches is not None and bi>=args.max_batches:break
                inputs=norm.apply(args.model,g.select(ids),builder);shared=sn.apply(g.select(ids),qvel[ids],args.state);target=data['actions'][ids]
                optimizer.zero_grad(set_to_none=True);raw=model.raw_action(inputs,shared)
                loss=bc_loss(raw,target,args.gripper)
                if not torch.isfinite(loss):raise FloatingPointError('BC loss')
                loss.backward();optimizer.step();steps+=1
                pred=raw.detach() if args.gripper=='regression' else torch.cat((raw[:,:6].detach(),torch.where(raw[:,6:].detach()>0,1.,-1.)),-1)
                predictions.append(pred);targets.append(target);losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                vp=torch.cat([model(norm.apply(args.model,g.select(ids),builder),sn.apply(g.select(ids),qvel[ids],args.state)) for ids in val.split(cfg['batch_size'])])
                vm=action_metrics(vp,data['actions'][val]);tm=action_metrics(torch.cat(predictions),torch.cat(targets))
            row=dict(epoch=epoch,train_mse=tm['action_mse'],val_mse=vm['action_mse'],train_metrics=tm,validation_metrics=vm,optimization_loss=sum(losses)/len(losses),seconds=perf_counter()-es,learning_rate=optimizer.param_groups[0]['lr'])
            history.append(row);scheduler.step()
            if row['val_mse']<best:
                best=row['val_mse']
                torch.save(dict(schema_version=2,model_name=args.model,config=cfg,state_mode=args.state,gripper_mode=args.gripper,model_state=model.state_dict(),normalizer=norm.as_dict(),state_normalizer=sn.as_dict(),epoch=epoch,validation_mse=best,dataset_sha256=dataset_hash,splits=data['manifest']['splits'],parameters=parameter_count(model),versions=versions(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict()),args.output/'best.pt')
                experiment.link_output(args.output / 'best.pt')
            write_json(args.output/'history.json',history)
            experiment.log_epoch(row)
            print(f"{args.model} {args.state}/{args.gripper} epoch {epoch}: train={row['train_mse']:.6f} val={best:.6f} {row['seconds']:.1f}s",flush=True)
        write_json(args.output/'training.json',dict(model=args.model,state_mode=args.state,gripper_mode=args.gripper,config=cfg,versions=versions(),parameters=parameter_count(model),training_seconds=perf_counter()-start,optimizer_steps=steps,dataset_sha256=dataset_hash,splits=data['manifest']['splits'],best_validation_mse=best,smoke_max_batches=args.max_batches,history=history))
        os.environ.setdefault('MPLCONFIGDIR',str(Path('outputs/graph/v2/.mplconfig').resolve()))
        from ..visualization import plot_curves
        plot_curves(history,args.output/'loss.png')

if __name__=='__main__':main()
