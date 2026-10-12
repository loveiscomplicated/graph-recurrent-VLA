"""Equal-budget BC ablations with train-only common-state normalization."""
import argparse
import os
import shutil
from pathlib import Path
from time import perf_counter
import torch
from ..common import config,DEFAULT_CONFIG,seed_all,versions,write_json
from ..data import load_dataset,split_indices,sha256
from ..graph_builder import GraphBuilder
from ..models import parameter_count
from ..normalization import FeatureNormalizer
from ..metrics import action_metrics
from ..v2.diagnostics import qvel_dataset
from ..v2.state import StateNormalizer
from ..v2.models import bc_loss
from .history_models import HistoryPolicy
from .history_dataset import HistoryDataset
from .devices import cpu_tree,synchronize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['flat','surface'],required=True)
    p.add_argument('--history',choices=['current','action','state','both'])
    p.add_argument('--length',type=int)
    
    p.add_argument('--dataset',type=Path,default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--config',type=Path,default=Path(__file__).with_name("config.toml"))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-batches',type=int)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--resume',type=Path,help='resume an epoch-boundary checkpoint into a fresh output directory')
    p.add_argument('--stop-after-epoch',type=int,help='partial-run diagnostic; does not change scheduler epoch budget')
    p.add_argument('--mps-memory-fraction',type=float,default=.4)
    args=p.parse_args()
    if args.epochs<1 or (args.max_batches is not None and args.max_batches<1):p.error('positive training budget required')
    if (args.output/'best.pt').exists():p.error('checkpoint already exists; choose a fresh output directory')
    args.state='goal';args.gripper='regression'
    cfg=config(args.config);cfg.update(epochs=args.epochs,seed=args.seed);cfg.setdefault('head_width',128)
    history_cfg=cfg.get('history',{})
    if args.length is None:args.length=history_cfg.get('length',4)
    if args.history is None:
        flags=(history_cfg.get('use_state_history',True),history_cfg.get('use_action_history',True))
        args.history={(False,False):'current',(False,True):'action',(True,False):'state',(True,True):'both'}[flags]
    if args.length<0:p.error('history length must be nonnegative')
    cfg['history']=dict(length=args.length,use_state_history=args.history in ('state','both'),use_action_history=args.history in ('action','both'))
    seed_all(cfg['seed'],cfg['threads'])
    if args.device=='mps' and not torch.backends.mps.is_available():p.error('MPS unavailable')
    if args.device=='mps':torch.use_deterministic_algorithms(True,warn_only=True)
    if args.device=='mps':torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    device=torch.device(args.device)
    data,g=load_dataset(args.dataset);qvel=qvel_dataset(data)
    for k in ('peg_points','hole_points','sampling_seed','control_mode'):
        if cfg[k]!=data['manifest']['config'][k]:raise ValueError(f'dataset configuration mismatch {k}')
    train,val=split_indices(data,'train'),split_indices(data,'validation')
    norm=FeatureNormalizer.fit(g.select(train));sn=StateNormalizer.fit(g.select(train),qvel[train],args.state)
    model=HistoryPolicy(args.model,cfg,args.history,args.length).to(device)
    windows=HistoryDataset(data,args.length)
    builder=GraphBuilder(cfg['surface_k'],cfg['interaction_k'],center=args.model=='center')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    args.output.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');steps=0;start=perf_counter()
    dataset_hash=sha256(args.dataset)
    start_epoch=0;resume_info=None
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=True)
        if ck['dataset_sha256']!=dataset_hash or ck['splits']!=data['manifest']['splits']:raise ValueError('resume dataset/split mismatch')
        if (ck['model_name'],ck['history_mode'],ck['history_length'])!=(args.model,args.history,args.length):raise ValueError('resume architecture mismatch')
        for key in ('epochs','seed','batch_size','learning_rate','weight_decay','head_width','layers','hidden_dim','flat_width','control_mode','surface_k','interaction_k'):
            if ck['config'][key]!=cfg[key]:raise ValueError(f'resume condition mismatch: {key}')
        model.load_state_dict(ck['model_state']);optimizer.load_state_dict(ck['optimizer_state']);scheduler.load_state_dict(ck['scheduler_state'])
        norm=FeatureNormalizer(**ck['normalizer']);sn=StateNormalizer(**ck['state_normalizer'])
        start_epoch=ck['epoch'];best=ck['validation_mse']
        old_history=__import__('json').loads((args.resume.parent/'history.json').read_text())
        history=[row for row in old_history if row['epoch']<=start_epoch]
        if len(history)!=start_epoch:raise ValueError('resume needs complete prefix history')
        steps=ck.get('optimizer_steps',start_epoch*((len(train)+cfg['batch_size']-1)//cfg['batch_size']))
        if 'generator_state' in ck:generator.set_state(ck['generator_state'])
        else:
            for _ in range(start_epoch):torch.randperm(len(train),generator=generator)
        if 'torch_rng_state' in ck:torch.set_rng_state(ck['torch_rng_state'])
        best_source=args.resume.parent/'best.pt' if ck.get('checkpoint_kind')=='last' or args.resume.name=='last.pt' else args.resume
        if not best_source.exists():raise ValueError('resume last.pt requires its associated best.pt')
        shutil.copy2(best_source,args.output/'best.pt')
        resume_info=dict(source=str(args.resume),source_sha256=sha256(args.resume),epoch=start_epoch,previous_device=ck.get('training_device','cpu'),new_device=args.device,determinism='warn_only on MPS: index_add lacks strict deterministic implementation' if args.device=='mps' else 'strict',exact_current_batch=False,shuffle_state='checkpoint or reconstructed from seed/per-epoch randperm',note='CPU/MPS numerical results need not be bit identical')
        write_json(args.output/'resume.json',resume_info)
        print(f'Resuming {args.model}/{args.history} after epoch {start_epoch} on {device}',flush=True)
    for epoch in range(start_epoch+1,cfg['epochs']+1):
        model.train();predictions=[];targets=[];losses=[];es=perf_counter()
        order=train[torch.randperm(len(train),generator=generator)]
        for bi,ids in enumerate(order.split(cfg['batch_size'])):
            if args.max_batches is not None and bi>=args.max_batches:break
            target=data['actions'][ids].to(device)
            optimizer.zero_grad(set_to_none=True);raw=windows.forward(model,ids,g,qvel,norm,sn,builder)
            loss=bc_loss(raw,target,args.gripper)
            if not torch.isfinite(loss):raise FloatingPointError('BC loss')
            loss.backward();optimizer.step();steps+=1
            pred=raw.detach() if args.gripper=='regression' else torch.cat((raw[:,:6].detach(),torch.where(raw[:,6:].detach()>0,1.,-1.)),-1)
            predictions.append(pred.cpu());targets.append(target.cpu());losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            vp=torch.cat([windows.forward(model,ids,g,qvel,norm,sn,builder).cpu() for ids in val.split(cfg['batch_size'])])
            vm=action_metrics(vp,data['actions'][val]);tm=action_metrics(torch.cat(predictions),torch.cat(targets))
        synchronize(device)
        row=dict(epoch=epoch,training_device=args.device,train_mse=tm['action_mse'],val_mse=vm['action_mse'],train_metrics=tm,validation_metrics=vm,optimization_loss=sum(losses)/len(losses),seconds=perf_counter()-es,learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row);scheduler.step()
        if row['val_mse']<best:
            best=row['val_mse']
            torch.save(cpu_tree(dict(schema_version=4,checkpoint_kind='best',training_device=args.device,resume=resume_info,optimizer_steps=steps,generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),history_mode=args.history,history_length=args.length,model_name=args.model,config=cfg,state_mode=args.state,gripper_mode=args.gripper,model_state=model.state_dict(),normalizer=norm.as_dict(),state_normalizer=sn.as_dict(),epoch=epoch,validation_mse=best,dataset_sha256=dataset_hash,splits=data['manifest']['splits'],parameters=parameter_count(model),versions=versions(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict())),args.output/'best.pt')
        torch.save(cpu_tree(dict(schema_version=4,checkpoint_kind='last',training_device=args.device,resume=resume_info,optimizer_steps=steps,generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),history_mode=args.history,history_length=args.length,model_name=args.model,config=cfg,state_mode=args.state,gripper_mode=args.gripper,model_state=model.state_dict(),normalizer=norm.as_dict(),state_normalizer=sn.as_dict(),epoch=epoch,validation_mse=best,dataset_sha256=dataset_hash,splits=data['manifest']['splits'],parameters=parameter_count(model),versions=versions(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict())),args.output/'last.pt')
        write_json(args.output/'history.json',history)
        print(f"{args.model} {args.history} K={args.length} epoch {epoch}: train={row['train_mse']:.6f} val={best:.6f} {row['seconds']:.1f}s",flush=True)
        if args.stop_after_epoch is not None and epoch>=args.stop_after_epoch:break
    write_json(args.output/'training.json',dict(training_device=args.device,resume=resume_info,history_mode=args.history,history_length=args.length,model=args.model,state_mode=args.state,gripper_mode=args.gripper,config=cfg,versions=versions(),parameters=parameter_count(model),training_seconds=sum(r['seconds'] for r in history),resumed_process_seconds=perf_counter()-start,optimizer_steps=steps,dataset_sha256=dataset_hash,splits=data['manifest']['splits'],best_validation_mse=best,smoke_max_batches=args.max_batches,history=history))
    os.environ.setdefault('MPLCONFIGDIR',str(Path('outputs/graph/v4/.mplconfig').resolve()))
    from ..visualization import plot_curves
    plot_curves(history,args.output/'loss.png')

if __name__=='__main__':main()
