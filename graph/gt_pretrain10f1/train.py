"""Fixed12k-update spatial supervision. Labels never enter model.forward."""
import argparse
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,append_csv,tensor_identity,now,tree_identity
from .data import GraphPool
from .model import StateEncoder,RobotOnly,absolute_loss,displacement_loss


def rng_state(rng):
    return dict(numpy_generator=copy.deepcopy(rng.bit_generator.state),python=random.getstate(),
        numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def set_rng(state,rng):
    rng.bit_generator.state=state['numpy_generator'];random.setstate(state['python'])
    np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def save(path,payload):
    tmp=path.with_suffix('.partial.pt');torch.save(payload,tmp)
    loaded=torch.load(tmp,map_location='cpu',weights_only=False)
    assert tensor_identity(loaded['model'])==tensor_identity(payload['model'])
    tmp.replace(path)


@torch.inference_mode()
def validate(model,pool,indices):
    model.eval();values=[];byepisode={};physical=[]
    for start in range(0,len(indices),64):
        ids=indices[start:start+64];g,r,y=pool.select(ids);p,_,_=model(g,r)
        loss,_=absolute_loss(p,y,reduction=False)
        physical.append(torch.stack((
            (p[:,:3]-y[:,:3]).norm(dim=-1)*1000,
            (p[:,7:10]-y[:,7:10]).norm(dim=-1)*1000,
            2*torch.acos((p[:,3:7]*y[:,3:7]).sum(-1).abs().clamp(0,1))*180/np.pi,
            2*torch.acos((p[:,10:14]*y[:,10:14]).sum(-1).abs().clamp(0,1))*180/np.pi),-1).cpu().numpy())
        for idx,value in zip(ids,loss.cpu().numpy()):
            byepisode.setdefault(pool.meta[idx]['episode_id'],[]).append(float(value));values.append(float(value))
    model.train()
    means=np.concatenate(physical).mean(0)
    return dict(macro_objective=float(np.mean([np.mean(v) for v in byepisode.values()])),
        frame_objective=float(np.mean(values)),episodes=len(byepisode),frames=len(values),
        peg_position_mm=float(means[0]),hole_position_mm=float(means[1]),
        peg_orientation_deg=float(means[2]),hole_orientation_deg=float(means[3]))


def gradient_check(model):
    report={}
    for name,module in [('gnn',model.graph_encoder),('state_head',model.state_head),
                        ('message_layer0',model.graph_encoder.layers[0]),('message_layer1',model.graph_encoder.layers[1])]:
        gradients=[p.grad for p in module.parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError('missing/nonfinite gradients '+name)
        norm=float(torch.stack([g.norm() for g in gradients]).norm())
        assert norm>0;report[name]=norm
    return report


def resume_check(model,pool,root,stats):
    """Real graph2-update trajectory vs serialized model/optimizer/RNG resume."""
    clone=copy.deepcopy(model);opt=torch.optim.AdamW(clone.parameters(),lr=.0003,weight_decay=.00001)
    rng=np.random.default_rng(4242);ids=rng.choice(pool.splits['train'],8)
    def update(m,o,ix):
        g,r,y=pool.select(ix);o.zero_grad(set_to_none=True);pred,_,_=m(g,r)
        loss,_=absolute_loss(pred,y);loss.backward();o.step();return float(loss)
    first=update(clone,opt,ids);grad=gradient_check(clone)
    payload=dict(model=copy.deepcopy(clone.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(rng),
        normalization=stats,step=1)
    smoke=root/'engineering';smoke.mkdir(exist_ok=False);save(smoke/'resume_step1.pt',payload)
    ids2=rng.choice(pool.splits['train'],8);continuous=update(clone,opt,ids2)
    other=copy.deepcopy(model);ck=torch.load(smoke/'resume_step1.pt',map_location='cpu',weights_only=False)
    other.load_state_dict(ck['model']);oo=torch.optim.AdamW(other.parameters(),lr=.0003,weight_decay=.00001)
    oo.load_state_dict(ck['optimizer']);set_rng(ck['rng'],rng)
    restored_ids=rng.choice(pool.splits['train'],8);np.testing.assert_array_equal(ids2,restored_ids)
    resumed=update(other,oo,restored_ids)
    exact=tensor_identity(clone.state_dict())==tensor_identity(other.state_dict())
    assert exact and resumed==continuous
    result=dict(status='PASS',first_loss=first,second_loss=continuous,gradients=grad,
        save_readback='PASS',resumed_next_update_weights_exact=True,primary_weights_untouched=True)
    write(root/'training_smoke_verification.json',result,immutable=True)
    return result


def train_one(root,pool,kind):
    cfg=read(root/'final_pretrain_config.json');stats=pool.normalization
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    rng=np.random.default_rng(cfg['seed'])
    model=(StateEncoder if kind=='GNN' else RobotOnly)(stats['robot_mean'],stats['robot_std'],stats['target_mean']).to('cuda')
    if kind=='GNN':
        initial=tensor_identity(model.state_dict());resume_check(model,pool,root,stats)
        assert tensor_identity(model.state_dict())==initial
        # Engineering RNG consumption is discarded before the primary run.
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
        model=StateEncoder(stats['robot_mean'],stats['robot_std'],stats['target_mean']).to('cuda')
    output=root/('training' if kind=='GNN' else 'robot_baseline_training');output.mkdir(exist_ok=False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],
        betas=tuple(cfg['betas']),eps=cfg['eps'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_updates'],eta_min=0)
    pairrows=[p for p in pool.pairs if p['split']=='train' and p['valid']]
    assert pairrows,'Paired delta loss was preregistered and requires valid pairs'
    absolute=np.asarray(pool.splits['train']);best=float('inf');beststep=None;start=time.perf_counter()
    with ExperimentLogger(team='graph',model='Task10F1_'+kind,seed=cfg['seed'],config=cfg,
            root=root/'experiments',job_type='train') as log:
        log.record(parent_run_id=read(root/'run_identity.json')['run_id'],normalization=stats,
            dataset_manifest_sha256=sha(root/'final_dataset_manifest.json'),graph_manifest_sha256=sha(root/'final_dataset_manifest.json'),
            initial_full_task_graph_manifest_sha256=sha(root/'graph_cache_manifest.json'),
            params=sum(p.numel() for p in model.parameters()),BC_weights_reused=False)
        for step in range(1,cfg['max_updates']+1):
            if (root/'STOP').exists():raise KeyboardInterrupt('cooperative STOP; last checkpoint preserved')
            ids=rng.choice(absolute,cfg['batch_absolute'],replace=True)
            selected=[pairrows[i] for i in rng.integers(0,len(pairrows),cfg['batch_pairs'])]
            indices=np.r_[ids,[p['base'] for p in selected],[p['index'] for p in selected]]
            g,r,y=pool.select(indices);opt.zero_grad(set_to_none=True)
            pred,_,_=model(g,r)
            abs_loss,parts=absolute_loss(pred,y)
            n=cfg['batch_pairs'];delta=displacement_loss(pred[-2*n:-n],pred[-n:],y[-2*n:-n],y[-n:])
            loss=abs_loss+cfg['delta_weight']*delta
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite primary loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
            opt.step();scheduler.step()
            if step==1 or step%25==0:
                row=dict(step=step,loss=float(loss),absolute_loss=float(abs_loss),delta_loss=float(delta),
                    position_loss=float(parts['position']),rotation_loss=float(parts['rotation']),size_loss=float(parts['size']),
                    lr=opt.param_groups[0]['lr'],gradient_norm=float(norm),wall_seconds=time.perf_counter()-start)
                append_csv(output/'train_metrics.csv',row);log.log(row,step=step)
            if step%cfg['validation_interval']==0 or step==1:
                metric=validate(model,pool,pool.splits['validation'])
                append_csv(output/'validation_metrics.csv',dict(step=step,**metric,wall_seconds=time.perf_counter()-start))
                log.log(metric,step=step,prefix='validation/')
                selected=metric['macro_objective']<best
                if selected:best=metric['macro_objective'];beststep=step
                payload=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(rng),
                    normalization=stats,step=step,best_step=beststep,best_validation=best,config=cfg,
                    run_id=log.run_id,parent_run_id=read(root/'run_identity.json')['run_id'],kind=kind,
                    source_manifest_sha256=sha(root/'training_execution_source.json'),
                    gt38_contract_sha256=sha(root/'gt38_feature_contract.json'),robot_schema=pool.schema,
                    geometry_contract=read(root/'config.json')['geometry'],
                    graph_manifest_sha256=sha(root/'final_dataset_manifest.json'),
                    initial_full_task_graph_manifest_sha256=sha(root/'graph_cache_manifest.json'),
                    slot_order=['TCP','Left','Right','Peg','Box'])
                save(output/'last.pt',payload)
                if selected:save(output/'best.pt',payload)
                print('TRAIN',kind,step,'val',metric['macro_objective'],'best',beststep,'seconds',time.perf_counter()-start,flush=True)
        for path in (output/'best.pt',output/'last.pt'):log.link_output(path,kind='checkpoint')
        for path in (output/'train_metrics.csv',output/'validation_metrics.csv'):log.link_output(path,kind='metrics')
        result=dict(run_id=log.run_id,kind=kind,best_step=beststep,best_validation=best,
            updates=cfg['max_updates'],wall_seconds=time.perf_counter()-start,
            checkpoint_sha256=sha(output/'best.pt'),last_sha256=sha(output/'last.pt'),
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        write(output/'result.json',result,immutable=True)
        return result


def run(root):
    import os
    if os.getenv('CONTAINER_ID')!='57987772':raise RuntimeError('Task10F1 CUDA training is restricted to instance57987772')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    # Latest steering forbids launching on the initial Full-task plan alone.
    release=read(root/'data_ready_release.json')
    if (release.get('status')!='READY_FOR_GNN_PRETRAINING' or
        release.get('Task10F_GT_validation')!='PASS' or release.get('transfer_readback')!='PASS' or
        release.get('runtime_camera_geometry_compatibility')!='PASS'):
        raise ValueError('Task10F DATA_READY/GT/transfer/runtime release is not verified')
    for filename,key in [('final_dataset_manifest.json','dataset_manifest_sha256'),
                         ('final_pretrain_config.json','pretrain_config_sha256')]:
        if sha(root/filename)!=release[key]:raise ValueError('final release freeze SHA mismatch')
    if not read(root/'final_dataset_manifest.json').get('contains_verified_Task10F'):
        raise ValueError('initial Full-task plan cannot stand in for priority Task10F data')
    import fcntl
    owner=(root/'primary_training.lock').open('a')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'training').exists():raise FileExistsError('primary stage already exists; inspect/resume without duplication')
    source=Path(__file__).resolve().parents[2]
    write(root/'training_execution_source.json',dict(created_at=now(),hashes=tree_identity(source)),immutable=True)
    started=time.perf_counter();pool=GraphPool(root,'cuda',allow_final=False)
    write(root/'training_data_verification.json',dict(status='PASS',absolute_counts={k:len(v) for k,v in pool.splits.items()},
        pair_counts={s:sum(p['split']==s and p['valid'] for p in pool.pairs) for s in ('train','validation','final')},
        final_access=False,sources=pool.sources,normalization=pool.normalization,
        load_seconds=time.perf_counter()-started),immutable=True)
    primary=train_one(root,pool,'GNN')
    import shutil
    shutil.copy2(root/'training/best.pt',root/'gnn_pretrained.pt')
    for name in ('train_metrics.csv','validation_metrics.csv'):shutil.copy2(root/'training'/name,root/name)
    # Freeze primary selection before any Final graphs or labels enter memory.
    write(root/'model_selection_frozen.json',dict(at=now(),primary=primary,
        checkpoint_sha256=sha(root/'gnn_pretrained.pt'),final_accessed=False),immutable=True)
    baseline=train_one(root,pool,'RobotOnly')
    write(root/'checkpoint_manifest.json',dict(primary=primary,robot_only=baseline,
        selected_checkpoint=dict(path='gnn_pretrained.pt',sha256=sha(root/'gnn_pretrained.pt'),state_head_included=True),
        same_GNN_for_A_B=True,optimizer_RNG_resume_included=True,
        source_manifest_sha256=sha(root/'training_execution_source.json')),immutable=True)
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
