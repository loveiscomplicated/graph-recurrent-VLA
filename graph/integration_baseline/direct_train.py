"""Existing Graph–GNN–GRU regression7 with episode-contiguous TBPTT."""
import copy
import os
from pathlib import Path
import random
import resource
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from graph.integration_baseline.data import collate_sequences
from graph.integration_baseline.direct_runtime import atomic, append, read, sha, finalize, stop_gate, StopRequested, check_space
from graph.integration_baseline.model import IntegrationPolicy, bc_loss
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer


def seed_all(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False


def rng_state(*,include_cuda=False):
    result=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if include_cuda:result['cuda']=torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        if not torch.cuda.is_available():raise RuntimeError('CUDA checkpoint cannot resume on CPU')
        torch.cuda.set_rng_state_all(state['cuda'])


def normalizer_state(n):
    return dict(schema=n.schema,mean=n.stats.mean.cpu(),std=n.stats.std.cpu(),fitted_episode_ids=n.fitted_episode_ids)


def load_normalizer(s):
    return RobotNormalizer(s['schema'],StateNormalizer(s['mean'],s['std']),s['fitted_episode_ids'])


def identity(run,config,dataset):
    run=Path(run)
    return dict(dataset_sha256=sha(run/'dataset_manifest.json'),split_sha256=sha(run/'split_manifest.json'),
                vision_sha256=sha(run/('vision_manifest.json' if (run/'vision_manifest.json').exists() else 'vision_smoke_manifest.json')),
                config_sha256=sha(run/'config.yaml'),training_config=config['training'],graph_config=config['graph'],model_schema=config['model'],
                robot_schema=dataset.schema,action_schema=read(run/'expert_action_schema.json'),
                feature_normalization='unchanged baseline finite feature view; no fitted graph transform')


def optimizer(model,config):
    t=config['training']
    opt=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'],
        betas=tuple(t['betas']),eps=t['eps'],amsgrad=t['amsgrad'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['max_epochs'],eta_min=t['eta_min'])
    return opt,scheduler


def memory(model,opt):
    result=dict(model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
        process_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        memory_scope='process RSS high-water including imports/graph collation; CUDA tensor allocator measured separately')
    if next(model.parameters()).is_cuda:
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def save_checkpoint(run,name,model,opt,scheduler,normalizer,ident,progress,*,pointers=()):
    root=Path(run)/'training/checkpoints';root.mkdir(parents=True,exist_ok=True)
    path=root/name;tmp=path.with_name(path.name+'.partial')
    if path.exists() or tmp.exists():raise FileExistsError('checkpoint collision: '+str(path))
    payload=dict(version='direct_bc_v1',model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        normalizer=normalizer_state(normalizer),identity=ident,progress=progress,
        rng=rng_state(include_cuda=next(model.parameters()).is_cuda))
    torch.save(payload,tmp)
    restored=torch.load(tmp,map_location='cpu',weights_only=False)
    if restored['identity']!=ident or restored['progress']['global_step']!=progress['global_step']:
        raise ValueError('checkpoint readback failure')
    for k,v in model.state_dict().items():
        if not torch.equal(restored['model'][k],v.cpu()):raise ValueError('checkpoint model readback mismatch')
    finalize(tmp,path)
    receipt=dict(path=str(path.relative_to(run)),sha256=sha(path),epoch=progress.get('completed_epoch',progress['epoch']),global_step=progress['global_step'])
    atomic(path.with_suffix('.json'),receipt,immutable=True)
    for pointer in pointers:atomic(Path(run)/'training'/f'{pointer}.json',receipt)
    return receipt


def restore_checkpoint(run,pointer,model,opt,scheduler,ident):
    receipt=read(Path(run)/'training'/f'{pointer}.json')
    path=Path(run)/receipt['path']
    if sha(path)!=receipt['sha256']:raise ValueError('checkpoint checksum failure')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['identity']!=ident:raise ValueError('checkpoint config/dataset/split/schema mismatch')
    model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
    restore_rng(cp['rng'])
    return load_normalizer(cp['normalizer']),cp['progress'],receipt


def batch(dataset,ids,start,length,device,config):
    seq=[dataset.sequence(e,start,min(length,dataset.episodes[e]['actions']-start)) for e in ids]
    graph,state,target,valid=collate_sequences(seq,**config['graph'])
    return graph.to(device),state.to(device),target.to(device),valid.to(device)


def finite_update(model,opt,pred,target,mask,clip):
    loss=bc_loss(pred[mask],target[mask],gripper_mode='regression')
    if not torch.isfinite(loss):raise FloatingPointError('NaN/Inf BC loss')
    opt.zero_grad(set_to_none=True);loss.backward()
    for name,p in model.named_parameters():
        if p.grad is None:raise RuntimeError('missing gradient: '+name)
        if not torch.isfinite(p.grad).all():raise FloatingPointError('NaN/Inf gradient: '+name)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
    opt.step()
    for p in model.parameters():
        if not torch.isfinite(p).all():raise FloatingPointError('NaN/Inf parameter after update')
    return float(loss.detach()),float(norm)


def gate(run,config,dataset,*,smoke=False):
    """Real packets, candidate B2×T32, step equivalence, exact optimizer/RNG resume."""
    run=Path(run);seed_all(config['seed'],config['threads']);device=config['device']
    if device not in ('cpu','cuda'):raise ValueError('explicit CPU or CUDA backend required')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    ids=dataset.splits['train'][:config['training']['batch_size']]
    length=config['training']['sequence_length']
    if len(ids)!=2 or any(dataset.episodes[e]['actions']<length for e in ids):
        raise ValueError('gate requires two real full candidate sequences')
    norm=dataset.fit_normalizer();model=IntegrationPolicy(dataset.schema,**config['model']).to(device)
    opt,scheduler=optimizer(model,config);ident=identity(run,config,dataset)
    graph,state,target,valid=batch(dataset,ids,0,length,device,config)
    model.eval();initial=torch.randn(1,len(ids),config['model']['gru_hidden'],device=device)
    with torch.no_grad():
        output,hidden=model(graph,state,norm,initial)
        h=initial.clone();steps=[]
        for t in range(length):
            g,s,_,_=batch(dataset,ids,t,1,device,config)
            y,h=model(g,s,norm,h);steps.append(y[:,0])
        torch.testing.assert_close(output,torch.stack(steps,1),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(hidden,h,atol=1e-6,rtol=1e-5)
    model.train();pred,h=model(graph,state,norm)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    loss,gn=finite_update(model,opt,pred,target,valid,config['training']['gradient_clip'])
    if not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('no optimizer update')
    progress=dict(epoch=0,global_step=1,best=None,patience=0,cursor=None)
    receipt=save_checkpoint(run,'gate-step1-'+uuid4().hex[:8]+'.pt',model,opt,scheduler,norm,ident,progress,pointers=('gate',))
    # Nonzero hidden and next optimizer update must agree after save/reload.
    h=h.detach();expected_rng=(random.random(),float(np.random.rand()),torch.rand(3),torch.rand(3,device=device))
    opt.zero_grad(set_to_none=True);pred,_=model(graph,state,norm,h)
    finite_update(model,opt,pred,target,valid,config['training']['gradient_clip'])
    expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    restored_model=IntegrationPolicy(dataset.schema,**config['model']).to(device);restored_opt,restored_scheduler=optimizer(restored_model,config)
    restored_norm,_,_=restore_checkpoint(run,'gate',restored_model,restored_opt,restored_scheduler,ident)
    actual_rng=(random.random(),float(np.random.rand()),torch.rand(3),torch.rand(3,device=device))
    if expected_rng[:2]!=actual_rng[:2] or any(not torch.equal(expected_rng[i],actual_rng[i]) for i in (2,3)):raise ValueError('RNG resume mismatch')
    pred,_=restored_model(graph,state,restored_norm,h)
    finite_update(restored_model,restored_opt,pred,target,valid,config['training']['gradient_clip'])
    for k,v in restored_model.state_dict().items():torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
    for key,value in restored_opt.state_dict()['state'].items():
        for k,v in value.items():
            if torch.is_tensor(v):torch.testing.assert_close(v,expected_opt['state'][key][k],atol=0,rtol=0)
    report=dict(status='PASS',scope='REAL_PACKET_ALIGNED',device=device,sequence_length=length,batch_size=len(ids),
        finite_loss=loss,gradient_norm=gn,optimizer_update=True,exact_model_optimizer_rng_resume=True,
        sequence_step_max_error=float((output-torch.stack(steps,1)).abs().max()),
        memory=memory(restored_model,restored_opt),checkpoint=receipt,smoke=smoke)
    report_path=run/('training_smoke_gate.json' if smoke else 'training_gate.json')
    if report_path.exists():report_path=report_path.with_name(report_path.stem+'-'+uuid4().hex[:8]+'.json')
    atomic(report_path,report,immutable=True)
    atomic(run/'latest_training_gate.json',dict(path=str(report_path.relative_to(run)),sha256=sha(report_path)))
    return report


def evaluate(dataset,model,norm,ids,config):
    sums=np.zeros(7,np.float64);absolute=np.zeros(7,np.float64);count=0;tick=perf_counter()
    model.eval()
    with torch.no_grad():
        for eid in ids:
            hidden=None
            for start in range(0,dataset.episodes[eid]['actions'],config['training']['sequence_length']):
                g,s,y,mask=batch(dataset,[eid],start,config['training']['sequence_length'],config['device'],config)
                out,hidden=model(g,s,norm,hidden);hidden=hidden.detach()
                if not torch.isfinite(out).all():raise FloatingPointError('nonfinite offline prediction')
                error=(out[mask]-y[mask]).cpu().numpy().astype(np.float64)
                sums+=(error**2).sum(0);absolute+=np.abs(error).sum(0);count+=len(error)
    if not count:raise ValueError('empty evaluation split')
    return dict(mse=float(sums.sum()/(7*count)),translation_mse=float(sums[:3].sum()/(3*count)),
        rotation_mse=float(sums[3:6].sum()/(3*count)),gripper_mse=float(sums[6]/count),
        translation_mae=float(absolute[:3].sum()/(3*count)),rotation_mae=float(absolute[3:6].sum()/(3*count)),
        gripper_mae=float(absolute[6]/count),per_dimension_mse=(sums/count).tolist(),
        count=count,elapsed_seconds=perf_counter()-tick,units='native continuous controller commands')


def train(run,config,dataset,events,logger):
    run=Path(run);t=config['training'];seed_all(config['seed'],config['threads'])
    model=IntegrationPolicy(dataset.schema,**config['model']).to(config['device'])
    opt,scheduler=optimizer(model,config);ident=identity(run,config,dataset)
    norm=dataset.fit_normalizer()
    progress=dict(epoch=1,completed_epoch=0,global_step=0,best=None,patience=0,cursor=None,history=[],elapsed_seconds=0.)
    if (run/'training/last_valid.json').exists():
        norm,progress,receipt=restore_checkpoint(run,'last_valid',model,opt,scheduler,ident)
        logger.record(resume_checkpoint=receipt,source_dataset_id=run.name)
    else:
        if (run/'training/progress.json').exists() or list((run/'training/checkpoints').glob('initial-*.pt')):
            raise RuntimeError('previous training exists without last valid checkpoint; refusing fresh initialization')
        save_checkpoint(run,'initial-e1-s0-'+uuid4().hex[:8]+'.pt',model,opt,scheduler,norm,ident,progress,pointers=('last_valid',))
    # Restored normalizer is immutable and must be fitted on precisely the frozen training split.
    if norm.fitted_episode_ids!=sorted(dataset.splits['train']):raise ValueError('normalizer train-only identity')
    if (run/'training/completion.json').exists():
        completed=read(run/'training/completion.json')
        if progress['epoch']<=t['max_epochs'] and progress['patience']<t['patience']:
            raise ValueError('training completion conflicts with checkpoint progress')
        for key in ['best','last_valid']:
            if completed[key]!=read(run/'training'/f'{key}.json') or sha(run/completed[key]['path'])!=completed[key]['sha256']:
                raise ValueError('completed training checkpoint integrity mismatch')
        return completed
    events.emit('training_start',epoch=progress['epoch'],step=progress['global_step'],device=config['device'],
                max_epochs=t['max_epochs'],random_initialization=True)
    tick=perf_counter();base_elapsed=progress['elapsed_seconds']
    def save_stop(cursor):
        progress['cursor']=cursor;progress['elapsed_seconds']=base_elapsed+perf_counter()-tick
        return save_checkpoint(run,f'stop-e{progress["epoch"]}-s{progress["global_step"]}-{uuid4().hex[:6]}.pt',
            model,opt,scheduler,norm,ident,progress,pointers=('last_valid',))
    while progress['epoch']<=t['max_epochs'] and progress['patience']<t['patience']:
        epoch=progress['epoch']; model.train()
        cursor=progress['cursor']
        if cursor is None:
            order=np.random.permutation(dataset.splits['train']).tolist()
            cursor=dict(order=order,group_index=0,offset=0,hidden={},sums=[0.]*7,count=0,gradient_sum=0.,updates=0)
        sums=np.asarray(cursor['sums'],np.float64);epoch_tick=perf_counter()
        order=cursor['order']
        for group_index in range(cursor['group_index'],len(order),t['batch_size']):
            ids=order[group_index:group_index+t['batch_size']]
            offset=cursor['offset'] if group_index==cursor['group_index'] else 0
            hidden=cursor['hidden'] if group_index==cursor['group_index'] else {}
            for start in range(offset,max(dataset.episodes[e]['actions'] for e in ids),t['sequence_length']):
                active=[e for e in ids if start<dataset.episodes[e]['actions']]
                cursor.update(group_index=group_index,offset=start,hidden=hidden,sums=sums.tolist())
                if (run/'STOP').exists():save_stop(cursor);raise StopRequested()
                check_space(run,config)
                g,s,y,mask=batch(dataset,active,start,t['sequence_length'],config['device'],config)
                h=None if start==0 else torch.cat([hidden[e] for e in active],dim=1).to(config['device'])
                pred,next_h=model(g,s,norm,h)
                loss,gradient=finite_update(model,opt,pred,y,mask,t['gradient_clip'])
                # Keep each stream's identity across shrinking padded batches.
                hidden={e:next_h[:,i:i+1].detach().cpu() for i,e in enumerate(active)}
                err=(pred[mask].detach()-y[mask]).cpu().numpy().astype(np.float64)
                sums+=(err**2).sum(0);cursor['count']+=len(err);cursor['gradient_sum']+=gradient;cursor['updates']+=1
                progress['global_step']+=1
                cursor.update(offset=start+t['sequence_length'],hidden=hidden,sums=sums.tolist())
                atomic(run/'training/progress.json',dict(epoch=epoch,step=progress['global_step'],group=group_index,
                    episode_ids=active,next_timestep=cursor['offset'],loss=loss))
            cursor.update(group_index=group_index+t['batch_size'],offset=0,hidden={},sums=sums.tolist())
        count=cursor['count']
        validation=evaluate(dataset,model,norm,dataset.splits['validation'],config)
        improved=progress['best'] is None or validation['mse']<progress['best']
        if improved:progress['best']=validation['mse'];progress['patience']=0
        else:progress['patience']+=1
        elapsed=perf_counter()-epoch_tick
        row=dict(epoch=epoch,global_step=progress['global_step'],train_mse=float(sums.sum()/(7*count)),
            train_translation_mse=float(sums[:3].sum()/(3*count)),train_rotation_mse=float(sums[3:6].sum()/(3*count)),
            train_gripper_mse=float(sums[6]/count),val_mse=validation['mse'],validation=validation,
            learning_rate=opt.param_groups[0]['lr'],gradient_norm=cursor['gradient_sum']/cursor['updates'],
            elapsed_seconds=elapsed,frames_per_second=count/elapsed,memory=memory(model,opt),best_val=progress['best'])
        progress['history'].append(row);progress['epoch']=epoch+1;progress['completed_epoch']=epoch;progress['cursor']=None
        progress['elapsed_seconds']=base_elapsed+perf_counter()-tick;scheduler.step()
        pointers=('last_valid','best') if improved else ('last_valid',)
        receipt=save_checkpoint(run,f'epoch{epoch:03d}-s{progress["global_step"]}-{uuid4().hex[:8]}.pt',model,opt,scheduler,norm,ident,progress,pointers=pointers)
        logger.link_output(run/receipt['path'],kind='checkpoint')
        append(run/'training/train_log.jsonl',row)
        atomic(run/'training/metrics.json',dict(history=progress['history'],best=read(run/'training/best.json')))
        logger.log_epoch(row)
        events.emit('epoch_summary',epoch=epoch,max_epochs=t['max_epochs'],train_loss=row['train_mse'],
            val_loss=row['val_mse'],best_val=progress['best'],lr=row['learning_rate'],elapsed=progress['elapsed_seconds'],
            checkpoint=receipt['path'],status='RUNNING')
        if improved:events.emit('best_checkpoint',epoch=epoch,validation_mse=validation['mse'],checkpoint=receipt['path'])
        stop_gate(run)
    early=progress['patience']>=t['patience']
    if early:events.emit('early_stopping',patience=progress['patience'],epoch=progress['epoch']-1)
    result=dict(status='COMPLETE',early_stopping=early,epochs=progress['epoch']-1,
        best=read(run/'training/best.json'),last_valid=read(run/'training/last_valid.json'),
        elapsed_seconds=progress['elapsed_seconds'],history=progress['history'],memory=memory(model,opt),
        timing_scope='active training wall; excludes stopped downtime, collection/vision, final checkpoint/report overhead')
    atomic(run/'training/completion.json',result,immutable=True)
    events.emit('training_complete',epochs=result['epochs'],best=result['best'],elapsed_seconds=result['elapsed_seconds'])
    return result


def offline_test(run,config,dataset,events,logger):
    run=Path(run)
    if (run/'training/test_metrics.json').exists():return read(run/'training/test_metrics.json')
    # A consumed test without an atomic result requires intervention, not an automatic repeated test.
    if (run/'training/test_started.json').exists():
        raise RuntimeError('test already consumed without result; manual intervention required')
    atomic(run/'training/test_started.json',dict(best=read(run/'training/best.json')),immutable=True)
    model=IntegrationPolicy(dataset.schema,**config['model']).to(config['device']);opt,scheduler=optimizer(model,config)
    norm,progress,receipt=restore_checkpoint(run,'best',model,opt,scheduler,identity(run,config,dataset))
    report=dict(status='COMPLETE',checkpoint=receipt,selection='validation only',test_evaluations=1,
        metrics=evaluate(dataset,model,norm,dataset.splits['test'],config),closed_loop_success='NOT_EXECUTED')
    atomic(run/'training/test_metrics.json',report,immutable=True);logger.log(report['metrics'],prefix='test')
    events.emit('offline_evaluation_complete',**report);return report
