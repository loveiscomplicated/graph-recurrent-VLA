"""Independent CUDA worker, exact cursor/hidden/RNG checkpoint resume."""
import argparse
import fcntl
import signal
import shutil
import copy
import csv
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,digest,tensor_identity,now
from graph.plan_a_bc.model import normalizer
from graph.plan_a_bc.train import assert_nested_equal
from graph.integration_baseline.direct_train import seed_all,rng_state,restore_rng
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.gt_pretrain10f1.model import absolute_loss,displacement_loss
from .model import Policy
from .data import Dataset


def setup(root,condition,device='cuda'):
    cfg=read(root/'config.json');seed_all(cfg['seed'],1)
    f1=torch.load(cfg['f1_checkpoint'],map_location='cpu',weights_only=False)
    model=Policy(read(root/'normalizer.json')['schema'],f1,condition,cfg['seed']).to(device)
    norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'));ds=Dataset(root)
    params=[p for p in model.parameters() if p.requires_grad]
    t=cfg['training'];opt=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'],
        betas=tuple(t['betas']),eps=t['eps'],amsgrad=t['amsgrad'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['updates'],eta_min=0)
    return cfg,model,norm,codec,ds,opt,sched


def geometry_loss(model,ds,g,r,target,step,cfg,device):
    pred=model.geometry_prediction(model.last_readout,model.last_presence,r)
    # Same F1 absolute/sign-invariant/shape objective, plus actual object pairs.
    n=cfg['geometry_pairs_per_update'];rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,707]))
    selected=[ds.pairs[i] for i in rng.integers(0,len(ds.pairs),n)]
    ix=[p['base'] for p in selected]+[p['index'] for p in selected]
    pg,pr,py=ds.select(ix,device);z,pres=model.graph_encoder(pg)
    pp=model.geometry_prediction(z,pres,pr)
    abs_loss,_=absolute_loss(torch.cat((pred,pp)),torch.cat((target,py)))
    delta=displacement_loss(pp[:n],pp[n:],py[:n],py[n:])
    return abs_loss+delta,dict(geometry_absolute=float(abs_loss.detach()),geometry_delta=float(delta.detach()),extra_graphs=2*n)


def gradient_norm(parameters):
    vals=[p.grad.norm() for p in parameters if p.grad is not None]
    if not vals:return 0.
    return float(torch.stack(vals).norm())


def update(model,opt,sched,ds,norm,codec,progress,entry,cfg,device,coefficient):
    ids=entry['ids'];start=entry['start']
    if start==0:progress['hidden']={}
    hidden=None if start==0 else torch.cat([progress['hidden'][str(e)].to(device) for e in ids],1)
    g,r,y,mask,gt,ix=ds.batch(ids,start,32,device)
    model.train();opt.zero_grad(set_to_none=True)
    pred,next_hidden=model(g,r,norm,hidden)
    metrics=dict(BC_loss=0.,geometry_loss=0.,geometry_absolute=0.,geometry_delta=0.,extra_graphs=0,
        gradient_norm=0.,gnn_gradient_norm=0.,state_head_gradient_norm=0.)
    if bool(mask.any())!=entry['supervised']:raise ValueError('schedule supervision mismatch')
    if entry['supervised']:
        bc=codec.loss(pred,y,mask);loss=bc
        metrics['BC_loss']=float(bc.detach())
        if model.condition=='C' and coefficient!=0:
            geo,parts=geometry_loss(model,ds,g,r,gt,entry['update'],cfg,device)
            loss=bc+coefficient*geo;metrics.update(parts,geometry_loss=float(geo.detach()))
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward()
        metrics['gnn_gradient_norm']=gradient_norm(model.graph_encoder.parameters())
        if model.condition=='A':
            if any(p.grad is not None for p in model.graph_encoder.parameters()):raise ValueError('frozen gradient')
        elif metrics['gnn_gradient_norm']<=0:raise ValueError('joint encoder gradient missing')
        if model.condition=='C' and coefficient:
            metrics['state_head_gradient_norm']=gradient_norm(model.state_head.parameters())
            if metrics['state_head_gradient_norm']<=0:raise ValueError('C state gradient missing')
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
            cfg['training']['gradient_clip'],error_if_nonfinite=True)
        metrics['gradient_norm']=float(grad);opt.step();sched.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite weight')
        progress['step']+=1
        if progress['step']!=entry['update']:raise ValueError('duplicate/missing optimizer update')
    for b,e in enumerate(ids):progress['hidden'][str(e)]=next_hidden[:,b:b+1].detach().cpu().clone()
    progress['cursor']+=1
    metrics.update(step=progress['step'],cursor=progress['cursor'],valid_BC_frames=int(mask.sum()),
        scanned_frames=len(ix),lr=opt.param_groups[0]['lr'])
    return metrics


@torch.inference_mode()
def validate(model,ds,norm,codec,device):
    model.eval();total=0.;count=0
    for eid,e in ds.episodes.items():
        if e['split']!='validation':continue
        hidden=None
        for start in range(0,e['actions'],32):
            g,r,y,mask,_,_=ds.batch([int(eid)],start,32,device)
            p,hidden=model(g,r,norm,hidden)
            if mask.any():total+=float(codec.loss(p,y,mask))*int(mask.sum());count+=int(mask.sum())
    if not count:raise ValueError('no validation labels')
    return dict(BC_loss=total/count,frames=count,episodes=48)


def save(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    torch.save(payload,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    back=torch.load(tmp,map_location='cpu',weights_only=False)
    assert_nested_equal(payload,back);os.replace(tmp,path)
    write(path.with_suffix('.json'),dict(sha256=sha(path),step=back['progress']['step'],
        cursor=back['progress']['cursor'],weights=tensor_identity(back['model']),readback='PASS'))


def checkpoint(root,model,opt,sched,progress,log,initial):
    return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=copy.deepcopy(opt.state_dict()),scheduler=sched.state_dict(),progress=copy.deepcopy(progress),
        rng=rng_state(include_cuda=next(model.parameters()).is_cuda),condition=model.condition,
        config_sha256=sha(root/'config.json'),protocol_sha256=sha(root/'training_protocol_frozen.json'),
        dataset_sha256=sha(root/'dataset_identity.json'),schedule_sha256=sha(root/'sampling_schedule.json'),
        codec=read(root/'action_codec.json'),normalizer=read(root/'normalizer.json'),initial=initial,
        run_id=log.run_id)


def restore(root,ck,model,opt,sched):
    for name,key in [('config.json','config_sha256'),('training_protocol_frozen.json','protocol_sha256'),
            ('dataset_identity.json','dataset_sha256'),('sampling_schedule.json','schedule_sha256')]:
        if sha(root/name)!=ck[key]:raise ValueError('resume identity '+name)
    if ck['condition']!=model.condition:raise ValueError('condition mismatch')
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
    restore_rng(ck['rng']);return copy.deepcopy(ck['progress'])


def csv_from_ledger(folder,out):
    rows=[read(p) for p in sorted(folder.glob('*.json'))]
    if not rows:return
    tmp=out.with_suffix('.partial')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(tmp,out)


def run(root,condition,mode='train',out=None,steps=None,barrier=None):
    out=out or root/('train_'+condition);out.mkdir(parents=True,exist_ok=True)
    owner=(out/'worker.lock').open('a')
    fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    device='cuda';cfg,model,norm,codec,ds,opt,sched=setup(root,condition,device)
    initial=model.initial_hashes();schedule=read(root/'sampling_schedule.json')
    coefficient=read(root/'lambda_calibration.json')['lambda'] if (root/'lambda_calibration.json').exists() else cfg['calibration']['fallback']
    out=out or root/('train_'+condition);out.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):raise InterruptedError('worker signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    progress=dict(step=0,cursor=0,hidden={},best_loss=None,best_step=None)
    target=steps if steps else cfg['training']['updates'];started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if barrier:
        write(out/'ready.json',dict(pid=os.getpid(),at=now()))
        deadline=time.monotonic()+180
        while not barrier.exists():
            if time.monotonic()>deadline:raise TimeoutError('concurrent barrier')
            time.sleep(.05)
    ckpath=out/'last.pt'
    with ExperimentLogger(team='graph',model='Task10G_'+condition,seed=cfg['seed'],
        config=dict(**cfg,worker_mode=mode),root=root/'experiments',job_type='train') as log:
        log.record(condition=condition,f1_sha256=cfg['f1_checkpoint_sha256'],
            dataset_sha256=sha(root/'dataset_identity.json'),schedule_sha256=sha(root/'sampling_schedule.json'))
        if mode=='train' and ckpath.exists():
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            if sha(ckpath)!=read(ckpath.with_suffix('.json'))['sha256']:raise ValueError('corrupt checkpoint')
            progress=restore(root,ck,model,opt,sched)
            # Preserve uncommitted observations in an attempt archive before replay.
            archive=out/'uncommitted_attempts'/str(time.time_ns())
            for kind in ('ledger','validation'):
                for p in (out/kind).glob('*.json'):
                    row=read(p)
                    later=row.get('cursor',0)>progress['cursor'] if kind=='ledger' else row['step']>progress['step']
                    if later:
                        dest=archive/kind/p.name;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
            # A crash after committing last.pt but before writing its scheduled copy
            # can recover the missing copy byte-for-byte from that valid commit.
            if progress['step'] in cfg['training']['checkpoint_steps']:
                candidate=out/f'step{progress["step"]:06d}.pt'
                if not candidate.exists():
                    shutil.copy2(ckpath,candidate);shutil.copy2(ckpath.with_suffix('.json'),candidate.with_suffix('.json'))
            log.link_output(ckpath,checkpoint=True)
        if mode=='train' and not ckpath.exists():
            save(ckpath,checkpoint(root,model,opt,sched,progress,log,initial))
        try:
            if barrier:started=time.perf_counter()
            while progress['step']<target:
                entry=schedule[progress['cursor']%len(schedule)]
                if mode!='train':
                    entry=dict(entry,update=progress['step']+int(entry['supervised']))
                metrics=update(model,opt,sched,ds,norm,codec,progress,entry,cfg,device,coefficient)
                if not entry['supervised']:continue
                step=progress['step'];metrics.update(wall_seconds=time.perf_counter()-started,
                    allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
                if mode=='train':
                    write(out/'ledger'/f'{step:06d}.json',metrics)
                    write(out/'status.json',dict(status='RUNNING',pid=os.getpid(),condition=condition,**metrics))
                    if step%25==0 or step==1:
                        log.log(metrics,step=step);print(condition,step,metrics,flush=True)
                    if step%cfg['training']['validation_interval']==0 or step in cfg['training']['checkpoint_steps'] or step==target:
                        val=validate(model,ds,norm,codec,device)
                        write(out/'validation'/f'{step:06d}.json',dict(step=step,**val))
                        if progress['best_loss'] is None or val['BC_loss']<progress['best_loss']:
                            progress.update(best_loss=val['BC_loss'],best_step=step)
                            save(out/'validation_best.pt',checkpoint(root,model,opt,sched,progress,log,initial))
                    if step%250==0 or step in cfg['training']['checkpoint_steps'] or step==target:
                        payload=checkpoint(root,model,opt,sched,progress,log,initial);save(ckpath,payload)
                        if step in cfg['training']['checkpoint_steps']:save(out/f'step{step:06d}.pt',payload)
                    if (root/'STOP').exists():
                        save(ckpath,checkpoint(root,model,opt,sched,progress,log,initial))
                        raise InterruptedError('cooperative STOP')
                if mode=='train' and condition=='A' and model.initial_hashes()['gnn']!=initial['gnn']:
                    raise ValueError('A encoder changed')
            torch.cuda.synchronize()
            result=dict(status='COMPLETE',condition=condition,steps=progress['step'],cursor=progress['cursor'],
                wall_seconds=time.perf_counter()-started,updates_per_second=progress['step']/(time.perf_counter()-started),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                initial=initial,final=model.initial_hashes(),all_loss_gradient_finite=True,pid=os.getpid())
            if mode=='train':
                result.update(schedule_sha256=sha(root/'sampling_schedule.json'),checkpoint_sha256=sha(ckpath))
                log.link_output(ckpath,kind='checkpoint')
                csv_from_ledger(out/'ledger',root/f'train_{condition}.csv')
                csv_from_ledger(out/'validation',root/f'validation_{condition}.csv')
            write(out/'result.json',result);write(out/'status.json',result);return result
        except BaseException as exc:
            # Never serialize partially applied optimizer state after a CUDA failure.
            write(out/'failure_'+str(time.time_ns())+'.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                last_committed_checkpoint=str(ckpath),progress_step=progress['step'],mode=mode))
            write(out/'status.json',dict(status='ERROR',error=repr(exc),step=progress['step']))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=list('ABC'),required=True)
    p.add_argument('--mode',default='train');p.add_argument('--out',type=Path);p.add_argument('--steps',type=int);p.add_argument('--barrier',type=Path)
    a=p.parse_args();run(a.root,a.condition,a.mode,a.out,a.steps,a.barrier)
