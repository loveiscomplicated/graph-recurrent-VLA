"""BC only on cached predicted State38; Task10G schedule and loss preserved."""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,tensor_identity
from graph.graph_recurrent10g.train_worker import save,csv_from_ledger
from graph.integration_baseline.direct_train import seed_all,rng_state,restore_rng
from graph.integration_baseline.bc08_codec import ActionCodec
from .model import Policy


class Features:
    """No graph, sidecar, object GT or robot bypass is exposed to BC."""
    def __init__(self,root):
        self.root=root;self.episodes={};self.arrays={}
        for row in read(root/'state38_feature_cache_manifest.json')['episodes']:
            eid=row['episode'];self.episodes[eid]=row
            with np.load(root/row['path'],allow_pickle=False) as z:
                self.arrays[eid]={k:z[k].copy() for k in ('state38','action7','supervision_mask','timestep')}

    def batch(self,ids,start,length=32,device='cuda'):
        count=max(min(length,self.episodes[str(e)]['frames']-start) for e in ids)
        x=np.zeros((len(ids),count,38),'f');y=np.zeros((len(ids),count,7),'f')
        mask=np.zeros((len(ids),count),bool);occupied=mask.copy()
        for b,e in enumerate(ids):
            a=self.arrays[str(e)];part=slice(start,start+length);n=len(a['state38'][part])
            x[b,:n]=a['state38'][part];y[b,:n]=a['action7'][part]
            mask[b,:n]=a['supervision_mask'][part];occupied[b,:n]=True
            np.testing.assert_array_equal(a['timestep'][part],np.arange(start,start+n))
        return tuple(torch.as_tensor(v,device=device) for v in (x,y,mask,occupied))


def setup(root):
    cfg=read(root/'config.json');seed_all(42,1)
    model=Policy(cfg['f1_checkpoint'],read(root/'normalizer.json')['schema'],read(root/'state38_normalizer.json'))
    params=list(model.policy.parameters());frozen={id(p) for p in model.frozen.parameters()}
    assert not frozen&{id(p) for p in params}
    t=cfg['training'];opt=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'],
        betas=tuple(t['betas']),eps=t['eps'],amsgrad=t['amsgrad'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['updates'],eta_min=t['eta_min'])
    return cfg,model,Features(root),ActionCodec(read(root/'action_codec.json')),opt,scheduler


def update(model,ds,codec,opt,sched,progress,entry,cfg):
    ids=entry['ids'];start=entry['start']
    if start==0:progress['hidden']={}
    hidden=None if start==0 else torch.cat([progress['hidden'][str(e)].cuda() for e in ids],1)
    x,y,mask,occupied=ds.batch(ids,start)
    model.train();opt.zero_grad(set_to_none=True)
    pred,nh=model.from_states(x,hidden,occupied)
    assert bool(mask.any())==entry['supervised']
    loss=codec.loss(pred,y,mask) if mask.any() else pred.sum()*0
    grad=0.
    if mask.any():
        if not torch.isfinite(loss):raise FloatingPointError('BC loss')
        loss.backward()
        assert not model.frozen.training and not any(p.grad is not None for p in model.frozen.parameters())
        for module in [model.policy.encoder,model.policy.rnn,model.policy.head]:
            assert any(p.grad is not None and p.grad.norm()>0 for p in module.parameters())
        grad=float(torch.nn.utils.clip_grad_norm_(model.policy.parameters(),cfg['training']['gradient_clip'],error_if_nonfinite=True))
        opt.step();sched.step();progress['step']+=1
        assert progress['step']==entry['update']
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    for b,e in enumerate(ids):progress['hidden'][str(e)]=nh[:,b:b+1].detach().cpu().clone()
    progress['cursor']+=1
    return dict(step=progress['step'],cursor=progress['cursor'],BC_loss=float(loss.detach()),gradient_norm=grad,
        valid_BC_frames=int(mask.sum()),scanned_frames=int(occupied.sum()),lr=opt.param_groups[0]['lr'])


@torch.inference_mode()
def validate(model,ds,codec):
    model.eval();total=0.;count=0;episodes=0
    for eid,row in ds.episodes.items():
        if row['split']!='validation':continue
        hidden=None;episodes+=1
        for start in range(0,row['frames'],32):
            x,y,mask,occupied=ds.batch([int(eid)],start)
            pred,hidden=model.from_states(x,hidden,occupied)
            if mask.any():total+=float(codec.loss(pred,y,mask))*int(mask.sum());count+=int(mask.sum())
    assert count>0 and episodes==48
    return dict(BC_loss=total/count,frames=count,episodes=episodes)


def payload(root,model,opt,sched,progress,initial,run_id):
    return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=copy.deepcopy(opt.state_dict()),scheduler=copy.deepcopy(sched.state_dict()),
        rng=rng_state(include_cuda=True),progress=copy.deepcopy(progress),initial=initial,run_id=run_id,
        initial_rng=model.initial_rng,condition='S',frozen_hash=model.frozen_hash(),
        identities={p:sha(root/p) for p in ['config.json','training_protocol_frozen.json',
            'sampling_schedule.json','state38_normalizer.json','state38_feature_cache_manifest.json','action_codec.json',
            'execution_source_identity.json','state38_feature_cache_runtime_identity.json']})


def restore(root,ck,model,opt,sched):
    for p,h in ck['identities'].items():assert sha(root/p)==h,p
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
    assert model.frozen_hash()==ck['frozen_hash'];restore_rng(ck['rng'])
    return copy.deepcopy(ck['progress'])


def run(root):
    out=root/'train_S';out.mkdir(exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'last.pt').exists():raise FileExistsError('Explicit inspected resume required; automatic retry disabled')
    cfg,m,ds,codec,opt,sched=setup(root);schedule=read(root/'sampling_schedule.json')
    p=dict(step=0,cursor=0,hidden={},best_loss=None,best_step=None)
    initial=dict(frozen=m.frozen_hash(),controller=m.control_hash(),seed=42,
        parameters=dict(frozen=sum(v.numel() for v in m.frozen.parameters()),controller=sum(v.numel() for v in m.policy.parameters())))
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with ExperimentLogger(team='graph',model='Task10I_predicted_State38_GRU',seed=42,config=cfg,
             root=root/'experiments',job_type='train') as log:
        log.record(schedule_sha256=sha(root/'sampling_schedule.json'),F1_sha256=cfg['f1_checkpoint_sha256'])
        save(out/'initial.pt',payload(root,m,opt,sched,p,initial,log.run_id))
        while p['step']<cfg['training']['updates']:
            entry=schedule[p['cursor']]
            metrics=update(m,ds,codec,opt,sched,p,entry,cfg)
            if not entry['supervised']:continue
            step=p['step'];metrics['wall_seconds']=time.perf_counter()-started
            write(out/'ledger'/f'{step:06d}.json',metrics)
            write(out/'status.json',dict(status='RUNNING',**metrics))
            if step==1 or step%25==0:print('TRAIN_S',metrics,flush=True);log.log(metrics,step=step)
            if step%cfg['training']['validation_interval']==0 or step in cfg['training']['checkpoint_steps']:
                val=validate(m,ds,codec);write(out/'validation'/f'{step:06d}.json',dict(step=step,**val))
                log.log(val,step=step,prefix='validation')
                if p['best_loss'] is None or val['BC_loss']<p['best_loss']:
                    p.update(best_loss=val['BC_loss'],best_step=step)
                    save(out/'validation_best.pt',payload(root,m,opt,sched,p,initial,log.run_id))
            if step%250==0 or step in cfg['training']['checkpoint_steps']:
                assert m.frozen_hash()==initial['frozen']
                ck=payload(root,m,opt,sched,p,initial,log.run_id);save(out/'last.pt',ck)
                if step in cfg['training']['checkpoint_steps']:save(out/f'step{step:06d}.pt',ck)
            if (root/'STOP').exists():
                save(out/'last.pt',payload(root,m,opt,sched,p,initial,log.run_id))
                raise InterruptedError('STOP; checkpoint preserved')
        assert p['step']==11487 and p['cursor']==len(schedule)
        assert initial['frozen']==m.frozen_hash()
        shutil.copy2(out/'last.pt',root/'last.pt')
        csv_from_ledger(out/'ledger',root/'train_metrics.csv')
        csv_from_ledger(out/'validation',root/'validation_metrics.csv')
        write(out/'result.json',dict(status='COMPLETE',updates=p['step'],cursor=p['cursor'],
            initial=initial,final_frozen=m.frozen_hash(),peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),wall_seconds=time.perf_counter()-started))
        log.link_output(out/'last.pt',kind='checkpoint')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
