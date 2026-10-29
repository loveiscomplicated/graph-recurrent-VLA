"""Episode-contiguous Task08 Method B BC with blocking periodic rollout hooks."""
import argparse
import copy
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer
from graph.integration_baseline.direct_train import seed_all,rng_state,restore_rng,memory
from .cache import CachedDataset
from .common import read,write,sha,digest,tensor_identity,checkpoint_load,append_csv,npz,now
from .model import CachedGraphPolicy,gradient_report,normalizer


def chunks(config,epoch):
    order=np.random.default_rng(np.random.SeedSequence([42,epoch])).permutation(config['splits']['train']).tolist()
    result=[];t=config['training']
    for group in range(0,len(order),t['batch_size']):
        pair=order[group:group+t['batch_size']]
        for start in range(0,max(config['episode_lengths'][str(i)] for i in pair),t['sequence_length']):
            result.append(([i for i in pair if start<config['episode_lengths'][str(i)]],start))
    return result


def optimizer(model,config):
    t=config['training']
    opt=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'],
        betas=tuple(t['betas']),eps=t['eps'],amsgrad=t['amsgrad'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=t['max_epochs'],eta_min=t['eta_min'])
    return opt,sched


def hidden_for(progress,ids,start,device):
    if start==0:return None
    if any(str(e) not in progress['hidden'] for e in ids):raise ValueError('missing hidden carry')
    return torch.cat([progress['hidden'][str(e)].to(device) for e in ids],1)


def train_update(model,opt,ds,norm,codec,progress,ids,start,config,device):
    g,s,y,mask=ds.batch(ids,start,config['training']['sequence_length'],device)
    hidden=hidden_for(progress,ids,start,device)
    model.train();opt.zero_grad(set_to_none=True)
    pred,next_hidden=model(g,s,norm,hidden)
    loss=codec.loss(pred,y,mask)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();grads=gradient_report(model)
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'],error_if_nonfinite=True)
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite weights')
    if start==0:progress['hidden']={}
    for b,eid in enumerate(ids):progress['hidden'][str(eid)]=next_hidden[:,b:b+1].detach().cpu().clone()
    native=codec.decode(codec.tag(pred.detach()))
    err=(native[mask]-y[mask]).cpu().numpy()
    return dict(optimization_loss=float(loss.detach()),native_action7_mse=float((err**2).mean()),
        native_action7_mae=float(abs(err).mean()),translation_mse=float((err[:,:3]**2).mean()),
        rotation_mse=float((err[:,3:6]**2).mean()),gripper_mse=float((err[:,6]**2).mean()),
        gradient_norm=float(grad),encoder_gradient_norm=grads['encoder'],gru_gradient_norm=grads['gru'],
        head_gradient_norm=grads['head'],valid_timesteps=int(mask.sum()),lr=opt.param_groups[0]['lr'])


@torch.inference_mode()
def offline(ds,model,norm,codec,ids,config,device,output=None):
    model.eval();p=[];y=[];loss_sum=0.;count=0;first=[]
    for eid in ids:
        hidden=None;ep_p=[];ep_y=[]
        for start in range(0,ds.episodes[eid]['actions'],config['training']['sequence_length']):
            g,s,target,mask=ds.batch([eid],start,config['training']['sequence_length'],device)
            raw,hidden=model(g,s,norm,hidden);hidden=hidden.detach()
            loss_sum+=float(codec.loss(raw,target,mask))*int(mask.sum());count+=int(mask.sum())
            native=codec.decode(codec.tag(raw))
            ep_p.append(native[mask].cpu().numpy());ep_y.append(target[mask].cpu().numpy())
        pp,yy=np.concatenate(ep_p),np.concatenate(ep_y);p.append(pp);y.append(yy)
        first.append(dict(episode_id=eid,prediction=pp[0].tolist(),target=yy[0].tolist(),mae=float(abs(pp[0]-yy[0]).mean())))
        if output is not None:npz(Path(output)/f'ep{eid}.npz',dict(prediction=pp,target=yy,timestep=np.arange(len(yy))))
    p,y=np.concatenate(p),np.concatenate(y);e=p.astype(float)-y.astype(float)
    cp,cy=controller_commands(p),controller_commands(y)
    result=dict(optimization_loss=loss_sum/count,native_action7_mse=float((e**2).mean()),
        native_action7_mae=float(abs(e).mean()),translation_mse=float((e[:,:3]**2).mean()),
        rotation_mse=float((e[:,3:6]**2).mean()),gripper_mse=float((e[:,6]**2).mean()),
        gripper_accuracy=float(((p[:,6]>0)==(y[:,6]>0)).mean()),
        controller_translation_mae_m=float(abs(cp[:,:3]-cy[:,:3]).mean()),
        controller_rotation_component_mae_rad=float(abs(cp[:,3:6]-cy[:,3:6]).mean()),
        controller_gripper_target_mae_m=float(abs(cp[:,6]-cy[:,6]).mean()),
        episodes=len(ids),actions=count)
    if output is not None:write(Path(output)/'first_actions.json',first,immutable=True)
    return result


def save(run,root,model,opt,sched,progress,variant):
    step=progress['step'];path=run/'checkpoints'/f'step{step:06d}.pt';path.parent.mkdir(exist_ok=True)
    if path.exists():
        old=checkpoint_load(path)
        if tensor_identity(old['model'])!=tensor_identity(model.state_dict()) or old['progress']['step']!=step:
            raise ValueError('checkpoint collision')
        return read(path.with_suffix('.json'))
    payload=dict(version='plan_a_cached_graph_bc_v1',variant=variant,model=model.state_dict(),optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),progress=copy.deepcopy(progress),rng=rng_state(include_cuda=next(model.parameters()).is_cuda),
        config_sha256=sha(root/'config.json'),cache_manifest_sha256=sha(root/'cache_manifest.json'),
        codec=read(root/'action_codec.json'),normalizer=read(root/'normalizer.json'),config=read(root/'config.json'))
    tmp=path.with_suffix('.partial');torch.save(payload,tmp)
    back=checkpoint_load(tmp)
    if tensor_identity(back['model'])!=tensor_identity(model.state_dict()) or back['progress']['step']!=step:
        raise ValueError('checkpoint readback mismatch')
    # Tensor values in optimizer/hidden/RNG must also survive serialization exactly.
    assert_nested_equal(payload,back)
    os.link(tmp,path);tmp.unlink()
    receipt=dict(path=str(path.resolve()),sha256=sha(path),step=step,variant=variant,
        model_sha256=tensor_identity(back['model']),readback='PASS',run_id=run.name,created_at=now())
    write(path.with_suffix('.json'),receipt,immutable=True)
    existing=read(run/'checkpoint_manifest.json') if (run/'checkpoint_manifest.json').exists() else []
    write(run/'checkpoint_manifest.json',existing+[receipt]);write(run/'last.json',receipt)
    return receipt


def assert_nested_equal(a,b):
    if torch.is_tensor(a):
        if not torch.equal(a.cpu(),b.cpu()):raise ValueError('tensor state changed on readback')
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('state keys')
        for k in a:assert_nested_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('state length')
        for x,y in zip(a,b):assert_nested_equal(x,y)
    elif a!=b:raise ValueError('state scalar changed')


def periodic(root,run,receipt,model,opt,sched,progress,device):
    """A separate evaluation process cannot mutate the training RNG/optimizer/hidden."""
    before_rng=rng_state(include_cuda=device=='cuda');before_hash=tensor_identity(model.state_dict())
    before_progress=copy.deepcopy(progress);before_opt=copy.deepcopy(opt.state_dict());before_sched=copy.deepcopy(sched.state_dict())
    # Policy training remains resident (<1GiB); the resource smoke measures combined
    # policy/perception residency before Full training is allowed.
    subprocess.run([sys.executable,'-B','-m','graph.plan_a_bc.rollout','--root',str(root),
        '--run',str(run),'--checkpoint',receipt['path'],'--device',device],check=True,env=dict(os.environ))
    assert_nested_equal(before_rng,rng_state(include_cuda=device=='cuda'));assert_nested_equal(before_opt,opt.state_dict())
    assert_nested_equal(before_sched,sched.state_dict());assert_nested_equal(before_progress,progress)
    if before_hash!=tensor_identity(model.state_dict()):raise ValueError('evaluation changed training weights')
    write(run/f'evaluation_state_guard_step{receipt["step"]:06d}.json',dict(status='PASS',
        rng_equal=True,optimizer_equal=True,scheduler_equal=True,hidden_cursor_equal=True,weights_equal=True),immutable=True)


def smoke(root,device):
    root=Path(root).resolve();cfg=read(root/'config.json');ds=CachedDataset(root);norm=normalizer(root)
    codec=ActionCodec(read(root/'action_codec.json'));out=root/'training_smoke';out.mkdir(exist_ok=False)
    result={}
    for variant in ('E0','E1'):
        seed_all(42,1);model=CachedGraphPolicy(norm.schema,variant,**cfg['model']).to(device);opt,sched=optimizer(model,cfg)
        progress=dict(step=0,epoch=1,chunk=0,hidden={})
        available=sorted(ds.arrays)[:2]
        batches=[([i for i in available if start<ds.episodes[i]['actions']],start) for start in (0,32,64)]
        losses=[]
        for ids,start in batches:
            row=train_update(model,opt,ds,norm,codec,progress,ids,start,cfg,device);progress['step']+=1;losses.append(row)
        payload=dict(model=copy.deepcopy(model.state_dict()),opt=copy.deepcopy(opt.state_dict()),sched=copy.deepcopy(sched.state_dict()),
            progress=copy.deepcopy(progress),rng=rng_state(include_cuda=device=='cuda'))
        path=out/(variant+'.pt');torch.save(payload,path);back=checkpoint_load(path);assert_nested_equal(payload,back)
        # One update after reloading must exactly match an uninterrupted update.
        ids,start=[i for i in available if 96<ds.episodes[i]['actions']],96
        if not ids:ids,start=available,0
        one=copy.deepcopy(progress);train_update(model,opt,ds,norm,codec,one,ids,start,cfg,device)
        expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        model.load_state_dict(back['model']);opt.load_state_dict(back['opt']);sched.load_state_dict(back['sched']);restore_rng(back['rng'])
        two=copy.deepcopy(back['progress']);train_update(model,opt,ds,norm,codec,two,ids,start,cfg,device)
        assert_nested_equal(expected,model.state_dict());assert_nested_equal(expected_opt,opt.state_dict());assert_nested_equal(one,two)
        result[variant]=dict(status='PASS',updates=4,gradients=losses,save_resume_exact=True,
            memory=memory(model,opt),parameters=sum(p.numel() for p in model.parameters()))
        # Actual current-RGB-D integration before Full training; these discarded
        # smoke weights are never adopted by either Full run.
        smoke_run=out/variant;smoke_run.mkdir()
        progress['step']=4
        receipt=save(smoke_run,root,model,opt,sched,progress,variant)
        before=rng_state(include_cuda=device=='cuda')
        subprocess.run([sys.executable,'-B','-m','graph.plan_a_bc.rollout','--root',str(root),
            '--run',str(smoke_run),'--checkpoint',receipt['path'],'--device',device,'--pretraining-smoke'],check=True)
        assert_nested_equal(before,rng_state(include_cuda=device=='cuda'))
        result[variant]['live_rollout_integration']='PASS'
    write(root/'training_smoke_verification.json',dict(status='PASS',variants=result,
        weights_discarded=True,full_training_fresh_seed42=True),immutable=True)
    print(result,flush=True)


def train(root,variant,device,resume=None):
    root=Path(root).resolve();cfg=read(root/'config.json')
    if read(root/'training_smoke_verification.json')['status']!='PASS':raise ValueError('smoke gate')
    if read(root/'runtime_verification.json')['status']!='PASS':raise ValueError('rollout runtime gate')
    ds=CachedDataset(root)
    if not ds.manifest['complete_train_validation']:raise ValueError('Full96/Val12 cache incomplete')
    seed_all(42,1);norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    model=CachedGraphPolicy(norm.schema,variant,**cfg['model']).to(device);opt,sched=optimizer(model,cfg)
    progress=dict(step=0,epoch=1,chunk=0,hidden={},best_loss=None,best_step=None)
    if resume:
        cp=checkpoint_load(resume)
        if cp['config_sha256']!=sha(root/'config.json') or cp['cache_manifest_sha256']!=sha(root/'cache_manifest.json') or cp['variant']!=variant:
            raise ValueError('resume identity mismatch')
        if cp['normalizer']!=read(root/'normalizer.json') or cp['codec']!=read(root/'action_codec.json'):
            raise ValueError('resume normalization changed')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sched.load_state_dict(cp['scheduler'])
        progress=cp['progress'];restore_rng(cp['rng'])
    with ExperimentLogger(team='integration',model='plan_a_'+variant+'_GRU_BC',seed=42,
        config=dict(comparison_uuid=root.name,config=cfg,variant=variant,resume=str(resume) if resume else None),
        root=root/'training_runs',job_type='train') as log:
        run=log.directory;started=time.perf_counter()
        log.record(comparison_group=root.name,dataset_sha256=ds.identity['dataset_sha256'],graph_cache_sha256=sha(root/'cache_manifest.json'),
            parameters=sum(p.numel() for p in model.parameters()),source_identity=cfg['source_sha256'])
        write(root/(variant+'_run.json'),dict(run_id=log.run_id,directory=str(run)))
        try:
            while progress['step']<cfg['training']['updates']:
                schedule=chunks(cfg,progress['epoch'])
                if progress['chunk']>=len(schedule):
                    progress.update(epoch=progress['epoch']+1,chunk=0,hidden={});sched.step();continue
                ids,start=schedule[progress['chunk']]
                metrics=train_update(model,opt,ds,norm,codec,progress,ids,start,cfg,device)
                progress['step']+=1;progress['chunk']+=1;step=progress['step']
                append_csv(run/'train_metrics.csv',dict(step=step,epoch=progress['epoch'],**metrics))
                if step%25==0:
                    log.log(metrics,step=step)
                    write(run/'progress.json',dict(step=step,epoch=progress['epoch'],chunk=progress['chunk'],
                        status='RUNNING',wall_seconds=time.perf_counter()-started))
                    print(f'{variant} step={step} epoch={progress["epoch"]} loss={metrics["optimization_loss"]:.6f}',flush=True)
                # Frequent validation for model selection; rollout checkpoints also
                # evaluate the complete Train96 and Validation12 on the same step.
                scheduled=step in cfg['evaluation']['schedule_steps']
                if step%250==0 or scheduled:
                    val=offline(ds,model,norm,codec,cfg['splits']['validation'],cfg,device)
                    append_csv(run/'validation_metrics.csv',dict(step=step,epoch=progress['epoch'],**val))
                    if scheduled:
                        fit=offline(ds,model,norm,codec,cfg['splits']['train'],cfg,device)
                        append_csv(run/'train_full_metrics.csv',dict(step=step,epoch=progress['epoch'],**fit))
                    best=progress['best_loss'] is None or val['optimization_loss']<progress['best_loss']
                    if best:progress.update(best_loss=val['optimization_loss'],best_step=step)
                    if best or scheduled:
                        receipt=save(run,root,model,opt,sched,progress,variant);log.link_output(receipt['path'],kind='checkpoint')
                        if best:write(run/'validation_best.json',receipt)
                    if scheduled:periodic(root,run,receipt,model,opt,sched,progress,device)
                if (root/'STOP').exists():
                    receipt=save(run,root,model,opt,sched,progress,variant)
                    write(run/'status.json',dict(status='STOPPED',step=step,last=receipt));return
            receipt=save(run,root,model,opt,sched,progress,variant)
            write(run/'status.json',dict(status='COMPLETE',updates=progress['step'],last=receipt,
                validation_best=read(run/'validation_best.json'),wall_seconds=time.perf_counter()-started),immutable=True)
        except BaseException as exc:
            receipt=save(run,root,model,opt,sched,progress,variant)
            write(run/'status.json',dict(status='FAIL',step=progress['step'],error=type(exc).__name__,message=str(exc),last=receipt))
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=['E0','E1'])
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.smoke:smoke(a.root,a.device)
    else:train(a.root,a.variant,a.device,a.resume)


if __name__=='__main__':main()
