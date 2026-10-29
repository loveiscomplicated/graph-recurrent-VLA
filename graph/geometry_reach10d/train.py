"""Matched-budget E0 Reach-only BC; delegates original model/loss/update/save."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,tensor_identity,checkpoint_load,append_csv,npz
from graph.plan_a_bc.cache import CachedDataset
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.train import optimizer,train_update,offline,save,assert_nested_equal
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from graph.integration_baseline.direct_train import seed_all,rng_state,restore_rng
from graph.geometry_reach10d.resident import ResidentDataset
from graph.geometry_reach10d.profile import Resources


def extra_metrics(folder):
    ps=[];ys=[]
    for path in folder.glob('ep*.npz'):
        with np.load(path) as z:ps.append(z['prediction']);ys.append(z['target'])
    p,y=map(np.concatenate,(ps,ys));p=controller_commands(p);y=controller_commands(y)
    a,b=p[:,:3],y[:,:3];dot=(a*b).sum(1);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    valid=(np.linalg.norm(b,axis=1)>=.001)&(np.linalg.norm(a,axis=1)>1e-9)
    cos=np.divide(dot,den,out=np.zeros_like(dot),where=den>0)
    return dict(translation_vector_error_mm=float(np.linalg.norm(a-b,axis=1).mean()*1000),
        direction_cosine=float(cos[valid].mean()) if valid.any() else None,
        direction_reversal_fraction=float((dot[valid]<0).mean()) if valid.any() else None,
        early_close_fraction=float((p[:,6]<.015).mean()),
        gripper_close_timing='No close action exists in Reach-only labels; report predicted close frequency',
        direction_valid_frames=int(valid.sum()))


def datasets(root):
    ds=CachedDataset(root);resident=ResidentDataset(ds)
    return ds,resident


def smoke(root):
    cfg=read(root/'config.json');plan=read(root/'dataset_ab_manifest.json')
    ds,resident=datasets(root);norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    schedule=np.load(root/'paired_training_schedule.npz')
    ids=[int(i) for i in schedule['B'][0] if i>=0]
    checks=[]
    for start in [0,32]:
        ii=[i for i in ids if start<ds.episodes[i]['actions']]
        if not ii:continue
        a=ds.batch(ii,start,32,'cuda');b=resident.batch(ii,start,32,'cuda')
        for k in vars(a[0]):torch.testing.assert_close(getattr(a[0],k),getattr(b[0],k),rtol=0,atol=0)
        for aa,bb in zip(a[1:],b[1:]):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        checks.append(dict(start=start,ids=ii,packed_graph_robot_target_mask_exact=True))
    timings={}
    for label,data in [('cpu_pack_transfer',ds),('resident_GPU',resident)]:
        seed_all(42,1);model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda()
        opt,sched=optimizer(model,cfg);progress=dict(step=0,epoch=1,chunk=0,hidden={})
        if label=='cpu_pack_transfer':initial=tensor_identity(model.state_dict())
        else:assert initial==tensor_identity(model.state_dict())
        torch.cuda.synchronize();begin=time.perf_counter()
        for k in range(4):
            start=0 if k%2==0 else 32;ii=[i for i in ids if start<ds.episodes[i]['actions']]
            if not ii:start=0;ii=ids
            train_update(model,opt,data,norm,codec,progress,ii,start,cfg,'cuda');progress['step']+=1
        torch.cuda.synchronize();timings[label]=time.perf_counter()-begin
        state=copy.deepcopy(model.state_dict())
        if label=='cpu_pack_transfer':expected=state
        else:assert_nested_equal(expected,state)
    # Exact optimizer/weight/RNG/hidden continuation through a real checkpoint.
    before=dict(model=copy.deepcopy(model.state_dict()),opt=copy.deepcopy(opt.state_dict()),
        sched=copy.deepcopy(sched.state_dict()),progress=copy.deepcopy(progress),rng=rng_state(include_cuda=True))
    smoke_folder=root/'training_smoke';smoke_folder.mkdir(exist_ok=True)
    torch.save(before,smoke_folder/'resume.pt');back=checkpoint_load(smoke_folder/'resume.pt');assert_nested_equal(before,back)
    train_update(model,opt,resident,norm,codec,progress,ids,0,cfg,'cuda');after=copy.deepcopy(model.state_dict());after_opt=copy.deepcopy(opt.state_dict())
    model.load_state_dict(back['model']);opt.load_state_dict(back['opt']);restore_rng(back['rng']);p2=copy.deepcopy(back['progress'])
    train_update(model,opt,resident,norm,codec,p2,ids,0,cfg,'cuda')
    assert_nested_equal(after,model.state_dict());assert_nested_equal(after_opt,opt.state_dict());assert_nested_equal(progress,p2)
    receipt=save(smoke_folder,root,model,opt,sched,dict(progress,step=5),'E0')
    subprocess.run([sys.executable,'-B','-m','graph.geometry_reach10d.evaluate','--root',str(root),
        '--checkpoint',receipt['path'],'--condition','SMOKE','--mode','smoke'],check=True)
    write(root/'training_smoke_verification.json',dict(status='PASS',checks=checks,timings_seconds=timings,
        same_initial_weight_sha256=initial,exact_weight_update=True,save_resume_exact=True,
        CUDA_GNN_GRU_gradients='PASS via original train_update gradient_report',
        preload_seconds=resident.preload_seconds,weights_discarded=True,AMP='not adopted; original FP32',
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()),immutable=True)


def train(root,condition):
    assert condition in ['A','B']
    assert read(root/'training_smoke_verification.json')['status']=='PASS'
    freeze=read(root/'dataset_ab_manifest.json');cfg=read(root/'config.json')
    assert freeze['config_sha256']==sha(root/'config.json')
    assert freeze['sampler_sha256']==sha(root/'paired_training_schedule.npz')
    assert freeze['cache_manifest_sha256']==sha(root/'cache_manifest.json')
    source_root=Path(__file__).resolve().parents[2]
    for name,value in freeze['source_hashes'].items():
        assert sha(source_root/name)==value,('frozen Task10D source changed',name)
    if (root/(condition+'_complete.json')).exists():return
    ds,resident=datasets(root);norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    seed_all(42,1);model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda()
    initial=tensor_identity(model.state_dict());opt,sched=optimizer(model,cfg)
    progress=dict(step=0,epoch=1,chunk=0,hidden={},best_loss=None,best_step=None)
    schedule=np.load(root/'paired_training_schedule.npz');resume=None
    if (root/(condition+'_last.json')).exists():
        resume=read(root/(condition+'_last.json'));assert sha(resume['path'])==resume['sha256']
        cp=checkpoint_load(resume['path']);assert cp['config_sha256']==sha(root/'config.json')
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);sched.load_state_dict(cp['scheduler'])
        progress=cp['progress'];restore_rng(cp['rng'])
    proto=read(root/'collection_protocol.json');start_clock=time.perf_counter()
    with ExperimentLogger(team='graph',model='Task10D_'+condition+'_E0_GRU_Reach_BC',seed=42,
            config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],condition=condition,
                dataset_AB_sha256=sha(root/'dataset_ab_manifest.json'),config=cfg),root=root/'training_runs',job_type='train') as log,Resources(root,'training_'+condition):
        run=log.directory
        log.record(initial_weight_sha256=initial,source_hashes=freeze['source_hashes'],
            normalizer_sha256=freeze['normalizer_sha256'],codec_sha256=freeze['codec_sha256'])
        if resume:log.link_output(resume['path'],checkpoint=True)
        write(root/(condition+'_run.json'),dict(run_id=log.run_id,path=str(run)))
        total_frames=int(schedule['frames'][:progress['step']].sum())
        while progress['step']<11487:
            step=progress['step'];ids=[int(i) for i in schedule[condition][step] if i>=0];start=int(schedule['start'][step])
            tick=time.perf_counter()
            row=train_update(model,opt,resident,norm,codec,progress,ids,start,cfg,'cuda')
            assert row['valid_timesteps']==int(schedule['frames'][step])
            total_frames+=row['valid_timesteps'];progress.update(step=step+1,chunk=step+1,epoch=int(schedule['cycle'][step]))
            step+=1
            if step in freeze['reference_scheduler_boundaries']:sched.step()
            row.update(step=step,episode_ids=';'.join(map(str,ids)),start=start,
                cumulative_timesteps=total_frames,update_seconds=time.perf_counter()-tick,timestamp=time.time())
            append_csv(run/'train_metrics.csv',row)
            if step%100==0:
                print('TRAIN',condition,step,row['optimization_loss'],row['update_seconds'],flush=True)
                log.log(row,step=step);write(run/'progress.json',dict(step=step,total_frames=total_frames,wall_seconds=time.perf_counter()-start_clock))
            if step in proto['offline_validation_steps']:
                pred_dir=run/'validation_predictions'/f'step{step:06d}'
                val=offline(resident,model,norm,codec,cfg['splits']['validation'],cfg,'cuda',pred_dir)
                val.update(extra_metrics(pred_dir));val.update(step=step,condition=condition)
                append_csv(run/'validation_metrics.csv',val)
                original_ids=[int(k) for k,v in freeze['exposures'][condition].items() if v['data_kind']=='original']
                train_val=offline(resident,model,norm,codec,original_ids,cfg,'cuda')
                append_csv(run/'train_offline_metrics.csv',dict(step=step,**train_val))
                improved=progress['best_loss'] is None or val['optimization_loss']<progress['best_loss']
                if improved:progress.update(best_loss=val['optimization_loss'],best_step=step)
                receipt=save(run,root,model,opt,sched,progress,'E0');log.link_output(receipt['path'],kind='checkpoint')
                write(root/(condition+'_last.json'),receipt)
                if improved:write(root/(condition+'_selected.json'),receipt)
            if step in proto['live_development_steps']:
                receipt=save(run,root,model,opt,sched,progress,'E0')
                before=copy.deepcopy(dict(weights=model.state_dict(),optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),progress=progress,rng=rng_state(include_cuda=True)))
                subprocess.run([sys.executable,'-B','-m','graph.geometry_reach10d.evaluate','--root',str(root),
                    '--checkpoint',receipt['path'],'--condition',condition,'--mode','development'],check=True)
                assert_nested_equal(before,dict(weights=model.state_dict(),optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),progress=progress,rng=rng_state(include_cuda=True)))
            if (root/'STOP').exists():
                receipt=save(run,root,model,opt,sched,progress,'E0');write(root/(condition+'_last.json'),receipt)
                raise RuntimeError('cooperative STOP after checkpoint')
        assert total_frames==freeze['total_timesteps'][condition]
        result=dict(status='COMPLETE',condition=condition,updates=11487,total_timesteps=total_frames,
            initial_weight_sha256=initial,selected=read(root/(condition+'_selected.json')),
            last=read(root/(condition+'_last.json')),wall_seconds=time.perf_counter()-start_clock,
            run_id=log.run_id,run_directory=str(run),cuda_peak_bytes=torch.cuda.max_memory_allocated())
        write(root/(condition+'_complete.json'),result,immutable=True)
        log.record(result=result);print('TRAIN COMPLETE',condition,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--condition',choices=['A','B']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    smoke(a.root) if a.smoke else train(a.root,a.condition)
