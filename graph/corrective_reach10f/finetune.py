"""Fixed-budget warm-start R0/R1, reusing the original optimizer/update/save."""
import argparse,copy,os,time
from pathlib import Path
import numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,checkpoint_load,tensor_identity,append_csv
from graph.plan_a_bc.cache import CachedDataset
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.train import train_update,optimizer,save,assert_nested_equal
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.integration_baseline.direct_train import seed_all
from graph.corrective_reach10f.window_data import freeze,WindowResident,validate_windows
from utils.experiment_logger import ExperimentLogger


@torch.inference_mode()
def validation(ds,model,norm,codec):
    model.eval();stats={k:[0.,0] for k in ['nominal','recovery']}
    for eid,e in sorted(ds.episodes.items()):
        if e['split']!='validation':continue
        with np.load(e['supervision']['path']) as a:supervision=a['supervision_mask']
        hidden=None
        for start in range(0,e['actions'],32):
            g,s,target,valid=ds.batch([eid],start,32,'cuda');raw,hidden=model(g,s,norm,hidden);hidden=hidden.detach()
            mask=valid.clone();selected=torch.tensor(supervision[start:start+valid.shape[1]],device='cuda');mask[0,:len(selected)]&=selected
            if mask.any():
                count=int(mask.sum());stats[e['kind']][0]+=float(codec.loss(raw,target,mask))*count;stats[e['kind']][1]+=count
    assert all(v[1] for v in stats.values())
    values={k+'_loss':v[0]/v[1] for k,v in stats.items()};values['selection_loss']=sum(values.values())/2
    values.update({k+'_supervised_timesteps':v[1] for k,v in stats.items()});return values


def one(root,ds,manifest,condition,initial,config):
    seed_all(42,1);norm=normalizer(ds.root);codec=ActionCodec(read(ds.root/'action_codec.json'));cp=checkpoint_load(initial)
    model=CachedGraphPolicy(norm.schema,'E0',**config['model']).cuda();model.load_state_dict(cp['model'],strict=True)
    assert tensor_identity(model.state_dict())==read(root/'initial_checkpoint_identity.json')['model_sha256']
    opt,sched=optimizer(model,config);progress=dict(step=0,epoch=0,chunk=0,hidden={});begin=time.perf_counter()
    with ExperimentLogger(team='graph',model='Task10F_'+condition,seed=42,config=dict(comparison_uuid=read(root/'ab_identity.json')['run_id'],plan_sha256=sha(root/'full_experiment_plan.json'),sampling_sha256=sha(root/'sampling_manifest.json'),condition=condition,initial_checkpoint_sha256=sha(initial),optimizer_reset=True),root=root/'training_runs',job_type='train') as log:
        run=log.directory;log.link_output(initial,checkpoint=True);write(root/(condition+'_run.json'),dict(run_id=log.run_id,directory=str(run)),immutable=True)
        view=WindowResident(ds,manifest,condition)
        validate_windows(view,model,norm,sorted(view.episodes)[:4])
        receipt=save(run,ds.root,model,opt,sched,progress,'E0');log.link_output(receipt['path'],kind='checkpoint')
        initial_cp=checkpoint_load(receipt['path']);assert initial_cp['optimizer']['state']=={}
        metrics=validation(ds,model,norm,codec);append_csv(root/'validation_metrics.csv',dict(condition=condition,update=0,**metrics));best=metrics['selection_loss'];selected=receipt;checkpoints=[receipt]
        write(root/(condition+'_selected.json'),selected)
        exposure=0;burn=0;norm_exposure=0;rec_exposure=0
        for row in manifest['schedule']:
            step=row['update'];ids=row[condition];model.train();progress['hidden']={str(vid):view.current_hidden(model,norm,vid) for vid in ids}
            # Offset1 is only a window coordinate to invoke inherited hidden
            # carry. Actual source time and every preceding frame are retained.
            metrics=train_update(model,opt,view.masked,norm,codec,progress,ids,1,config,'cuda')
            assert metrics['valid_timesteps']==row['supervised_timesteps'];exposure+=metrics['valid_timesteps']
            for vid in ids:
                w=view.specs[vid];burn+=w['start'];kind=ds.episodes[w['episode_id']]['kind']
                if kind=='nominal':norm_exposure+=w['supervised']
                else:rec_exposure+=w['supervised']
            progress.update(step=step,epoch=step//60,chunk=step%60)
            if step%60==0:sched.step()
            append_csv(root/'train_metrics.csv',dict(condition=condition,update=step,burn_in_frames=burn,supervised_exposure=exposure,nominal_exposure=norm_exposure,recovery_exposure=rec_exposure,**metrics));log.log(metrics,step=step)
            if step in [500,1000,2000,3000]:
                receipt=save(run,ds.root,model,opt,sched,progress,'E0');log.link_output(receipt['path'],kind='checkpoint');checkpoints.append(receipt)
                val=validation(ds,model,norm,codec);append_csv(root/'validation_metrics.csv',dict(condition=condition,update=step,**val));log.log({'validation':val},step=step)
                if val['selection_loss']<best:best=val['selection_loss'];selected=receipt;write(root/(condition+'_selected.json'),selected)
                print('FINETUNE',condition,step,exposure,val['selection_loss'],flush=True)
        assert exposure==manifest['supervised_timesteps']
        if condition=='R1':assert norm_exposure==rec_exposure
        else:assert rec_exposure==0
        write(root/(condition+'_complete.json'),dict(status='PASS',updates=3000,supervised_timesteps=exposure,burn_in_frames=burn,nominal_exposure=norm_exposure,recovery_exposure=rec_exposure,selected=selected,checkpoints=checkpoints,wall_seconds=time.perf_counter()-begin),immutable=True)
        log.link_output(root/(condition+'_complete.json'),kind='metrics')
    del view,model,opt,sched;torch.cuda.empty_cache();return selected


def run(root):
    assert os.getenv('CONTAINER_ID')=='57640710';assert os.getenv('CUBLAS_WORKSPACE_CONFIG')==':4096:8';seed_all(42,1)
    assert read(root/'full_data_complete.json')['status']=='PASS';assert read(root/'matched_hold_v4/verification.json')['completed_branches']==40
    hold=read(root/'matched_hold_v4_exact_audit.json')
    assert hold['branches']==40 and hold['p_all_array_exact'] and hold['n_hold_all_under_1mm'] and hold['n_mask_all']
    assert read(root/'matched_hold_v4_completion_gate.json')['status']=='PASS'
    assert not (root/'matched_hold_v4/failure.json').exists()
    cache=root/'full_cache';assert read(cache/'corrective_cache_manifest.json')['status']=='PASS'
    cache_manifest=read(cache/'cache_manifest.json');assert cache_manifest['complete'] and len(cache_manifest['episodes'])==212
    assert sha(cache/'raw_content_manifest.json')==cache_manifest['raw_content_manifest_sha256']
    for episode in cache_manifest['episodes']:
        assert episode['readback']=='PASS' and sha(cache/episode['path'])==episode['sha256']
        assert sha(episode['raw_path'])==episode['raw_episode_sha256']
        side=episode['supervision'];assert side['raw_sha256']==episode['raw_episode_sha256'] and sha(side['path'])==side['sha256']
        with np.load(side['path']) as data:
            mask=data['supervision_mask'];assert len(mask)==episode['actions'] and int(mask.sum())==side['supervised']
            if episode['kind']=='nominal':assert mask.all()
            else:
                assert np.any(mask) and not mask[0]
    config=read(cache/'config.json');config['training'].update(updates=3000,scheduler_clock='Cosine T_max50; advance every60 optimizer updates',validation_best='first minimum balanced nominal/recovery development Method B loss')
    write(cache/'config.json',config)
    ds=CachedDataset(cache);manifest=freeze(root,ds);initial=root/'initial_B_step006000.pt';initial_sha=sha(initial)
    assert initial_sha==read(root/'initial_checkpoint_identity.json')['sha256']
    train_parents={r['parent'] for r in ds.episodes.values() if r['split']=='train'};dev_parents={r['parent'] for r in ds.episodes.values() if r['split']=='validation'};assert not train_parents&dev_parents
    with ExperimentLogger(team='graph',model='Task10F_R0_R1_finetune_comparison',seed=42,config=dict(plan=read(root/'full_experiment_plan.json'),sampling_sha256=sha(root/'sampling_manifest.json'),cache_manifest_sha256=sha(cache/'cache_manifest.json')),root=root/'experiments',job_type='train') as log:
        write(root/'ab_identity.json',dict(run_id=log.run_id,parent_pilot=read(root/'pilot03/identity.json')['run_id']),immutable=True)
        sources=[Path(__file__),Path(__file__).with_name('window_data.py'),Path(__file__).with_name('masked.py')]
        sources += [Path(__file__).resolve().parents[1]/'plan_a_bc'/n for n in ['model.py','train.py','cache.py','geometry.py']]
        write(root/'ab_training_config.json',dict(plan_sha256=sha(root/'full_experiment_plan.json'),config=config,config_sha256=sha(cache/'config.json'),sampling_sha256=sha(root/'sampling_manifest.json'),initial_sha256=initial_sha,source_sha256={str(p):sha(p) for p in sources}),immutable=True)
        selected={c:one(root,ds,manifest,c,initial,config) for c in ['R0','R1']}
        a,b=[read(root/(c+'_complete.json')) for c in ['R0','R1']];assert a['updates']==b['updates']==3000 and a['supervised_timesteps']==b['supervised_timesteps'];assert sha(initial)==initial_sha
        # Compare actual saved initial weights and fresh optimizer/scheduler.
        checkpoints=[checkpoint_load(read(root/(c+'_complete.json'))['checkpoints'][0]['path']) for c in ['R0','R1']]
        for k in ['model','optimizer','scheduler']:assert_nested_equal(checkpoints[0][k],checkpoints[1][k])
        write(root/'model_selection_frozen.json',dict(selected=selected,plan_sha256=sha(root/'full_experiment_plan.json'),final_cases=read(root/'full_experiment_plan.json')['final_cases'],no_final_access_yet=True),immutable=True)
        write(root/'checkpoint_manifest.json',{c:read(root/(c+'_complete.json'))['checkpoints'] for c in ['R0','R1']},immutable=True)
        log.link_output(root/'model_selection_frozen.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
