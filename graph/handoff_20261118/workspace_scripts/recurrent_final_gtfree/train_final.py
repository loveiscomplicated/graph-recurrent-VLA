"""Fine-tune the five original Mamba models on new-Graph/GT paired BC episodes."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

import exp2_train as original
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write

ALPHAS = (1.,)
VAL = 3100120
STEPS = 1000
BATCH = 8
INTERVAL = 50
PATIENCE = 5
MIN_GAIN = .005
RATES = (8e-5, 2.5e-5)


def data_episodes(root, stats):
    data, sources = [], []
    for path in sorted(root.glob('shard_*/demos.npz')):
        summary = json.loads(path.with_name('summary.json').read_text())
        digest = file_sha256(path)
        assert summary['sha256'] == digest
        meta = json.loads((path.parent / 'experiments' / summary['run_id'] / 'metadata.json').read_text())
        assert meta['status'] == 'finished' and meta['git_commit'].startswith('b2e8e4c')
        sources.append(dict(run_id=summary['run_id'], sha256=digest))
        with np.load(path, allow_pickle=False) as d:
            off = d['episode_offsets']
            y = d['expert_action']
            gt, g = d['state36_gt'], d['state36_transformer']
            np.testing.assert_array_equal(gt[:,:15],g[:,:15])
            assert len(gt) == off[-1] == len(y)
            for seed, a, b in zip(d['reset_seed'], off[:-1], off[1:], strict=True):
                assert (d['phase'][a:b] == 'insert').any()
                moved = bool((np.linalg.norm(d['box_offset'][a:b], axis=1) > .001).any())
                levels = []
                for alpha in ALPHAS:
                    assert alpha == 1
                    x = g[a:b]
                    f = original.features(x, hole_frame=True, r6=True)
                    assert f.shape == (b-a,44) and np.isfinite(f).all()
                    levels.append(original.prepare([(f,y[a:b])],stats)[0])
                data.append((int(seed),moved,levels))
    assert len(sources) == 8 and len({d[0] for d in data}) == len(data)
    groups = {(phase,moved): [levels for seed, was_moved, levels in data
                              if (seed >= VAL) == (phase == 'validation') and was_moved == moved]
              for phase in ('train','validation') for moved in (False,True)}
    assert all(len(g) >= (7 if k[0]=='validation' else 30) for k,g in groups.items())
    return groups, sources


@torch.inference_mode()
def validation(model, groups, device):
    model.eval()
    results = {}
    for moved in (False,True):
        sequences = [levels[-1] for levels in groups['validation',moved]]
        count, loss = 0., 0.
        for i in range(0,len(sequences),BATCH):
            x, y, grip, mask = original.pad(sequences[i:i+BATCH],1,device)
            prediction, _ = model(x)
            n = float(mask.sum().item())
            loss += float(original.loss_fn(prediction,y,grip,mask,1).item())*n
            count += n
        results['moving' if moved else 'fixed'] = loss/count
    results['balanced'] = (results['fixed']+results['moving'])/2
    return results


def main():
    p = argparse.ArgumentParser()
    for name in ('data','prior','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    device=torch.device('cuda:0')
    prior=torch.load(a.prior,map_location='cpu',weights_only=True)
    assert prior['seed']==a.seed and prior['chunk']==1 and prior['rot6'] and prior['obs_dim']==44 and prior['hole_frame']
    assert prior['mix']=='0,0.25,0.5,0.75,1'
    stats={key:np.asarray(v,dtype=np.float32) for key,v in prior['stats'].items()}
    groups,sources=data_episodes(a.data,stats)
    model=original.build('mamba',44,7).to(device)
    model.load_state_dict(prior['state_dict'],strict=True)
    opt=torch.optim.Adam(model.parameters(),lr=RATES[0])
    rng=np.random.default_rng(a.seed)
    cfg=dict(sources=sources, prior_sha256=file_sha256(a.prior), alpha=ALPHAS,batch=BATCH,
             steps=STEPS, interval=INTERVAL,patience_checks=PATIENCE,minimum_gain=MIN_GAIN,
             init_lr=RATES[0],final_lr=RATES[-1],script_sha256=file_sha256(__file__),
             normalization='frozen original checkpoint', filter=False,
             counts={f'{phase}_{moved}':len(eps) for (phase,moved),eps in groups.items()},
             selection='Graph-only BC loss, equal fixed-moving weights')
    a.output.mkdir(parents=True,exist_ok=False)
    started=monotonic()
    with ExperimentLogger(team='recurrent',model='mamba_final_gtfree',job_type='train',
                          seed=a.seed,config=cfg,root=a.output/'experiments') as log:
        log.record(runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
        best_path=a.output/'last.pt'

        def save(step):
            ck=dict(prior,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    kin=False,steps=step,run_id=log.run_id,training_config=cfg)
            tmp=best_path.with_suffix('.tmp')
            torch.save(ck,tmp)
            tmp.replace(best_path)

        val=validation(model,groups,device)
        hist=[dict(step=0,validation=val)]
        best_loss=threshold=val['balanced']
        best_step=last_gain=rate_idx=0
        stop='budget'
        save(0)
        log.log(val,step=0,prefix='validation')
        print(a.seed, 'baseline', val, flush=True)
        losses=[]
        for step in range(1,STEPS+1):
            model.train()
            batch=[]
            for moved in (False,True):
                pool=groups['train',moved]
                ix=rng.choice(len(pool),BATCH//2,replace=False)
                levels=rng.integers(len(ALPHAS),size=len(ix))
                batch.extend(pool[i][j] for i,j in zip(ix,levels,strict=True))
            x,y,grip,mask=original.pad(batch,1,device)
            pred,_=model(x)
            loss=original.loss_fn(pred,y,grip,mask,1)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            losses.append(loss.detach())
            if step % INTERVAL:
                continue
            val=validation(model,groups,device)
            avg=float(torch.stack(losses).mean().item())
            losses.clear()
            hist.append(dict(step=step,train_loss=avg,validation=val,lr=RATES[rate_idx]))
            log.log({'loss':avg},step=step,prefix='train')
            log.log(val,step=step,prefix='validation')
            if val['balanced']<best_loss:
                best_loss,best_step = val['balanced'],step
                save(step)
            print(a.seed,step,f'train={avg:.5f}',val,f'best={best_step}',f'{monotonic()-started:.1f}s',flush=True)
            if val['balanced'] <= threshold*(1-MIN_GAIN):
                threshold,last_gain=val['balanced'],step
            if step-last_gain >= PATIENCE*INTERVAL:
                if rate_idx+1 == len(RATES):
                    stop='plateau'
                    break
                rate_idx+=1
                opt.param_groups[0]['lr']=RATES[rate_idx]
                threshold,last_gain=best_loss,step
        path=a.output/'summary.json'
        write(path,dict(run_id=log.run_id,config=cfg,history=hist,best_step=best_step,best_loss=best_loss,
                        final_step=step,stop=stop,seconds=monotonic()-started,checkpoint_sha256=file_sha256(best_path)))
        log.link_output(path)
        log.link_output(best_path,checkpoint=True)
    print('Completed Mamba training',a.seed,flush=True)


if __name__=='__main__':
    main()
