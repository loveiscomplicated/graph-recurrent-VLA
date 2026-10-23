"""Fixed 30-epoch v2 Flat BC; validation-only selection, one final test read."""
from utils.experiment_logger import ExperimentLogger
import argparse
from pathlib import Path
from time import perf_counter
import torch

from graph.prototype.common import config, seed_all, versions, write_json
from graph.prototype.data import load_dataset, split_indices, sha256
from graph.prototype.metrics import action_metrics
from graph.prototype.models import parameter_count
from graph.prototype.v2.models import CommonPolicy, bc_loss
from .policy import DATASET, ROOT, V2_CONFIG, fit_inputs, inputs


def setup(dataset):
    seed_all(0,4)
    data, g = load_dataset(dataset)
    cfg = config(V2_CONFIG)
    if not data['manifest'].get('trimming'):
        raise ValueError('use verified first-success dataset')
    for key in ('peg_points','hole_points','sampling_seed','control_mode'):
        if cfg[key] != data['manifest']['config'][key]:
            raise ValueError('dataset config mismatch')
    norm, sn = fit_inputs(data,g)
    x, s = inputs(g,data['qvel'],norm,sn)
    model = CommonPolicy('flat',cfg,'goal','regression')
    return data,g,cfg,norm,sn,x,s,model


def smoke(dataset, output):
    data,g,cfg,norm,sn,x,s,model = setup(dataset)
    ids = split_indices(data,'train')[:32]
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    with torch.no_grad():
        initial = float(bc_loss(model(x[ids],s[ids]),data['actions'][ids]))
    losses = []
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = bc_loss(model(x[ids],s[ids]),data['actions'][ids])
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('nonfinite loss/gradients')
        optimizer.step(); losses.append(float(loss.detach()))
    with torch.no_grad():
        pred = model(x[ids],s[ids]); final = float(bc_loss(pred,data['actions'][ids]))
    if final >= initial*.1:
        raise AssertionError('small-batch overfit failed')
    path = Path(output)/'reports/smoke_checkpoint.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path)
    clone=CommonPolicy('flat',cfg,'goal','regression')
    clone.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(clone(x[ids],s[ids]),pred,atol=0,rtol=0)
    report=dict(batch_size=len(ids),iterations=300,initial_mse=initial,final_mse=final,
                finite_gradients=True,checkpoint_roundtrip_exact=True,parameters=parameter_count(model),
                dataset_sha256=sha256(dataset),losses=losses,versions=versions())
    write_json(Path(output)/'reports/training_smoke.json',report)
    print({k:v for k,v in report.items() if k!='losses'})
    return report


def train(dataset, output):
    output=Path(output)
    if (output/'checkpoints/best.pt').exists():
        raise FileExistsError('choose a new experiment directory')
    smoke_report=output/'reports/training_smoke.json'
    audit_report=output/'reports/trimming_audit.json'
    import json
    if not smoke_report.exists() or not audit_report.exists() or json.loads(smoke_report.read_text())['dataset_sha256']!=sha256(dataset):
        raise ValueError('run replay audit and training smoke first')
    data,g,cfg,norm,sn,x,s,model=setup(dataset)
    with ExperimentLogger(team='graph', config=dict(parameters=cfg, dataset=dataset, output=output), seed=cfg['seed'], model='CommonPolicy/flat') as experiment:
        experiment.record(dataset=str(Path(dataset).resolve()))
        experiment.record(dataset_sha256=sha256(dataset), environment=data['manifest']['config'])
        train_ids,val_ids=split_indices(data,'train'),split_indices(data,'validation')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'])
        generator=torch.Generator().manual_seed(cfg['seed'])
        best=float('inf');history=[];start=perf_counter()
        (output/'checkpoints').mkdir(parents=True,exist_ok=True)
        specification=dict(model='graph.prototype.v2.models.CommonPolicy',encoder='unchanged FlatEncoder',
                            state_mode='goal',gripper_mode='regression',flat_features=x.shape[1],shared_features=s.shape[1],
                            encoder_dimensions=[x.shape[1],128,128,128],head_dimensions=[159,128,7],
                            config=cfg,normalization='train-only FeatureNormalizer + StateNormalizer, population std floor .001',
                            loss='unweighted 7D normalized action MSE',device='cpu',parameters=parameter_count(model),
                            scheduler='CosineAnnealingLR T_max=30',selection='best validation MSE',splits=data['manifest']['splits'])
        write_json(output/'training/config.json',specification)
        torch.save(dict(geometry=norm.as_dict(),state=sn.as_dict(),train_episode_ids=data['manifest']['splits']['train']),output/'training/normalization.pt')
        experiment.link_output(output / 'training/normalization.pt')
        for epoch in range(1,cfg['epochs']+1):
            es=perf_counter();model.train();predictions=[];targets=[]
            order=train_ids[torch.randperm(len(train_ids),generator=generator)]
            for ids in order.split(cfg['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                raw=model.raw_action(x[ids],s[ids]);target=data['actions'][ids]
                loss=bc_loss(raw,target,'regression')
                loss.backward()
                if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite BC loss/gradients')
                optimizer.step();predictions.append(raw.detach());targets.append(target)
            model.eval()
            with torch.inference_mode():
                val_pred=torch.cat([model(x[ids],s[ids]) for ids in val_ids.split(cfg['batch_size'])])
            tm=action_metrics(torch.cat(predictions),torch.cat(targets));vm=action_metrics(val_pred,data['actions'][val_ids])
            row=dict(epoch=epoch,train_mse=tm['action_mse'],val_mse=vm['action_mse'],train_metrics=tm,validation_metrics=vm,
                     learning_rate=optimizer.param_groups[0]['lr'],seconds=perf_counter()-es,parameters=parameter_count(model))
            history.append(row);scheduler.step()
            if row['val_mse']<best:
                best=row['val_mse']
                torch.save(dict(schema_version=2,model_name='flat',config=cfg,state_mode='goal',gripper_mode='regression',
                                model_state=model.state_dict(),normalizer=norm.as_dict(),state_normalizer=sn.as_dict(),
                                optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
                                epoch=epoch,validation_mse=best,dataset_sha256=sha256(dataset),splits=data['manifest']['splits'],
                                versions=versions(),parameters=parameter_count(model)),output/'checkpoints/best.pt')
                experiment.link_output(output / 'checkpoints/best.pt')
            write_json(output/'training/history.json',history)
            experiment.log_epoch(row)
            print(f'epoch={epoch} train={row["train_mse"]:.7f} validation={row["val_mse"]:.7f} seconds={row["seconds"]:.2f}',flush=True)
        ck=torch.load(output/'checkpoints/best.pt',weights_only=True);model.load_state_dict(ck['model_state']);model.eval()
        with torch.inference_mode():
            # Only after all epochs and the validation-selected checkpoint is fixed.
            test_ids=split_indices(data,'test')
            test=action_metrics(model(x[test_ids],s[test_ids]),data['actions'][test_ids])
            final_train=action_metrics(model(x[train_ids],s[train_ids]),data['actions'][train_ids])
        report=dict(**specification,training_seconds=perf_counter()-start,versions=versions(),best_epoch=ck['epoch'],
                    best_validation_mse=best,test_metrics=test,selected_train_metrics=final_train,history=history,
                    train_samples=len(train_ids),validation_samples=len(val_ids),test_samples=len(test_ids),
                    dataset_sha256=sha256(dataset),checkpoint_sha256=sha256(output/'checkpoints/best.pt'),
                    converged_at_budget=not all(history[i]['val_mse']<history[i-1]['val_mse'] for i in range(len(history)-5,len(history))))
        write_json(output/'training/training.json',report)
        experiment.log(report, prefix='summary')
        experiment.link_output(output / 'training/training.json')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (output/'figures').mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for key,label in [('train_mse','train'),('val_mse','validation')]:
            axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],label=label)
        axes[0].set(xlabel='Epoch',ylabel='7D action MSE',yscale='log');axes[0].legend()
        for key in ('translation_rmse','rotation_rmse','gripper_mae'):
            axes[1].plot([r['epoch'] for r in history],[r['validation_metrics'][key] for r in history],label=key)
        axes[1].set(xlabel='Epoch',ylabel='Normalized validation error');axes[1].legend()
        fig.tight_layout();fig.savefig(output/'figures/loss_components.png',dpi=160);plt.close(fig)
        print('SELECTED',ck['epoch'],'TEST',test,flush=True)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATASET);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();(smoke if a.smoke else train)(a.dataset,a.output)
