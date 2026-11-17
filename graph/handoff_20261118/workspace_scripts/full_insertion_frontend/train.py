"""Fine-tune the fixed-ID Transformer and then the Object21 GNN on full insertion."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from collect import VAL, TEST
from data_collection.episode_dataset import file_sha256
from evaluate_hybrid import apply_existing_correction
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train_finetune import prepare_tensors, train as fit_gnn
from train_gt20 import describe, error_metrics, predict
from train_tracker import CornerTransformer, batches, evaluate, model_args, prepare, test_causality
from utils.experiment_logger import ExperimentLogger, write

BASE = Path('/workspace')
TRACKER = BASE/'spatiotemporal_corner_tracker_convergence/train/spatial_temporal/best.pt'
GNN = BASE/'hybrid_input_gnn_finetune/train/best.pt'
RATES = (1e-4, 3e-5, 1e-5)
PATIENCE = 40
MAX_EPOCHS = 220


def load(root):
    sources, shards = [], []
    for folder in sorted(root.glob('shard_*')):
        if not folder.is_dir():
            continue
        summary = json.loads((folder/'summary.json').read_text())
        path = folder/'data.npz'
        digest = file_sha256(path)
        assert summary['sha256'] == digest
        metadata = json.loads((folder/'experiments'/summary['run_id']/'metadata.json').read_text())
        assert metadata['status']=='finished' and not metadata['git_dirty']
        sources.append(dict(run_id=summary['run_id'], sha256=digest))
        with np.load(path,allow_pickle=False) as part:
            shards.append(dict(part))
        assert len(shards[-1]['seed']) == summary['frames']
    assert len(shards)==10
    data={key:np.concatenate([part[key] for part in shards]) for key in shards[0]}
    ids = np.unique(data['seed'])
    indices=[ids[ids<VAL],ids[(ids>=VAL)&(ids<TEST)],ids[ids>=TEST]]
    for seed in ids:
        mask=data['seed']==seed
        run=data['step'][mask]
        np.testing.assert_array_equal(run,np.arange(len(run)))
        assert data['phase'][mask][-1]=='success' and data['depth_mm'][mask].max()>20
    for split in indices:
        assert len(split)>=20
    return data,indices,sources


def metrics(pred, data, selected):
    distance=np.linalg.norm(pred-data['xyz'][selected],axis=-1)*1000
    missing=~data['observed'][selected]
    inside=data['depth_mm'][selected]>0
    before=distance[missing&~inside[:,None]]
    after=distance[missing&inside[:,None]]
    score=(before.mean()+after.mean())/2
    return dict(before_mean=float(before.mean()), inside_mean=float(after.mean()), score=float(score))


def fit_tracker(data, indices, args, sources, device):
    initial = torch.load(TRACKER,map_location='cpu',weights_only=True)
    assert initial['epoch']==954
    fallback = initial['fallback'].numpy()
    train, val = [prepare(data,fallback,ids,device) for ids in indices[:2]]
    selected=np.isin(data['seed'],indices[1])
    model=CornerTransformer(spatial=True).to(device)
    model.load_state_dict(initial['model'],strict=True)
    old_model=CornerTransformer(spatial=True).to(device)
    old_model.load_state_dict(initial['model'],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=RATES[0],weight_decay=.01)
    rng=torch.Generator(device=device).manual_seed(0)
    folder=args.output/'tracker'
    folder.mkdir()
    cfg=dict(initial['config'], initial_sha256=file_sha256(TRACKER), sources=sources,
             script_sha256=file_sha256(__file__), objective='missing-only Smooth L1',
             selection='mean of validation pre-entry and inside missing-point means',
             rates=RATES, max_epochs=MAX_EPOCHS, patience=PATIENCE, min_gain_mm=.05)
    with ExperimentLogger(team='graph',model='full_insertion_transformer',job_type='train',
                          config=cfg,seed=0,root=folder/'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name()))
        test_causality(model,val)
        best=anchor=metrics(evaluate(model,val,6),data,selected)['score']
        best_epoch=last_gain=rate=0
        history=[]
        path=folder/'best.pt'

        def save(epoch):
            tmp=path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),epoch=epoch,config=cfg,run_id=logger.run_id,
                            fallback=initial['fallback']),tmp)
            tmp.replace(path)

        save(0)
        stop='epoch_limit'
        for epoch in range(MAX_EPOCHS+1):
            if epoch:
                model.train()
                for batch in batches(train,torch.randperm(len(train['gt']),device=device,generator=rng),6):
                    missing=batch['valid'][...,None]&~batch['observed']
                    pred=model(*model_args(batch))
                    loss=F.smooth_l1_loss(pred[missing]*100,batch['gt'][missing]*100)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step()
            val_metrics=metrics(evaluate(model,val,6),data,selected)
            score=val_metrics['score']
            history.append(dict(epoch=epoch,lr=RATES[rate],**val_metrics))
            if score<best:
                best,best_epoch=score,epoch
                save(epoch)
            if score<=anchor-.05:
                anchor,last_gain=score,epoch
            logger.log(val_metrics,step=epoch)
            if epoch%10==0 or epoch==best_epoch:
                print('Transformer',epoch,val_metrics,'best',best_epoch,flush=True)
            if epoch-last_gain>=PATIENCE:
                if rate+1==len(RATES):
                    stop='plateau_at_lowest_lr'
                    break
                rate+=1
                opt.param_groups[0]['lr']=RATES[rate]
                anchor,last_gain=best,epoch
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        test_causality(model,val)
        result=folder/'summary.json'
        write(result,dict(run_id=logger.run_id,config=cfg,best_epoch=best_epoch,final_epoch=epoch,
                          stop=stop,history=history,sha256=file_sha256(path)))
        logger.link_output(result)
        logger.link_output(path,checkpoint=True)
    return old_model,model,fallback


def predict_coordinates(model,data,device,fallback):
    ids=np.unique(data['seed'])
    output=evaluate(model,prepare(data,fallback,ids,device),6)
    observed=data['observed']
    np.testing.assert_array_equal(output[observed],data['detected_xyz'][observed])
    seen=np.zeros_like(observed)
    for seed in ids:
        ix=np.flatnonzero(data['seed']==seed)
        seen[ix]=np.logical_or.accumulate(observed[ix],axis=0)
    corrected,*_=apply_existing_correction(dict(data,spatial_temporal=output,ever=seen))
    np.testing.assert_array_equal(corrected[observed],data['detected_xyz'][observed])
    return output,corrected


def heldout(output,data, old_tracker,new_tracker,old_gnn,new_gnn,device):
    selected=data['seed']>=TEST
    keys=('xyz','observed','target','tcp','seed','step','depth_mm','offset_m','phase')
    test={key:data[key][selected] for key in keys}
    old_raw,old_xyz=(part[selected] for part in old_tracker)
    new_raw,new_xyz=(part[selected] for part in new_tracker)
    coords={'old':old_xyz,'new':new_xyz}
    tensors={key:prepare_tensors(dict(test,xyz=xyz),device) for key,xyz in coords.items()}
    models={'old_front':(old_gnn,'old'),'new_tracker_only':(old_gnn,'new'),'new_front':(new_gnn,'new')}
    index=torch.arange(len(test['seed']),device=device)
    predicted={key:predict(model,tensors[name],index,96).numpy() for key,(model,name) in models.items()}
    depth=test['depth_mm']
    bands=dict(all=np.ones(len(depth),bool),pre=depth<=0,shallow=(depth>0)&(depth<=20),deep=depth>20)
    results={}
    for mode,(model,key) in models.items():
        diff=torch.as_tensor(predicted[mode])
        errors=error_metrics(diff,torch.as_tensor(test['target']))
        point=np.linalg.norm(coords[key]-test['xyz'],axis=-1)*1000
        results[mode]={}
        for band,mask in bands.items():
            unseen=~test['observed'][mask]
            peg=point[mask,:8][unseen[:,:8]]
            all_missing=point[mask][unseen]
            results[mode][band]=dict(frames=int(mask.sum()),
                                     episodes=len(np.unique(test['seed'][mask])),
                                     missing_p90_mm=float(np.percentile(all_missing,90)),
                                     missing_peg_p90_mm=float(np.percentile(peg,90)),
                                     **describe(errors,mask))
    np.savez_compressed(output/'heldout.npz',**test,**predicted,
                        old_raw=old_raw,new_raw=new_raw,old_xyz=old_xyz,new_xyz=new_xyz)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device=torch.device('cuda:0')
    data,splits,sources=load(args.input)
    args.output.mkdir(parents=True,exist_ok=False)
    print('Successful episodes by split',[(len(ids),int(np.isin(data['seed'],ids).sum())) for ids in splits],flush=True)
    old,new,fallback=fit_tracker(data,splits,args,sources,device)
    old_tracker=predict_coordinates(old,data,device,fallback)
    new_tracker=predict_coordinates(new,data,device,fallback)
    initial=torch.load(GNN,map_location='cpu',weights_only=True)
    assert initial['epoch']==54
    cfg={key:initial['config'][key] for key in ('gnn','loss','training')}
    cfg.update(initial_sha256=file_sha256(GNN), tracker_sha256=file_sha256(args.output/'tracker'/'best.pt'),
               sources=sources,script_sha256=file_sha256(__file__), validation_boundary=VAL, test_boundary=TEST)
    # The existing GNN trainer reads its fixed boundaries from its own module.
    fit_gnn.__globals__['VAL'],fit_gnn.__globals__['TEST']=VAL,TEST
    args.output=args.output/'gnn'
    args.output.mkdir()
    tensors=prepare_tensors(dict(data,xyz=new_tracker[1]),device)
    with ExperimentLogger(team='graph',model='full_insertion_object21',job_type='train',
                          config=cfg,seed=0,root=args.output/'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name()))
        trained,stopping=fit_gnn(args,data,tensors,initial,cfg,logger)
        old_gnn=VertexEdgeGatedMPNN(**cfg['gnn']).to(device).eval()
        old_gnn.load_state_dict(initial['model'],strict=True)
        trained.eval()
        results=heldout(args.output,data,old_tracker,new_tracker,old_gnn,trained,device)
        summary=args.output/'summary.json'
        write(summary,dict(run_id=logger.run_id,config=cfg,stopping=stopping,results=results,
                          split_seeds=[ids.tolist() for ids in splits],
                          best_sha256=file_sha256(args.output/'best.pt'),
                          heldout_sha256=file_sha256(args.output/'heldout.npz')))
        logger.link_output(summary)
        for name in ('best.pt','resume.pt','history.json','heldout.npz'):
            logger.link_output(args.output/name,checkpoint=name.endswith('.pt'))
    print('Completed full insertion Transformer and GNN',flush=True)


if __name__=='__main__':
    main()
