"""Train on mixed fixed and moving GT20; compare on held-out paired seeds."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from collect_gt20 import FIRST_SEED,SPLIT_SIZES,END
from train_gt20 import read_collection,write_checkpoint,predict,error_metrics,groups,describe

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn import run_epoch
from utils.experiment_logger import ExperimentLogger,write


def read_moving(root):
    batches,sources,terms=[],[],{}
    for start in range(FIRST_SEED,END,44):
        folder=root/f'data_{start}_{start+44}'
        cfg=json.loads((folder/'summary.json').read_text())
        meta=json.loads((folder/'experiments'/cfg['run_id']/'metadata.json').read_text())
        path=folder/'dataset.npz'
        assert meta['status']=='finished'
        assert file_sha256(path)==cfg['dataset_sha256']
        with np.load(path,allow_pickle=False) as file:
            a={k:file[k] for k in file.files}
        assert np.unique(a['seed']).tolist()==list(range(start,start+44))
        assert a['xyz'].shape==(len(a['seed']),20,3) and a['target'].shape==(len(a['seed']),21)
        assert np.all(a['kinds']==[0]*8+[1]*12)
        batches.append(a)
        sources.append(cfg)
        terms.update(cfg['terminals'])
    return {k:np.concatenate([a[k] for a in batches]) for k in batches[0]},sources,terms


def metrics(data, terminal, predictions):
    truth=torch.from_numpy(data['target'])
    evaluated={k:error_metrics(v,truth) for k,v in predictions.items()}
    masks=groups(data,terminal)
    if 'offset_m' in data:
        masks['while_box_moving']=data['moving_now']
        masks['after_box_motion']=(~data['moving_now'])&(np.linalg.norm(data['offset_m'],axis=1)>1e-6)
    return {name:dict(frames=int(mask.sum()),episodes=len(np.unique(data['seed'][mask])),
                      **{k:describe(v,mask) for k,v in evaluated.items()})
            for name,mask in masks.items()}


def main():
    p=argparse.ArgumentParser()
    for field in ('fixed','moving','output'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    args=p.parse_args()
    fixed,fs,ft=read_collection(args.fixed)
    moving,ms,mt=read_moving(args.moving)
    for key in ('xyz','tcp','target'):
        np.testing.assert_allclose(fixed[key][fixed['step']==0],moving[key][moving['step']==0],
                                   atol=1e-6,rtol=0)
    splits=FIRST_SEED+np.cumsum((0,*SPLIT_SIZES))
    graph_fields=('xyz','kinds','tcp','target')
    train_end,test_start=splits[1:3]
    train_fixed=(fixed['seed']<train_end)&(fixed['seed']%2==1)
    train_moving=(moving['seed']<train_end)&(moving['seed']%2==0)
    val_fixed=(train_end<=fixed['seed'])&(fixed['seed']<test_start)
    val_moving=(train_end<=moving['seed'])&(moving['seed']<test_start)
    assert len(np.unique(fixed['seed'][train_fixed]))==len(np.unique(moving['seed'][train_moving]))==60
    cols=[(fixed,train_fixed),(moving,train_moving),(fixed,val_fixed),(moving,val_moving)]
    n_train=int(train_fixed.sum()+train_moving.sum())
    n_val=int(val_fixed.sum()+val_moving.sum())
    device=torch.device('cuda:0')
    def to_tensors(data):
        out={k:torch.as_tensor(data[k],device=device) for k in graph_fields}
        out['valid']=torch.ones_like(out['kinds'],dtype=torch.bool)
        out['observed']=out['valid']
        return out
    inputs=to_tensors({k:np.concatenate([d[k][mask] for d,mask in cols]) for k in graph_fields})
    ckpath=args.fixed/'train/best.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    previous=ck['config']
    settings=dict(previous['training'],epochs=args.epochs,log_every_batches=100000)
    conf=dict(gnn=previous['gnn'],loss=previous['loss'],training=settings,seed=0,
              device=str(device),split_seed_ranges=splits.tolist(),fixed_checkpoint_sha256=file_sha256(ckpath),
              fixed_sources=[{'sha256':s['dataset_sha256'],'run_id':s['run_id']} for s in fs],
              moving_sources=[{'sha256':s['dataset_sha256'],'run_id':s['run_id']} for s in ms],
              training_frames={'fixed':int(train_fixed.sum()),'moving':int(train_moving.sum()),
                               'moving_with_translation':int((train_moving&moving['moving_now']).sum())},
              validation_frames={'fixed':int(val_fixed.sum()),'moving':int(val_moving.sum())},
              script_sha256=file_sha256(__file__))
    torch.set_num_threads(settings['cpu_threads'])
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    rng=np.random.default_rng(0)
    model=VertexEdgeGatedMPNN(**conf['gnn']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    val=torch.arange(n_train,n_train+n_val,device=device)
    args.output.mkdir(parents=True,exist_ok=False)
    history=[]
    best=(float('inf'),0)
    start=monotonic()
    with ExperimentLogger(team='graph',model='moving_box_gt20_mixed_training',config=conf,
                          seed=0,root=args.output/'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
        for i in range(1,args.epochs+1):
            t=monotonic()
            tr=run_epoch(model,inputs,torch.as_tensor(rng.permutation(n_train),device=device),
                         conf,optimizer,i,logger,args.output)
            va=run_epoch(model,inputs,val,conf,None,i,logger,args.output)
            if va['loss']<best[0]:
                best=(va['loss'],i)
                write_checkpoint(model,i,logger,conf,args.output/'best.pt')
            history.append(dict(epoch=i,train=tr,validation=va,seconds=monotonic()-t))
            write(args.output/'history.json',history)
            print(f'epoch {i}/{args.epochs} train={tr["loss"]:.6f} val={va["loss"]:.6f} best={best[1]} '
                  f'epoch_s={monotonic()-t:.1f}',flush=True)
        state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True)
        model.eval()
        old=VertexEdgeGatedMPNN(**conf['gnn']).to(device)
        old.load_state_dict(ck['model'],strict=True)
        old.eval()
        result={}
        for name,group,term in [('fixed',fixed,ft),('moving',moving,mt)]:
            mask=group['seed']>=test_start
            test={k:v[mask] for k,v in group.items()}
            tensors=to_tensors(test)
            indices=torch.arange(int(mask.sum()),device=device)
            preds={k:predict(m,tensors,indices,settings['batch_size'])
                   for k,m in (('fixed_model',old),('mixed_model',model))}
            result[name]=metrics(test,term,preds)
            tosave={k:test[k] for k in ('seed','step','phase','grasp','target')}
            if name=='moving':
                tosave.update(offset_m=test['offset_m'],moving_now=test['moving_now'])
            np.savez_compressed(args.output/f'test_{name}.npz',**tosave,**{k:v.numpy() for k,v in preds.items()})
        summary=dict(run_id=logger.run_id,best_epoch=best[1],validation_loss=best[0],config=conf,
                     seconds=monotonic()-start,test=result,
                     moving_test_manifest={s:{**m,'terminal':mt[s]} for page in ms for s,m in page['motions'].items()
                                           if int(s)>=test_start})
        write(args.output/'summary.json',summary)
        logger.log(result,prefix='eval/test')
        for name in ('best.pt','summary.json','history.json','test_fixed.npz','test_moving.npz'):
            logger.link_output(args.output/name)
        print('Completed',logger.run_id,flush=True)


if __name__=='__main__':
    main()
