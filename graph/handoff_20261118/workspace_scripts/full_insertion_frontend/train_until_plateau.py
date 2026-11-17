"""Fine-tune the fixed-ID Transformer and then the Object21 GNN on full insertion."""
import argparse
import json
from pathlib import Path
from time import sleep

import numpy as np
import torch
from torch.nn import functional as F

from collect import VAL, TEST
from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train import load, metrics, predict_coordinates, heldout
from train_finetune import prepare_tensors, train as fit_gnn
from train_tracker import CornerTransformer, batches, evaluate, model_args, prepare, test_causality
from utils.experiment_logger import ExperimentLogger, write

BASE = Path('/workspace')
TRACKER = BASE/'spatiotemporal_corner_tracker_convergence/train/spatial_temporal/best.pt'
GNN = BASE/'hybrid_input_gnn_finetune/train/best.pt'
RATES = (5e-5, 2e-5, 7e-6)
PATIENCE = 55
MAX_EPOCHS = 600


def fit_tracker(data, indices, args, sources, device):
    original = torch.load(TRACKER,map_location='cpu',weights_only=True)
    initial_path=args.input/'trained/tracker/best.pt'
    prior_summary=json.loads(initial_path.with_name('summary.json').read_text())
    assert file_sha256(initial_path)==prior_summary['sha256']
    initial = torch.load(initial_path,map_location='cpu',weights_only=True)
    assert original['epoch']==954 and initial['epoch']==prior_summary['best_epoch']
    torch.testing.assert_close(original['fallback'], initial['fallback'], rtol=0, atol=0)
    fallback = initial['fallback'].numpy()
    train, val = [prepare(data,fallback,ids,device) for ids in indices[:2]]
    selected=np.isin(data['seed'],indices[1])
    model=CornerTransformer(spatial=True).to(device)
    model.load_state_dict(initial['model'],strict=True)
    old_model=CornerTransformer(spatial=True).to(device)
    old_model.load_state_dict(original['model'],strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=RATES[0],weight_decay=.01)
    rng=torch.Generator(device=device).manual_seed(0)
    folder=args.output/'tracker'
    folder.mkdir()
    cfg=dict(initial['config'], initial_sha256=file_sha256(initial_path), sources=sources,
             original_tracker_sha256=file_sha256(TRACKER), start_epoch=initial['epoch'],
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
    # Preserve the original comparison model and use the already trained GNN
    # as initialization. All updates below use the final tracker's coordinates.
    prior_gnn=args.input/'trained/gnn'
    for attempt in range(180):
        if (prior_gnn/'summary.json').is_file():
            break
        sleep(5)
    prior_summary=json.loads((prior_gnn/'summary.json').read_text())
    initial_path=prior_gnn/'best.pt'
    assert file_sha256(initial_path)==prior_summary['best_sha256']
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    original=torch.load(GNN,map_location='cpu',weights_only=True)
    assert original['epoch']==54
    cfg={key:initial['config'][key] for key in ('gnn','loss','training')}
    cfg.update(initial_sha256=file_sha256(initial_path), original_gnn_sha256=file_sha256(GNN), tracker_sha256=file_sha256(args.output/'tracker'/'best.pt'),
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
        old_gnn.load_state_dict(original['model'],strict=True)
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
    print('Completed plateau follow-up Transformer and GNN',flush=True)


if __name__=='__main__':
    main()
