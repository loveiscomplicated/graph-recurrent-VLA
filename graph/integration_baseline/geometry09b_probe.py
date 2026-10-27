"""Small decoder-only diagnostics on detached arrays; no policy import/optimizer."""
from copy import deepcopy
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from graph.integration_baseline.io import save_npz, write_json

PROBES=['P0','P1','P2','P3','P4']
MODELS=['Full_B','Full_C','Single_B','Single_C','Full_A','Single_A']


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def standardize(x,train):
    if not np.any(train):raise ValueError('empty train')
    mean=x[train].mean(0,dtype=np.float64);std=np.maximum(x[train].std(0,dtype=np.float64),1e-6)
    return (x-mean)/std,mean,std


def absolute(y,p):
    if len(y)==0:return dict(n=0,mean_mm=None)
    err=np.linalg.norm(p-y,axis=1)*1000;mae=np.abs(p-y).mean(0)*1000
    ss=((y-y.mean(0))**2).sum(0);r2=np.divide(((y-p)**2).sum(0),ss,out=np.full(3,np.nan),where=ss>1e-14)
    return dict(n=len(y),mean_mm=float(err.mean()),median_mm=float(np.median(err)),
        p90_mm=float(np.percentile(err,90)),p95_mm=float(np.percentile(err,95)),maximum_mm=float(err.max()),
        mae_x_mm=float(mae[0]),mae_y_mm=float(mae[1]),mae_z_mm=float(mae[2]),
        r2_x=float(1-r2[0]) if np.isfinite(r2[0]) else None,
        r2_y=float(1-r2[1]) if np.isfinite(r2[1]) else None,
        r2_z=float(1-r2[2]) if np.isfinite(r2[2]) else None)


def delta(y,p):
    norm=float(np.linalg.norm(y));pn=float(np.linalg.norm(p));e=p-y
    return dict(true_norm_mm=norm*1000,predicted_norm_mm=pn*1000,delta_error_mm=float(np.linalg.norm(e)*1000),
        direction_cosine=float(np.dot(y,p)/(norm*pn)) if norm>1e-7 and pn>1e-9 else None,
        signed_gain=float(np.dot(y,p)/(norm*norm)) if norm>1e-7 else None,
        delta_x_error_mm=float(abs(e[0])*1000),delta_y_error_mm=float(abs(e[1])*1000),delta_z_error_mm=float(abs(e[2])*1000),
        true_dx_mm=float(y[0]*1000),true_dy_mm=float(y[1]*1000),true_dz_mm=float(y[2]*1000),
        predicted_dx_mm=float(p[0]*1000),predicted_dy_mm=float(p[1]*1000),predicted_dz_mm=float(p[2]*1000))


def ridge_fit(x,y,alpha):
    # Center the standardized X again for intercept; alpha0 uses stable SVD min norm.
    xm=x.mean(0);ym=y.mean(0);xx=x-xm;yy=y-ym
    u,s,vt=np.linalg.svd(xx,full_matrices=False)
    if alpha==0:
        cutoff=np.finfo(np.float64).eps*max(xx.shape)*s.max()
        coeff=np.divide(1,s,out=np.zeros_like(s),where=s>cutoff)
    else:coeff=s/(s*s+alpha)
    w=(vt.T*coeff)@(u.T@yy);b=ym-xm@w
    return w,b


def linear(x,y,train,val,c):
    best=None;hist=[]
    for alpha in c['alphas']:
        w,b=ridge_fit(x[train],y[train],alpha);p=x@w+b
        score=float(np.mean((p[val]-y[val])**2));hist.append(dict(alpha=alpha,val_mse=score))
        if best is None or score<best[0]:best=(score,w,b,alpha,p)
    return best[-1],dict(kind='ridge',weight=best[1],bias=best[2],alpha=np.array(best[3])),dict(candidates=hist,selected_alpha=best[3])


def mlp(x,y,train,val,c):
    # Fixed one hidden layer and two pre-registered weight decays. CPU, deterministic.
    xx=torch.from_numpy(x.astype(np.float32));yy=torch.from_numpy(y.astype(np.float32))
    tt=torch.from_numpy(train);vv=torch.from_numpy(val);winner=None;hist=[]
    for wd in c['weight_decay']:
        torch.manual_seed(42)
        model=nn.Sequential(nn.Linear(x.shape[1],c['hidden']),nn.ReLU(),nn.Linear(c['hidden'],3))
        opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=wd)
        params={id(p) for group in opt.param_groups for p in group['params']}
        assert params=={id(p) for p in model.parameters()}
        initial=deepcopy(model.state_dict());best_loss=float('inf');state=None;best_epoch=0;stale=0;curve=[]
        for epoch in range(1,c['max_epochs']+1):
            model.train();opt.zero_grad(set_to_none=True);loss=((model(xx[tt])-yy[tt])**2).mean()
            loss.backward();opt.step();model.eval()
            with torch.no_grad():score=float(((model(xx[vv])-yy[vv])**2).mean())
            if score<best_loss-c['min_delta']:
                best_loss=score;state=deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            if epoch==1 or epoch%25==0:curve.append(dict(epoch=epoch,train_mse=float(loss.detach()),val_mse=score))
            if stale>=c['patience']:break
        model.load_state_dict(state);model.eval()
        assert any(not torch.equal(initial[k],state[k]) for k in state),'decoder parameters must update'
        with torch.no_grad():pred=model(xx).numpy().astype(np.float64)
        # Select on exact score of reloaded checkpoint.
        score=float(np.mean((pred[val]-y[val])**2))
        hist.append(dict(weight_decay=wd,best_epoch=best_epoch,epochs=epoch,val_mse=score,curve=curve))
        if winner is None or score<winner[0]:winner=(score,pred,state,wd,best_epoch)
    arrays={k:v.detach().numpy() for k,v in winner[2].items()}
    return winner[1],dict(kind='mlp',**arrays),dict(candidates=hist,selected_weight_decay=winner[3],selected_epoch=winner[4],
        decoder_only_optimizer=True,decoder_weights_changed=True)


def features(data,model,probe):
    z=data[model+'_readout320'];r=data['robot76']
    return {'P0':r,'P1':z,'P2':np.concatenate([z,r],1),'P3':z[:,192:320],
        'P4':data['raw_geometry15'][:,:3]}[probe]


def train_all(root,rows,c,logger):
    root=Path(root);buffers={}
    for row in rows:
        with np.load(root/row['sample_id']/'features.npz') as f:
            for k in f.files:buffers.setdefault(k,[]).append(f[k])
    data={k:np.stack(v) for k,v in buffers.items()};n=len(rows)
    save_npz(root/'frozen_readouts320.npz',**data,sample_id=np.array([r['sample_id'] for r in rows]),
        seed=np.array([r['seed'] for r in rows]),timestep=np.array([r['t'] for r in rows]),
        split=np.array([r['split'] for r in rows]),validity=np.array([r['validity'] for r in rows]))
    y=data['target_peg_position_tcp'].astype(np.float64)
    valid=np.array([r['physical_valid'] and r['validity']=='VISIBLE_VALID' for r in rows])
    primary=np.array([r['group'] in ('baseline','primary') for r in rows])
    train=np.array([r['split']=='train' for r in rows])&valid&primary
    val=np.array([r['split']=='validation' for r in rows])&valid&primary
    assert train.sum()>=20 and val.sum()>=10,'insufficient independent valid probe data'
    assert not set(np.array([r['seed'] for r in rows])[train])&set(np.array([r['seed'] for r in rows])[val])
    ym=y[train].mean(0);ys=np.maximum(y[train].std(0),.001);yn=(y-ym)/ys
    rng=np.random.default_rng(43);permutation=rng.permutation(train.sum());shuffled=yn.copy();shuffled[train]=yn[train][permutation]
    save_npz(root/'probe_training_masks.npz',train=train,validation=val,target_mean=ym,target_std=ys,
        shuffle_permutation=permutation)
    metrics=[];episodes=[];pairs=[];traces=[];predictions={};index={(r['seed'],r['t'],r['condition']):i for i,r in enumerate(rows)}
    pred_dir=root/'probes';pred_dir.mkdir(exist_ok=False)

    def record(model,probe,decoder,control,pred,identity):
        key=f'{model}_{probe}_{decoder}_{control}'
        predictions[key]=pred
        save_npz(pred_dir/(key+'_predictions.npz'),prediction=pred.astype(np.float32),target=y.astype(np.float32),
            sample_id=np.array([r['sample_id'] for r in rows]))
        for part in ('train','validation','final'):
            for category in ('VISIBLE_VALID','VISIBLE_MISSING','OCCLUDED','GEOMETRY_CONTAMINATED','ALL'):
                mask=np.array([r['split']==part and r['physical_valid'] and r['group']!='repeat' and
                    (category=='ALL' or r['validity']==category) for r in rows])
                # Final primary absolute and secondary are kept separate.
                if part=='final':mask &= primary
                m=dict(model=model,probe=probe,decoder=decoder,control=control,split=part,validity=category,
                    **absolute(y[mask],pred[mask]));metrics.append(m)
        for part in ('train','validation','final'):
            for seed in sorted({r['seed'] for r in rows if r['split']==part}):
                mask=np.array([r['seed']==seed for r in rows])&valid&primary
                episodes.append(dict(model=model,probe=probe,decoder=decoder,control=control,split=part,seed=seed,
                    **absolute(y[mask],pred[mask])))
        for i,r in enumerate(rows):
            if r['group']=='baseline':continue
            bi=index[(r['seed'],r['t'],'baseline')];base=rows[bi]
            np.testing.assert_array_equal(data['robot76'][i],data['robot76'][bi])
            if probe=='P0':np.testing.assert_array_equal(pred[i],pred[bi])
            dy=y[i]-y[bi];dp=pred[i]-pred[bi]
            # Audit centroid support parallel to09A; not a target/feature filter for fitting.
            world=np.asarray(r.get('support',[])) if False else None
            rawdelta=data['raw_geometry15'][i,:3]-data['raw_geometry15'][bi,:3]
            projected=float(np.dot(rawdelta,dy)/max(np.linalg.norm(dy),1e-30)*1000)
            pairs.append(dict(model=model,probe=probe,decoder=decoder,control=control,
                split=r['split'],seed=r['seed'],t=r['t'],condition=r['condition'],group=r['group'],
                validity=r['validity'],baseline_validity=base['validity'],physical_valid=bool(r['physical_valid'] and base['physical_valid']),
                primary_valid=bool(valid[i] and valid[bi] and r['group']=='primary'),
                centroid_projected_response_mm=projected,geometry09a_response_support=bool(projected>=1 and valid[i] and valid[bi]),
                **delta(dy,dp)))

    const=np.repeat(y[train].mean(0)[None],n,0);record('Shared','Constant','constant','normal',const,{})
    # Direct raw centroid error, no learned origin correction; partial-surface bias retained.
    record('Shared','RawCentroid','direct','normal',data['raw_geometry15'][:,:3].astype(np.float64),{})
    for model in MODELS:
        for probe in PROBES:
            x=features(data,model,probe).astype(np.float64);xx,xm,xs=standardize(x,train)
            for control,labels in [('normal',yn),('shuffle',shuffled)]:
                for decoder,fn in [('linear',linear),('mlp',mlp)]:
                    tick=time.monotonic();pp,state,selection=fn(xx,labels,train,val,c[decoder])
                    pred=pp*ys+ym;key=f'{model}_{probe}_{decoder}_{control}'
                    arrays={k:v for k,v in state.items() if k!='kind'}
                    save_npz(pred_dir/(key+'_checkpoint.npz'),**arrays,feature_mean=xm,feature_std=xs,target_mean=ym,target_std=ys)
                    write_json(pred_dir/(key+'_selection.json'),dict(model=model,probe=probe,decoder=decoder,control=control,
                        feature_dimension=x.shape[1],train_rows=int(train.sum()),validation_rows=int(val.sum()),
                        train_groups=c['train'],validation_groups=c['validation'],final_used=False,
                        elapsed_s=time.monotonic()-tick,**selection))
                    record(model,probe,decoder,control,pred,selection)
                    logger.log(dict(probe_train_mm=absolute(y[train],pred[train])['mean_mm'],
                        probe_validation_mm=absolute(y[val],pred[val])['mean_mm']),prefix=key)
                    print('PROBE',key,'val_mm',absolute(y[val],pred[val])['mean_mm'],flush=True)
        # Freeze no parameter tensors exist in this module, inputs read as numpy.
    write_json(root/'probe_metrics.json',metrics);csv_write(root/'absolute_metrics.csv',metrics)
    csv_write(root/'per_episode.csv',episodes);csv_write(root/'per_pair.csv',pairs)
    write_json(root/'probe_fit_audit.json',dict(train_rows=int(train.sum()),validation_rows=int(val.sum()),
        primary_train_rows_only=True,final_used_for_fit_or_selection=False,train_statistics_only=True,
        train_seeds=sorted({rows[i]['seed'] for i in np.flatnonzero(train)}),
        validation_seeds=sorted({rows[i]['seed'] for i in np.flatnonzero(val)}),
        optimizers='only independent Linear64/ReLU3 decoder parameters; no policy parameter references',
        train_label_shuffle_seed=43,number_of_fits=120,all_model_inputs_detached_numpy=True))
    return metrics
