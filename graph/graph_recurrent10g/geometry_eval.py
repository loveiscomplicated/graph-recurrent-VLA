"""Post-Final geometry reconstruction and equal-budget train-only probes."""
from pathlib import Path
import copy
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,npz
from graph.gt_pretrain10f1.data import GraphPool
from graph.gt_pretrain10f1.model import SpatialHead,absolute_loss,displacement_loss
from graph.gt_pretrain10f1.evaluate import summary,delta_metrics
from .eval_runner import load_model,csv_write,vision


def run(root):
    cfg=read(root/'config.json')
    if not (root/'final_verification.json').exists():raise ValueError('main Final must finish first')
    pool=GraphPool(Path(cfg['f1_root']),'cpu',allow_final=True)
    gt=pool.targets.numpy();presence=pool.presence.numpy().astype(bool);predictions={};deltas=[];probe_records={}
    training=np.asarray(pool.splits['train']);pairs=[p for p in pool.pairs if p['split']=='train' and p['valid']]
    finalpairs=[p for p in pool.pairs if p['split']=='final' and p['valid']]
    if cfg['miniature']:finalpairs=finalpairs[:4]
    with ExperimentLogger(team='graph',model='Task10G_geometry_equal_probe',seed=42,config=cfg,
        root=root/'experiments',job_type='evaluate') as log:
        for condition in 'ABC':
            selected=read(root/'model_selection_frozen.json')['conditions'][condition]
            _,model,_,_,_=load_model(root,condition,root/selected['path'])
            f1=torch.load(cfg['f1_checkpoint'],map_location='cpu',weights_only=False);stats=f1['normalization']
            head=SpatialHead(398,stats['target_mean']).cuda()
            head.load_state_dict({k.removeprefix('state_head.'):v for k,v in f1['model'].items() if k.startswith('state_head.')})
            features=[]
            with torch.inference_mode():
                for start in range(0,len(pool.meta),64):
                    ix=np.arange(start,min(start+64,len(pool.meta)));g,r,_=pool.select(ix)
                    z,pres=model.graph_encoder(g.to('cuda'));rr=r.cuda()
                    features.append(torch.cat((z,pres,(rr-rr.new_tensor(stats['robot_mean']))/rr.new_tensor(stats['robot_std'])),-1).cpu().numpy())
            x=np.concatenate(features);xt=torch.from_numpy(x).cuda()
            with torch.inference_mode():
                diagnostic=model.state_head if condition=='C' else head
                pred=torch.cat([diagnostic(xt[s:s+64]) for s in range(0,len(x),64)]).cpu().numpy()
            predictions[condition+('_joint_head' if condition=='C' else '_frozen_F1_head')]=pred
            # Every encoder gets the same independent head, initialization/schedule/budget.
            probe=copy.deepcopy(head).train().requires_grad_(True)
            opt=torch.optim.AdamW(probe.parameters(),lr=.0003,weight_decay=.00001)
            rng=np.random.default_rng(4242);updates=3 if cfg['miniature'] else 500
            yt=torch.from_numpy(gt).cuda()
            for step in range(updates):
                ids=rng.choice(training,64);pp=[pairs[j] for j in rng.integers(0,len(pairs),16)]
                ix=np.r_[ids,[p['base'] for p in pp],[p['index'] for p in pp]]
                y=yt[ix];p=probe(xt[ix]);loss,_=absolute_loss(p,y)
                loss=loss+displacement_loss(p[-32:-16],p[-16:],y[-32:-16],y[-16:])
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),1.,error_if_nonfinite=True);opt.step()
            with torch.inference_mode():pred=torch.cat([probe(xt[s:s+64]) for s in range(0,len(x),64)]).cpu().numpy()
            predictions[condition+'_matched_probe']=pred
            torch.save(dict(head=probe.cpu().state_dict(),updates=updates,seed=4242,train_only=True),root/('geometry_probe_'+condition+'.pt'))
            probe_records[condition]=dict(updates=updates,initial_head='same F1 head',train_only=True,gnn_frozen=True)
            for name,prediction in list(predictions.items()):
                if not name.startswith(condition+'_'):continue
                for pair in finalpairs:
                    s=slice(0,3) if pair['object']=='peg' else slice(7,10)
                    b,a=pair['base'],pair['index']
                    deltas.append(dict(model=name,parent=pool.meta[a]['parent'],pair_id=pair['pair_id'],object=pair['object'],
                        offset_mm=pair['offset_mm'],previously_observed_F1_Final=True,
                        **delta_metrics(gt[a,s]-gt[b,s],prediction[a,s]-prediction[b,s])))
            del model,xt,yt,head,probe;torch.cuda.empty_cache()
        metrics=summary(predictions,gt,pool.meta,presence,only_split='final')
        csv_write(root/'geometry_metrics.csv',metrics);csv_write(root/'paired_displacement_metrics.csv',deltas)
        npz(root/'geometry_predictions.npz',dict(target18=gt,**predictions))
        write(root/'geometry_verification.json',dict(status='PASS',probes=probe_records,
            geometry_scope='previously examined F1 Final91000-91011; separate from new closed-loop Final',
            B_frozen_head_drift_is_not_representation_proof=True,matched_probe_primary=True,
            final_pairs=len(finalpairs),new_BLIND_geometry=False))
        for name in ('geometry_metrics.csv','paired_displacement_metrics.csv'):log.link_output(root/name,kind='metrics')
    return dict(status='PASS',steps=sum(v['updates'] for v in probe_records.values()),files=['geometry_metrics.csv','paired_displacement_metrics.csv','geometry_verification.json','geometry_predictions.npz'])


def action_correction(root):
    import shutil
    from graph.geometry_reach10d.evaluate import interventions
    cfg=read(root/'config.json');f1=Path(cfg['f1_root']);reach=Path(read(f1/'run_identity.json')['reach'])
    source=reach/'collection_protocol.json'
    if not source.exists():
        csv_write(root/'action_correction_metrics.csv',[dict(status='NOT_TESTED',reason='validated Task10D intervention protocol unavailable')])
        return dict(status='PASS',files=['action_correction_metrics.csv'],limitation='NOT_TESTED')
    proto=read(source);proto['intervention_offsets_mm']=[[-5,0,0],[5,0,0],[-10,0,0],[10,0,0]]
    write(root/'collection_protocol.json',proto)
    detector,tracker=vision(root)
    with ExperimentLogger(team='graph',model='Task10G_actual_action_interventions',seed=42,config=cfg,
            root=root/'experiments',job_type='evaluate') as log:
        for c in 'ABC':
            chosen=read(root/'model_selection_frozen.json')['conditions'][c]
            _,m,norm,codec,ck=load_model(root,c,root/chosen['path'])
            for seed in cfg['evaluation']['dev_seeds'][:1 if cfg['miniature'] else 2]:
                interventions(root,root/'action_interventions'/c/str(seed),seed,m,norm,codec,detector,tracker,c,chosen['step'])
            del m;torch.cuda.empty_cache()
        # Rebuild the final table from completed fixtures, never from a partial append log.
        rows=[]
        for c in 'ABC':
            for seed in cfg['evaluation']['dev_seeds'][:1 if cfg['miniature'] else 2]:
                result=read(root/'action_interventions'/c/str(seed)/'complete.json')
                rows.extend({k:v for k,v in row.items() if k not in ('metrics','reference')} for row in result['rows'])
        csv_write(root/'action_correction_metrics.csv',rows)
        log.link_output(root/'action_correction_metrics.csv',kind='metrics')
    return dict(status='PASS',files=['action_correction_metrics.csv'])
