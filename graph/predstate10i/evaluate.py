"""Development selection and independent paired autonomous/handoff evaluation."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,tensor_identity,now
from graph.plan_a_bc.model import normalizer
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.graph_recurrent10g.eval_runner import load_model as load_graph,vision,csv_write,nested_hash
from .model import Policy
from .rollout import episode


def load_s(root,path=None):
    cfg=read(root/'config.json')
    model=Policy(cfg['f1_checkpoint'],read(root/'normalizer.json')['schema'],read(root/'state38_normalizer.json'))
    if path:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        for name,h in ck['identities'].items():assert sha(root/name)==h
        model.load_state_dict(ck['model']);model.checkpoint_step=ck['progress']['step']
        assert model.frozen_hash()==ck['frozen_hash']
    else:model.checkpoint_step=0
    model.eval().requires_grad_(False)
    return model


def models(root,s_checkpoint=None):
    result={'S':load_s(root,s_checkpoint)};cfg=read(root/'config.json')
    for family,row in read(root/'comparison_policy_manifest.json').items():
        assert sha(row['path'])==row['sha256']
        _,m,_,_,ck=load_graph(Path(cfg['reference_root']),row['condition'],Path(row['path']))
        m.checkpoint_step=ck['progress']['step'];result[family]=m
    return result


def paired_verify(root,rows,stage):
    checks=[]
    for seed in sorted({r['seed'] for r in rows}):
        rr=[r for r in rows if r['seed']==seed]
        for key in ['initial_snapshot_sha256','initial_input_sha256','initial_rgbd_sha256']:
            assert len({r[key] for r in rr})==1,(seed,key)
        refrow=next(r for r in rr if r['condition']=='E')
        ref=read(root/refrow['folder']/'trace.json')
        for row in rr:
            tr=read(root/row['folder']/'trace.json')
            assert [r['timestep'] for r in tr]==list(range(len(tr)))
            assert tr[0]['hidden_before_sha256']==nested_hash(None)
            assert all(a['hidden_after_sha256']==b['hidden_before_sha256'] for a,b in zip(tr,tr[1:]))
            if row['condition'][-1:] not in ('1','2'):continue
            end=row['handoff_timestep']
            n=end+1 if end is not None else min(len(ref),len(tr))
            family=row['condition'][0]
            for t in range(n):
                for key in ['input_sha256','rgbd_sha256','state_sha256']:
                    assert tr[t][key]==ref[t][key],(seed,row['condition'],t,key)
                expected=ref[t] if family=='S' else ref[t]['observers'][family]
                for key in ['hidden_before_sha256','hidden_after_sha256','policy_action']:
                    assert tr[t][key]==expected[key],(seed,row['condition'],t,key)
                if end is None or t<end:assert tr[t]['action']==ref[t]['action']
            checks.append(dict(seed=seed,condition=row['condition'],observations_compared=n,valid_anchor=row['valid_anchor']))
    write(root/(stage+'_paired_verification.json'),dict(status='PASS',checks=checks))
    return checks


def development(root):
    cfg=read(root/'config.json');norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    detector,tracker=vision(root);rows=[]
    with ExperimentLogger(team='graph',model='Task10I_Development',seed=42,config=cfg,
           root=root/'experiments',job_type='evaluate') as log:
        for step in cfg['training']['checkpoint_steps']:
            cp=root/'train_S'/f'step{step:06d}.pt';m=load_s(root,cp);before=tensor_identity(m.state_dict())
            for seed in cfg['evaluation']['dev_seeds']:
                row=episode(root,root,m,norm,codec,detector,tracker,seed,'B0',f'development/step{step:06d}','S0')
                row.update(step=step,checkpoint_sha256=sha(cp));rows.append(row)
                log.link_output(root/row['folder']/'episode.json',kind='metrics')
            assert tensor_identity(m.state_dict())==before
            del m;torch.cuda.empty_cache()
        csv_write(root/'development_rollouts.csv',rows)


def select(root):
    import csv
    cfg=read(root/'config.json');rows=list(csv.DictReader((root/'development_rollouts.csv').open()));candidates=[]
    for step in cfg['training']['checkpoint_steps']:
        rr=[r for r in rows if int(r['step'])==step]
        assert len(rr)==len(cfg['evaluation']['dev_seeds'])
        counts=[sum(r[k]=='True' for r in rr) for k in ['insertion','lift','grasp','alignment']]
        error=float(np.mean([float(read(root/r['folder']/'trace.json')[-1]['tcp_peg_distance_m']) for r in rr]))
        val=read(root/'train_S'/'validation'/f'{step:06d}.json')['BC_loss']
        candidates.append(dict(step=step,rank=[-s for s in counts]+[error,val,step]))
    best=min(candidates,key=lambda r:tuple(r['rank']));cp=root/'train_S'/f"step{best['step']:06d}.pt"
    shutil.copy2(cp,root/'best.pt');assert sha(cp)==sha(root/'best.pt')
    write(root/'development_selection.json',dict(status='FROZEN',selected_step=best['step'],
        checkpoint_sha256=sha(cp),path='best.pt',candidates=candidates,rule=cfg['evaluation']['selection'],
        development_seeds=cfg['evaluation']['dev_seeds'],final_accessed=False,created_at=now()),immutable=True)
    manifest=[]
    for p in sorted((root/'train_S').glob('*.pt')):
        ck=torch.load(p,map_location='cpu',weights_only=False)
        manifest.append(dict(path=str(p.relative_to(root)),sha256=sha(p),step=ck['progress']['step'],
            frozen_hash=ck['frozen_hash'],optimizer_scheduler_rng=True))
    write(root/'checkpoint_manifest.json',dict(checkpoints=manifest,best=sha(root/'best.pt'),last=sha(root/'last.pt')))


def final(root,mode):
    assert read(root/'development_selection.json')['status']=='FROZEN'
    cfg=read(root/'config.json');mm=models(root,root/'best.pt');norm=normalizer(root)
    codec=ActionCodec(read(root/'action_codec.json'));detector,tracker=vision(root);rows=[]
    before={k:tensor_identity(m.state_dict()) for k,m in mm.items()}
    conditions=['E','S0','R0','J0'] if mode=='final' else ['S1','S2','R1','R2','J1','J2']
    with ExperimentLogger(team='graph',model='Task10I_'+mode,seed=42,config=cfg,
             root=root/'experiments',job_type='evaluate') as log:
        for seed in read(root/'final_seed_manifest.json')['parents']:
            for label in conditions:
                m=mm['S' if label=='E' else label[0]]
                engine='E' if label=='E' else {'0':'B0','1':'H1','2':'H2'}[label[-1]]
                write(root/'evaluation_progress.json',dict(stage=mode,seed=seed,condition=label,at=now()))
                row=episode(root,root,m,norm,codec,detector,tracker,seed,engine,'final',label,
                    observers={k:mm[k] for k in ['R','J']} if label=='E' else None)
                rows.append(row);log.link_output(root/row['folder']/'episode.json',kind='metrics')
                assert before=={k:tensor_identity(v.state_dict()) for k,v in mm.items()}
        csv_write(root/('final_functional_success.csv' if mode=='final' else 'handoff_success.csv'),rows)
    if mode=='handoff':
        allrows=[read(p) for p in sorted((root/'rollouts/final').glob('seed*/*/episode.json'))]
        assert len(allrows)==120
        paired_verify(root,allrows,'final')


def smoke(root):
    cfg=read(root/'config.json');mm=models(root);norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    detector,tracker=vision(root);seed=cfg['evaluation']['dev_seeds'][0];rows=[]
    for label,engine,cap in [('E','E',250),('S0','B0',2),('S1','H1',250),('S2','H2',250),('R0','B0',2),('J0','B0',2)]:
        m=mm['S' if label=='E' else label[0]]
        row=episode(root,root,m,norm,codec,detector,tracker,seed,engine,'evaluation_smoke',label,
                    horizon=cap,post_handoff_steps=2 if label in ['S1','S2'] else None,
                    observers={k:mm[k] for k in ['R','J']} if label=='E' else None)
        rows.append(row)
    for label in ['S1','S2']:
        row=next(r for r in rows if r['condition']==label)
        assert row['valid_anchor'],label+' engineering anchor not reached'
        assert row['actions']-row['handoff_timestep']==2
    paired_verify(root,rows,'evaluation_smoke')
    for r in rows:
        import cv2
        cap=cv2.VideoCapture(str(root/r['folder']/'rollout.mp4'));ok,frame=cap.read();cap.release()
        assert ok and frame.shape[:2]==(448,768)
    write(root/'evaluation_smoke_verification.json',dict(status='PASS',episodes=rows,
        random_policy=True,GT_student_input=False,prefix_observations_once=True,
        original_H1_H2_anchors=True,media_readback=True,performance_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['development','select','final','handoff','smoke'],required=True)
    a=p.parse_args();torch.set_num_threads(1)
    {'development':development,'select':select,'final':lambda r:final(r,'final'),
     'handoff':lambda r:final(r,'handoff'),'smoke':smoke}[a.stage](a.root)
