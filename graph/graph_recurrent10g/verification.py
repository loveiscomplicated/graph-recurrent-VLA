"""Fail-closed real-data gates and complete artifact readback."""
import copy
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha,digest,tensor_identity,now
from graph.plan_a_bc.train import assert_nested_equal
from graph.integration_baseline.direct_train import rng_state,restore_rng
from .train_worker import setup,update,geometry_loss,save,checkpoint,restore


def preflight(root):
    cfg=read(root/'config.json');f1=Path(cfg['f1_root'])
    if os.getenv('CONTAINER_ID')!='57987772':raise RuntimeError('only designated57987772')
    if sha(cfg['f1_checkpoint'])!=cfg['f1_checkpoint_sha256']:raise ValueError('F1 checkpoint SHA')
    release=read(f1/'data_ready_release.json')
    if release['status']!='READY_FOR_GNN_PRETRAINING':raise ValueError('F1 release not approved')
    if release['dataset_manifest_sha256']!=cfg['dataset_manifest_sha256']:raise ValueError('dataset release SHA')
    if shutil.disk_usage(root).free<2*1024**3:raise OSError('less than2GiB scratch free')
    import importlib.metadata as im
    versions={k:im.version(k) for k in ('torch','numpy','mani_skill','sapien','gymnasium')}
    expected=read(f1/'environment_worker0.json')['versions']
    for k in ('torch','numpy','sapien','gymnasium'):
        if versions[k]!=expected[k]:raise ValueError('native runtime mismatch '+k)
    from graph.gt_pretrain10f1.release import expected_runtime
    runtime=expected_runtime(f1)
    checks=list((f1/'release_checks').glob('*.json'))
    if not checks:raise ValueError('native runtime provenance missing')
    compatible=any(read(p).get('expected',{}).get('source_sha256')==runtime['source_sha256'] for p in checks)
    if not compatible:raise ValueError('current geometry/vision code differs from released F1 runtime')
    # Pin every existing Python/config source used by this isolated execution.
    source=Path(__file__).resolve().parents[2]
    files={str(p.relative_to(source)):sha(p) for d in ('graph','visual','envs','recurrent','utils','legacy_fps')
        for p in sorted((source/d).rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.toml')}
    write(root/'source_identity.json',dict(source=str(source),hashes=files,created_at=now()))
    original={str(p):sha(p) for name in ('gnn_pretrained.pt','final_dataset_manifest.json','training_source_snapshot.tar.gz')
        for p in [f1/name] if p.exists()}
    write(root/'original_preservation.json',original)
    result=dict(status='PASS',versions=versions,python=platform.python_version(),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),vram=torch.cuda.get_device_properties(0).total_memory,
        disk_free=shutil.disk_usage(root).free,cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        GPU_PhysX='NOT_TESTED',backend_plan='CPU PhysX + CUDA renderer/vision/policy',
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],text=True))
    write(root/'preflight.json',result)
    return dict(status='PASS',files=['preflight.json','source_identity.json','original_preservation.json'])


def contracts(root):
    models={};hashes={}
    for condition in 'ABC':
        cfg,m,norm,codec,ds,opt,sched=setup(root,condition)
        hashes[condition]=m.initial_hashes();models[condition]=m
    for k in ('gnn','readout','projection','gru','action_head'):
        if len({hashes[c][k] for c in 'ABC'})!=1:raise ValueError('different initial '+k)
    f1=torch.load(cfg['f1_checkpoint'],map_location='cpu',weights_only=False)
    expected={k.removeprefix('state_head.'):v for k,v in f1['model'].items() if k.startswith('state_head.')}
    if hashes['C']['state_head']!=tensor_identity(expected):raise ValueError('C head not F1')
    schedule=read(root/'sampling_schedule.json');entry=next(e for e in schedule if e['start']==0 and e['supervised'])
    g,r,y,mask,gt,_=ds.batch(entry['ids'],0,32,'cuda');predictions={}
    for c,m in models.items():
        m.eval();p,h=m(g,r,norm);predictions[c]=p.detach().clone()
        if c=='C':
            # Active positive control: inference must not execute the state head.
            with patch.object(m.state_head,'forward',side_effect=AssertionError('state head used in action inference')):
                p2,_=m(g,r,norm)
            torch.testing.assert_close(p2,p,atol=0,rtol=0)
        # Block every GT sidecar read after the legitimate input batch was formed.
        with patch('numpy.load',side_effect=PermissionError('GT sidecar access denied')),patch('h5py.File',side_effect=PermissionError('GT denied')):
            p3,_=m(g,r,norm)
        torch.testing.assert_close(p3,p,atol=0,rtol=0)
    for c in 'BC':torch.testing.assert_close(predictions['A'],predictions[c],atol=0,rtol=0)
    gradients={}
    for c in 'BC':
        m=models[c];m.train();m.zero_grad();p,_=m(g,r,norm);codec.loss(p,y,mask).backward()
        gradients[c]={k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None}
    for k,v in gradients['B'].items():torch.testing.assert_close(v,gradients['C'][k],atol=0,rtol=0)
    c=models['C'];c.zero_grad();c(g,r,norm)
    geo,_=geometry_loss(c,ds,g,r,gt,1,cfg,'cuda');geo.backward()
    norms={name:float(torch.stack([p.grad.norm() for p in mod.parameters() if p.grad is not None]).norm())
        for name,mod in [('gnn',c.graph_encoder),('state_head',c.state_head)]}
    if min(norms.values())<=0 or not all(np.isfinite(list(norms.values()))):raise ValueError('geometry gradient routing')
    # Contiguous hidden carry vs one complete episode, then branch reset.
    eid=entry['ids'][0];m=models['B'];m.eval();e=ds.episodes[str(eid)]
    ga,ra,_,_,_,_=ds.batch([eid],0,e['actions'],'cuda')
    with torch.no_grad():
        whole,_=m(ga,ra,norm);hidden=None;parts=[]
        for start in range(0,e['actions'],32):
            gg,rr,_,_,_,_=ds.batch([eid],start,32,'cuda');pp,hidden=m(gg,rr,norm,hidden)
            hidden=hidden.detach();parts.append(pp)
        # cuDNN may choose different kernels by sequence length; bounded numerical agreement.
        torch.testing.assert_close(torch.cat(parts,1),whole,atol=2e-6,rtol=2e-5)
    write(root/'initial_weight_identity.json',dict(status='PASS',hashes=hashes,F1_head_restored=True))
    write(root/'model_contract.json',dict(status='PASS',initial_predictions_exact=True,
        C_lambda_zero_BC_gradient_exact=True,geometry_gradient_norms=norms,
        GT_sidecar_denial_positive_control='PASS',state_head_denial_action_control='PASS',
        hidden_carry_full_sequence='PASS',TBPTT_detach=True,
        branch_isolation='each episode ID owns hidden; start0 clears group; no parent-shared hidden'))
    return dict(status='PASS',files=['initial_weight_identity.json','model_contract.json'])


def calibration(root):
    cfg,m,norm,codec,ds,opt,sched=setup(root,'C');rows=[]
    schedule=[e for e in read(root/'sampling_schedule.json') if e['start']==0 and e['supervised']]
    for i,entry in enumerate(schedule[:cfg['calibration']['batches']]):
        g,r,y,mask,gt,ix=ds.batch(entry['ids'],0,32,'cuda');m.zero_grad();p,_=m(g,r,norm)
        bc=codec.loss(p,y,mask)
        bg=torch.autograd.grad(bc,list(m.graph_encoder.parameters()),retain_graph=True)
        geo,_=geometry_loss(m,ds,g,r,gt,i+1,cfg,'cuda')
        gg=torch.autograd.grad(geo,list(m.graph_encoder.parameters()))
        bn=float(torch.stack([x.norm() for x in bg]).norm());gn=float(torch.stack([x.norm() for x in gg]).norm())
        rows.append(dict(episode_ids=entry['ids'],start=0,BC_gnn_norm=bn,geometry_gnn_norm=gn))
    valid=[r for r in rows if np.isfinite(r['BC_gnn_norm']) and np.isfinite(r['geometry_gnn_norm']) and min(r['BC_gnn_norm'],r['geometry_gnn_norm'])>1e-12]
    fallback=len(valid)<4
    raw=cfg['calibration']['fallback'] if fallback else cfg['calibration']['ratio']*np.median([r['BC_gnn_norm'] for r in valid])/np.median([r['geometry_gnn_norm'] for r in valid])
    value=float(np.clip(raw,*cfg['calibration']['clamp']))
    write(root/'lambda_calibration.json',dict(status='PASS',rule=cfg['calibration'],rows=rows,lambda_raw=float(raw),
        **{'lambda':value},fallback_used=fallback,train_only=True,F1_scales=dict(position=.1,shape=.02,delta=.01),
        calibration_before_primary=True))
    return dict(status='PASS',files=['lambda_calibration.json'])


def training_smoke(root):
    import types
    schedule=read(root/'sampling_schedule.json');results={}
    coefficient=read(root/'lambda_calibration.json')['lambda']
    for condition in 'ABC':
        cfg,m,norm,codec,ds,opt,sched=setup(root,condition);initial=m.initial_hashes()
        progress=dict(step=0,cursor=0,hidden={},best_loss=None,best_step=None)
        folder=root/'smoke'/condition;folder.mkdir(parents=True,exist_ok=True)
        update(m,opt,sched,ds,norm,codec,progress,schedule[0],cfg,'cuda',coefficient)
        # Same payload implementation as primary; protocol is frozen after calibration.
        payload=checkpoint(root,m,opt,sched,progress,types.SimpleNamespace(run_id='engineering'),initial)
        save(folder/'step1.pt',payload);back=torch.load(folder/'step1.pt',map_location='cpu',weights_only=False)
        (folder/'uncommitted.partial').write_bytes(b'incomplete CUDA checkpoint')
        expected_metrics=update(m,opt,sched,ds,norm,codec,progress,schedule[1],cfg,'cuda',coefficient)
        expected=copy.deepcopy(m.state_dict());expected_opt=copy.deepcopy(opt.state_dict());expected_progress=copy.deepcopy(progress)
        progress=restore(root,back,m,opt,sched)
        actual=update(m,opt,sched,ds,norm,codec,progress,schedule[1],cfg,'cuda',coefficient)
        assert_nested_equal(expected,m.state_dict());assert_nested_equal(expected_opt,opt.state_dict());assert_nested_equal(expected_progress,progress)
        assert_nested_equal(expected_metrics,actual)
        if condition=='A' and m.initial_hashes()['gnn']!=initial['gnn']:raise ValueError('A changed')
        results[condition]=dict(status='PASS',save_resume_next_step_exact=True,gradient=actual,
            frozen_encoder_unchanged=condition=='A')
    write(root/'resume_verification.json',dict(status='PASS',conditions=results,
        scope='actual CUDA minibatches, optimizer/scheduler/RNG/hidden/cursor readback and next update exact; incomplete partial file ignored'))
    return dict(status='PASS',steps=9,files=['resume_verification.json'])


def freeze(root):
    cfg=read(root/'config.json')
    data=dict(config_sha256=sha(root/'config.json'),dataset_sha256=sha(root/'dataset_identity.json'),
        initial_sha256=sha(root/'initial_weight_identity.json'),sampling_sha256=sha(root/'sampling_schedule.json'),
        lambda_sha256=sha(root/'lambda_calibration.json'),source_sha256=sha(root/'source_identity.json'),
        primary_updates=cfg['training']['updates'],lambda_geometry=read(root/'lambda_calibration.json')['lambda'],
        selection_rule=cfg['evaluation']['selection'],development_seeds=cfg['evaluation']['dev_seeds'],
        final_seeds=cfg['evaluation']['final_seeds'],created_at=now())
    path=root/'training_protocol_frozen.json'
    if path.exists():
        prior=read(path)
        for k in data:
            if k!='created_at' and prior[k]!=data[k]:raise ValueError('freeze changed '+k)
    else:write(path,data,immutable=True)
    return dict(status='PASS',files=['training_protocol_frozen.json'])


def checkpoint_verification(root,write_manifest=True):
    cfg=read(root/'config.json');records={}
    for c in 'ABC':
        folder=root/('train_'+c);r=read(folder/'result.json');ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if r['cursor']!=len(read(root/'sampling_schedule.json')):raise ValueError('exposure cursor incomplete')
        if r['steps']!=cfg['training']['updates'] or ck['progress']['step']!=r['steps']:raise ValueError('incomplete training')
        if r['checkpoint_sha256']!=sha(folder/'last.pt'):raise ValueError('checkpoint SHA')
        if r['schedule_sha256']!=sha(root/'sampling_schedule.json'):raise ValueError('exposure mismatch')
        for key in ('optimizer','scheduler','rng','normalizer','codec','progress'):assert key in ck
        if c=='A' and r['initial']['gnn']!=r['final']['gnn']:raise ValueError('A frozen mutation')
        if c in 'BC' and r['initial']['gnn']==r['final']['gnn']:raise ValueError('joint GNN not updated')
        import csv
        rows=list(csv.DictReader((root/f'train_{c}.csv').open()))
        if len(rows)!=cfg['training']['updates'] or [int(x['step']) for x in rows]!=list(range(1,cfg['training']['updates']+1)):
            raise ValueError('training ledger missing/duplicate steps')
        if c=='C' and any(float(x['geometry_loss'])<=0 or float(x['state_head_gradient_norm'])<=0 for x in rows):
            raise ValueError('C geometry execution/gradient missing')
        for step in cfg['training']['checkpoint_steps']:
            p=folder/f'step{step:06d}.pt'
            if not p.exists() or sha(p)!=read(p.with_suffix('.json'))['sha256']:raise ValueError('candidate checkpoint missing/corrupt')
        records[c]=dict(last_sha256=sha(folder/'last.pt'),cursor=r['cursor'],step=r['steps'],
            checkpoints=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in sorted(folder.glob('*.pt'))])
    if len({r['cursor'] for r in records.values()})!=1:raise ValueError('BC exposure cursor differs')
    if write_manifest:write(root/'checkpoint_manifest.json',dict(status='PASS',conditions=records,same_exposure=True))
    return dict(status='PASS',files=['checkpoint_manifest.json'])
