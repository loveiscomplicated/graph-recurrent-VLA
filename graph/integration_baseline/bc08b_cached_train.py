"""Task08-B worker with exact tensor memoization for one immutable episode."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

from graph.integration_baseline.bc08_codec import ActionCodec, digest, native_metrics
from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.direct_runtime import atomic, append, read, sha, lock, Events
from graph.integration_baseline.direct_train import (seed_all, batch as original_batch, memory, rng_state,
    restore_rng, normalizer_state, load_normalizer)
from graph.integration_baseline.model import IntegrationPolicy
from utils.experiment_logger import ExperimentLogger
from tools.discord_notifier import Sanitizer


VERSION = 'task08_bc_v1'
_BATCH_CACHE = None
_CACHE_DATASET = None


def batch(dataset, ids, start, length, device, config):
    if (dataset is _CACHE_DATASET and _BATCH_CACHE is not None and
            ids == [30003] and length == config['training']['sequence_length'] and
            start in _BATCH_CACHE and device == config['device']):
        return _BATCH_CACHE[start]
    return original_batch(dataset, ids, start, length, device, config)


def cache_episode_batches(dataset, plan):
    """Memoize exact original Graph/Robot76/Action tensors, with byte equality audit."""
    global _BATCH_CACHE, _CACHE_DATASET
    if plan['overfit_ids'] != [30003]:
        raise ValueError('cached worker only supports fixed Episode 30003')
    config = plan['config']; device = config['device']
    length = config['training']['sequence_length']
    cache = {}; hashes = {}; shape = {}
    for start in range(0, dataset.episodes[30003]['actions'], length):
        a = original_batch(dataset, [30003], start, length, device, config)
        b = original_batch(dataset, [30003], start, length, device, config)
        for name, value in vars(a[0]).items():
            if not torch.equal(value, getattr(b[0], name)):
                raise ValueError('Graph tensor differs across original batch constructions: '+name)
        for i, name in enumerate(('robot76', 'action7', 'valid_mask'), 1):
            if not torch.equal(a[i], b[i]):
                raise ValueError('input tensor differs across original batch constructions: '+name)
        h = hashlib.sha256()
        for name, value in sorted(vars(a[0]).items()):
            data = value.detach().cpu().contiguous().numpy()
            h.update(name.encode()); h.update(str(data.shape).encode()); h.update(data.tobytes())
        for i in (1,2,3):
            data = a[i].detach().cpu().contiguous().numpy()
            h.update(str(data.shape).encode()); h.update(data.tobytes())
        cache[start] = a
        hashes[str(start)] = h.hexdigest()
        shape[str(start)] = dict(frames=int(a[3].sum()), graph_nodes=int(a[0].node_input.shape[0]),
                                 graph_edges=int(a[0].edge_index.shape[1]))
    _CACHE_DATASET = dataset
    _BATCH_CACHE = cache
    return dict(status='PASS',starts=sorted(cache),tensor_sha256=hashes,shapes=shape,
                rule='original DirectDataset.frame and direct_train.batch twice per chunk; all Graph/Robot76/Action/valid tensors equal; exact tensors reused on GPU')


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def normalizer_hash(n):
    return digest(dict(schema=n['schema'], fitted_episode_ids=n['fitted_episode_ids'],
        mean=n['mean'].tolist(), std=n['std'].tolist()))


def restricted_load(path):
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                          np.dtype, type(np.dtype('uint32'))]):
        return torch.load(path, map_location='cpu', weights_only=True)


def load_plan(path):
    path = Path(path).resolve(); plan = read(path)
    if plan['version'] != VERSION:
        raise ValueError('Task08 frozen plan required')
    for p, expected in plan['input_hashes'].items():
        if sha(p) != expected:
            raise ValueError('frozen input changed: '+p)
    for p, expected in plan['source_hashes'].items():
        if sha(p) != expected:
            raise ValueError('source changed since protocol freeze: '+p)
    return plan


def make_identity(plan, plan_path, phase, method):
    return dict(version=VERSION, plan_sha256=sha(plan_path), phase=phase, method=method,
        codec_sha256=digest(plan['codecs'][phase][method]), initial_sha256=plan['initial_weight_identity'],
        normalizer_sha256=plan['robot_normalizer_identity'], input_hashes=plan['input_hashes'],
        training_config=plan['config']['training'], source_hashes=plan['source_hashes'],
        train_ids=plan['overfit_ids'] if phase == 'overfit' else plan['splits']['train'])


def optimizer(model, plan, phase):
    t = plan['config']['training']
    opt = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'],
        betas=tuple(t['betas']), eps=t['eps'], amsgrad=t['amsgrad'])
    # Overfit uses precisely 2000 updates; Full retains the original epoch cosine.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
        T_max=plan['overfit_updates'] if phase == 'overfit' else t['max_epochs'], eta_min=t['eta_min'])
    return opt, scheduler


def update(model, opt, codec, pred, target, mask, clip):
    loss = codec.loss(pred, target, mask)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite optimization loss')
    opt.zero_grad(set_to_none=True); loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError('missing/nonfinite gradient: '+name)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite parameter')
    return float(loss.detach()), float(norm)


def evaluate(dataset, model, norm, codec, ids, config, output=None):
    """Full causal episode, hidden carry across TBPTT boundaries, native decode."""
    model.eval(); episodes = {}; all_p = []; all_y = []; weighted_loss = 0; count = 0;gripper_loss=0.;motion_z_error=0.
    for eid in ids:
        hidden = None; pp = []; yy = []; rr = []; hh = []
        with torch.inference_mode():
            for start in range(0, dataset.episodes[eid]['actions'], config['training']['sequence_length']):
                g, s, y, valid = batch(dataset, [eid], start, config['training']['sequence_length'], config['device'], config)
                prior=hidden
                raw, hidden = model(g, s, norm, hidden); hidden = hidden.detach()
                native = codec.decode(codec.tag(raw))
                n = int(valid.sum()); weighted_loss += float(codec.loss(raw, y, valid))*n; count += n
                rp,ty=raw[valid],y[valid]
                if codec.method=='C':
                    gripper_loss+=float(torch.nn.functional.binary_cross_entropy_with_logits(rp[:,6],(ty[:,6]>0).float()))*n
                else:gripper_loss+=float(((rp[:,6]-ty[:,6])**2).mean())*n
                mean,std=codec._stats(ty)
                motion_z_error+=float((((native[valid][:,:6]-ty[:,:6])/std)**2).mean())*n
                if output is not None:
                    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'ep{eid}_chunk{start:04d}.npz',
                        **{name:value.cpu().numpy() for name,value in vars(g).items()},
                        robot_state=s.cpu().numpy(),native_target=y.cpu().numpy(),valid_mask=valid.cpu().numpy(),
                        encoded_output=raw.cpu().numpy(),native_prediction=native.cpu().numpy(),
                        hidden_before=np.zeros_like(hidden.cpu().numpy()) if prior is None else prior.cpu().numpy(),
                        hidden_after=hidden.cpu().numpy(),episode_id=eid,start_timestep=start,codec_identity=codec.identity)
                pp.append(native[valid].cpu().numpy()); yy.append(y[valid].cpu().numpy())
                rr.append(raw[valid].cpu().numpy()); hh.append(hidden.cpu().numpy())
        p, y = np.concatenate(pp), np.concatenate(yy)
        episodes[str(eid)] = native_metrics(p, y); all_p.append(p); all_y.append(y)
        if output is not None:
            output = Path(output); output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output/f'ep{eid}.npz', native_prediction=p, native_target=y,
                encoded_output=np.concatenate(rr), chunk_final_hidden=np.stack(hh),
                timestep=np.arange(len(y)), episode_id=eid, codec_identity=codec.identity)
    # Do not concatenate transition events across episode boundaries.
    metrics = native_metrics(np.concatenate(all_p), np.concatenate(all_y))
    metrics.pop('gripper_transition')
    metrics.update(optimization_loss=weighted_loss/count, episodes=episodes,
        native_motion_z_mse=motion_z_error/count,gripper_optimization_loss=gripper_loss/count,
        gripper_optimization_type='BCEWithLogits' if codec.method=='C' else 'continuous MSE',
        gripper_transition_boundary_error_rate=float(np.average(
            [v['gripper_transition']['boundary_error_rate'] for v in episodes.values()],
            weights=[max(v['count']-1, 1) for v in episodes.values()])),
        hidden_contract='reset at episode t0, carry/detach between contiguous chunks')
    return metrics


def save(run, model, opt, sched, norm, codec, ident, progress, *, best=False):
    root = Path(run)/'checkpoints'; root.mkdir(exist_ok=True)
    path = root/f'step{progress["step"]:06d}-epoch{progress["epoch"]:03d}.pt'
    # A terminal/error save can reuse the already-valid exact same cursor.
    if path.exists():
        previous = restricted_load(path)
        if previous['identity'] != ident or previous['progress']['step'] != progress['step'] or tensor_hash(previous['model']) != tensor_hash(model.state_dict()):
            raise ValueError('checkpoint collision')
        return read(path.with_suffix('.json'))
    payload = dict(version=VERSION, model=model.state_dict(), optimizer=opt.state_dict(),
        scheduler=sched.state_dict(), normalizer=normalizer_state(norm), action_codec=codec.state,
        identity=ident, progress=copy.deepcopy(progress), rng=rng_state(include_cuda=next(model.parameters()).is_cuda))
    tmp = path.with_suffix('.partial'); torch.save(payload, tmp)
    restored = restricted_load(tmp)
    if restored['identity'] != ident or tensor_hash(restored['model']) != tensor_hash(model.state_dict()):
        raise ValueError('checkpoint readback failure')
    with tmp.open('rb') as f: os.fsync(f.fileno())
    os.link(tmp, path); tmp.unlink()
    receipt = dict(path=str(path.resolve()), sha256=sha(path), epoch=progress['epoch'], global_step=progress['step'],
        codec_identity=codec.identity, model_identity=tensor_hash(restored['model']), run_id=Path(run).name)
    atomic(path.with_suffix('.json'), receipt, immutable=True)
    atomic(Path(run)/'last_valid.json', receipt)
    if best: atomic(Path(run)/'best.json', receipt)
    return receipt


def load_checkpoint(receipt, ident=None):
    receipt = read(receipt) if isinstance(receipt, (str, Path)) else receipt
    if sha(receipt['path']) != receipt['sha256']:
        raise ValueError('checkpoint receipt SHA mismatch')
    cp = restricted_load(receipt['path'])
    if cp['version'] != VERSION or (ident is not None and cp['identity'] != ident):
        raise ValueError('checkpoint experiment/model/dataset/codec identity mismatch')
    codec = ActionCodec(cp['action_codec'])
    if codec.identity != cp['identity']['codec_sha256'] or codec.identity != receipt['codec_identity']:
        raise ValueError('checkpoint ActionCodec mismatch')
    if normalizer_hash(cp['normalizer']) != cp['identity']['normalizer_sha256']:
        raise ValueError('checkpoint Robot76 normalizer mismatch')
    if tensor_hash(cp['model']) != receipt['model_identity'] or cp['progress']['step'] != receipt['global_step']:
        raise ValueError('checkpoint model/progress mismatch')
    return cp, codec


def chunks(plan, phase, epoch):
    """Stateless schedule: independent of model RNG, loss and concurrent process."""
    ids = plan['overfit_ids'] if phase == 'overfit' else plan['splits']['train']
    order = np.random.default_rng(np.random.SeedSequence([42, epoch])).permutation(ids).tolist()
    t = plan['config']['training']; result = []
    for group in range(0, len(order), t['batch_size']):
        pair = order[group:group+t['batch_size']]
        for start in range(0, max(plan['episode_lengths'][str(e)] for e in pair), t['sequence_length']):
            result.append((tuple(e for e in pair if start < plan['episode_lengths'][str(e)]), start))
    return result


def error_bundle(run, exc, ident):
    clean = Sanitizer()
    atomic(Path(run)/'error_bundle.json', json.loads(clean(json.dumps(dict(
        error_type=type(exc).__name__, traceback=traceback.format_exc(), identity=ident,
        last_valid=read(Path(run)/'last_valid.json') if (Path(run)/'last_valid.json').exists() else None,
        callback='NOT_CONFIGURED: no automatic Codex invocation or experiment mutation')))))


def worker(plan_path, phase, method, *, resume=None, probe=False, paired=False):
    plan = load_plan(plan_path); config = plan['config']; seed_all(42, config['threads'])
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    ds = DirectDataset(plan['dataset_directory']); codec = ActionCodec(plan['codecs'][phase][method])
    cache_audit = cache_episode_batches(ds, plan) if phase == 'overfit' else None
    init = restricted_load(plan['initial_path'])
    if tensor_hash(init['model']) != plan['initial_weight_identity'] or normalizer_hash(init['normalizer']) != plan['robot_normalizer_identity']:
        raise ValueError('initial weight/normalizer identity mismatch')
    model = IntegrationPolicy(ds.schema, **config['model']).to(device)
    model.load_state_dict(init['model'], strict=True); norm = load_normalizer(init['normalizer'])
    opt, sched = optimizer(model, plan, phase); ident = make_identity(plan, plan_path, phase, method)
    progress = dict(step=0, epoch=1, chunk=0, hidden={}, best=None, best_epoch=None, history=[], optimization_sum=0., targets=0)
    if resume is not None:
        original_run=Path(read(resume)['path']).parent.parent
        if phase=='full' and (original_run/'test_started.json').exists():
            raise RuntimeError('test already consumed; refusing automatic repeat on resume')
        cp, _ = load_checkpoint(resume, ident); model.load_state_dict(cp['model'], strict=True)
        opt.load_state_dict(cp['optimizer']); sched.load_state_dict(cp['scheduler']); restore_rng(cp['rng'])
        progress = cp['progress']
    kind = 'probe' if probe else 'train'
    with lock(Path(plan_path).parent/f'{phase}-{method}-{kind}.lock'):
        with ExperimentLogger(team='integration', model=f'task08_{phase}_{method}_{kind}', seed=42,
            config=dict(identity=ident, action_codec=codec.state, config=config, resume=str(resume)),
            root=plan['experiments_root'], job_type=kind) as logger:
            run = logger.directory; events = Events(run)
            atomic(Path(plan_path).parent/f'{phase}-{method}-{kind}.json', dict(run_id=logger.run_id, directory=str(run.resolve()), pid=os.getpid(), status='RUNNING'))
            atomic(run/'identity.json', ident, immutable=True); atomic(run/'action_codec.json', codec.state, immutable=True)
            if cache_audit is not None: atomic(run/'batch_cache_audit.json', cache_audit, immutable=True)
            if resume is not None:
                logger.record(source_run_id=read(resume)['run_id'],resume_checkpoint=read(resume))
                if (original_run/'best.json').exists():
                    load_checkpoint(original_run/'best.json',ident)
                    atomic(run/'best.json',read(original_run/'best.json'))
                for name in ['history.jsonl','batch_identity.jsonl']:
                    if (original_run/name).exists():
                        rows=[json.loads(line) for line in (original_run/name).read_text().splitlines()]
                        for row in rows:
                            if row['step']<=progress['step']:append(run/name,row)
            # Source snapshot and reproducibility records are part of every run.
            from graph.integration_baseline.bc08_report import snapshot
            snapshot(run, plan)
            stop = [False]
            def request_stop(*_): stop[0] = True
            signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
            try:
                events.emit(f'{phase}_{kind}_start', method=method, pid=os.getpid(), run_id=logger.run_id)
                if probe:
                    # Real B2 x T32 backward plus paired simultaneous resident probe.
                    probe_batch=plan.get('probe_batch',dict(episode_ids=list(chunks(plan,phase,1)[0][0]),start=0))
                    ids=probe_batch['episode_ids'];length=config['training']['sequence_length']
                    g, s, y, valid = batch(ds, ids, probe_batch['start'], length, device, config)
                    model.eval()
                    with torch.no_grad():
                        initial_hidden=torch.randn(1,len(ids),config['model']['gru_hidden'],device=device)
                        sequence_output,sequence_hidden=model(g,s,norm,initial_hidden)
                        running_hidden=initial_hidden.clone();steps=[]
                        for offset in range(length):
                            gt,st,_,_=batch(ds,ids,probe_batch['start']+offset,1,device,config)
                            one,running_hidden=model(gt,st,norm,running_hidden);steps.append(one[:,0])
                        torch.testing.assert_close(sequence_output,torch.stack(steps,1),atol=1e-6,rtol=1e-5)
                        torch.testing.assert_close(sequence_hidden,running_hidden,atol=1e-6,rtol=1e-5)
                    model.train()
                    p, h = model(g, s, norm); loss, gn = update(model, opt, codec, p, y, valid, 1.)
                    probe_progress=dict(step=1,epoch=1,hidden=h.detach().cpu())
                    receipt=save(run,model,opt,sched,norm,codec,ident,probe_progress)
                    from copy import deepcopy
                    rng_expected=(float(np.random.rand()),torch.rand(3),torch.rand(3,device=device))
                    p2,_=model(g,s,norm,h.detach());update(model,opt,codec,p2,y,valid,1.)
                    expected=tensor_hash(model.state_dict());expected_opt=deepcopy(opt.state_dict())
                    cp,_=load_checkpoint(receipt,ident)
                    clone=IntegrationPolicy(ds.schema,**config['model']).to(device);clone.load_state_dict(cp['model'],strict=True)
                    clone_opt,clone_sched=optimizer(clone,plan,phase)
                    clone_opt.load_state_dict(cp['optimizer']);clone_sched.load_state_dict(cp['scheduler']);restore_rng(cp['rng'])
                    rng_actual=(float(np.random.rand()),torch.rand(3),torch.rand(3,device=device))
                    if rng_expected[0]!=rng_actual[0] or not all(torch.equal(rng_expected[i],rng_actual[i]) for i in [1,2]):
                        raise ValueError('real-device RNG resume mismatch')
                    p3,_=clone(g,s,norm,cp['progress']['hidden'].to(device));update(clone,clone_opt,codec,p3,y,valid,1.)
                    if tensor_hash(clone.state_dict())!=expected:raise ValueError('real-device optimizer/hidden resume mismatch')
                    for key,values in clone_opt.state_dict()['state'].items():
                        for name,value in values.items():
                            if torch.is_tensor(value):torch.testing.assert_close(value,expected_opt['state'][key][name],atol=0,rtol=0)
                    torch.cuda.synchronize() if device == 'cuda' else None
                    atomic(run/'probe.json', dict(status='PASS', memory=memory(model,opt), loss=loss, gradient_norm=gn,
                        pid=os.getpid(), batch_size=len(ids), sequence_length=length,paired=paired,
                        sequence_step_equivalence=True,exact_model_optimizer_rng_hidden_resume=True), immutable=True)
                    if paired:
                        barrier=Path(plan_path).parent/'probe_release'
                        import time
                        deadline=perf_counter()+120
                        while not barrier.exists():
                            if perf_counter()>deadline: raise TimeoutError('paired probe release missing')
                            time.sleep(.2)
                    from graph.integration_baseline.bc08_report import verify
                    atomic(run/'verification.json',verify(run),immutable=True)
                    return dict(directory=str(run.resolve()), status='PROBE_COMPLETE')
                save(run,model,opt,sched,norm,codec,ident,progress)
                previous_initial=original_run/'initial_train_metrics.json' if resume is not None else None
                initial_metrics = read(previous_initial) if previous_initial is not None and previous_initial.exists() else evaluate(ds, model, norm, codec, ident['train_ids'], config)
                atomic(run/'initial_train_metrics.json', initial_metrics, immutable=True)
                t = config['training']; tick = perf_counter()
                done = False
                while not done:
                    schedule = chunks(plan,phase,progress['epoch'])
                    for index in range(progress['chunk'],len(schedule)):
                        if stop[0] or (Path(plan_path).parent/'STOP').exists():
                            save(run,model,opt,sched,norm,codec,ident,progress)
                            atomic(run/'completion.json',dict(status='STOPPED',last_valid=read(run/'last_valid.json')))
                            return dict(directory=str(run.resolve()),status='STOPPED')
                        if __import__('shutil').disk_usage(run).free < plan['minimum_free_bytes']:
                            raise OSError('disk resource gate')
                        ids,start=schedule[index]
                        hidden=None if start==0 else torch.cat([progress['hidden'][e] for e in ids],1).to(device)
                        model.train();g,s,y,valid=batch(ds,list(ids),start,t['sequence_length'],device,config)
                        raw,h=model(g,s,norm,hidden)
                        loss,gn=update(model,opt,codec,raw,y,valid,t['gradient_clip'])
                        progress['hidden']={e:h[:,i:i+1].detach().cpu() for i,e in enumerate(ids)}
                        progress['step']+=1;progress['chunk']=index+1
                        progress['optimization_sum']+=loss*int(valid.sum());progress['targets']+=int(valid.sum())
                        append(run/'batch_identity.jsonl',dict(step=progress['step'],epoch=progress['epoch'],episode_ids=list(ids),start=start,
                            input_identity=digest([list(ids),start,[plan['episode_input_hashes'][str(e)] for e in ids]])))
                        atomic(run/'progress.json',dict(step=progress['step'],epoch=progress['epoch'],chunk=index+1,loss=loss,pid=os.getpid()))
                        if phase=='overfit':sched.step()
                        milestone=phase=='overfit' and progress['step'] in tuple(plan['milestones'])
                        epoch_end=index==len(schedule)-1
                        if milestone or (phase=='full' and epoch_end):
                            train_metrics=evaluate(ds,model,norm,codec,ident['train_ids'],config,
                                run/'offline'/f'step{progress["step"]}' if milestone else None)
                            val=evaluate(ds,model,norm,codec,plan['splits']['validation'],config) if phase=='full' else None
                            if phase=='overfit':
                                reduction=1-train_metrics['native_action7_mse']/initial_metrics['native_action7_mse']
                                passes=all(train_metrics[k]<=v for k,v in plan['offline_fit_thresholds'].items()) and reduction>=.95 and train_metrics['gripper_accuracy']>=.99
                                improved=passes and (progress['best'] is None or train_metrics['native_action7_mse']<progress['best'])
                            else:
                                improved=val is not None and (progress['best'] is None or val['native_action7_mse']<progress['best'])
                            if improved:
                                progress['best']=(train_metrics if phase=='overfit' else val)['native_action7_mse']
                                progress['best_epoch']=progress['epoch']
                            row=dict(step=progress['step'],epoch=progress['epoch'],train=train_metrics,validation=val,
                                optimization_loss=progress['optimization_sum']/progress['targets'],gradient_norm=gn,
                                learning_rate=opt.param_groups[0]['lr'],memory=memory(model,opt),elapsed_seconds=perf_counter()-tick)
                            progress['history'].append(row);append(run/'history.jsonl',row);logger.log(row,step=progress['step'])
                            events.emit('epoch_summary' if phase=='full' else 'overfit_summary',method=method,step=progress['step'],
                                epoch=progress['epoch'],train_action7_mse=train_metrics['native_action7_mse'],
                                val_action7_mse=None if val is None else val['native_action7_mse'])
                            # Full scheduler update occurs once at each epoch; all methods same budget.
                            if phase=='full':sched.step()
                            receipt=save(run,model,opt,sched,norm,codec,ident,progress,best=improved)
                            logger.link_output(receipt['path'],kind='checkpoint')
                            if improved:events.emit('best_checkpoint',method=method,receipt=receipt)
                        if phase=='overfit' and progress['step']==plan['overfit_updates']:
                            done=True;break
                    if phase=='full' and progress['epoch']==t['max_epochs']:
                        done=True
                    if not done:
                        progress.update(epoch=progress['epoch']+1,chunk=0,hidden={},optimization_sum=0.,targets=0)
                receipt=read(run/'best.json') if phase=='full' or (run/'best.json').exists() else read(run/'last_valid.json')
                cp,_=load_checkpoint(receipt,ident);model.load_state_dict(cp['model'],strict=True)
                # Save/reload output equivalence is checked on an actual input.
                g,s,y,valid=batch(ds,list(ident['train_ids'][:2]),0,t['sequence_length'],device,config)
                model.eval()
                with torch.inference_mode(): expected=model(g,s,norm)[0]
                clone=IntegrationPolicy(ds.schema,**config['model']).to(device);clone.load_state_dict(cp['model'],strict=True);clone.eval()
                with torch.inference_mode():torch.testing.assert_close(clone(g,s,norm)[0],expected,atol=0,rtol=0)
                final=evaluate(ds,model,norm,codec,ident['train_ids'],config,run/'offline'/'selected_train')
                if phase=='full':
                    atomic(run/'test_started.json',dict(checkpoint=receipt),immutable=True)
                    test=evaluate(ds,model,norm,codec,plan['splits']['test'],config,run/'offline'/'test')
                    atomic(run/'test_metrics.json',test,immutable=True);logger.log(test,prefix='test')
                reduction=1-final['native_action7_mse']/initial_metrics['native_action7_mse']
                fit='OFFLINE_FIT_PASS' if all(final[k]<=v for k,v in plan['offline_fit_thresholds'].items()) and reduction>=.95 and final['gripper_accuracy']>=.99 else 'OFFLINE_FIT_INSUFFICIENT'
                result=dict(status='COMPLETE',directory=str(run.resolve()),selected=receipt,train=final,
                    offline_fit=fit if phase=='overfit' else None,relative_train_error_reduction=reduction,
                    updates=progress['step'],epochs=progress['epoch'],best_epoch=progress['best_epoch'],
                    best_validation_native_action7_mse=progress['best'],
                    memory=memory(model,opt),reload_output_exact=True,phase=phase,method=method,
                    selection_rule='lowest native Action7 MSE among milestone checkpoints passing all four offline thresholds; otherwise final 20000-update checkpoint',
                    selection_scope='offline-only; closed-loop outcomes cannot alter checkpoint selection')
                atomic(run/'completion.json',result,immutable=True);logger.link_output(run/'completion.json',kind='metrics')
                from graph.integration_baseline.bc08_report import verify
                atomic(run/'verification.json',verify(run),immutable=True)
                logger.link_output(run/'verification.json',kind='metrics')
                if phase=='full':logger.link_output(run/'test_metrics.json',kind='metrics')
                events.emit(f'{phase}_training_complete',method=method,run_id=logger.run_id,selected=receipt)
                return result
            except BaseException as exc:
                # Last atomic valid checkpoint survives OOM/kill; never serialize a
                # potentially partially-mutated optimizer after an update failure.
                error_bundle(run,exc,ident);events.emit('error',error_type=type(exc).__name__,bundle=str(run/'error_bundle.json'))
                raise
            finally:
                events.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--phase',choices=['overfit','full'],required=True);p.add_argument('--method',choices=['A','B','C'],required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--probe',action='store_true')
    a=p.parse_args();result=worker(a.plan,a.phase,a.method,resume=a.resume,probe=a.probe)
    print(json.dumps(result),flush=True)
    if result['status']=='STOPPED':raise SystemExit(75)


if __name__=='__main__':main()
