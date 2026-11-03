"""Real-data contract gates. A failure prevents the long pipeline from starting."""
import argparse
import builtins
import copy
import os
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha,tensor_identity
from graph.graph_recurrent10g.data import Dataset
from graph.graph_recurrent10g.train_worker import save
from graph.plan_a_bc.train import assert_nested_equal
from graph.integration_baseline.graph import Graph,pack_graphs
from graph.gt_pretrain10f1.contract import split_fields,measured_robot,assemble_gt38
from graph.integration_baseline.direct_train import rng_state,restore_rng
from .model import Policy
from .train import setup,update,validate,payload,restore,Features
from .prepare import verify_cache


def close(a,b,cfg,action=False):
    tol=cfg['tolerances'];key='action' if action else 'state'
    np.testing.assert_allclose(a,b,atol=tol[key+'_atol'],rtol=tol[key+'_rtol'])


def state_contract(root):
    cfg,m,features,codec,opt,sched=setup(root);ds=Dataset(root)
    # Native real frames: full object presence, peg/hole missing, differing poses.
    presence=ds.pool.presence.numpy().astype(bool)
    ids=[];strata={}
    for name,mask in [('both',presence.all(1)),('peg_missing',~presence[:,0]),('hole_missing',~presence[:,1]),('both_missing',~presence.any(1))]:
        found=np.flatnonzero(mask)
        strata[name]=int(len(found))
        ids+=found[:2].tolist()
    ids=list(dict.fromkeys(ids+[0,100,1000,2000]))
    g,r,_=ds.select(ids,'cuda');before=m.frozen_hash()
    with torch.no_grad():
        obj,z,p=m.frozen(g,r)
        assert obj.shape==(len(ids),18) and z.shape==(len(ids),320) and p.shape==(len(ids),2)
        official=m.frozen.extract(g,r,ds.schema)['state38']
        wrapper=m.extract_state(g,r).cpu().numpy();close(official,wrapper,cfg)
        single=[]
        for i in ids:
            gi,ri,_=ds.select([i],'cuda');single.append(m.extract_state(gi,ri).cpu().numpy())
        close(wrapper,np.concatenate(single),cfg)
        np.testing.assert_allclose(np.linalg.norm(wrapper[:,18:22],axis=1),1,atol=2e-5,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(wrapper[:,28:32],axis=1),1,atol=2e-5,rtol=0)
        assert (wrapper[:,32:36]>0).all() and (wrapper[:,32:36]<1).all()
        np.testing.assert_array_equal(wrapper[:,36:],np.tile([1.,0.],(len(ids),1)))
        measured=measured_robot(r.cpu().numpy(),ds.schema)
        np.testing.assert_array_equal(wrapper[:,:3],measured['tcp_pose'][:,:3])
        close(wrapper[:,7:10],measured['left_tip']-measured['tcp_pose'][:,:3],cfg)
        # Oracle branch is diagnostic only; never passed to controller.
        target=ds.pool.targets[ids].numpy()
        oracle=assemble_gt38(split_fields(target),measured)
        from recurrent.features import build_features
        raw=dict(measured,peg_pose=np.concatenate([target[:,:3]+measured['tcp_pose'][:,:3],target[:,3:7]],-1),
            hole_pose=np.concatenate([target[:,7:10]+measured['tcp_pose'][:,:3],target[:,10:14]],-1),
            peg_half=target[:,14:17],hole_radius=target[:,17:18])
        expected=build_features(raw,np.ones(len(ids),'f'),np.zeros(len(ids),'f'))
        np.testing.assert_allclose(oracle,expected,atol=cfg['tolerances']['oracle_assembly_atol'],rtol=0)
        # A valid empty current object graph retains mandatory measured robot nodes.
        gi,ri,_=ds.select([ids[0]],'cpu')
        for kinds in [(0,1,2),(0,1,2,3),(0,1,2,4)]:
            keep=torch.tensor([int(k) in kinds for k in gi.node_type]);remap=torch.full((len(keep),),-1,dtype=torch.long)
            remap[keep]=torch.arange(int(keep.sum()));edge=keep[gi.edge_index].all(0)
            gg=Graph(gi.node_input[keep],gi.node_type[keep],remap[gi.edge_index[:,edge]],gi.edge_attr[edge],gi.edge_type[edge],{}, {})
            packed=pack_graphs([gg],[(0,0)]).to('cuda')
            state=m.extract_state(packed,ri.cuda())
            assert state.shape==(1,38) and torch.isfinite(state).all()
        # Make sidecar/expert/object GT access fatal while forwarding actual data.
        original=builtins.open
        def guarded(file,*args,**kwargs):
            text=str(file).lower()
            if any(k in text for k in ['gt_supervision','gt38','sidecar','expert_stages']):
                raise AssertionError('GT file access in sensor-only forward: '+text)
            return original(file,*args,**kwargs)
        with patch('builtins.open',guarded),patch('recurrent.common.raw_gt',side_effect=AssertionError('GT accessor forbidden')):
            action,hidden=m(g,r,None,None)
            assert action.shape[-1]==7 and torch.isfinite(action).all()
        assert before==m.frozen_hash()
    write(root/'state38_contract_verification.json',dict(status='PASS',frames=ids,observed_strata=strata,
        readout_dimension=320,head_dimension=18,state_dimension=38,single_batch_equivalent=True,
        oracle_assembly=True,empty_graph='no object nodes; three named robot nodes remain under original PlanA contract',
        synthesized_missing_from_real_graph=True,GT_access_intentionally_blocked=True,
        dtype=str(state.dtype),device=str(state.device),shape_range_m=[float(wrapper[:,32:36].min()),float(wrapper[:,32:36].max())],
        tolerance=cfg['tolerances']))
    # Equivalence of direct single-frame online input and stored batch-extracted cache.
    rows=[]
    for eid,e in list(ds.episodes.items())[:3]:
        for t in [0,min(31,e['actions']-1),e['actions']-1]:
            gi,ri,_=ds.select([e['indices'][t]],'cuda')
            with torch.no_grad():
                direct=m.extract_state(gi,ri);cached=torch.tensor(features.arrays[eid]['state38'][t:t+1],device='cuda')
                close(direct.cpu(),cached.cpu(),cfg)
                assert np.all(np.isclose(m.normalize(direct).cpu().numpy(),m.normalize(cached).cpu().numpy(),atol=cfg['tolerances']['state_atol']/m.state_std.cpu().numpy(),rtol=cfg['tolerances']['state_rtol']))
                h=torch.randn(1,1,144,device='cuda')
                a,_=m.from_states(direct[:,None],h);b,_=m.from_states(cached[:,None],h)
                close(a.cpu(),b.cpu(),cfg,True)
                rows.append(dict(episode=eid,timestep=t,state_max_abs=float((direct-cached).abs().max()),action_max_abs=float((a-b).abs().max())))
    write(root/'offline_online_feature_equivalence.json',dict(status='PASS',cases=rows,normalizer='same model buffers',tolerance=cfg['tolerances']))
    # Actual graph-forward BC backward, not merely cache-only gradients.
    mixed=None
    for eid,e in ds.episodes.items():
        for start in range(0,e['actions'],32):
            selection=ds.mask[e['indices'][start:start+32]]
            if selection.any() and not selection.all():mixed=([int(eid)],start);break
        if mixed:break
    assert mixed is not None,'actual disturbance/supervised mixed chunk required'
    gg,rr,y,mask,_,_=ds.batch(mixed[0],mixed[1],32,'cuda')
    assert mask.any() and (~mask).any()
    m.train();opt.zero_grad(set_to_none=True);a,_=m(gg,rr)
    old=[p.detach().clone() for p in m.policy.parameters()];loss=codec.loss(a,y,mask);loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in m.frozen.parameters())
    assert all(not module.training for module in m.frozen.modules())
    grads={name:float(torch.stack([p.grad.norm() for p in module.parameters()]).norm()) for name,module in
        [('projection',m.policy.encoder),('gru',m.policy.rnn),('action',m.policy.head)]}
    assert all(v>0 and np.isfinite(v) for v in grads.values())
    assert all(torch.isfinite(p.grad).all() for p in m.policy.parameters())
    frozen_ids={id(p) for p in m.frozen.parameters()}
    assert not frozen_ids&{id(p) for group in opt.param_groups for p in group['params']}
    opt.step();assert any(not torch.equal(a,b) for a,b in zip(old,m.policy.parameters()))
    assert before==m.frozen_hash()
    # Masked labels cannot affect loss; valid recovery labels do affect loss.
    yy=y.clone();yy[~mask]=1000000
    assert torch.equal(codec.loss(a,y,mask),codec.loss(a,yy,mask))
    changed=y.clone();changed[mask]+=100
    assert not torch.equal(codec.loss(a,y,mask),codec.loss(a,changed,mask))
    write(root/'gradient_freeze_verification.json',dict(status='PASS',loss=float(loss.detach()),gradients=grads,
        frozen_hash_before=before,frozen_hash_after=m.frozen_hash(),optimizer_excludes_frozen=True,
        changed_controller_only=True,masked_label_invariance=True,supervised_labels_affect_loss=True,all_frozen_modules_eval=True,actual_masked_frames=int((~mask).sum()),actual_supervised_frames=int(mask.sum())))


def training_contract(root):
    cfg,m,ds,codec,opt,sched=setup(root);schedule=read(root/'sampling_schedule.json')
    p=dict(step=0,cursor=0,hidden={},best_loss=None,best_step=None)
    torch.cuda.reset_peak_memory_stats();initial=m.frozen_hash();initial_control=m.control_hash()
    # Five real mixed-episode scheduled updates; smoke weights never reach full training.
    while p['step']<5:update(m,ds,codec,opt,sched,p,schedule[p['cursor']],cfg)
    val=validate(m,ds,codec)
    path=root/'preflight/smoke.pt';save(path,payload(root,m,opt,sched,p,{'frozen':initial},'engineering'))
    ck=torch.load(path,map_location='cpu',weights_only=False)
    next_entry=schedule[p['cursor']]
    metric_a=update(m,ds,codec,opt,sched,p,next_entry,cfg)
    state_a=copy.deepcopy(m.state_dict());opt_a=copy.deepcopy(opt.state_dict());sched_a=copy.deepcopy(sched.state_dict())
    cfg2,m2,ds2,codec2,opt2,sched2=setup(root)
    p2=restore(root,ck,m2,opt2,sched2)
    metric_b=update(m2,ds2,codec2,opt2,sched2,p2,next_entry,cfg2)
    assert_nested_equal(state_a,m2.state_dict());assert_nested_equal(opt_a,opt2.state_dict())
    assert_nested_equal(sched_a,sched2.state_dict());assert_nested_equal(p,p2);assert metric_a==metric_b
    assert initial==m2.frozen_hash() and initial_control!=m2.control_hash()
    # Confirm deterministic fresh initialization after smoke, with no smoke state loaded.
    _,fresh,_,_,_,_=setup(root);assert fresh.control_hash()==initial_control
    torch.save(dict(initial_rng=fresh.initial_rng,controller=fresh.policy.state_dict(),
                    cuda_rng=torch.cuda.get_rng_state_all()),root/'preflight/fresh_initialization.pt')
    write(root/'training_smoke_verification.json',dict(status='PASS',optimizer_updates=5,
        validation=val,checkpoint_save_readback=True,exact_next_update_resume=True,
        normalizer_save_load=True,fresh_full_training_initialization=True,
        frozen_hash=initial,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    # Test branch lifecycle, non-supervised observations, final partial chunks.
    md=read(root/'host_data/metadata.json');phases=md['phases'];actions=np.load(root/'host_data/actions.npy',mmap_mode='r')
    masks=np.load(root/'host_data/bc_mask.npy',mmap_mode='r');disturbance=0;recovery=0
    for e in md['episodes'].values():
        for i in e['indices']:
            if phases[i]=='disturbance':assert not masks[i];disturbance+=1
            if phases[i]=='recovery':assert masks[i];recovery+=1
    assert disturbance>0 and recovery>0
    partial=next(e for e in ds.episodes.values() if e['frames']%32!=0)
    # Batched vs sequential valid inputs and hidden; includes inactive padded stream.
    eids=[int(min(ds.episodes,key=lambda e:ds.episodes[e]['frames'])),int(max(ds.episodes,key=lambda e:ds.episodes[e]['frames']))]
    x,y,mask,occ=ds.batch(eids,0)
    with torch.no_grad():
        a,h=m.from_states(x,None,occ)
        for b in range(len(eids)):
            hh=None;aa=[]
            for t in range(int(occ[b].sum())):
                z,hh=m.from_states(x[b:b+1,t:t+1],hh);aa.append(z)
            close(a[b:b+1,:len(aa)].cpu(),torch.cat(aa,1).cpu(),cfg,True)
            close(h[:,b:b+1].cpu(),hh.cpu(),cfg,True)
    last_eid=next(e for e in ds.episodes if ds.episodes[e] is partial)
    xx,yy,ms,occupied=ds.batch([int(last_eid)],(partial['frames']//32)*32)
    assert xx.shape[1]==partial['frames']%32 and occupied.all()
    with torch.no_grad():m.from_states(xx,None,occupied)
    assert all(v.grad_fn is None for v in p['hidden'].values())
    # Fixed schedule guarantees no branch hidden reuse; verify ordered cursors per group.
    previous=None
    for entry in schedule:
        if entry['start']==0:previous=set(entry['ids'])
        else:assert set(entry['ids'])<=previous
        assert entry['start']%32==0
    write(root/'sequence_hidden_verification.json',dict(status='PASS',disturbance_frames_retained=disturbance,
        recovery_BC_frames=recovery,episode_reset=True,TBPTT_carry_detach=True,branch_separation=True,
        batched_sequential_equivalence=True,partial_episode_frames=partial['frames'],schedule_sha256=sha(root/'sampling_schedule.json'),
        dataloader='in-memory immutable numpy features, zero worker processes; worker exceptions/restart not applicable',
        iterator_restart_test='Features re-created in setup and exact next-update replay',
        CUDA_OOM=False,free_disk_bytes=shutil.disk_usage(root).free,allocator=os.getenv('PYTORCH_ALLOC_CONF')))


def gate(root):
    names=['source_checkpoint_audit','state38_contract_verification','offline_online_feature_equivalence',
           'gradient_freeze_verification','sequence_hidden_verification','training_smoke_verification','evaluation_smoke_verification',
           'train_feature_cache_verification','data_reconstruction_verification','pipeline_engine_verification','regression_verification','final_source_runtime_verification']
    result={name:dict(status=read(root/(name+'.json'))['status'],evidence=name+'.json',sha256=sha(root/(name+'.json'))) for name in names}
    assert all(v['status']=='PASS' for v in result.values())
    write(root/'preflight_verification.json',dict(status='PASS',checks=result,
        required_contracts_all_pass=True,full_training_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['contracts','gate'],required=True);a=p.parse_args()
    try:
        if a.stage=='gate':gate(a.root)
        else:verify_cache(a.root);state_contract(a.root);training_contract(a.root)
    except BaseException:
        write(a.root/'preflight_verification.json',dict(status='FAIL',stage=a.stage,traceback=traceback.format_exc(),full_training_started=False))
        raise
