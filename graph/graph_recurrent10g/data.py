"""Read-only host mmap graphs; separate loss labels; bounded CUDA minibatches."""
from pathlib import Path
from collections import Counter
import json
import os
import shutil
import h5py
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha,digest
from graph.gt_pretrain10f1.data import GraphPool
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from graph.integration_baseline.graph import PackedGraphs


def prepare(root):
    cfg=read(root/'config.json');f1=Path(cfg['f1_root'])
    shared=cfg.get('reuse_verified_data') or cfg.get('miniature_root')
    if shared:return reuse_prepare(root,Path(shared))
    if sha(f1/'final_dataset_manifest.json')!=cfg['dataset_manifest_sha256']:
        raise ValueError('F1 dataset changed')
    pool=GraphPool(f1,'cpu',allow_final=False)
    manifest=pool.manifest;specs={s['episode_id']:s for s in manifest['episode_specs']}
    episodes={};actions=np.zeros((len(pool.meta),7),np.float32)
    masks=np.zeros(len(pool.meta),bool);phases=['pair']*len(pool.meta);sources=[]
    # Validate masked branch prefixes against their approved nominal expert source.
    nominal={}
    for e in manifest['episodes']:
        spec=specs[e['episode_id']]
        if spec['split']=='final':continue
        if spec.get('branch_id')=='nominal':nominal[spec['seed']]=e
    counts=Counter();parents={}
    for e in manifest['episodes']:
        eid=e['episode_id'];spec=specs[eid]
        if spec['split']=='final':continue
        parent=spec['seed'];split=spec['split'];n=e['graphs']
        if parent in parents and parents[parent]!=split:raise ValueError('parent leakage')
        parents[parent]=split
        path=f1/e['path'];raw=Path(e.get('source_raw_path',f1/'raw'/str(eid)/'episode.h5'))
        if sha(raw)!=e['raw_episode_sha256']:raise ValueError('raw SHA mismatch')
        with np.load(path,allow_pickle=False) as z,h5py.File(raw) as f:
            np.testing.assert_array_equal(z['timestep'],np.arange(n))
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1))
            np.testing.assert_array_equal(z['robot76'],f['robot_state'][:n])
            np.testing.assert_array_equal(z['action7'],f['actions'][:])
            if len(f['robot_state'])!=n+1 or len(f['actions'])!=n:
                raise ValueError('T actions / T+1 observations required')
            y=f['actions'][:];mask=np.ones(n,bool)
            if 'segment' in f:
                segment=f['segment'][:];mask=f['supervision_mask'][:]
                if mask.dtype!=bool or mask.shape!=(n,):raise ValueError('BC mask schema')
                if (mask & (segment==1)).any():raise ValueError('disturbance has expert label')
                prefix=np.flatnonzero((segment==0)&~mask)
                if len(prefix):
                    ne=nominal[parent]
                    with h5py.File(ne['source_raw_path']) as nf:
                        # Source collector explicitly copied these expert prefixes.
                        for key in ('actions','robot_state','timestep'):
                            np.testing.assert_array_equal(f[key][:len(prefix)],nf[key][:len(prefix)])
                    if not np.array_equal(prefix,np.arange(len(prefix))):raise ValueError('prefix order')
                    mask=mask.copy();mask[prefix]=True
                if not np.array_equal(mask,segment!=1):raise ValueError('prefix/disturbance/recovery labels')
                if np.any(np.diff(segment.astype(int))<0):raise ValueError('segment order')
                phase=np.array(['expert_prefix','disturbance','recovery'])[segment]
            else:
                stages=read(raw.parent/'expert_stages.json')
                phase=np.asarray([s['phase'] for s in stages])
            if not np.isfinite(y).all() or np.abs(y[:,6]).max()>1.000001:raise ValueError('finite Action7 / gripper contract')
            scaled=controller_commands(y)
            if 'controller_input' in f:
                np.testing.assert_array_equal(y,f['controller_input'][:])
                np.testing.assert_allclose(scaled[:,:6],f['arm_scaled_input'][:],atol=2e-8,rtol=1e-6)
                np.testing.assert_allclose(scaled[:,6:],f['gripper_scaled_input'][:],atol=1e-8,rtol=0)
            else:
                trace=read(raw.parent/'trace.json')
                for tr in trace:
                    t=int(tr['clock'])-1
                    np.testing.assert_array_equal(y[t],np.asarray(tr['action'],np.float32))
                    np.testing.assert_allclose(scaled[t,:6],tr['scaled_action'],atol=2e-8,rtol=1e-6)
            indices=[pool.index[eid,t] for t in range(n)]
            actions[indices]=y;masks[indices]=mask
            for idx,phase_name in zip(indices,phase):phases[idx]=str(phase_name)
        counts.update({split+'_episodes':1,split+'_frames':n,split+'_BC_frames':int(mask.sum())})
        episodes[str(eid)]=dict(indices=indices,actions=n,split=split,parent=parent,
            branch=spec.get('branch_id','full_task'),source=spec.get('source','full_task'))
        sources.append(dict(path=str(raw),sha256=sha(raw),graphs=str(path),graph_sha256=sha(path)))
    if (counts['train_episodes'],counts['validation_episodes'],counts['train_frames'],counts['validation_frames'])!=(272,48,21121,3704):
        raise ValueError('released split counts differ')
    folder=root/'host_data';folder.mkdir(exist_ok=True)
    values={**{k:v.numpy() for k,v in pool.values.items()},'ptr':pool.ptr.numpy(),'eptr':pool.eptr.numpy(),
        'node_counts':pool.node_counts,'edge_counts':pool.edge_counts,'robot':pool.robot.numpy(),
        'targets':pool.targets.numpy(),'presence':pool.presence.numpy(),'actions':actions,'bc_mask':masks}
    hashes={}
    for k,v in values.items():
        p=folder/(k+'.npy');tmp=p.with_suffix('.partial')
        with tmp.open('wb') as f:np.save(f,v,allow_pickle=False);f.flush();os.fsync(f.fileno())
        np.testing.assert_array_equal(np.load(tmp,mmap_mode='r'),v);os.replace(tmp,p)
        hashes[p.name]=sha(p)
    write(folder/'metadata.json',dict(episodes=episodes,meta=pool.meta,pairs=pool.pairs,
        phases=phases,schema=pool.schema,normalization=pool.normalization,hashes=hashes))
    for name in ('normalizer.json','runtime_paths.json','renderer_gpu_gate.json'):
        shutil.copy2(f1/name,root/name)
    if not (root/'inputs').exists():shutil.copytree(f1/'inputs',root/'inputs')
    make_schedule(root,episodes,actions,masks,pool.meta)
    write(root/'dataset_identity.json',dict(status='PASS',counts=dict(counts),sources=sources,
        action_contract='unaltered raw expert commands; installed controller translation component clipping and rotation unit-norm clipping verified against stored scaled traces; raw rotation components may exceed1',
        original_manifest_sha256=cfg['dataset_manifest_sha256'],host_metadata_sha256=sha(folder/'metadata.json'),
        native_runtime_only=True,GT_input=False,terminal_Tplus1_excluded=True))
    return dict(status='PASS',counts=dict(counts),files=[str(p.relative_to(root)) for p in folder.glob('*')]+[
        'dataset_identity.json','sampling_schedule.json','sampling_manifest.json','action_codec.json','normalizer.json'])


def make_schedule(root,episodes,actions,masks,meta):
    cfg=read(root/'config.json')
    train_actions={int(e):actions[s['indices']][masks[s['indices']]] for e,s in episodes.items() if s['split']=='train'}
    codec=ActionCodec.fit('B',train_actions,list(train_actions))
    write(root/'action_codec.json',codec.state)
    schedule=[];updates=0;epoch=0;exposure=Counter()
    while updates<cfg['training']['updates']:
        order=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch])).permutation(sorted(train_actions)).tolist()
        for group in range(0,len(order),2):
            ids=order[group:group+2]
            for start in range(0,max(episodes[str(e)]['actions'] for e in ids),32):
                active=[e for e in ids if start<episodes[str(e)]['actions']]
                idx=[i for e in active for i in episodes[str(e)]['indices'][start:start+32]]
                supervised=bool(masks[idx].any())
                if supervised:updates+=1
                schedule.append(dict(ids=active,start=start,epoch=epoch,update=updates,supervised=supervised))
                exposure.update(idx)
                if updates>=cfg['training']['updates']:break
            if updates>=cfg['training']['updates']:break
        epoch+=1
    write(root/'sampling_schedule.json',schedule)
    write(root/'sampling_manifest.json',dict(schedule_sha256=sha(root/'sampling_schedule.json'),
        optimizer_updates=updates,scan_chunks=len(schedule),frame_exposures=sum(exposure.values()),
        source_frame_exposures=dict(Counter(meta[i]['source'] for i,n in exposure.items() for _ in range(n))),
        train_parent_count=len({e['parent'] for e in episodes.values() if e['split']=='train'}),train_episode_count=272,
        corrective_prefix_rule='BC enabled only after exact approved nominal expert action/robot/timestep comparison',
        disturbance='scan history, never BC',epoch_permutation_seed=42,full_corrective_ratio='natural episode mixture96:176, fixed schedule',
        exposure_sha256=digest(sorted(exposure.items()))))


def reuse_prepare(root,source):
    cfg=read(root/'config.json');identity=read(source/'dataset_identity.json')
    if identity['status']!='PASS' or identity['original_manifest_sha256']!=cfg['dataset_manifest_sha256']:
        raise ValueError('shared host data release identity')
    md=read(source/'host_data/metadata.json')
    for name,h in md['hashes'].items():
        if sha(source/'host_data'/name)!=h:raise ValueError('shared host data hash '+name)
    for record in identity['sources']:
        if sha(record['path'])!=record['sha256'] or sha(record['graphs'])!=record['graph_sha256']:
            raise ValueError('original data source changed')
    if not (root/'host_data').exists():(root/'host_data').symlink_to((source/'host_data').resolve(),target_is_directory=True)
    for name in ('normalizer.json','runtime_paths.json','renderer_gpu_gate.json','dataset_identity.json'):
        shutil.copy2(source/name,root/name)
    if not (root/'inputs').exists():shutil.copytree(source/'inputs',root/'inputs')
    actions=np.load(root/'host_data/actions.npy',mmap_mode='r');masks=np.load(root/'host_data/bc_mask.npy',mmap_mode='r')
    make_schedule(root,md['episodes'],actions,masks,md['meta'])
    write(root/'host_data_reuse.json',dict(source=str(source),metadata_sha256=sha(source/'host_data/metadata.json'),
        full_hash_readback=True,graph_recache=False,shared_readonly_host_mapping=True))
    return dict(status='PASS',files=[str(p.relative_to(root)) for p in (root/'host_data').glob('*')]+[
        'dataset_identity.json','sampling_schedule.json','sampling_manifest.json','action_codec.json','normalizer.json','host_data_reuse.json'])


class Dataset:
    def __init__(self,root):
        self.root=Path(root);md=read(self.root/'host_data/metadata.json')
        self.md=md;self.episodes=md['episodes'];self.schema=md['schema']
        self.pool=object.__new__(GraphPool);p=self.pool;p.device='cpu'
        p.values={k:torch.from_numpy(np.load(self.root/'host_data'/(k+'.npy'),mmap_mode='c'))
            for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
        for k in ('ptr','eptr','robot','targets','presence'):
            setattr(p,k,torch.from_numpy(np.load(self.root/'host_data'/(k+'.npy'),mmap_mode='c')))
        p.node_counts=np.load(self.root/'host_data/node_counts.npy',mmap_mode='r')
        p.edge_counts=np.load(self.root/'host_data/edge_counts.npy',mmap_mode='r')
        self.actions=np.load(self.root/'host_data/actions.npy',mmap_mode='r')
        self.mask=np.load(self.root/'host_data/bc_mask.npy',mmap_mode='r')
        self.pairs=[p for p in md['pairs'] if p['split']=='train' and p['valid']]

    def select(self,indices,device):
        g,r,y=self.pool.select(indices)
        return g.to(device),r.to(device),y.to(device)

    def batch(self,ids,start,length,device):
        indices=[];fm=[];count=max(min(length,self.episodes[str(e)]['actions']-start) for e in ids)
        target=np.zeros((len(ids),count,7),np.float32);mask=np.zeros((len(ids),count),bool)
        for b,e in enumerate(ids):
            ix=self.episodes[str(e)]['indices'][start:start+length]
            indices+=ix;fm +=[(b,t) for t in range(len(ix))]
            target[b,:len(ix)]=self.actions[ix];mask[b,:len(ix)]=self.mask[ix]
        g,r,y=self.select(indices,device);g.frame_map=torch.tensor(fm,device=device)
        return g,r,torch.tensor(target,device=device),torch.tensor(mask,device=device),y,indices
