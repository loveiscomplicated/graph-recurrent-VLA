"""Single CUDA vision owner + inherited CPU geometry workers and atomic writer."""
import argparse,copy,os,time,shutil
from pathlib import Path
import h5py,numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,digest
from graph.plan_a_bc.cache import make_vision,cache_episode,input_identity
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode as parallel_episode
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from graph.geometry_reach10d.profile import Resources
from utils.experiment_logger import ExperimentLogger

def candidates(root):
    plan=read(root/'full_experiment_plan.json');rows=[]
    for i,s in enumerate(plan['train_parent96']):rows.append(dict(episode_id=5100000+i,seed=s,parent=s,split='train',kind='nominal',folder=str(root/'nominal/raw'/str(5100000+i))))
    for i,s in enumerate(plan['development_nominal_seeds']):rows.append(dict(episode_id=5200000+i,seed=s,parent=s,split='validation',kind='nominal',folder=str(root/'nominal/raw'/str(5200000+i))))
    for split,parents in [('train',plan['corrective_train_parents']),('validation',plan['development_corrective_parents'])]:
        for i,s in enumerate(parents):
            for j,anchor in enumerate(plan['anchors']):
                for k,offset in enumerate(plan['corrective_train_offsets_mm']):
                    identity=f'{s}_t{anchor}_{offset[0]}_{offset[1]}'
                    base=root/('pilot03' if k<4 else 'additional10mm') if split=='train' else root/'development'
                    eid=(6100000 if split=='train' else 6200000)+i*16+j*8+k
                    if split=='train' and i==0 and j==0 and k==0:eid=4100000
                    if split=='train' and i==4 and j==1 and k==0:eid=4100010
                    rows.append(dict(episode_id=eid,seed=s,parent=s,split=split,kind='recovery',anchor=anchor,offset_mm=offset,folder=str(base/'raw'/identity/'P')))
    return rows

def source_entry(c):
    folder=Path(c['folder']);p=folder/'receipt.json'
    if not p.exists():return None
    r=read(p)
    assert r.get('approved',r.get('status')=='ACCEPTED'),('rejected source cannot be silently excluded',p)
    raw=folder/'episode.h5';assert sha(raw)==r['sha256']
    return dict(episode_id=c['episode_id'],seed=c['seed'],split=c['split'],actions=r['actions'],path=str(raw),sha256=r['sha256'])

def mask_sidecar(folder,entry,candidate):
    target=folder/'supervision.npz'
    with h5py.File(entry['path']) as f:
        mask=f['supervision_mask'][:] if 'supervision_mask' in f else np.ones(entry['actions'],bool)
        segment=f['segment'][:] if 'segment' in f else np.full(entry['actions'],2,np.uint8)
        if not target.exists():
            temp=target.with_suffix('.partial')
            with temp.open('xb') as stream:np.savez_compressed(stream,supervision_mask=mask,segment=segment)
            os.link(temp,target);temp.unlink()
        with np.load(target) as a:
            np.testing.assert_array_equal(a['supervision_mask'],mask);np.testing.assert_array_equal(a['segment'],segment)
    return dict(path=str(target),sha256=sha(target),supervised=int(mask.sum()),raw_sha256=entry['sha256'],parent=candidate['parent'],kind=candidate['kind'])

def adopt(root,cache,c,entry,log):
    sources={4100000:root/'cache_gate/optimized/cache/ep4100000',4100010:root/'parallel_gate/w2/cache/ep4100010'}
    if c['episode_id'] not in sources:return None
    source=sources[c['episode_id']];r=read(source/'receipt.json');expected=input_identity(cache)
    assert r['raw_episode_sha256']==entry['sha256'] and sha(source/'graphs.npz')==r['sha256']
    for k in ['graph_config','frozen_geometry_protocol_sha256','frozen_source_sha256','cache_version']:assert r['identity'][k]==expected[k]
    folder=cache/'cache'/f'ep{c["episode_id"]}';folder.mkdir(parents=True,exist_ok=False)
    for n in ['graphs.npz','gate_trace.json']:os.link(source/n,folder/n)
    r.update(identity=expected,path=str((folder/'graphs.npz').relative_to(cache)),split=c['split'],adopted_from=str(source))
    write(folder/'receipt.json',r,immutable=True);log.link_output(folder/'receipt.json',kind='metrics');return r

def run(root,dino,sam):
    assert os.getenv('CONTAINER_ID')=='57640710';assert read(root/'pilot_completion_gate.json')['status']=='PASS';assert read(root/'parallel_gate/parallel_gate.json')['status']=='PASS'
    torch.set_num_threads(1);torch.manual_seed(42);cache=root/'full_cache';cache.mkdir(exist_ok=True);cs=candidates(root)
    for n in ['inputs','normalizer.json','action_codec.json']:
        if not (cache/n).exists():(cache/n).symlink_to(root/'pilot03'/n)
    cfg=read(root/'pilot03/config.json');cfg['training']['updates']=3000;cfg['splits']={k:[c['episode_id'] for c in cs if c['split']==k] for k in ['train','validation']};cfg['splits']['test']=[];cfg['episode_lengths']={}
    if not (cache/'config.json').exists():write(cache/'config.json',cfg,immutable=True)
    if not (cache/'dataset_audit.json').exists():write(cache/'dataset_audit.json',dict(dataset_sha256=sha(root/'full_experiment_plan.json'),split_sha256=digest(cfg['splits']),episodes=[],identity_semantics='fixed logical plan; each raw source SHA in receipt; complete content manifest frozen before training'),immutable=True)
    if not (cache/'cache_candidates.json').exists():write(cache/'cache_candidates.json',cs,immutable=True)
    detector,tracker=make_vision('cuda',dino,sam);identity=read(root/'cache_gate/corrective_cache_gate.json')['model_identity'];memo=DetectionMemo(detector,'Task10F_full',identity)
    import graph.plan_a_geometry.core as core
    core.components=components;pending=list(cs);completed=[];sidecars=[];pool=None;workers=None
    with ExperimentLogger(team='graph',model='Task10F_full_causal_graph_cache',config=dict(plan_sha256=sha(root/'full_experiment_plan.json'),single_cuda_vision_owner=True),root=root/'experiments',job_type='prepare') as log,Resources(cache,'full_causal_cache'):
        try:
            while pending:
                progress=False
                data_complete=(root/'full_data_complete.json').exists()
                desired=read(root/'parallel_gate/parallel_gate.json')['adopted_workers'] if data_complete else 1
                if desired!=workers:
                    if pool:pool.shutdown(wait=True)
                    workers=desired;pool=make_pool(cache,workers) if workers else None
                for c in list(pending):
                    entry=source_entry(c)
                    if entry is None:continue
                    folder=cache/'cache'/f'ep{c["episode_id"]}'
                    if not (folder/'receipt.json').exists() and folder.exists():folder.rename(folder.with_name(folder.name+'.interrupted.'+str(time.time_ns())))
                    r=None
                    if not (folder/'receipt.json').exists():r=adopt(root,cache,c,entry,log)
                    if r is None:
                        r=parallel_episode(cache,Path('/'),entry,memo,tracker,log,pool,workers) if workers else cache_episode(cache,Path('/'),entry,memo,tracker,log)
                    side=mask_sidecar(folder,entry,c);r=dict(r,parent=c['parent'],kind=c['kind'],raw_path=entry['path'],supervision=side)
                    completed.append(r);sidecars.append(side);pending.remove(c);progress=True
                    write(cache/'cache_manifest.json',dict(identity=input_identity(cache),episodes=completed,complete=False))
                    write(cache/'progress.json',dict(completed=len(completed),candidates=len(cs),graphs=sum(r['actions'] for r in completed),workers=workers))
                    print('FULL_CACHE',len(completed),'/',len(cs),c['episode_id'],r['actions'],flush=True)
                    # Detector results are pure; cap memo memory without touching
                    # causal per-camera SAM2 episode ownership.
                    if len(memo.cache)>1000:memo.cache.clear()
                    log.log(dict(episodes=len(completed),graphs=sum(r['actions'] for r in completed)),step=len(completed))
                if not progress:
                    if (root/'full_data_identity.json').exists():
                        identity=read(root/'full_data_identity.json');meta=read(root/'experiments'/identity['run_id']/'metadata.json')
                        if meta['status']=='failed':raise RuntimeError('source collection failed; cache stops without approving dataset')
                    if data_complete:raise RuntimeError('complete collection missing a frozen candidate source')
                    time.sleep(5)
            assert len(completed)==212 and (root/'full_data_complete.json').exists()
            cfg=read(cache/'config.json');cfg['episode_lengths']={str(r['episode_id']):r['actions'] for r in completed};write(cache/'config.json',cfg)
            write(cache/'raw_content_manifest.json',dict(episodes=[dict(episode_id=r['episode_id'],raw_path=r['raw_path'],raw_sha256=r['raw_episode_sha256']) for r in completed]),immutable=True)
            write(cache/'cache_manifest.json',dict(identity=input_identity(cache),episodes=sorted(completed,key=lambda r:r['episode_id']),complete=True,raw_content_manifest_sha256=sha(cache/'raw_content_manifest.json')))
            write(cache/'corrective_cache_manifest.json',dict(status='PASS',episodes=completed,mask_sidecars=sidecars,config_sha256=sha(cache/'config.json'),logical_plan_sha256=sha(root/'full_experiment_plan.json'),raw_content_manifest_sha256=sha(cache/'raw_content_manifest.json')),immutable=True)
            log.link_output(cache/'corrective_cache_manifest.json',kind='metrics')
        finally:
            if pool:pool.shutdown(wait=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);a=p.parse_args();run(a.root,a.dino,a.sam)
