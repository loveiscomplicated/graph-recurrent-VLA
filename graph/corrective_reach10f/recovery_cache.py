"""Task-local mask/provenance sidecar around unchanged Plan A graph writers.

No trainable representation is cached. This engineering entrypoint refuses to
run without a complete engineering-parent physical gate (or all-40 gate),
and never regenerates valid receipts. Engineering caching cannot unlock training.
"""
import argparse,copy,time
from pathlib import Path
import h5py,numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,digest
from graph.plan_a_bc.cache import cache_episode,make_vision
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from utils.experiment_logger import ExperimentLogger


def run(root,pilot,dino,sam,engineering=False):
    if engineering:
        entries=read(pilot/'recovery_pilot_manifest.json')
        first=[r for r in entries if r['seed']==30000]
        assert len(first)==8 and all(r['approved'] for r in first)
        for r in first:assert read(pilot/'raw'/r['branch']/'N/receipt.json')['approved']
    else:
        gate=read(pilot/'verification.json');assert gate['completed_branches']==40 and 'FAIL' not in gate['checks'].values()
    torch.set_num_threads(1);root.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','normalizer.json','action_codec.json']:
        if not (root/name).exists():(root/name).symlink_to(pilot/name)
    cfg=read(pilot/'config.json');entries=[]
    # The first +/-X pair was fixed by protocol order, independent of policy.
    for eid,label in [(4100000,'30000_t10_5_0/P'),(4100001,'30000_t10_5_0/N')]:
        p=pilot/'raw'/label/'episode.h5';r=read(p.with_name('receipt.json'));assert r['approved'];assert sha(p)==r['sha256']
        entries.append(dict(episode_id=eid,seed=30000,split='pilot',path=str(p),actions=r['actions'],sha256=r['sha256']))
    cfg['splits']=dict(train=[e['episode_id'] for e in entries],validation=[],test=[])
    write(root/'config.json',cfg);write(root/'dataset_audit.json',dict(dataset_sha256=digest(entries),split_sha256=digest(cfg['splits']),episodes=entries))
    detector,tracker=make_vision('cuda',dino,sam)
    model_identity=dict(dino_files={p.name:sha(p) for p in dino.iterdir() if p.is_file()},sam_sha256=sha(sam))
    import graph.plan_a_geometry.core as core
    original_components=core.components;results=[]
    with ExperimentLogger(team='graph',model='Task10F_causal_corrective_cache_gate',config=dict(source_pilot_uuid=read(pilot/'identity.json')['run_id'],same_source_geometry=True),root=root/'experiments',job_type='prepare') as log:
        for mode in ['original','optimized']:
            folder=root/mode;folder.mkdir(exist_ok=True)
            for name in ['inputs','normalizer.json','action_codec.json','config.json','dataset_audit.json']:
                if not (folder/name).exists():(folder/name).symlink_to(root/name)
            core.components=original_components if mode=='original' else components
            current=detector if mode=='original' else DetectionMemo(detector,'Task10F_equivalence',model_identity)
            start=time.perf_counter()
            for e in entries:
                r=cache_episode(folder,Path('/'),e,current,tracker,log);results.append(dict(mode=mode,receipt=r))
                if isinstance(current,DetectionMemo):current.cache.clear()
                # Masks and segment metadata cannot be added as new features.
                target=folder/'cache'/f'ep{e["episode_id"]}'/'supervision.npz'
                if not target.exists():
                    with h5py.File(e['path']) as f:
                        with target.open('xb') as stream:np.savez_compressed(stream,supervision_mask=f['supervision_mask'][:],segment=f['segment'][:],executed_action=f['executed_action'][:])
                    with np.load(target) as a:
                        with h5py.File(e['path']) as f:np.testing.assert_array_equal(a['supervision_mask'],f['supervision_mask'][:])
            write(root/(mode+'_wall.json'),dict(seconds=time.perf_counter()-start))
        checks=[]
        for e in entries:
            paths=[root/m/'cache'/f'ep{e["episode_id"]}' for m in ['original','optimized']]
            with np.load(paths[0]/'graphs.npz') as a,np.load(paths[1]/'graphs.npz') as b:
                assert set(a.files)==set(b.files)
                for k in a.files:
                    if k!='generation_seconds':np.testing.assert_array_equal(a[k],b[k],err_msg=k)
            assert read(paths[0]/'gate_trace.json')==read(paths[1]/'gate_trace.json')
            checks.append(dict(episode=e['episode_id'],graph_mask_gate_all_arrays_exact=True,sha_original=sha(paths[0]/'graphs.npz'),sha_optimized=sha(paths[1]/'graphs.npz')))
        write(root/'corrective_cache_gate.json',dict(status='PASS',checks=checks,results=results,model_identity=model_identity),immutable=True);log.link_output(root/'corrective_cache_gate.json',kind='metrics')
    core.components=original_components

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);p.add_argument('--engineering',action='store_true');a=p.parse_args();run(a.root,a.pilot,a.dino,a.sam,a.engineering)
