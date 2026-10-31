"""Reuse Task10D CUDA-vision / CPU-geometry writer, then prefix-fixed pairs."""
import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from graph.plan_a_bc.cache import make_vision, graph_at, GRAPH_KEYS
from graph.plan_a_bc.geometry import CurrentGeometry, validate_graph
from graph.plan_a_bc.common import read,write,sha,npz,now
from graph.geometry_reach10d.optimizations import DetectionMemo, components
from graph.geometry_reach10d.parallel_cache import make_pool, cache_episode
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.geometry09b import memory,restore
from utils.experiment_logger import ExperimentLogger


def equality(root,entry,detector,memo):
    """New full-task frames: original/optimized graph/gate exact equivalence."""
    import graph.plan_a_geometry.core as core
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    def original(xyz,radius):
        if not len(xyz):return np.empty(0,int)
        pairs=cKDTree(xyz).query_pairs(radius,output_type='ndarray')
        return connected_components(coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),
            shape=(len(xyz),len(xyz))),directed=False)[1]
    checks=[]
    g0=CurrentGeometry(root/'inputs',read(root/'normalizer.json')['schema'])
    with h5py.File(root/entry['path']) as f,np.load(root/'cache'/f'ep{entry["episode_id"]}'/'graphs.npz') as z:
        for t in sorted({0,min(20,entry['actions']-1),min(60,entry['actions']-1),min(100,entry['actions']-1)}):
            cams,data,surf=g0.from_h5(f,t,str(entry['episode_id']))
            mm=np.unpackbits(z['mask_bits'][t])[:2*2*128*128].reshape(2,2,128,128).astype(bool)
            masks=dict(zip(('base_camera','hand_camera'),mm))
            state=np.random.get_state();torchstate=torch.get_rng_state()
            core.components=original
            before,records0,gate0=g0.build(cams,data,surf,masks)
            np.random.set_state(state);torch.set_rng_state(torchstate)
            core.components=components
            after,records1,gate1=g0.build(cams,data,surf,masks)
            saved=graph_at(z,t)
            for k in GRAPH_KEYS:
                np.testing.assert_array_equal(getattr(before,k).numpy(),getattr(after,k).numpy(),err_msg=k)
                np.testing.assert_array_equal(getattr(after,k).numpy(),getattr(saved,k).numpy(),err_msg=k)
            assert gate0==gate1
            checks.append(dict(t=t,nodes=len(before.node_type),edges=before.edge_type.numel(),graph_exact=True,gate_exact=True))
        rgb=f['base_camera/rgb'][0]
        for prompt in ('peg','box'):
            a=detector(rgb,prompt);b=memo(rgb,prompt);c=memo(rgb,prompt)
            for x,y in ((a,b),(a,c)):
                for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
    return dict(status='PASS',episode_id=entry['episode_id'],frames=checks,
        detection_memo_original_exact=True,component_source_sha256=sha(Path(__file__).resolve().parents[1]/'geometry_reach10d/optimizations.py'))


def paired_cache(root,spec,detector,tracker,log):
    folder=root/'pairs'/str(spec['episode_id']);receipt=folder/'receipt.json'
    raw=root/'raw'/str(spec['episode_id'])
    entries=read(raw/'paired_receipt.json')['pairs']
    if not entries:return None
    if receipt.exists():
        r=read(receipt)
        assert sha(folder/'graphs.npz')==r['sha256']
        assert r['raw_sha256']==sha(raw/'paired_rgbd.h5')
        return r
    folder.mkdir(parents=True,exist_ok=False)
    import graph.plan_a_geometry.core as core
    core.components=components
    geo=CurrentGeometry(root/'inputs',read(root/'normalizer.json')['schema'])
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ('base_camera','hand_camera')}
    identity=str(spec['episode_id'])
    for a in adapters.values():a.reset(identity)
    values={k:[] for k in GRAPH_KEYS};nodes=[0];edges=[0];robots=[];pres=[];pairids=[];costs=[]
    anchor_ts=sorted({e['baseline_t'] for e in entries})
    def append(g,robot,pair_id):
        validate_graph(g)
        for k in GRAPH_KEYS:values[k].append(getattr(g,k).numpy().copy())
        nodes.append(nodes[-1]+len(g.node_type));edges.append(edges[-1]+g.edge_type.numel())
        robots.append(robot);pres.append([(g.node_type==k).any().item() for k in (3,4)]);pairids.append(pair_id)
    started=time.perf_counter()
    try:
        with h5py.File(raw/'episode.h5') as base,h5py.File(raw/'paired_rgbd.h5') as branch, \
                np.load(root/'cache'/f'ep{identity}'/'graphs.npz') as cached:
            for t in range(max(anchor_ts)+1):
                pre={c:memory(a) for c,a in adapters.items()} if t in anchor_ts else None
                masks={c:a.process(base[c+'/rgb'][t],identity,t)[0] for c,a in adapters.items()}
                if t not in anchor_ts:continue
                baseline_masks=np.unpackbits(cached['mask_bits'][t])[:4*128*128].reshape(2,2,128,128).astype(bool)
                np.testing.assert_array_equal(np.stack(list(masks.values())),baseline_masks)
                post={c:memory(a) for c,a in adapters.items()}
                # Tracker.start may reread observed JPEGs on late detection.
                # Restore the actual prefix files as well as in-memory tensors.
                jpeg_bytes={c:(Path(a.directory.name)/f'{t:06d}.jpg').read_bytes() for c,a in adapters.items()}
                for entry in (e for e in entries if e['baseline_t']==t):
                    tick=time.perf_counter()
                    for c,a in adapters.items():restore(a,pre[c])
                    group=branch[entry['raw_hdf5_group']]
                    branchm={c:a.process(group[c+'/rgb'][0],identity,t)[0] for c,a in adapters.items()}
                    np.testing.assert_array_equal(base['robot_state'][t],group['robot_state'][0])
                    for c in adapters:
                        for k in ('intrinsic','extrinsic_cv','cam2world_gl'):
                            np.testing.assert_array_equal(base[c+'/'+k][t],group[c+'/'+k][0])
                    cams,data,surf=geo.from_h5(group,0,identity)
                    # from_h5's t argument is a row index; metadata must retain action t.
                    # Rebuild the CameraObservation metadata without changing any pixels.
                    from dataclasses import replace
                    cams=[replace(c,timestep=t) for c in cams]
                    data['timestep']=np.int64(t);data['timestamp']=np.float64(t/20)
                    g,_,_=geo.build(cams,data,surf,branchm)
                    append(g,data['robot76'],entry['pair_id']);costs.append(time.perf_counter()-tick)
                for c,a in adapters.items():
                    restore(a,post[c])
                    (Path(a.directory.name)/f'{t:06d}.jpg').write_bytes(jpeg_bytes[c])
                print('PAIRCACHE',spec['split'],spec['seed'],t,len(pairids),flush=True)
        arrays={k:np.concatenate(v,axis=1 if k=='edge_index' else 0) for k,v in values.items()}
        arrays.update(node_ptr=np.asarray(nodes,np.int64),edge_ptr=np.asarray(edges,np.int64),
            robot76=np.asarray(robots,np.float32),presence2=np.asarray(pres,bool),pair_id=np.asarray(pairids),
            episode_id=np.int64(spec['episode_id']),generation_seconds=np.asarray(costs))
        npz(folder/'graphs.npz',arrays)
        with np.load(folder/'graphs.npz') as z:
            for i in range(len(pairids)):
                for k in GRAPH_KEYS:np.testing.assert_array_equal(getattr(graph_at(z,i),k).numpy(),values[k][i])
        r=dict(spec=spec,pairs=entries,graphs=len(pairids),sha256=sha(folder/'graphs.npz'),raw_sha256=sha(raw/'paired_rgbd.h5'),
            label_sha256=sha(raw/'paired_gt_supervision.npz'),path=str((folder/'graphs.npz').relative_to(root)),
            readback='PASS',same_robot_camera_prefix='PASS',baseline_mask_replay_exact=True,wall_seconds=time.perf_counter()-started)
        write(receipt,r,immutable=True);log.link_output(receipt,kind='metrics');return r
    finally:
        for a in adapters.values():a.close()


def run(root,worker=0,workers=1):
    torch.set_num_threads(1);torch.manual_seed(42)
    paths=read(root/'runtime_paths.json');detector,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    memo=DetectionMemo(detector,'Task10F1_full_task',dict(dino=paths['dino'],sam_sha256=sha(paths['sam'])))
    specs=read(root/'dataset_manifest.json')['episodes'][worker::workers];completed=[];pairs=[];started=time.perf_counter()
    with ExperimentLogger(team='graph',model='Task10F1_causal_graph_cache',seed=42,
            config=dict(parent_run_id=read(root/'run_identity.json')['run_id']),root=root/'experiments',job_type='prepare') as log,make_pool(root,4) as pool:
        for spec in specs:
            raw=root/'raw'/str(spec['episode_id'])/'receipt.json'
            while not raw.exists():
                if (root/'collection_failure.json').exists():raise RuntimeError('collection failed')
                time.sleep(2)
            receipt=read(raw)
            entry=dict(episode_id=spec['episode_id'],seed=spec['seed'],split=spec['split'],actions=receipt['actions'],
                path=str(raw.parent.relative_to(root)/'episode.h5'),sha256=receipt['hashes']['episode.h5'])
            tick=time.perf_counter()
            cached=cache_episode(root,root,entry,memo,tracker,log,pool,4)
            cached.update(parent_seed=spec['seed'],gt_sidecar_sha256=receipt['hashes']['gt_supervision.npz'])
            completed.append(cached)
            if len(completed)==1 and not (root/'optimization_equivalence.json').exists():
                write(root/'optimization_equivalence.json',equality(root,entry,detector,memo),immutable=True)
            if spec['paired']:
                paired=paired_cache(root,spec,memo,tracker,log)
                if paired:pairs.append(paired)
            memo.cache.clear();torch.cuda.empty_cache()
            progress_name='graph_cache_progress.json' if workers==1 else f'graph_cache_progress_worker{worker}.json'
            write(root/progress_name,dict(episodes=len(completed),graphs=sum(r['graphs'] for r in completed),
                paired_graphs=sum(r['graphs'] for r in pairs),wall_seconds=time.perf_counter()-started,at=now()))
            print('CACHE COMPLETE',spec['split'],spec['seed'],len(completed),'seconds',time.perf_counter()-tick,flush=True)
        manifest_name='graph_cache_manifest.json' if workers==1 else f'graph_cache_manifest_worker{worker}.json'
        write(root/manifest_name,dict(episodes=completed,paired_episodes=pairs,
            source_before_GNN=True,source_graph_contract=read(root/'config.json')['geometry'],
            wall_seconds=time.perf_counter()-started),immutable=True)
        print('ALL CACHE COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();run(a.root,a.worker,a.workers)
