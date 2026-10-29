"""Bounded CUDA-vision/CPU-geometry overlap with original cache tensor schema.

The sink follows Plan A cache_episode's pointer/provenance/readback algorithm.
Workers call the unchanged CurrentGeometry.from_h5/build and never see GT.
"""
import concurrent.futures
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from graph.plan_a_bc.cache import GRAPH_KEYS,graph_at,input_identity
from graph.plan_a_bc.geometry import CurrentGeometry,validate_graph
from graph.plan_a_bc.common import read,write,sha,npz,now,append_csv
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.geometry_reach10d.optimizations import components

_geometry=None
_root=None


def initialize(root):
    global _geometry,_root
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized(),'CPU worker must not initialize CUDA'
    _root=Path(root)
    _geometry=CurrentGeometry(_root/'inputs',read(_root/'normalizer.json')['schema'])
    import graph.plan_a_geometry.core as core
    core.components=components


def geometry_frame(raw_path,eid,t,masks):
    wall=time.perf_counter();cpu=time.process_time()
    # PlanA has no temporal map; mount is the sole calibrated episode invariant.
    # Reset it on every job; the parent independently checks every calibration
    # against the episode's first link-to-camera mount below.
    _geometry.reset()
    with h5py.File(raw_path) as f:
        cameras,data,surfaces=_geometry.from_h5(f,t,str(eid))
        graph,records,gate=_geometry.build(cameras,data,surfaces,masks)
        wrist=next(c for c in cameras if c.camera=='hand_camera')
        link=next(s for s in surfaces if s.link=='camera_link')
        mount=np.linalg.inv(link.world_from_link)@wrist.world_from_camera
    assert not torch.cuda.is_initialized(),'geometry candidate unexpectedly used CUDA'
    return graph,records,gate,data['robot76'],mount,dict(wall=time.perf_counter()-wall,cpu=time.process_time()-cpu)


def make_pool(root,workers):
    return concurrent.futures.ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(root),))


def cache_episode(root,dataset,entry,detector,tracker,log,pool,workers):
    folder=root/'cache'/f'ep{entry["episode_id"]}';receipt=folder/'receipt.json';identity=input_identity(root)
    if receipt.exists():
        r=read(receipt)
        assert r['identity']==identity and sha(folder/'graphs.npz')==r['sha256']
        assert sha(dataset/entry['path'])==r['raw_episode_sha256']
        return r
    folder.mkdir(parents=True,exist_ok=False)
    raw=dataset/entry['path'];assert sha(raw)==entry['sha256']
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in ['base_camera','hand_camera']}
    ident=str(entry['episode_id'])
    for a in adapters.values():
        a.reset(ident);assert not (a.states or a.found or a.rgb or a.records)
    nodes=[0];edges=[0];values={k:[] for k in GRAPH_KEYS};actions=[];robots=[];presence=[];masks=[];gates=[]
    source_point=[];source_camera=[];source_pixel=[];point_world=[];point_f5=[];point_valid=[];point_graph_node=[];point_frame=[];point_structural=[]
    costs=[];pending={};started=time.perf_counter();mount=None
    def consume(t,future,mm,action,tick):
        nonlocal mount
        g,records,gate,robot,current_mount,profile=future.result()
        if mount is None:mount=current_mount
        np.testing.assert_allclose(current_mount,mount,atol=2e-5,rtol=0)
        by_id={r['persistent_node_id']:r for r in records}
        ordered=[by_id[k] for k in g.packet['node_ids']];point_offset=len(point_world)
        for j,r in enumerate(ordered):
            point_world.append(r['world_xyz']);point_f5.append(r['last_f5']);point_valid.append(r['last_feature_valid'])
            point_graph_node.append(nodes[-1]+3+j);point_frame.append(t);point_structural.append(r.get('structural',False))
            for _,_,camera,pixel in r['source_observation_ids']:
                source_point.append(point_offset+j);source_camera.append(0 if camera=='base_camera' else 1);source_pixel.append(pixel)
        for k in GRAPH_KEYS:values[k].append(getattr(g,k).numpy().copy())
        actions.append(action);robots.append(robot);presence.append([(g.node_type==k).any().item() for k in [3,4]])
        masks.append(np.packbits(np.stack([mm[c] for c in ['base_camera','hand_camera']]).ravel()))
        gates.append(gate);nodes.append(nodes[-1]+len(g.node_type));edges.append(edges[-1]+g.edge_index.shape[1])
        costs.append(time.perf_counter()-tick)
        append_csv(root/'pipeline_profile.csv',dict(stage='parallel_geometry_worker_CPU',episode_id=entry['episode_id'],t=t,seconds=profile['wall']))
        if t%20==0:print('parallel cache episode',ident,'t',t,'/',entry['actions'],flush=True)
    try:
        with h5py.File(raw) as f:
            for t in range(entry['actions']):
                tick=time.perf_counter()
                mm={c:adapters[c].process(f[c+'/rgb'][t],ident,t)[0] for c in adapters}
                future=pool.submit(geometry_frame,str(raw),entry['episode_id'],t,mm)
                pending[t]=(future,mm,f['actions'][t],tick)
                if len(pending)>=2*workers:
                    first=min(pending);consume(first,*pending.pop(first))
            for t in sorted(pending):consume(t,*pending[t])
        arrays={k:np.concatenate(v,axis=1 if k=='edge_index' else 0) for k,v in values.items()}
        arrays.update(node_ptr=np.array(nodes,np.int64),edge_ptr=np.array(edges,np.int64),
            robot76=np.asarray(robots,np.float32),action7=np.asarray(actions,np.float32),presence2=np.asarray(presence,bool),
            timestep=np.arange(entry['actions'],dtype=np.int64),timestamp=np.arange(entry['actions'],dtype=np.float64)/20,
            episode_id=np.int64(entry['episode_id']),mask_bits=np.stack(masks),
            point_world=np.asarray(point_world,np.float64).reshape(-1,3),point_f5=np.asarray(point_f5,np.float64).reshape(-1,5),
            point_valid=np.asarray(point_valid,bool).reshape(-1,3),point_graph_node=np.asarray(point_graph_node,np.int64),
            point_frame=np.asarray(point_frame,np.int64),point_structural=np.asarray(point_structural,bool),
            source_point=np.asarray(source_point,np.int64),source_camera=np.asarray(source_camera,np.uint8),
            source_pixel=np.asarray(source_pixel,np.int64),generation_seconds=np.asarray(costs))
        npz(folder/'graphs.npz',arrays)
        with np.load(folder/'graphs.npz',allow_pickle=False) as z:
            for t in range(entry['actions']):
                g=graph_at(z,t);validate_graph(g)
                for k in GRAPH_KEYS:np.testing.assert_array_equal(getattr(g,k).numpy(),values[k][t])
        assert sha(raw)==entry['sha256']
        write(folder/'gate_trace.json',gates,immutable=True)
        n=np.diff(nodes);e=np.diff(edges);pp=np.asarray(presence)
        row=dict(identity=identity,episode_id=entry['episode_id'],seed=entry['seed'],split=entry['split'],actions=len(actions),
            graphs=len(actions),sha256=sha(folder/'graphs.npz'),path=str((folder/'graphs.npz').relative_to(root)),raw_episode_sha256=entry['sha256'],
            node_min=int(n.min()),node_mean=float(n.mean()),node_max=int(n.max()),edge_min=int(e.min()),edge_mean=float(e.mean()),edge_max=int(e.max()),
            object_missing_rate=(1-pp.mean(0)).tolist(),conversion_failures=0,readback='PASS',tracker_episode_reset='PASS',
            wall_seconds=time.perf_counter()-started,completed_at=now(),geometry_workers=workers,
            overlap='ordered causal CUDA vision producer; bounded independent CPU current-frame geometry; ordered sink',
            optimized_writer_sha256=sha(__file__),generation_seconds_definition='queue-inclusive latency, not additive wall cost')
        write(receipt,row,immutable=True);log.link_output(receipt,kind='metrics');return row
    except BaseException as exc:
        write(folder/'failure.json',dict(status='FAIL',completed_graphs=len(actions),exception=type(exc).__name__,message=str(exc)));raise
    finally:
        for a in adapters.values():a.close()
