"""Sequential causal RGB-D -> variable Graph21 cache, never GNN outputs."""
import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from graph.integration_baseline.graph import Graph, pack_graphs
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_stop_clock import compare_saved_sequence
from visual.runtime.grounding import GDino
from visual.runtime.tracking import SAM2Tracker
from utils.experiment_logger import ExperimentLogger
from .common import read,write,sha,npz,digest,tree_identity,now
from .geometry import CurrentGeometry,validate_graph


GRAPH_KEYS=('node_input','node_type','edge_index','edge_attr','edge_type')


def make_vision(device,dino,sam):
    detector=GDino(device=device,model_id=str(dino))
    tracker=SAM2Tracker(device=device,checkpoint=sam)
    return detector,tracker


def input_identity(root):
    cfg=read(root/'config.json');audit=read(root/'dataset_audit.json')
    frozen=Path(__file__).resolve().parents[1]/'plan_a_geometry/protocol.json'
    return dict(dataset_sha256=audit['dataset_sha256'],split_sha256=audit['split_sha256'],
        graph_config=cfg['geometry'],frozen_geometry_protocol_sha256=sha(frozen),
        frozen_source_sha256=cfg['source_sha256'],cache_version='graphs_before_trainable_encoder_v1')


def cache_episode(root,dataset,entry,detector,tracker,log):
    cache=root/'cache'; folder=cache/f'ep{entry["episode_id"]}'
    receipt=folder/'receipt.json';ident=input_identity(root)
    if receipt.exists():
        r=read(receipt)
        if r['identity']!=ident or sha(folder/'graphs.npz')!=r['sha256'] or sha(dataset/entry['path'])!=entry['sha256']:
            raise ValueError('stale/corrupt cache cannot be reused')
        return r
    folder.mkdir(parents=True,exist_ok=False)
    if sha(dataset/entry['path'])!=entry['sha256']: raise ValueError('original episode changed')
    schema=read(root/'normalizer.json')['schema'];geometry=CurrentGeometry(root/'inputs',schema)
    adapters={name:OnlineVisionAdapter(detector,tracker) for name in ('base_camera','hand_camera')}
    identity=str(entry['episode_id'])
    for v in adapters.values():
        v.reset(identity)
        if v.states or v.found or v.rgb or v.records: raise ValueError('tracker reset leak')
    nodes=[0];edges=[0]; values={k:[] for k in GRAPH_KEYS}; actions=[];robots=[];presence=[];masks=[];gates=[]
    source_point=[];source_camera=[];source_pixel=[];point_world=[];point_f5=[];point_valid=[];point_graph_node=[];point_frame=[];point_structural=[]
    costs=[];started=time.perf_counter()
    try:
        with h5py.File(dataset/entry['path'],'r') as f:
            for t in range(entry['actions']):
                tick=time.perf_counter();cameras,data,surfaces=geometry.from_h5(f,t,identity)
                mm={c.camera:adapters[c.camera].process(c.rgb,identity,t)[0] for c in cameras}
                g,records,gate=geometry.build(cameras,data,surfaces,mm)
                records_by_id={r['persistent_node_id']:r for r in records}
                ordered=[records_by_id[k] for k in g.packet['node_ids']]
                point_offset=len(point_world)
                for j,r in enumerate(ordered):
                    point_world.append(r['world_xyz']);point_f5.append(r['last_f5']);point_valid.append(r['last_feature_valid'])
                    point_graph_node.append(nodes[-1]+3+j);point_frame.append(t);point_structural.append(r.get('structural',False))
                    for _,_,camera,pixel in r['source_observation_ids']:
                        source_point.append(point_offset+j);source_camera.append(0 if camera=='base_camera' else 1);source_pixel.append(pixel)
                for k in GRAPH_KEYS:values[k].append(getattr(g,k).numpy().copy())
                actions.append(f['actions'][t]);robots.append(data['robot76'])
                presence.append([(g.node_type==k).any().item() for k in (3,4)])
                masks.append(np.packbits(np.stack([mm[c] for c in ('base_camera','hand_camera')]).ravel()))
                gates.append(gate);nodes.append(nodes[-1]+len(g.node_type));edges.append(edges[-1]+g.edge_index.shape[1])
                costs.append(time.perf_counter()-tick)
                if t%20==0:print(f'cache episode={identity} t={t}/{entry["actions"]} wall={time.perf_counter()-started:.1f}s',flush=True)
        arrays={k:np.concatenate(v,axis=1 if k=='edge_index' else 0) for k,v in values.items()}
        arrays.update(node_ptr=np.array(nodes,np.int64),edge_ptr=np.array(edges,np.int64),
            robot76=np.asarray(robots,np.float32),action7=np.asarray(actions,np.float32),
            presence2=np.asarray(presence,bool),timestep=np.arange(entry['actions'],dtype=np.int64),
            timestamp=np.arange(entry['actions'],dtype=np.float64)/20,episode_id=np.int64(entry['episode_id']),
            mask_bits=np.stack(masks),point_world=np.asarray(point_world,np.float64).reshape(-1,3),
            point_f5=np.asarray(point_f5,np.float64).reshape(-1,5),point_valid=np.asarray(point_valid,bool).reshape(-1,3),
            point_graph_node=np.asarray(point_graph_node,np.int64),point_frame=np.asarray(point_frame,np.int64),
            point_structural=np.asarray(point_structural,bool),source_point=np.asarray(source_point,np.int64),
            source_camera=np.asarray(source_camera,np.uint8),source_pixel=np.asarray(source_pixel,np.int64),
            generation_seconds=np.asarray(costs))
        npz(folder/'graphs.npz',arrays)
        # Reconstruct every original graph from saved arrays and check numeric equality.
        with np.load(folder/'graphs.npz',allow_pickle=False) as saved:
            for t in range(entry['actions']):
                g=graph_at(saved,t);validate_graph(g)
                for k in GRAPH_KEYS:
                    if not np.array_equal(getattr(g,k).numpy(),values[k][t]): raise ValueError('cache graph readback mismatch')
        write(folder/'gate_trace.json',gates,immutable=True)
        n=np.diff(nodes);e=np.diff(edges); pp=np.asarray(presence)
        row=dict(identity=ident,episode_id=entry['episode_id'],seed=entry['seed'],split=entry['split'],
            actions=len(actions),graphs=len(actions),sha256=sha(folder/'graphs.npz'),path=str((folder/'graphs.npz').relative_to(root)),
            raw_episode_sha256=entry['sha256'],node_min=int(n.min()),node_mean=float(n.mean()),node_max=int(n.max()),
            edge_min=int(e.min()),edge_mean=float(e.mean()),edge_max=int(e.max()),
            object_missing_rate=(1-pp.mean(0)).tolist(),conversion_failures=0,readback='PASS',
            tracker_episode_reset='PASS',wall_seconds=time.perf_counter()-started,completed_at=now())
        write(receipt,row,immutable=True);log.link_output(receipt,kind='metrics');return row
    except BaseException as exc:
        write(folder/'failure.json',dict(status='FAIL',completed_graphs=len(actions),exception=type(exc).__name__,message=str(exc)))
        raise
    finally:
        for v in adapters.values():v.close()


def graph_at(arrays,t):
    lo,hi=arrays['node_ptr'][t:t+2];a,b=arrays['edge_ptr'][t:t+2]
    values={k:torch.from_numpy(np.array(arrays[k][:,a:b] if k=='edge_index' else arrays[k][a:b] if k.startswith('edge_') else arrays[k][lo:hi],copy=True)) for k in GRAPH_KEYS}
    return Graph(**values,packet={},timings_ms={})


class CachedDataset:
    def __init__(self,root, *, allow_test=False):
        self.root=Path(root);self.config=read(self.root/'config.json');self.identity=input_identity(self.root)
        self.manifest=read(self.root/'cache_manifest.json');self.episodes={e['episode_id']:e for e in self.manifest['episodes']}
        self.arrays={};self.allow_test=allow_test
        if self.manifest['identity']!=self.identity:raise ValueError('cache config/dataset/source identity mismatch')
        for e in self.episodes.values():
            if e['split']=='test' and not allow_test:continue
            path=self.root/e['path']
            if sha(path)!=e['sha256']:raise ValueError('cached input hash mismatch')
            with np.load(path,allow_pickle=False) as z:
                # Input graph only; never store frozen encoder/readout/GRU states.
                self.arrays[e['episode_id']]={k:z[k].copy() for k in (*GRAPH_KEYS,'node_ptr','edge_ptr','robot76','action7','presence2','timestep')}

    def batch(self,ids,start,length,device):
        graphs=[];mapping=[];robots=[]
        count=min(length,max(self.episodes[i]['actions']-start for i in ids))
        target=torch.zeros((len(ids),count,7));valid=torch.zeros((len(ids),count),dtype=torch.bool)
        for b,eid in enumerate(ids):
            if eid not in self.arrays:raise ValueError('test or uncached episode access')
            a=self.arrays[eid]
            for t in range(start,min(start+length,self.episodes[eid]['actions'])):
                g=graph_at(a,t);graphs.append(g);mapping.append((b,t-start));robots.append(a['robot76'][t])
                target[b,t-start]=torch.from_numpy(a['action7'][t]);valid[b,t-start]=True
        return pack_graphs(graphs,mapping).to(device),torch.tensor(np.stack(robots),device=device),target.to(device),valid.to(device)


def run(root,dataset,dino,sam,device, *, pilot=False,final_test=False):
    root=Path(root).resolve();dataset=Path(dataset).resolve();audit=read(root/'dataset_audit.json');config=read(root/'config.json')
    if sha(dataset/'dataset_manifest.json')!=audit['dataset_sha256']:raise ValueError('dataset changed')
    if final_test and not (root/'model_selection_frozen.json').exists():raise ValueError('test gate locked')
    detector,tracker=make_vision(device,dino,sam)
    entries=[e for e in audit['episodes'] if e['split']==('test' if final_test else 'train') or (not final_test and e['split']=='validation')]
    order={eid:i for i,eid in enumerate(config['splits']['train']+config['splits']['validation']+config['splits']['test'])}
    entries.sort(key=lambda e:order[e['episode_id']])
    if pilot:entries=entries[:2]
    if not pilot and not final_test and not (root/'cache_pilot_verification.json').exists():raise ValueError('pilot gate required')
    with ExperimentLogger(team='integration',model='plan_a_graph_cache',seed=42,config={'comparison_uuid':root.name,'pilot':pilot,'final_test':final_test,'device':device},root=root/'stage_runs',job_type='prepare') as log:
        if pilot:
            checkdir=root/'vision_contract';checkdir.mkdir(exist_ok=False)
            first=entries[0]
            with h5py.File(dataset/first['path']) as f:
                checks=[]
                for name in ('base_camera','hand_camera'):
                    folder=checkdir/name;folder.mkdir()
                    result=compare_saved_sequence(f[name+'/rgb'][:8],detector,tracker,folder)
                    if result['status']!='PASS':raise ValueError('offline/streaming causal contract')
                    checks.append(result)
            write(root/'vision_contract.json',dict(status='PASS',cameras=checks),immutable=True)
        rows=[]
        for entry in entries:
            rows.append(cache_episode(root,dataset,entry,detector,tracker,log))
            log.log(dict(completed_episodes=len(rows),graphs=sum(e['graphs'] for e in rows)))
        receipts=[read(p) for p in (root/'cache').glob('ep*/receipt.json')]
        manifest=dict(identity=input_identity(root),episodes=sorted(receipts,key=lambda e:e['episode_id']),
            source_before_gnn=True,generated_at=now(),complete_train_validation=all(
                eid in {e['episode_id'] for e in receipts} for eid in config['splits']['train']+config['splits']['validation']))
        write(root/'cache_manifest.json',manifest)
        if not pilot and not final_test:
            write(root/'cache_manifest_train_validation.json',manifest,immutable=True)
        if final_test:write(root/'cache_manifest_test.json',manifest,immutable=True)
        if pilot:write(root/'cache_pilot_verification.json',dict(status='PASS',episodes=rows,
            vision_contract_sha256=sha(root/'vision_contract.json'),full_conversion_permitted=True),immutable=True)
        print(dict(status='PASS',episodes=len(rows),graphs=sum(r['graphs'] for r in rows)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--pilot',action='store_true');p.add_argument('--final-test',action='store_true')
    a=p.parse_args();run(a.root,a.dataset,a.dino,a.sam,a.device,pilot=a.pilot,final_test=a.final_test)


if __name__=='__main__':main()
