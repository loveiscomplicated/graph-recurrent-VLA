"""Causal diagnostic history forks, restored only from completed past frames.

This is not simulator reset or a deployment API. An actual episode reset always
uses PersistentMap.update/reset and clears identities. Forks keep the exact
parent prefix and use a distinct source-observation namespace for the current
counterfactual frame.
"""
import json
from pathlib import Path
import numpy as np

from .map import PersistentMap,Surfel,ObjectMap,INVALID_REMOVED
from .oracle import OracleMotionMap
from .contracts import transform


def restore_history(geometry,episode,variant,parent_timestep,branch_episode):
    oracle=variant in ('O2','O3')
    memory=OracleMotionMap() if oracle else PersistentMap()
    memory.reset(branch_episode)
    if parent_timestep<0:
        return memory
    root=Path(geometry)/'geometry'/f'seed{episode["seed"]}'
    folder=root/f'f{parent_timestep:04d}'
    state=json.loads((folder/(variant+'_tracker.json')).read_text())
    if state['episode']!=episode['episode'] or state['timestep']!=parent_timestep:
        raise ValueError('wrong/foreign/future history snapshot')
    if state.get('geometry_variant',variant)!=variant:
        raise ValueError('map variant identity mismatch')
    if variant.startswith('G') and state.get('condition_type','estimated')!='estimated':
        raise ValueError('GT oracle history cannot enter a practical variant')
    if ('oracle_only' in state)!=oracle:
        raise ValueError('oracle GT state cannot be restored into a practical map')
    sources={}
    for t in range(parent_timestep+1):
        path=root/f'f{t:04d}'/(variant+'_source_observations.npz')
        with np.load(path,allow_pickle=False) as a:
            for node,source_episode,source_t,camera,pixel in zip(a['persistent_node_id'],a['source_episode'],
                    a['source_timestep'],a['source_camera'],a['source_pixel']):
                if int(source_t)!=t or int(source_t)>parent_timestep or str(source_episode)!=episode['episode']:
                    raise ValueError('future/foreign source observation in history')
                source=(str(source_episode),int(source_t),str(camera),int(pixel))
                sources.setdefault(str(node),[]).append(source)
    for oid,value in state['objects'].items():
        obj=ObjectMap(int(oid))
        for key,item in value.items():
            setattr(obj,key,np.asarray(item,float) if key=='world_from_object' else item)
        memory.objects[int(oid)]=obj
    with np.load(folder/(variant+'_map.npz'),allow_pickle=False) as a:
        for i,identity in enumerate(a['persistent_node_id']):
            identity=str(identity)
            if identity not in sources:
                raise ValueError('history node has no past sensor provenance')
            if int(a['last_seen_timestep'][i])>parent_timestep or int(a['first_seen_timestep'][i])>parent_timestep:
                raise ValueError('future node lifecycle in past snapshot')
            oid=int(a['object_id'][i])
            node=Surfel(identity,oid,a['object_local_xyz'][i].copy(),a['object_local_normal'][i].copy(),
                bool(a['normal_valid'][i]),float(a['confidence'][i]),int(a['first_seen_timestep'][i]),
                int(a['last_seen_timestep'][i]),str(a['last_observed_camera'][i]),int(a['observation_count'][i]),
                bool(a['currently_observed'][i]),bool(a['estimated_only'][i]),float(a['position_uncertainty'][i]),
                sources[identity],a['last_f5'][i].copy(),a['last_feature_valid'][i].copy(),
                int(a['contradiction_count'][i]),int(a['status'][i])==INVALID_REMOVED)
            np.testing.assert_allclose(transform(node.object_local_xyz,memory.objects[oid].world_from_object),
                                       a['world_xyz'][i],atol=1e-12,rtol=0)
            memory.objects[oid].nodes.append(node)
    memory.next_id=state['next_id']
    memory.clock.episode=branch_episode
    memory.clock.timestep=parent_timestep
    if oracle:
        memory.gt_reference={int(k):np.array(v) for k,v in state['oracle_only']['gt_reference'].items()}
        memory.map_reference={int(k):np.array(v) for k,v in state['oracle_only']['map_reference'].items()}
    return memory
