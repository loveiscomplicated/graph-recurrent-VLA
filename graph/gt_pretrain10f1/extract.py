"""Sensor-only extraction CLI. It does not load a dataset/GT sidecar or optimizer."""
import argparse
from pathlib import Path

import numpy as np
import torch

from graph.plan_a_bc.common import read,write,npz,sha,tensor_identity
from graph.plan_a_bc.cache import graph_at,GRAPH_KEYS
from graph.plan_a_bc.geometry import validate_graph
from graph.integration_baseline.graph import pack_graphs
from .model import restore_checkpoint


def run(checkpoint,graphs,schema_path,receipt,out,frames,device):
    # Match the primary/evaluation reduction contract for CUDA index_add.
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    schema=read(schema_path)
    if 'schema' in schema:schema=schema['schema']
    model,ck=restore_checkpoint(checkpoint,device,True)
    if ck['robot_schema']!=schema:raise ValueError('measured robot feature schema/order mismatch')
    r=read(receipt)
    if sha(graphs)!=r['sha256']:raise ValueError('graph receipt SHA mismatch')
    if r['identity']['graph_config']!=ck['geometry_contract']:raise ValueError('geometry contract mismatch')
    before=tensor_identity(model.state_dict());values={k:[] for k in ('readout320','presence2','robot76','state38','state38_identifiability_mask')}
    with np.load(graphs,allow_pickle=False) as z:
        n=len(z['node_ptr'])-1
        ids=list(range(n)) if frames is None else frames
        if not ids or min(ids)<0 or max(ids)>=n:raise ValueError('valid frame indices required')
        sensor={k:z[k].copy() for k in (*GRAPH_KEYS,'node_ptr','edge_ptr','robot76')}
    for start in range(0,len(ids),64):
        selected=ids[start:start+64];gs=[graph_at(sensor,t) for t in selected]
        for g in gs:validate_graph(g)
        g=pack_graphs(gs).to(device);robot=torch.as_tensor(sensor['robot76'][selected],device=device)
        result=model.extract(g,robot,schema)
        for key in values:
            x=result[key];values[key].append(x.cpu().numpy() if torch.is_tensor(x) else x)
    assert before==tensor_identity(model.state_dict())
    arrays={k:np.concatenate(v) for k,v in values.items()}
    arrays.update(timestep=np.asarray(ids,np.int64),checkpoint_sha256=np.array(sha(checkpoint)),weight_sha256=np.array(before))
    npz(out,arrays)
    write(out.with_suffix('.identity.json'),dict(checkpoint_sha256=sha(checkpoint),graph_sha256=sha(graphs),
        weight_sha256=before,source_graph_receipt_sha256=sha(receipt),robot_schema=schema,
        slot_order=ck['slot_order'],GT_sidecar_read=False,
        unsupported_universal_dimensions=list(range(18,25))+list(range(28,36)),
        identifiability_mask='metadata only, not additional GRU/Graph features; presence does not certify geometry accuracy',
        state38='estimated compatible fields; missing/symmetry observability limitations remain',
        frames=len(ids),feature_cache_sha256=sha(out)),immutable=True)
    print('EXTRACTED',len(ids),'same frozen checkpoint; no GT sidecar',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--graphs',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--schema',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(a.checkpoint,a.graphs,a.schema,a.receipt,a.out,a.frames,a.device)
