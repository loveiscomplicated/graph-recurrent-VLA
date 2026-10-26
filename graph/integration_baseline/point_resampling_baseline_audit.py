"""Read-only full-policy API audit of saved sampling diagnostic tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.graph import Graph,build_graph,pack_graphs
from graph.integration_baseline.rollout_checkpoint import load_bundle
from graph.integration_baseline.io import write_json,sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint-manifest',type=Path,required=True)
    parser.add_argument('--training-run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    cfg=json.loads((run/'config.yaml').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction'])
    model,norm,provenance=load_bundle(args.checkpoint_manifest,device='cuda')
    assert provenance['manifest']['files']['checkpoint']['sha256']==cfg['checkpoint_sha256']
    dataset=DirectDataset(args.training_run);reports=[]
    with torch.inference_mode():
        for spec in cfg['episodes']:
            folder=run/f"{spec['split']}_{spec['episode_id']}"
            with np.load(folder/'tensors.npz',allow_pickle=False) as f:tensors={k:f[k] for k in f.files}
            maxerr={c:dict(action=0.,hidden=0.,z=0.) for c in cfg['conditions']}
            hidden=None
            for t in range(len(tensors['expert'])):
                al=dataset.frame(spec['episode_id'],t)
                np.testing.assert_array_equal(al.robot_state,tensors['robot76'][t])
                np.testing.assert_array_equal(al.action,tensors['expert'][t])
                state=torch.tensor(al.robot_state[None],device='cuda')
                original=build_graph(al.packet,al.robot_positions_ee,**cfg['graph'])
                packed=pack_graphs([original],[(0,0)]).to('cuda')
                z,_=model.graph_encoder(packed)
                action,hidden=model.step(packed,state,norm,hidden)
                def compare(c,a,h,z,ak,hk):
                    for name,x,key in [('action',a[0],ak),('hidden',h[0,0],hk),('z',z[0],'z_'+c)]:
                        y=torch.tensor(tensors[key][t],device='cuda')
                        torch.testing.assert_close(x,y,atol=cfg['atol'],rtol=cfg['rtol'])
                        maxerr[c][name]=max(maxerr[c][name],float((x-y).abs().max()))
                compare('phase0',action,hidden,z,'action_phase0','hidden_phase0')
                for c in cfg['conditions'][1:]:
                    with np.load(folder/'graphs'/f'{t:04d}_{c}.npz',allow_pickle=False) as f:
                        graph=Graph(torch.tensor(f['node_input']),torch.tensor(np.r_[np.arange(3),f['object_id']+3]),
                            torch.tensor(f['edge_index']),torch.tensor(f['edge_attr']),torch.tensor(f['edge_type']),{}, {})
                    p=pack_graphs([graph],[(0,0)]).to('cuda');zb,_=model.graph_encoder(p)
                    before=torch.tensor(tensors['pre_hidden_phase0'][t][None,None],device='cuda')
                    ai,hi=model.step(p,state,norm,before)
                    compare(c,ai,hi,zb,'action_instant_'+c,'hidden_instant_'+c)
                    hb=torch.zeros_like(before) if t==0 else torch.tensor(tensors['hidden_history_'+c][t-1][None,None],device='cuda')
                    ab,h=model.step(p,state,norm,hb)
                    compare(c,ab,h,zb,'action_history_'+c,'hidden_history_'+c)
            reports.append(dict(**spec,status='PASS',frames=len(tensors['expert']),maximum_absolute_errors=maxerr))
    write_json(run/'full_policy_api_audit.json',dict(status='PASS',device='CUDA on Vast57640710',episodes=reports,
        source_sha256=sha256(__file__),atol=cfg['atol'],rtol=cfg['rtol'],
        scope='Canonical DirectDataset -> original full model.step compared with saved batched-encoder/GRU diagnostic; both instant and independent-history alternatives checked with correct previous hidden',
        simulator_executed=False,model_training=False))
    print('FULL_POLICY_API_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
