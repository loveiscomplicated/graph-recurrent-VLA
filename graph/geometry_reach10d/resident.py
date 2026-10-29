"""GPU-resident packed graph chunks; identical temporal and graph ordering."""
import time
import numpy as np
import torch
from graph.integration_baseline.graph import PackedGraphs,pack_graphs
from graph.plan_a_bc.cache import graph_at


class ResidentDataset:
    def __init__(self,ds,device='cuda'):
        self.episodes=ds.episodes;self.config=ds.config;self.arrays=ds.arrays
        self.chunks={};self.device=device;started=time.perf_counter()
        for eid,a in self.arrays.items():
            for start in range(0,self.episodes[eid]['actions'],32):
                stop=min(start+32,self.episodes[eid]['actions'])
                graphs=[graph_at(a,t) for t in range(start,stop)]
                p=pack_graphs(graphs).to(device)
                self.chunks[eid,start]=(p,torch.from_numpy(a['robot76'][start:stop]).to(device),
                    torch.from_numpy(a['action7'][start:stop]).to(device))
        torch.cuda.synchronize();self.preload_seconds=time.perf_counter()-started

    def batch(self,ids,start,length,device):
        assert length==32 and str(device)==str(self.device)
        groups=[self.chunks[e,start] for e in ids]
        count=max(len(r) for _,r,_ in groups)
        target=torch.zeros((len(ids),count,7),device=device)
        mask=torch.zeros((len(ids),count),device=device,dtype=torch.bool)
        nodeoff=0;graphoff=0
        values={k:[] for k in vars(groups[0][0])};robots=[]
        for b,(p,r,y) in enumerate(groups):
            target[b,:len(y)]=y;mask[b,:len(y)]=True;robots.append(r)
            for k in ['node_input','node_type','edge_attr','edge_type']:values[k].append(getattr(p,k))
            values['edge_index'].append(p.edge_index+nodeoff)
            values['node_graph'].append(p.node_graph+graphoff)
            values['ptr'].append(p.ptr[1:]+nodeoff)
            fm=p.frame_map.clone();fm[:,0]=b;values['frame_map'].append(fm)
            nodeoff+=len(p.node_type);graphoff+=p.count
        values={k:torch.cat(v,dim=1 if k=='edge_index' else 0) for k,v in values.items()}
        values['ptr']=torch.cat([torch.zeros(1,dtype=torch.int64,device=device),values['ptr']])
        return PackedGraphs(**values),torch.cat(robots),target,mask
