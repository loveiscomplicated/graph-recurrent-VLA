"""Packed, reusable GPU graphs and a separate training-only label store."""
from pathlib import Path

import h5py
import numpy as np
import torch

from graph.plan_a_bc.common import read,sha
from graph.integration_baseline.graph import PackedGraphs
from .contract import measured_robot


class GraphPool:
    def __init__(self,root,device='cuda',allow_final=False):
        self.root=Path(root);self.device=device
        if allow_final and not (self.root/'model_selection_frozen.json').exists():
            raise ValueError('final data locked before model selection freeze')
        released=self.root/'final_dataset_manifest.json'
        manifest=read(released if released.exists() else self.root/'graph_cache_manifest.json')
        self.manifest=manifest
        specs=manifest.get('episode_specs',read(self.root/'dataset_manifest.json')['episodes'])
        allowed={s['episode_id']:s for s in specs if s['split']!='final' or allow_final}
        self.meta=[];self.index={};self.pairs=[];self.splits={s:[] for s in ('train','validation','final')}
        arrays={k:[] for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
        robots=[];targets=[];presence=[];node_counts=[];edge_counts=[];total_nodes=0;self.sources=[]
        schema=read(self.root/'normalizer.json')['schema'];self.schema=schema
        def append(path,gtpath,rows,spec,pair=False):
            nonlocal total_nodes
            with np.load(path,allow_pickle=False) as z,np.load(gtpath,allow_pickle=False) as label:
                gcount=len(z['node_ptr'])-1
                assert len(rows)==gcount
                mask_path=spec.get('gnn_mask_path')
                eligible=np.ones(gcount,bool);physical=np.ones(gcount,bool)
                if mask_path:
                    with np.load(mask_path,allow_pickle=False) as masks:
                        eligible=masks['gnn_supervision_mask'].copy();physical=masks['physical_valid'].copy()
                    if eligible.shape!=(gcount,) or eligible.dtype!=bool or (eligible&~physical).any():
                        raise ValueError('approved pre-action GNN mask alignment')
                nc=np.diff(z['node_ptr']);ec=np.diff(z['edge_ptr'])
                # Whitelist only graph and measured robot arrays. action7 and all
                # GT arrays are excluded from PackedGraphs and model inputs.
                for k in arrays:
                    a=z[k].copy()
                    if k=='edge_index':a+=np.repeat(z['node_ptr'][:-1],ec)[None]+total_nodes
                    arrays[k].append(a)
                for i,row in enumerate(rows):
                    idx=len(self.meta);li=row['label_index'] if pair else i
                    md=dict(episode_id=spec['episode_id'],parent=spec['seed'],split=spec['split'],
                        t=row['baseline_t'] if pair else i,pair=pair,pair_id=row.get('pair_id'),
                        physical_valid=row.get('valid',bool(physical[i])),object=row.get('object'),offset_mm=row.get('offset_mm'),
                        gnn_supervised=bool(eligible[i]),branch_id=spec.get('branch_id'),
                        distribution=str(label['distribution'][li]) if 'distribution' in label.files else spec.get('distribution','full_task'),
                        source=spec.get('source','Task10F1_full_task'))
                    self.meta.append(md)
                    if not pair:
                        self.index[(spec['episode_id'],i)]=idx
                        if eligible[i]:self.splits[spec['split']].append(idx)
                    else:
                        self.pairs.append(dict(index=idx,base=self.index[(spec['episode_id'],md['t'])],
                            split=spec['split'],**row))
                    targets.append(label['target18'][li]);robots.append(z['robot76'][i]);presence.append(z['presence2'][i])
                    rawrobot=measured_robot(z['robot76'][i],schema)
                    for key in rawrobot:
                        np.testing.assert_allclose(rawrobot[key],label['raw_'+key][li],atol=1e-7,rtol=0)
                if not pair:
                    np.testing.assert_array_equal(z['timestep'],label['timestep'][:gcount])
                node_counts.extend(nc);edge_counts.extend(ec);total_nodes+=len(z['node_input'])
            self.sources.append(dict(path=str(path),sha256=sha(path),gt_path=str(gtpath),gt_sha256=sha(gtpath)))
        for e in manifest['episodes']:
            if e['episode_id'] not in allowed:continue
            spec=allowed[e['episode_id']];path=self.root/e['path']
            assert sha(path)==e['sha256']
            gt=Path(e['gt_path']) if e.get('gt_path') else self.root/'raw'/str(e['episode_id'])/'gt_supervision.npz'
            assert sha(gt)==e['gt_sidecar_sha256']
            append(path,gt,[{} for _ in range(e['graphs'])],spec)
        for e in manifest['paired_episodes']:
            spec=e['spec']
            if spec['episode_id'] not in allowed:continue
            path=self.root/e['path'];gt=self.root/'raw'/str(spec['episode_id'])/'paired_gt_supervision.npz'
            assert sha(path)==e['sha256'] and sha(gt)==e['label_sha256']
            append(path,gt,e['pairs'],spec,True)
        self.node_counts=np.asarray(node_counts,np.int64);self.edge_counts=np.asarray(edge_counts,np.int64)
        nptr=np.r_[0,np.cumsum(self.node_counts)];eptr=np.r_[0,np.cumsum(self.edge_counts)]
        self.ptr=torch.as_tensor(nptr,device=device);self.eptr=torch.as_tensor(eptr,device=device)
        self.values={k:torch.from_numpy(np.concatenate(v,axis=1 if k=='edge_index' else 0)).to(device) for k,v in arrays.items()}
        self.robot=torch.tensor(np.stack(robots),device=device);self.targets=torch.tensor(np.stack(targets),device=device)
        self.presence=torch.tensor(np.stack(presence),device=device)
        train=np.asarray(self.splits['train']);r=np.stack(robots)[train];y=np.stack(targets)[train]
        self.normalization=dict(robot_mean=r.mean(0).tolist(),robot_std=np.maximum(r.std(0),.001).tolist(),
            target_mean=y.mean(0).tolist(),fitted_split='train',fitted_frames=len(train))

    def select(self,indices):
        indices=np.asarray(indices,np.int64)
        ix=torch.as_tensor(indices,device=self.device)
        nc=self.node_counts[indices];ec=self.edge_counts[indices]
        n=int(nc.sum());e=int(ec.sum());count=len(indices)
        counts=torch.as_tensor(nc,device=self.device);ecounts=torch.as_tensor(ec,device=self.device)
        ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        edgeptr=torch.cat((ecounts.new_zeros(1),ecounts.cumsum(0)))
        owner=torch.repeat_interleave(torch.arange(count,device=self.device),counts,output_size=n)
        edgeowner=torch.repeat_interleave(torch.arange(count,device=self.device),ecounts,output_size=e)
        nodes=self.ptr[ix][owner]+torch.arange(n,device=self.device)-ptr[owner]
        edges=self.eptr[ix][edgeowner]+torch.arange(e,device=self.device)-edgeptr[edgeowner]
        edge_index=self.values['edge_index'][:,edges]-self.ptr[ix][edgeowner][None]+ptr[edgeowner][None]
        graph=PackedGraphs(self.values['node_input'][nodes],self.values['node_type'][nodes],edge_index,
            self.values['edge_attr'][edges],self.values['edge_type'][edges],owner,ptr,
            torch.stack((torch.arange(count,device=self.device),torch.zeros(count,device=self.device,dtype=torch.long)),-1))
        return graph,self.robot[ix],self.targets[ix]
