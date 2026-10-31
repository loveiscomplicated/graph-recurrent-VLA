"""Paired exposure sampler and pre-GNN GPU windows for inherited train_update."""
from collections import Counter
from pathlib import Path
import h5py,numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,digest,npz
from graph.plan_a_bc.cache import graph_at
from graph.integration_baseline.graph import pack_graphs
from graph.geometry_reach10d.resident import ResidentDataset
from graph.corrective_reach10f.masked import SupervisedDataset


def freeze(root,ds):
    destination=root/'sampling_manifest.json'
    if destination.exists():
        m=read(destination);assert m['cache_manifest_sha256']==sha(ds.root/'cache_manifest.json');return m
    rng=np.random.default_rng(42);nom=sorted(e for e,r in ds.episodes.items() if r['split']=='train' and r['kind']=='nominal');rec=sorted(e for e,r in ds.episodes.items() if r['split']=='train' and r['kind']=='recovery')
    parent_nom={ds.episodes[e]['parent']:e for e in nom};definitions=[];lookup={};schedule=[];exposure={c:Counter() for c in ['R0','R1']};parent_exposure={c:Counter() for c in ['R0','R1']};burn={c:0 for c in ['R0','R1']}
    def window(eid,start,k):
        key=(eid,start,k)
        if key not in lookup:
            vid=31000000+len(definitions);lookup[key]=vid;definitions.append(dict(virtual_id=vid,episode_id=eid,start=start,stop=min(start+32,ds.episodes[eid]['actions']),supervised=k))
        return lookup[key]
    for u in range(3000):
        if u%len(nom)==0:norder=rng.permutation(nom).tolist()
        if u%len(rec)==0:rorder=rng.permutation(rec).tolist()
        n1=norder[u%len(nom)];r=rorder[u%len(rec)];row=ds.episodes[r]
        n2=parent_nom[row['parent']]
        # Some valid nominal Reach episodes end before frame 32. Keep the
        # sequence capacity at 32, but pair only frames that exist in all
        # three sources for this update. Both conditions see exactly 2*k
        # supervised frames and R1 remains exactly half nominal/recovery.
        k=min(32,row['supervision']['supervised'],ds.episodes[n1]['actions'],ds.episodes[n2]['actions'])
        assert k>0
        with np.load(row['supervision']['path']) as a:mask=a['supervision_mask']
        handoff=int(np.flatnonzero(mask)[0]);assert np.all(mask[handoff:])
        start_r=handoff+(0 if u%2==0 else int(rng.integers(0,row['supervision']['supervised']-k+1)))
        starts=[int(rng.integers(0,ds.episodes[n]['actions']-k+1)) for n in [n1,n2]]
        common=window(n1,starts[0],k);s=dict(update=u+1,k=k,R0=[common,window(n2,starts[1],k)],R1=[common,window(r,start_r,k)],supervised_timesteps=2*k)
        schedule.append(s)
        for condition,items in [('R0',[(n1,starts[0]),(n2,starts[1])]),('R1',[(n1,starts[0]),(r,start_r)])]:
            for eid,start in items:
                parent_exposure[condition][ds.episodes[eid]['parent']]+=k;burn[condition]+=start
                for t in range(start,start+k):exposure[condition][eid,t]+=1
    assert parent_exposure['R0']==parent_exposure['R1']
    for condition in ['R0','R1']:
        values=sorted(exposure[condition]);npz(root/f'sampling_frequency_{condition}.npz',dict(episode_id=np.array([e for e,t in values]),timestep=np.array([t for e,t in values]),frequency=np.array([exposure[condition][v] for v in values])))
    m=dict(version='current_weight_prefix_windows_v1',cache_manifest_sha256=sha(ds.root/'cache_manifest.json'),config_sha256=sha(ds.root/'config.json'),windows=definitions,schedule=schedule,
           supervised_timesteps=sum(s['supervised_timesteps'] for s in schedule),R1_nominal_fraction=.5,parent_exposure_exact=True,parent_exposure={c:dict(v) for c,v in parent_exposure.items()},burn_in_frames=burn,
           sequence_length=32,logical_window_offset=1,logical_offset_semantics='internal carry flag for inherited train_update, never simulator time. Actual source frame/time is the window start/stop. All actual frames0..start-1 form independent current-model burn-in.')
    write(destination,m,immutable=True);return m


class WindowResident(ResidentDataset):
    """Reuse ResidentDataset.batch unchanged; only prepack arbitrary raw windows."""
    def __init__(self,base,manifest,condition,device='cuda'):
        self.device=device;self.config=base.config;self.arrays=base.arrays;self.base=base;self.chunks={};self.episodes={};self.masks={};self.prefix={};self.specs={w['virtual_id']:w for w in manifest['windows']}
        used={v for row in manifest['schedule'] for v in row[condition]}
        for vid in sorted(used):
            w=self.specs[vid];eid,start,stop=w['episode_id'],w['start'],w['stop'];a=base.arrays[eid]
            self.chunks[vid,1]=self.pack(eid,start,stop)
            count=stop-start;mask=np.zeros(1+count,bool);mask[1:1+w['supervised']]=True
            self.masks[vid]=mask;self.episodes[vid]=dict(actions=1+count)
            for begin in range(0,start,32):
                end=min(begin+32,start)
                if (eid,begin,end) not in self.prefix:self.prefix[eid,begin,end]=self.pack(eid,begin,end)[:2]
        self.masked=SupervisedDataset(self,self.masks)
    def pack(self,eid,start,stop):
        a=self.base.arrays[eid];graphs=[graph_at(a,t) for t in range(start,stop)]
        return pack_graphs(graphs).to(self.device),torch.from_numpy(a['robot76'][start:stop]).to(self.device),torch.from_numpy(a['action7'][start:stop]).to(self.device)
    def current_hidden(self,model,norm,vid):
        w=self.specs[vid];eid,start=w['episode_id'],w['start'];hidden=None
        with torch.no_grad():
            for begin in range(0,start,32):
                end=min(begin+32,start);g,s=self.prefix[eid,begin,end];_,hidden=model(g,s,norm,hidden)
        if hidden is None:hidden=next(model.parameters()).new_zeros(model.policy.rnn.num_layers,1,model.policy.rnn.hidden_size)
        return hidden.detach()
    def loss_batch(self,ids):return self.masked.batch(ids,1,32,self.device)


def validate_windows(view,model,norm,ids):
    """Actual raw-prefix vs prepacked-window forward, with fixed inherited tolerances."""
    model.eval()
    with torch.no_grad():
        for vid in ids:
            w=view.specs[vid];a=view.base.arrays[w['episode_id']]
            g=pack_graphs([graph_at(a,t) for t in range(w['stop'])]).to(view.device);s=torch.tensor(a['robot76'][:w['stop']],device=view.device)
            expected,_=model(g,s,norm,None);h=view.current_hidden(model,norm,vid)
            g,s,_,mask=view.loss_batch([vid]);actual,_=model(g,s,norm,h)
            torch.testing.assert_close(actual,expected[:,w['start']:w['stop']],rtol=1e-5,atol=1e-6)
            assert int(mask.sum())==w['supervised']
    model.train()
