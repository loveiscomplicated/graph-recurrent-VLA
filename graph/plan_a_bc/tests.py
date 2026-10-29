"""Executable graph/cache/training/evaluation contracts; no performance claims."""
import ast
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from graph.integration_baseline.graph import Graph,pack_graphs
from graph.integration_baseline.model import EpisodeStream
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer
from .geometry import validate_graph,CurrentGeometry
from .cache import graph_at
from .common import npz,read
from .model import CachedGraphPolicy,gradient_report
from .train import chunks,assert_nested_equal


def fixture(n=7):
    x=torch.randn(n,21);x[:,14:]=0
    types=torch.tensor(list(range(3))+[3]*(n-3))
    edge=torch.tensor([[0,3,1,3],[3,0,3,1]]) if n>3 else torch.zeros((2,0),dtype=torch.long)
    return Graph(x,types,edge,torch.randn(edge.shape[1],4),torch.ones(edge.shape[1],dtype=torch.long),{}, {})


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        self.schema=dict(dimension=76)
        self.norm=RobotNormalizer(self.schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[1])
        self.codec=ActionCodec.fit('B',{1:np.random.default_rng(42).normal(size=(20,7)).astype('float32')},[1])
        self.config=dict(d=64,L=2,gru_hidden=144,gru_layers=1,enc_hidden=128)

    def test_graph_schema_and_empty_objects(self):
        for n in [3,4,7]:
            g=fixture(n);validate_graph(g)
            p=pack_graphs([g]);m=CachedGraphPolicy(self.schema,'E0',**self.config)
            out,_=m(p,torch.zeros(1,76),self.norm)
            self.assertEqual(out.shape,(1,1,7));self.assertTrue(torch.isfinite(out).all())
        g=fixture();g.edge_index[0,0]=100
        with self.assertRaises(ValueError):validate_graph(g)

    def test_end_to_end_gradients_both_variants(self):
        for variant in ('E0','E1'):
            m=CachedGraphPolicy(self.schema,variant,**self.config)
            p=pack_graphs([fixture() for _ in range(3)])
            out,_=m(p,torch.randn(3,76),self.norm)
            self.codec.loss(out,torch.randn(1,3,7),torch.ones(1,3,dtype=torch.bool)).backward()
            report=gradient_report(m)
            self.assertTrue(all(v>0 for v in report.values()))

    def test_same_common_initial_weights(self):
        torch.manual_seed(42);a=CachedGraphPolicy(self.schema,'E0',**self.config)
        torch.manual_seed(42);b=CachedGraphPolicy(self.schema,'E1',**self.config)
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,a.state_dict()[k]),k)
        self.assertGreater(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))

    def test_E1_no_message_dependency(self):
        m=CachedGraphPolicy(self.schema,'E1',**self.config).eval();g=fixture()
        a=m.graph_encoder(pack_graphs([g]))[0]
        g.edge_attr*=10000;g.edge_index=g.edge_index.flip(0)
        b=m.graph_encoder(pack_graphs([g]))[0]
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_padding_mask_method_B_and_native_decode(self):
        raw=torch.randn(2,4,7,requires_grad=True);target=torch.randn(2,4,7)
        mask=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
        a=self.codec.loss(raw,target,mask);other=target.clone();other[~mask]=999
        torch.testing.assert_close(a,self.codec.loss(raw,other,mask),atol=0,rtol=0)
        a.backward();self.assertTrue((raw.grad[~mask]==0).all())
        mean=torch.tensor(self.codec.mean);std=torch.tensor(self.codec.std)
        encoded=torch.cat(((target[:,:,:6]-mean)/std,target[:,:,6:]),-1)
        torch.testing.assert_close(self.codec.decode(self.codec.tag(encoded)),target,atol=2e-7,rtol=1e-6)
        with self.assertRaises(ValueError):ActionCodec.fit('B',{2:np.ones((2,7),np.float32)},[1])

    def test_hidden_carry_detach_and_reset(self):
        m=CachedGraphPolicy(self.schema,'E0',**self.config).eval();gs=[fixture() for _ in range(5)];s=torch.randn(5,76)
        full,full_h=m(pack_graphs(gs),s,self.norm)
        first,h=m(pack_graphs(gs[:2]),s[:2],self.norm)
        tail,last=m(pack_graphs(gs[2:]),s[2:],self.norm,h.detach())
        torch.testing.assert_close(torch.cat([first,tail],1),full,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(last,full_h,atol=2e-7,rtol=1e-6)
        h.retain_grad();m(pack_graphs(gs[2:]),s[2:],self.norm,h.detach())[0].sum().backward()
        self.assertIsNone(h.grad)
        stream=EpisodeStream(m,self.norm);g=pack_graphs([gs[0]],[(0,0)])
        a=stream.step(g,s[:1],'episode1',0);stream.step(g,s[:1],'episode1',1)
        b=stream.step(g,s[:1],'episode2',0)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_graph_npz_exact_readback_and_alignment(self):
        gs=[fixture(3),fixture(7)]
        arrays={k:np.concatenate([getattr(g,k).numpy() for g in gs],axis=1 if k=='edge_index' else 0) for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
        arrays.update(node_ptr=np.array([0,3,10]),edge_ptr=np.array([0,0,4]),
            timestep=np.arange(2),action7=np.arange(14,dtype='float32').reshape(2,7))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'graphs.npz';npz(p,arrays)
            with np.load(p,allow_pickle=False) as a:
                for t,g in enumerate(gs):
                    back=graph_at(a,t)
                    for k in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                        self.assertTrue(torch.equal(getattr(g,k),getattr(back,k)))
                    self.assertEqual(a['timestep'][t],t)
                    np.testing.assert_array_equal(a['action7'][t],np.arange(t*7,t*7+7))

    def test_11487_step_schedule_and_episode_boundaries(self):
        root=Path(__file__).resolve().parents[2]
        # Actual lengths are an input fixture; no policy or test output is read.
        candidates=([root.parent/'config.json'] if (root.parent/'config.json').exists() else [])+list((root/'experiments').glob('3788798a719f4ec4a66c224ca42f5d63/config.json'))
        if not candidates:self.skipTest('actual preregistration not present in this source-only deployment')
        config=read(candidates[0]);schedule=[x for epoch in range(1,51) for x in chunks(config,epoch)]
        self.assertEqual(len(schedule),11487)
        for ids,start in schedule:
            self.assertEqual(len(ids),len(set(ids)));self.assertTrue(all(i in config['splits']['train'] for i in ids))
            self.assertTrue(all(start<config['episode_lengths'][str(i)] for i in ids))

    def test_live_rollout_has_no_expert_cache_or_teacher(self):
        tree=ast.parse(Path(__file__).with_name('rollout.py').read_text())
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertFalse({'CachedDataset','ScriptedExpert','DirectDataset'} & names)
        text=Path(__file__).with_name('rollout.py').read_text()
        self.assertIn('CurrentGeometry',text);self.assertIn('physics.unchanged(token)',text)


if __name__=='__main__':unittest.main()
