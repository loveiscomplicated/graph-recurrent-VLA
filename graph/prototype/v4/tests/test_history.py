import io
import unittest
from pathlib import Path
import torch
from graph.prototype.common import config, seed_all
from graph.prototype.data import load_dataset, split_indices
from graph.prototype.tests.fixtures import geometry
from graph.prototype.normalization import FeatureNormalizer
from graph.prototype.graph_builder import GraphBuilder
from graph.prototype.v2.models import CommonPolicy
from graph.prototype.v2.state import StateNormalizer
from graph.prototype.v2.diagnostics import qvel_dataset
from graph.prototype.v4.history_dataset import HistoryDataset
from graph.prototype.v4.history_buffer import HistoryBuffer
from graph.prototype.v4.history_models import HistoryPolicy, MODES
from graph.prototype.v4.evaluate import Runtime, HistoryReplaySession, run_episode


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(actions=torch.arange(56).reshape(8,7).float(),
            episode_id=torch.tensor([2,1,2,1,2,1,2,1]), timestep=torch.tensor([0,0,1,1,2,2,3,3]))

    def test_alignment_padding_boundaries_and_leakage(self):
        w = HistoryDataset(self.data,4)
        for i in range(8):
            states, actions, valid = w.window(torch.tensor([i]))
            t = int(self.data['timestep'][i])
            self.assertEqual(states[0,-1], i)
            self.assertTrue((self.data['episode_id'][states] == self.data['episode_id'][i]).all())
            self.assertEqual(valid.sum(), t)
            self.assertTrue((actions[~valid.bool()] == 0).all())
            self.assertTrue((self.data['timestep'][w.past[i][valid[0].bool()]] < t).all())
            old = actions.clone(); self.data['actions'][i] += 10000
            self.assertTrue(torch.equal(w.window(torch.tensor([i]))[1],old))
            self.data['actions'][i] -= 10000
        self.assertEqual(w.states[6].tolist(),[0,0,2,4,6])

    def test_buffer_offline_equivalence_applied_action_and_reset(self):
        w = HistoryDataset(self.data,4); b = HistoryBuffer(4)
        reps = torch.arange(8).float()[:,None].expand(-1,159)
        for i in [0,2,4,6]:
            states, actions, valid = w.window(torch.tensor([i]))
            actual = b.window(reps[i:i+1])
            self.assertTrue(torch.equal(actual[0], reps[states]))
            self.assertTrue(torch.equal(actual[1],actions))
            self.assertTrue(torch.equal(actual[2],valid))
            b.commit(reps[i:i+1],self.data['actions'][i:i+1])
        b.reset(); self.assertEqual(b.window(reps[1:2])[2].sum(),0)
        b.commit(reps[1:2],torch.ones(1,7))
        self.assertTrue(torch.equal(b.window(reps[3:4])[1][0,-1],torch.ones(7)))

    def test_lengths_and_reject_invalid_timesteps(self):
        for k in [0,1,4,8,16]:
            self.assertEqual(HistoryDataset(self.data,k).window(torch.tensor([6]))[0].shape,(1,k+1))
        with self.assertRaises(ValueError): HistoryDataset(self.data,-1)
        self.data['timestep'][6]=7
        with self.assertRaises(ValueError): HistoryDataset(self.data)


class ModelTests(unittest.TestCase):
    def test_v2_current_exact_weights_and_predictions(self):
        cfg=config(Path('graph/prototype/v4/config.toml')); g=geometry(2)
        norm=FeatureNormalizer.fit(g);q=torch.randn(2,9);sn=StateNormalizer.fit(g,q,'goal')
        for name in ['flat','surface']:
            seed_all(0,1); old=CommonPolicy(name,cfg)
            seed_all(0,1); new=HistoryPolicy(name,cfg,'current')
            self.assertEqual(old.state_dict().keys(),new.state_dict().keys())
            self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))
            inputs=norm.apply(name,g,GraphBuilder());s=sn.apply(g,q,'goal')
            self.assertTrue(torch.equal(old(inputs,s),new.from_history(new.encode(inputs,s)[:,None],torch.zeros(2,4,7),torch.zeros(2,4))))

    def test_shared_encoder_gradient_shapes_and_checkpoint_all_modes(self):
        cfg=config(Path('graph/prototype/v4/config.toml'));g=geometry(5);q=torch.randn(5,9)
        norm=FeatureNormalizer.fit(g);sn=StateNormalizer.fit(g,q,'goal')
        data=dict(actions=torch.randn(5,7),episode_id=torch.zeros(5,dtype=torch.long),timestep=torch.arange(5))
        windows=HistoryDataset(data)
        for name in ['flat','surface']:
            for mode in MODES:
                seed_all(0,1);m=HistoryPolicy(name,cfg,mode);builder=GraphBuilder()
                out=windows.forward(m,torch.tensor([0,4]),g,q,norm,sn,builder)
                self.assertEqual(out.shape,(2,7));out.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                self.assertEqual(sum(k.startswith('encoder.') for k in m.state_dict()),sum(k.startswith('encoder.') for k in CommonPolicy(name,cfg).state_dict()))
                blob=io.BytesIO();torch.save(m.state_dict(),blob);blob.seek(0)
                loaded=HistoryPolicy(name,cfg,mode);loaded.load_state_dict(torch.load(blob,weights_only=True))
                self.assertTrue(torch.equal(out.detach(),windows.forward(loaded,torch.tensor([0,4]),g,q,norm,sn,builder)))

    def test_each_state_receives_gradient_and_cache_requires_eval(self):
        cfg=config();m=HistoryPolicy('flat',cfg,'both');h=torch.randn(1,5,159,requires_grad=True)
        m.from_history(h,torch.randn(1,4,7),torch.ones(1,4)).sum().backward()
        self.assertTrue((h.grad.abs().sum(-1)>0).all())
        with self.assertRaises(ValueError): Runtime(m,None,None,None)


class LiveTests(unittest.TestCase):
    def test_train_split_normalizers_and_real_prefix_learner_transition(self):
        seed_all(0,1);data,g=load_dataset('outputs/graph/dataset.pt');q=qvel_dataset(data)
        cfg=config(Path('graph/prototype/v4/config.toml'));train=split_indices(data,'train')
        norm=FeatureNormalizer.fit(g.select(train));sn=StateNormalizer.fit(g.select(train),q[train],'goal')
        model=HistoryPolicy('flat',cfg,'both');model.eval()
        runtime=Runtime(model,norm,sn,GraphBuilder());session=HistoryReplaySession(data,runtime,horizon=2)
        ep=next(e for e in data['manifest']['episodes'] if e['episode_id']==29)
        try:
            with torch.inference_mode():
                error,_=session.prepare(ep,77);self.assertLessEqual(error,.001)
                ids=torch.where((data['episode_id']==29)&(data['timestep']==77))[0]
                _,_,actual,valid=runtime.predict(session.adapter(session.env),session.env.unwrapped.agent.robot.get_qvel())
                states,actions,mask=HistoryDataset(data).window(ids)
                self.assertTrue(torch.equal(actual,actions));self.assertTrue(torch.equal(valid,mask))
                live_h=runtime.buffer.window(runtime.encode(session.adapter(session.env),q[ids])[0])[0]
                sg=g.select(states.flatten());offline_h=model.encode(norm.apply('flat',sg,GraphBuilder()),sn.apply(sg,q[states.flatten()],'goal')).reshape(1,5,159)
                self.assertTrue(torch.allclose(live_h,offline_h,atol=2e-4))
                records=run_episode(session,ep,77)
                first,second=records['trace'];self.assertEqual(first['history_actions'],actions[0].tolist())
                self.assertEqual(second['history_actions'][-1],first['applied_action'])
                self.assertEqual(second['history_actions'][:-1],first['history_actions'][1:])
                full=run_episode(session,ep,0)
                self.assertEqual(full['trace'][0]['history_valid'],[0.]*4)
                self.assertEqual(full['trace'][1]['history_valid'],[0.,0.,0.,1.])
                self.assertEqual(full['trace'][1]['history_actions'][-1],full['trace'][0]['applied_action'])
        finally:session.close()
        # Windows remain in the episode's original split, including all padding.
        w=HistoryDataset(data)
        for split in ['train','validation','test']:
            ids=split_indices(data,split)
            self.assertTrue(torch.isin(data['episode_id'][w.states[ids]],torch.tensor(data['manifest']['splits'][split])).all())
        before=sn.mean.clone();sn.apply(g.select(split_indices(data,'test')),q[split_indices(data,'test')],'goal')
        self.assertTrue(torch.equal(before,sn.mean))


if __name__=='__main__':unittest.main()
