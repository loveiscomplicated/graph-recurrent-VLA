"""Meaningful data/model contracts; CUDA comparisons run on the server."""
import unittest
from pathlib import Path
import tempfile
import copy
import numpy as np
import torch
from graph.geometry_reach10d.resident import ResidentDataset
from graph.integration_baseline.graph import Graph


def fake_episode(length):
    n=5;edges=np.array([[0,3,1,4],[3,0,4,1]],np.int64)
    return dict(node_input=np.arange(length*n*21,dtype=np.float32).reshape(length*n,21)/100,
        node_type=np.tile(np.array([0,1,2,3,4]),length),edge_index=np.tile(edges,(1,length)),
        edge_attr=np.zeros((length*4,4),np.float32),edge_type=np.ones(length*4,np.int64),
        node_ptr=np.arange(length+1)*n,edge_ptr=np.arange(length+1)*4,
        robot76=np.zeros((length,76),np.float32),action7=np.ones((length,7),np.float32))


class Contracts(unittest.TestCase):
    def test_report_merges_already_tagged_validation_rows_without_mutating_inputs(self):
        from graph.geometry_reach10d.report import merge_condition_rows
        data={'A':[dict(step='6000',condition='A',optimization_loss='.043')],
              'B':[dict(step='6000',condition='B',optimization_loss='.040')]}
        before=copy.deepcopy(data)
        self.assertEqual(merge_condition_rows(data),[data['A'][0],data['B'][0]])
        self.assertEqual(data,before)
        self.assertEqual(merge_condition_rows({'A':[{'step':'1000'}]}),
                         [{'step':'1000','condition':'A'}])

    def test_frozen_reach_budget_and_every_update_mask(self):
        import json
        root=Path(__file__).resolve().parents[2].parent
        path=root/'dataset_ab_manifest.json'
        if not path.exists():self.skipTest('actual Reach budget fixture not present on this host')
        manifest=json.loads(path.read_text());cfg=json.loads((root/'config.json').read_text())
        expected=manifest['total_timesteps'];actual={c:0 for c in ['A','B']}
        with np.load(root/'paired_training_schedule.npz') as schedule:
            self.assertEqual(len(schedule['start']),11487)
            for step,start in enumerate(schedule['start']):
                counts={}
                for c in ['A','B']:
                    ids=[int(e) for e in schedule[c][step] if e>=0]
                    counts[c]=sum(min(32,cfg['episode_lengths'][str(e)]-int(start)) for e in ids)
                    self.assertTrue(all(int(start)<cfg['episode_lengths'][str(e)] for e in ids))
                    actual[c]+=counts[c]
                self.assertEqual(counts['A'],counts['B'])
                self.assertEqual(counts['A'],int(schedule['frames'][step]))
        self.assertEqual(actual,expected)

    def test_geometry_signal_audit_uses_cache_variant_id(self):
        from graph.geometry_reach10d.audit_graph_signal import run
        from graph.plan_a_bc.common import write
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);episodes=[]
            for eid,kind,shift in [(1,'original',0),(2,'perturbed',.005)]:
                x=np.zeros((4,21),np.float32);x[3,0]=shift
                np.savez(root/f'{eid}.npz',node_ptr=np.array([0,4]),node_type=np.array([0,1,2,3]),node_input=x)
                episodes.append(dict(episode_id=eid,seed=7,data_kind=kind,path=f'{eid}.npz',variant_id='P0' if eid==1 else 'P1'))
            write(root/'graph_cache_manifest.json',dict(episodes=episodes))
            write(root/'raw/2/receipt.json',dict(spec=dict(offset_mm=[5,0,0]),audit=dict(original=dict(ee_pose=[0,0,0,1,0,0,0]))))
            run(root)
            import json
            result=json.loads((root/'graph_conditional_signal.json').read_text())
            self.assertEqual(result['rows'][0]['variant'],'P1')
            self.assertEqual(result['valid_presence_pairs'],1)
            self.assertLess(result['rows'][0]['peg_centroid_response_error_mm'],1e-5)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not executed on this host')
    def test_resident_packing_and_partial_episode_temporal_order(self):
        from graph.plan_a_bc.cache import CachedDataset
        from types import SimpleNamespace
        ds=SimpleNamespace(episodes={1:dict(actions=33),2:dict(actions=51)},config={},
            arrays={1:fake_episode(33),2:fake_episode(51)})
        resident=ResidentDataset(ds)
        for ids,start in [([1,2],0),([1,2],32),([2],32)]:
            a=CachedDataset.batch(ds,ids,start,32,'cuda');b=resident.batch(ids,start,32,'cuda')
            for k in vars(a[0]):torch.testing.assert_close(getattr(a[0],k),getattr(b[0],k),rtol=0,atol=0)
            for aa,bb in zip(a[1:],b[1:]):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        self.assertEqual(int(resident.batch([1,2],32,32,'cuda')[3].sum()),20)

    def test_new_pipeline_delegates_frozen_model_and_loss(self):
        import inspect
        from graph.geometry_reach10d import train
        from graph.plan_a_bc.train import train_update
        self.assertIs(train.train_update,train_update)
        self.assertNotIn('E1',inspect.getsource(train))

    def test_same_evaluation_geometry_rng_namespace_for_AB(self):
        # Frame sampling derives from episode identity. Evaluation namespaces
        # must be identical for paired A/B at the same seed/offset/history.
        import inspect
        from graph.geometry_reach10d.evaluate import rollout,interventions
        for fn in [rollout,interventions]:
            source=inspect.getsource(fn)
            identity=[line for line in source.splitlines() if 'identity=f' in line][0]
            self.assertNotIn('condition',identity);self.assertNotIn('step}',identity)


if __name__=='__main__':unittest.main()
