import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from graph.prototype.data import load_dataset, sha256, split_indices
from graph.prototype.geometry import GeometryState
from graph.prototype.v2.models import CommonPolicy, bc_loss
from graph.prototype.common import config, seed_all
from graph.subtasks.dataset import load_audit, state_row
from graph.subtasks.specs import SubtaskSpec
from graph.subtasks.conditions import ConditionContext
from graph.subtasks.reach_bc.policy import ROOT, DATASET, V2_CONFIG, fit_inputs, inputs, decode, LivePolicy


@unittest.skipUnless(DATASET.exists(),'prepare real first-success dataset first')
class DatasetAndPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(0,4)
        cls.data,cls.g=load_dataset(DATASET)
        cls.audit=json.loads((ROOT/'reports/trimming_audit.json').read_text())

    def test_segment_alignment_original_and_splits_preserved(self):
        old,g=load_dataset('outputs/subtasks/generated_50/bc/reach.pt')
        self.assertEqual(self.data['manifest']['splits'],old['manifest']['splits'])
        for p,h in self.audit['source_hashes'].items():self.assertEqual(sha256(p),h)
        for ep in self.data['episodes']:
            self.assertEqual(len(ep['states']),len(ep['actions'])+1)
            self.assertEqual(len(ep['observations']),len(ep['actions'])+1)
            self.assertEqual(len(ep['qvel']),len(ep['actions'])+1)
            rows=torch.where(self.data['episode_id']==ep['episode_id'])[0]
            original=torch.where(old['episode_id']==ep['episode_id'])[0][:len(rows)]
            self.assertTrue(torch.equal(self.data['actions'][rows],old['actions'][original]))
            for k in self.data['geometry']:
                self.assertTrue(torch.equal(self.data['geometry'][k][rows],old['geometry'][k][original]))
            self.assertTrue(torch.equal(self.data['t'][rows],torch.arange(ep['start_t'],ep['end_t'])))

    def test_first_success_predicate_each_episode(self):
        index=json.loads(Path(self.data['manifest']['source_index']).read_text())
        spec=SubtaskSpec('reach',index['thresholds'])
        for ep in self.data['episodes']:
            source=next(e for e in index['episodes'] if e['episode_id']==ep['episode_id'])
            audit=load_audit(source['audit_file']);context=ConditionContext()
            for t in range(ep['end_t']+1):
                m=spec.compute_metrics(state_row(audit,t))
                context.update(m,t,spec.config,False)
                self.assertEqual(spec.check_success(m,context),t==ep['end_t'])

    def test_train_only_statistics_resist_heldout_poison(self):
        norm,sn=fit_inputs(self.data,self.g)
        ids=torch.cat([split_indices(self.data,s) for s in ('validation','test')])
        changed={k:v.clone() for k,v in self.data['geometry'].items()}
        for v in changed.values():v[ids]=10000.
        q=self.data['qvel'].clone();q[ids]=10000.
        n2,s2=fit_inputs(dict(self.data,qvel=q),GeometryState(**changed))
        for k,v in norm.as_dict().items():torch.testing.assert_close(v,n2.as_dict()[k],atol=0,rtol=0)
        for k,v in sn.as_dict().items():torch.testing.assert_close(v,s2.as_dict()[k],atol=0,rtol=0)

    def test_forward_backward_checkpoint_and_decoding(self):
        norm,sn=fit_inputs(self.data,self.g)
        x,s=inputs(self.g.select(slice(0,4)),self.data['qvel'][:4],norm,sn)
        self.assertEqual(tuple(x.shape),(4,1915));self.assertEqual(tuple(s.shape),(4,31))
        model=CommonPolicy('flat',config(V2_CONFIG),'goal','regression')
        pred=model(x,s);self.assertEqual(tuple(pred.shape),(4,7))
        loss=bc_loss(pred,self.data['actions'][:4]);loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';torch.save(model.state_dict(),path)
            clone=CommonPolicy('flat',config(V2_CONFIG),'goal','regression')
            clone.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(clone(x,s),pred,atol=0,rtol=0)
        torch.testing.assert_close(decode(torch.tensor([[2.,-2.,0.,0.,0.,0.,1.]])),torch.tensor([[1.,-1.,0.,0.,0.,0.,1.]]))
        with self.assertRaises(ValueError):decode(torch.full((1,7),float('nan')))
        with self.assertRaises(ValueError):decode(torch.zeros(1,8))

    def test_measured_overfit_and_full_feature_agreement(self):
        report=json.loads((ROOT/'reports/training_smoke.json').read_text())
        self.assertLess(report['final_mse'],.1*report['initial_mse'])
        self.assertTrue(report['finite_gradients']);self.assertTrue(report['checkpoint_roundtrip_exact'])
        self.assertLessEqual(self.audit['maximum_normalized_feature_error'],.002)
        self.assertLessEqual(self.audit['maximum_raw_feature_error'],2e-6)

    def test_seed_and_source_disjointness(self):
        ids=sum(self.data['manifest']['splits'].values(),[])
        self.assertEqual(len(ids),len(set(ids)))
        seeds=[e['seed'] for e in self.data['episodes']]
        self.assertEqual(len(seeds),len(set(seeds)))
        manifest=json.loads((ROOT/'evaluation/paired_starts.json').read_text())
        self.assertEqual(manifest['seed_overlap'],[])
        self.assertFalse(set(seeds)&set(manifest['fresh_seeds']))
        self.assertEqual(len(manifest['fresh_seeds']),50)
        held=[s['source_episode_id'] for s in manifest['starts'] if s['group']=='held_out']
        self.assertEqual(set(held),set(self.data['manifest']['splits']['test']))
        for start in manifest['starts']:
            if start['valid']:self.assertEqual(start['initial_state_fingerprint'],start['repeat_initial_state_fingerprint'])


@unittest.skipUnless(os.environ.get('REACH_BC_INTEGRATION')=='1','set REACH_BC_INTEGRATION=1 on Linux CPU')
class LiveCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):seed_all(0,4)

    def test_saved_state_live_features_and_success(self):
        from graph.subtasks.env import make_subtask_env
        data,g=load_dataset(DATASET)
        policy=LivePolicy(ROOT/'checkpoints/best.pt')
        env=make_subtask_env('reach',index=data['manifest']['source_index'])
        try:
            eid=data['manifest']['splits']['test'][0]
            env.reset(options={'source_episode_id':eid})
            ids=torch.where(data['episode_id']==eid)[0]
            for row in ids:
                offline=inputs(g.select(row.reshape(1)),data['qvel'][row].reshape(1,9),policy.norm,policy.sn)
                live=policy.features(env)
                for a,b in zip(offline,live):torch.testing.assert_close(a,b,atol=.002,rtol=0)
                _,_,term,trunc,info=env.step(data['actions'][row])
            self.assertTrue(term);self.assertFalse(trunc);self.assertTrue(info['subtask_success'])
            self.assertEqual(info['subtask_step'],len(ids))
        finally:env.close()

    def test_timeout_fresh_reproducibility_and_controller_decoding(self):
        from graph.subtasks.env import make_subtask_env
        from graph.subtasks.reach_bc.evaluate import fingerprint
        from graph.subtasks.reach_bc.diagnostics import reference_action
        # Match the benchmark's recorded environment configuration for fresh
        # starts too; avoids switching SAPIEN's process-global render device.
        data,_=load_dataset(DATASET)
        env=make_subtask_env('reach',index=data['manifest']['source_index'],max_steps=1)
        try:
            env.reset(seed=12000);fp=fingerprint(env)
            env.reset(seed=12000);self.assertEqual(fp,fingerprint(env))
            arm=env.unwrapped.agent.controller.controllers['arm']
            a=torch.tensor([[.2,-.3,.4,.6,.8,0.]])
            scaled=arm._clip_and_scale_action(a)
            torch.testing.assert_close(scaled[:,:3],.1*a[:,:3],atol=1e-7,rtol=0)
            torch.testing.assert_close(scaled[:,3:],-.1*a[:,3:],atol=1e-7,rtol=0)
            gripper=env.unwrapped.agent.controller.controllers['gripper']
            torch.testing.assert_close(gripper._clip_and_scale_action(torch.ones(1,1)),torch.full((1,1),.04),atol=1e-7,rtol=0)
            _,_,term,trunc,info=env.step(torch.tensor([0.,0,0,0,0,0,1.]))
            self.assertFalse(term);self.assertTrue(trunc);self.assertEqual(info['failure_reason'],'timeout')
            env.reset(seed=12000);before=dict(env.metrics)
            action=reference_action(env)
            env.step(action)
            self.assertLess(env.metrics['position_error'],before['position_error'])
            self.assertLess(env.metrics['rotation_error'],before['rotation_error'])
        finally:env.close()

    def test_actual_policy_evaluation_exact_repeat(self):
        report=json.loads((ROOT/'evaluation/flat_repeat.json').read_text())
        self.assertTrue(report['repeat_exact'])
        self.assertEqual(len(report['records']),55)


if __name__=='__main__':unittest.main()
