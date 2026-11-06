import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from graph.prototype.tests.fixtures import geometry
from graph.prototype.common import config,seed_all
from graph.prototype.data import load_dataset,split_indices
from graph.prototype.geometry import rotation,transform
from graph.prototype.graph_builder import GraphBuilder
from graph.prototype.normalization import FeatureNormalizer
from graph.prototype.models import make_policy
from graph.prototype.v2.goals import goal_geometry,rotation6
from graph.prototype.v2.state import shared_features,StateNormalizer,STATE_DIMS
from graph.prototype.v2.models import CommonPolicy,bc_loss
from graph.prototype.v2.diagnostics import ReplaySession,qvel_dataset,stage_summary,THRESHOLDS


class StateTests(unittest.TestCase):
    def test_dimensions_finite_and_shape_rejection(self):
        g=geometry();q=torch.zeros(4,9)
        for mode,dim in STATE_DIMS.items():
            f=shared_features(g,q,mode);self.assertEqual(f.shape,(4,dim));self.assertTrue(torch.isfinite(f).all())
        with self.assertRaises(ValueError):shared_features(g,q[:,:8])
        q[0,0]=float('nan')
        with self.assertRaises(ValueError):shared_features(g,q)

    def test_rotation_columns_quaternion_sign_and_goal_frame(self):
        g=geometry(1)
        r=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]])
        self.assertTrue(torch.equal(rotation6(r),torch.tensor([[0.,1.,0.,-1.,0.,0.]])))
        changed=replace(g,peg_pose=torch.cat((g.peg_pose[:,:3],-g.peg_pose[:,3:]),1))
        self.assertTrue(torch.allclose(shared_features(g,torch.zeros(1,9)),shared_features(changed,torch.zeros(1,9))))
        head_local=torch.tensor([[[g.peg_half_size[0,0],0.,0.]]])
        expected=(transform(head_local,g.peg_pose)-g.hole_pose[:,:3,None].transpose(1,2)) @ rotation(g.hole_pose[:,3:])
        self.assertTrue(torch.allclose(goal_geometry(g)['head_in_hole'],expected[:,0]))

    def test_common_world_rigid_transform(self):
        g=geometry(1);q=torch.tensor([[.70710678,0.,0.,.70710678]]);r=rotation(q);translation=torch.tensor([[.4,-.2,.1]])
        def pose(p):
            # All fixture quaternions identity. Rotate each frame together.
            return torch.cat((p[:,:3]@r[0].T+translation,q),-1)
        shifted=replace(g,ee_pose=pose(g.ee_pose),peg_pose=pose(g.peg_pose),hole_pose=pose(g.hole_pose))
        a,b=goal_geometry(g),goal_geometry(shifted)
        for k in ['ee_to_grasp','grasp_relative_rotation','head_in_hole','peg_relative_rotation']:
            self.assertTrue(torch.allclose(a[k],b[k],atol=1e-6),k)

    def test_train_only_normalization(self):
        g=geometry();q=torch.randn(4,9)
        norm=StateNormalizer.fit(g.select(torch.tensor([0,1])),q[:2],'goal')
        f=norm.apply(g.select(torch.tensor([0,1])),q[:2],'goal')
        self.assertTrue(torch.allclose(f.mean(0),torch.zeros(31),atol=1e-5))
        before=norm.mean.clone();q[2:]=10000
        self.assertTrue(torch.equal(before,norm.mean))


class ModelTests(unittest.TestCase):
    def test_v0_exact_initialization(self):
        cfg=config();seed_all(0,1);original=make_policy('flat',cfg)
        seed_all(0,1);new=CommonPolicy('flat',cfg,'none');g=geometry();n=FeatureNormalizer.fit(g);x=n.apply('flat',g,None)
        self.assertTrue(torch.equal(original(x),new(x,torch.zeros(4,0))))

    def test_forward_backward_overfit_checkpoint_all_models(self):
        cfg=config();g=geometry(2);q=torch.randn(2,9)
        norm=FeatureNormalizer.fit(g);sn=StateNormalizer.fit(g,q,'goal');s=sn.apply(g,q,'goal')
        for name in ['flat','center','surface']:
            seed_all(0,1);model=CommonPolicy(name,cfg)
            x=norm.apply(name,g,GraphBuilder(center=name=='center'))
            target=torch.full((2,7),.15);opt=torch.optim.Adam(model.parameters(),lr=.003)
            initial=float(bc_loss(model(x,s),target).detach())
            for _ in range(80):
                opt.zero_grad();loss=bc_loss(model(x,s),target);loss.backward();opt.step()
            model.eval();out=model(x,s).detach()
            self.assertEqual(model.encoder(x).shape,(2,128));self.assertEqual(out.shape,(2,7))
            self.assertLess(float(bc_loss(out,target)),initial*.1)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'checkpoint.pt';torch.save(model.state_dict(),path)
                loaded=CommonPolicy(name,cfg);loaded.load_state_dict(torch.load(path,weights_only=True));loaded.eval()
                self.assertTrue(torch.equal(out,loaded(x,s).detach()))
                self.assertTrue(torch.equal(out,model(x,s).detach()))

    def test_head_width_configuration(self):
        cfg=config();cfg['head_width']=64
        model=CommonPolicy('flat',cfg,'goal')
        self.assertEqual(model.action_head[0].out_features,64)

    def test_binary_loss_and_commands(self):
        cfg=config();g=geometry(2);model=CommonPolicy('flat',cfg,'proprio','binary')
        n=FeatureNormalizer.fit(g);x=n.apply('flat',g,None);s=torch.zeros(2,9)
        raw=model.raw_action(x,s);loss=bc_loss(raw,torch.ones(2,7),'binary');loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.all(model(x,s)[:,6].abs()==1))


class EvaluationTests(unittest.TestCase):
    def test_wilson_and_bounding(self):
        from graph.prototype.v2.compare import wilson
        low,high=wilson(0,9)
        self.assertAlmostEqual(low,0);self.assertAlmostEqual(high,.29915275,places=5)
        action=torch.tensor([[-2.,-.1,0.,.1,2.,1.,1.2]])
        applied=action.clamp(-1,1)
        self.assertTrue(torch.all(applied.abs()<=1));self.assertTrue(torch.equal(applied[:,1:4],action[:,1:4]))

    def test_actual_horizon_audit(self):
        audit=json.loads(Path('outputs/graph/v2/diagnostics/audit.json').read_text())
        self.assertEqual(audit['default_limit'],100)
        rows={(r['horizon'],r['t']):r for r in audit['records']}
        self.assertTrue(rows[100,100]['truncated']);self.assertFalse(rows[250,100]['truncated']);self.assertTrue(rows[250,250]['truncated'])

    def test_success_ever_and_retention(self):
        base=dict(grasped=True,lifted=True,alignment=False,insertion=False,approach=False)
        trace=[dict(after=base,success=t==2) for t in range(6)]
        result=stage_summary(trace)
        self.assertTrue(result['success']);self.assertTrue(result['acquisition']);self.assertTrue(result['transport'])
        self.assertIsNone(result['failure_stage'])

    def test_saved_expert_and_paired_reproduction(self):
        expert=json.loads(Path('outputs/graph/v2/diagnostics/expert.json').read_text())
        self.assertEqual(expert['settings']['full']['successes'],9)
        self.assertTrue(all(e['observation_max_error']==0 for e in expert['settings']['full']['episodes']))
        records=[json.loads(Path(f'outputs/graph/v2/diagnostics/{n}_v0.json').read_text()) for n in ['flat','center','surface']]
        for stage in ['full','grasp','transport','alignment']:
            keys=[[(e['episode_id'],e['prefix'],e['initial_geometry_sha256']) for e in r['settings'][stage]['episodes']] for r in records]
            self.assertEqual(keys[0],keys[1]);self.assertEqual(keys[0],keys[2])

    def test_live_feature_action_controller_consistency(self):
        data,g=load_dataset('outputs/graph/dataset.pt');q=qvel_dataset(data)
        session=ReplaySession(data)
        try:
            ep=next(e for e in data['manifest']['episodes'] if e['episode_id']==29)
            session.prepare(ep,56);live=session.adapter(session.env);vel=session.env.unwrapped.agent.robot.get_qvel()
            row=torch.where((data['episode_id']==29)&(data['timestep']==56))[0]
            self.assertTrue(torch.allclose(vel,q[row],atol=2e-5))
            self.assertTrue(torch.allclose(shared_features(live,vel),shared_features(g.select(row),q[row]),atol=2e-5))
            control=session.env.unwrapped.agent.controller.controllers['arm']
            a=torch.tensor([[.2,-.3,.4,.2,-.3,.4]])
            scaled=control._clip_and_scale_action(a.clone())
            self.assertTrue(torch.allclose(scaled[:,:3],a[:,:3]*.1,atol=1e-6))
            self.assertTrue(torch.allclose(scaled[:,3:],a[:,3:]*-.1,atol=1e-6))
        finally:session.close()

    def test_real_demonstration_overfit(self):
        data,_=load_dataset('outputs/graph/dataset.pt')
        results=json.loads(Path('outputs/graph/v2/diagnostics/overfit_real.json').read_text())
        for row in results.values():
            self.assertTrue(row['passed']);self.assertTrue(row['checkpoint_identical'])
            self.assertLess(row['final']['action_mse'],row['initial']['action_mse']*.1)
            self.assertTrue(set(row['training_episode_ids']).issubset(data['manifest']['splits']['train']))

    def test_main_common_normalization_budget_and_pairing(self):
        checkpoints=[torch.load(f'outputs/graph/v2/training/{n}_goal/best.pt',weights_only=True) for n in ['flat','center','surface']]
        fields=['epochs','seed','batch_size','learning_rate','weight_decay','threads','head_width','control_mode']
        for other in checkpoints[1:]:
            for key in fields:self.assertEqual(checkpoints[0]['config'][key],other['config'][key])
            self.assertEqual(checkpoints[0]['dataset_sha256'],other['dataset_sha256'])
            self.assertEqual(checkpoints[0]['splits'],other['splits'])
            for key in ['mean','std']:self.assertTrue(torch.equal(checkpoints[0]['state_normalizer'][key],other['state_normalizer'][key]))
            for key in checkpoints[0]['normalizer']:self.assertTrue(torch.equal(checkpoints[0]['normalizer'][key],other['normalizer'][key]))
        for kind,settings in [('goal',['full','grasp','transport','alignment']),('fresh',['full'])]:
            reports=[json.loads(Path(f'outputs/graph/v2/diagnostics/{n}_{kind}.json').read_text()) for n in ['flat','center','surface']]
            for stage in settings:
                reference=[(e['episode_id'],e['prefix'],e['initial_geometry_sha256']) for e in reports[0]['settings'][stage]['episodes']]
                for other in reports[1:]:
                    self.assertEqual(other['horizon'],250)
                    self.assertEqual(reference,[(e['episode_id'],e['prefix'],e['initial_geometry_sha256']) for e in other['settings'][stage]['episodes']])

    def test_episode_split_and_inputs_no_labels(self):
        data,g=load_dataset('outputs/graph/dataset.pt')
        splits=[set(data['episode_id'][split_indices(data,k)].tolist()) for k in ['train','validation','test']]
        self.assertFalse(splits[0]&splits[1]);self.assertFalse(splits[0]&splits[2]);self.assertFalse(splits[1]&splits[2])
        self.assertEqual(set(g.as_dict()),set(data['geometry']))
        self.assertNotIn('phase',g.as_dict());self.assertNotIn('success',g.as_dict())

if __name__=='__main__':unittest.main()
