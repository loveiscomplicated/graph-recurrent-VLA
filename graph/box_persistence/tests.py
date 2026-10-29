"""Mandatory synthetic negative contracts; live evidence is separate."""
import copy,json,unittest,inspect
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from graph.persistent_geometry.tests.test_contracts import surface,camera
from graph.persistent_geometry.map import current_records
from graph.persistent_geometry.registration import Registration
from graph.persistent_geometry.contracts import transform
from graph.plan_a_geometry.core import budget,PlanA
from .core import BoxMemory,paired_records

CFG=json.loads(Path(__file__).with_name('protocol.json').read_text())

def fixture(t=0,shift=0):
    # Dense nonplanar corner fixture, all points have measured source provenance.
    o=replace(surface(),object_id=1,timestep=t,camera='base_camera')
    xyz=o.xyz.copy();xyz[:,0]+=shift
    return current_records([replace(o,xyz=xyz)],fuse_views=False)

class Contracts(unittest.TestCase):
    def setUp(self):self.m=BoxMemory('B1',CFG['memory']);self.c=camera(name='base_camera')
    def update(self,t,rr=None):
        self.m.update_records('ep',t,fixture(t) if rr is None else rr,replace(self.c,timestep=t))
    def test_01_plan_a_identity_contract(self):
        cfg=json.loads(Path(__file__).parents[1].joinpath('plan_a_geometry/protocol.json').read_text())
        self.assertEqual(cfg['training']['updates'],CFG['training']['updates']);self.assertEqual(cfg['point_cap'],128)
    def test_02_peg_exact(self):
        peg=current_records([surface()],fuse_views=False)
        out=paired_records(peg+fixture(),[])
        for a,b in zip(peg,out):np.testing.assert_array_equal(a['world_xyz'],b['world_xyz']);np.testing.assert_array_equal(a['last_f5'],b['last_f5'])
    def test_03_robot76_unchanged(self):
        self.assertNotIn('robot76',inspect.signature(BoxMemory.update_records).parameters)
    def test_04_main_only(self):
        with self.assertRaises(ValueError):self.m.update_records('ep',0,fixture(),replace(self.c,camera='hand_camera'))
    def test_05_wrist_source_forbidden(self):
        r=fixture();r[0]['source_observation_ids'][0]=('ep',0,'hand_camera',0)
        with self.assertRaises(ValueError):self.update(0,r)
    def test_06_robot_filter_reused(self):
        self.assertIn('classify_robot',inspect.getsource(PlanA.prepare))
    def test_07_world_ee_transform(self):
        p=np.array([[1.,2.,3.]]);a=np.eye(4);a[:3,3]=[.1,.2,.3]
        np.testing.assert_allclose(transform(transform(p,a),np.linalg.inv(a)),p)
    def test_08_origin_gt_free(self):
        self.update(0);np.testing.assert_array_equal(self.m.objects[1].world_from_object,np.eye(4))
    def test_09_ids_persist(self):
        self.update(0);old={r['persistent_node_id'] for r in self.m.active_records()};self.update(1)
        self.assertTrue(old<={r['persistent_node_id'] for r in self.m.active_records()})
    def test_10_current_estimated(self):
        self.update(0);self.assertTrue(all(r['currently_observed'] for r in self.m.active_records()));self.update(1,[])
        self.assertTrue(all(r['estimated_only'] and not r['currently_observed'] for r in self.m.active_records()))
    def test_11_empty_memory(self):
        self.update(0);n=len(self.m.active_records());self.update(1,[]);self.assertEqual(len(self.m.active_records()),n)
    def test_12_unseen_empty(self):
        self.update(0,[]);self.assertEqual(self.m.active_records(),[])
    def test_13_reobservation(self):
        self.update(0);self.update(1);self.assertTrue(any(r['observation_count']==2 for r in self.m.active_records()))
    def test_14_new_surface(self):
        self.update(0);n=len(self.m.active_records());self.update(1,fixture(1,.004));self.assertGreater(len(self.m.active_records()),n)
    def test_15_duplicate_merge(self):
        self.update(0);n=len(self.m.active_records());self.update(1);self.assertEqual(len(self.m.active_records()),n)
    def test_16_contamination_reject(self):
        rr=fixture();r=copy.deepcopy(rr[0]);r['world_xyz']+=10;r['source_observation_ids']=[('ep',0,'base_camera',9999)];rr.append(r)
        self.update(0,rr);self.assertLess(len(self.m.active_records()),len(rr));self.assertFalse(any(np.linalg.norm(r['world_xyz'])>5 for r in self.m.active_records()))
    def test_17_static_no_transport(self):
        self.update(0);self.update(1);np.testing.assert_array_equal(self.m.objects[1].world_from_object,np.eye(4))
    def test_18_visible_motion_flag(self):
        self.update(0);p=np.eye(4);p[0,3]=.01
        with patch.object(self.m,'_register',return_value=Registration(p,True,'TRACKED',.001,1.,.8,.01,0.,6)):self.update(1,fixture(1,.01))
        self.assertTrue(self.m.trace[-1]['static_violation'])
    def test_19_hidden_unobservable(self):
        self.update(0);self.update(1,[]);self.assertFalse(self.m.trace[-1]['hidden_motion_observable']);self.assertEqual(self.m.trace[-1]['state'],'UNOBSERVABLE_UNKNOWN')
    def test_20_reappearance_conflict(self):
        self.update(0);self.update(1,[]);self.update(2,fixture(2,.5));self.assertEqual(self.m.trace[-1]['accepted_count'],0)
    def test_21_age_uncertainty(self):
        self.update(0);self.update(1,[]);self.assertTrue(all(r['age']==1 and r['position_uncertainty']>.003 for r in self.m.active_records()))
    def test_22_free_space_contract(self):
        from graph.persistent_geometry.graph import free_space_violation
        self.assertTrue(free_space_violation(np.array([[0,0,.5]]),[camera(z=1000)])[0])
    def test_23_budget(self):
        self.update(0);rr=self.m.active_records();self.assertEqual(len(budget(rr,{0:128,1:7})),min(len(rr),7));self.assertLessEqual(len(budget(rr,128)),128)
    def test_24_historical_f5(self):
        from graph.persistent_geometry.graph import build_graph
        self.update(0);self.update(1,[]);g=build_graph(self.m.active_records(),np.zeros((3,3)),self.c.tcp_pose,1,point_cap=128,include_observation_status=False)
        self.assertFalse(g.node_input[3:,9:11].any());self.assertFalse(g.node_input[3:,12:14].any())
    def test_25_edge_schema(self):
        from graph.persistent_geometry import EDGE_POLICY
        self.assertTrue(EDGE_POLICY)
    def test_26_aperture_audit_exists(self):
        from graph.persistent_geometry.geometry_audit import graph_audit
        self.assertIn('aperture_crossing_edges',inspect.getsource(graph_audit))
    def test_27_permutation(self):
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        from graph.persistent_geometry.train import permutation_audit
        from graph.persistent_geometry.graph import build_graph
        self.update(0);g=build_graph(self.m.active_records(),np.zeros((3,3)),self.c.tcp_pose,0,include_observation_status=False)
        for name in ('E0','E1'):self.assertEqual(permutation_audit(DiagnosticEncoder(name).eval(),g,device='cpu')['status'],'PASS')
    def test_28_reset(self):
        self.update(0);ids={r['persistent_node_id'] for r in self.m.active_records()};self.m.reset('other');self.assertFalse(self.m.records());self.assertEqual(self.m.next_id,0)
    def test_29_no_future_gt(self):
        with self.assertRaises(ValueError):self.update(0,fixture(1))
        self.assertNotIn('actor_pose',inspect.signature(BoxMemory.update_records).parameters)
    def test_30_final_split(self):self.assertFalse(set(CFG['final_seeds'])&set(CFG['excluded_final_seeds']))
    def test_31_finite_shape(self):
        from graph.persistent_geometry.graph import build_graph
        self.update(0);g=build_graph(self.m.active_records(),np.zeros((3,3)),self.c.tcp_pose,0,include_observation_status=False)
        self.assertEqual(g.node_input.shape[1],21);self.assertTrue(np.isfinite(g.node_input.numpy()).all())
    def test_32_no_policy(self):
        self.assertIn('policy_training',CFG['prohibited']);self.assertIn('overwrite_old_artifacts',CFG['prohibited'])
if __name__=='__main__':unittest.main()
