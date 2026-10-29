import json,unittest,inspect
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from graph.persistent_geometry.tests.test_contracts import camera,surface,cube
from graph.persistent_geometry.contracts import synchronized,transform
from graph.persistent_geometry.self_filter import LinkSurface,classify_robot
from graph.persistent_geometry.map import current_records
from graph.plan_a_geometry.core import PlanA,wrist_gate,deduplicate,select,budget

class Contracts(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
        self.main=camera(name='base_camera',z=100);self.wrist=camera(name='hand_camera',z=100)
        self.cameras=[self.main,self.wrist]
        self.masks={c.camera:np.zeros((2,24,24),bool) for c in self.cameras}
        self.masks['hand_camera'][0,9:15,9:15]=True
        self.data=dict(episode=np.array('ep'),timestep=np.array(0),tcp_pose=self.main.tcp_pose,
            robot_world_geometry=np.zeros((3,3)),robot76=np.zeros(76))
        self.system=PlanA('base_camera','hand_camera',self.cfg)
    def test_same_timestep(self):
        with self.assertRaises(ValueError):synchronized([self.main,replace(self.wrist,timestep=1)])
    def test_calibration_reject(self):
        with self.assertRaises(ValueError):replace(self.main,cam2world_gl=np.eye(4)).validate()
    def test_world_transform(self):
        xyz=self.main.world_points();np.testing.assert_allclose(transform(xyz,np.linalg.inv(self.main.world_from_camera)),self.main.camera_points())
    def test_filter_finger_gap(self):
        c=camera();links=[LinkSurface('l',cube([-.15,-.2,1],[-.03,.2,1.1]),np.eye(4)),LinkSurface('r',cube([.03,-.2,1],[.15,.2,1.1]),np.eye(4))]
        self.assertTrue(classify_robot(c,links)['candidate'][:,11:13].all())
    def test_wrist_only_recovery_and_box_block(self):
        self.masks['hand_camera'][1,2:5,2:5]=True
        p=self.system.prepare(self.cameras,self.data,self.masks,[])
        self.assertTrue(p['gate']['accepted']);self.assertFalse(p['conditions']['C3']['hand_camera'][1].any())
        self.assertTrue(p['conditions']['C3']['hand_camera'][0].any());self.assertFalse(p['conditions']['C0']['hand_camera'].any())
    def test_wrong_identity_reject(self):
        self.masks['base_camera'][1]=self.masks['hand_camera'][0]
        p=self.system.prepare(self.cameras,self.data,self.masks,[])
        self.assertIn('cross_camera_identity_conflict',p['gate']['reasons'])
    def test_empty_no_backfill(self):
        for v in self.masks.values():v[:]=False
        p=self.system.prepare(self.cameras,self.data,self.masks,[])
        self.assertFalse(p['gate']['accepted']);r,_,_,_=self.system.represent(self.cameras,p,'C3')
        self.assertTrue(all(not x for x in r.values()))
    def test_gate_no_gt_or_future(self):
        with self.assertRaisesRegex(ValueError,'forbidden'):self.system.prepare(self.cameras,{**self.data,'actor_pose':np.ones((2,7))},self.masks,[])
        with self.assertRaisesRegex(ValueError,'forbidden'):self.system.prepare(self.cameras,{**self.data,'future_depth':1},self.masks,[])
    def test_duplicate_source_f5_preserved(self):
        a=current_records([surface([[0,0,1]],name='base_camera')],fuse_views=False)[0]
        b=current_records([surface([[.001,0,1]],name='hand_camera')],fuse_views=False)[0]
        a['structural']=True;b['structural']=False
        rows,audit=deduplicate([b,a]);self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]['source_observation_ids']),2)
        np.testing.assert_array_equal(rows[0]['last_f5'],a['last_f5']);np.testing.assert_array_equal(rows[0]['world_xyz'],a['world_xyz'])
    def test_hybrid_anchors_and_shortfall(self):
        rows=current_records([surface()],fuse_views=False)
        for i,r in enumerate(rows):r['structural']=i<5
        out=select(rows,12,anchors=True);self.assertEqual(len(out),12)
        self.assertTrue({r['persistent_node_id'] for r in rows[:5]} <= {r['persistent_node_id'] for r in out})
        self.assertEqual(len(select(rows,128,anchors=True)),50)
        self.assertEqual(select(rows,0),[])
    def test_overflow_repeat_permutation(self):
        rows=current_records([surface()],fuse_views=False)
        for r in rows:r['structural']=True
        a=select(rows,8,anchors=True);b=select(rows[::-1],8,anchors=True)
        self.assertEqual([r['persistent_node_id'] for r in a],[r['persistent_node_id'] for r in b])
    def test_graph_schema_rows_and_permutation(self):
        import torch
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        from graph.persistent_geometry.train import permutation_audit
        rows=current_records([surface()],fuse_views=False);graph=self.system.graph(budget(rows,12),self.data,self.cameras)
        self.assertEqual(graph.node_input.shape,(15,21));self.assertFalse(graph.node_input[:,14:].any())
        self.assertTrue((graph.edge_index<15).all());self.assertEqual(len(graph.edge_attr),graph.edge_index.shape[1])
        for name in ('E0','E1'):
            self.assertEqual(permutation_audit(DiagnosticEncoder(name).eval(),graph,device='cpu')['status'],'PASS')
    def test_frozen_final_seeds(self):
        self.assertFalse(set(self.cfg['final_seeds'])&set(self.cfg['excluded_final_seeds']))
        self.assertEqual(self.cfg['training']['updates'],2000)
    def test_original_sf_exact_before_merge(self):
        from graph.persistent_geometry.fusion import make_packet
        p=self.system.prepare(self.cameras,self.data,self.masks,[])
        records,_,_,packets=self.system.represent(self.cameras,p,'C3',('SF',))
        packet,_=make_packet(self.wrist,p['conditions']['C3']['hand_camera'])
        for k in ('points','object_id','geometric_features'):
            np.testing.assert_array_equal(packet[k],packets['hand_camera'][k])
        for row in records['SF']:
            self.assertEqual(row['source_observation_ids'][0][2],'hand_camera')
    def test_hybrid_actual_support(self):
        p=self.system.prepare(self.cameras,self.data,self.masks,[])
        pools,_,_,_=self.system.represent(self.cameras,p,'C3')
        sources={tuple(r['source_observation_ids'][0]) for r in pools['SF']+pools['L']}
        self.assertTrue(all(tuple(r['source_observation_ids'][0]) in sources for r in pools['H']))
        structural={tuple(r['source_observation_ids'][0]) for r in pools['SF'] if r['structural']}
        hs={tuple(s) for r in pools['H'] for s in r['source_observation_ids']}
        self.assertTrue(structural<=hs)
    def test_hybrid_surface_pool_precedes_legacy_final_fps(self):
        from graph.persistent_geometry.contracts import CameraObservation
        from legacy_fps.adapter import frame_points
        from graph.plan_a_geometry.core import sor_surface_observations
        h,w=80,90
        c=CameraObservation('dense',0,'hand_camera',0.,np.zeros((h,w,3),'uint8'),np.full((h,w),100,'int16'),
            np.array([[90.,0,w/2],[0,90,h/2],[0,0,1]]),np.eye(4)[:3],np.diag([1.,-1,-1,1]),np.array([0,0,0,1,0,0,0.]))
        masks=np.zeros((2,h,w),bool);masks[0,5:75,5:85]=True
        legacy,points,_=frame_points(c,masks,f5=True)
        candidates=sor_surface_observations(c,masks,points)
        self.assertGreater(len(candidates[0].pixels),1024)
        self.assertEqual(len(legacy[0].pixels),1024)
        np.testing.assert_array_equal(candidates[0].pixels,points[0].stages['sor_source'])
        self.assertTrue(np.isin(legacy[0].pixels,candidates[0].pixels).all())
        self.assertTrue((~np.isin(candidates[0].pixels,legacy[0].pixels)).any())
        xyz=c.world_points().reshape(-1,3)[candidates[0].pixels]
        np.testing.assert_allclose(candidates[0].xyz,xyz,atol=1e-7)

if __name__=='__main__':unittest.main()
