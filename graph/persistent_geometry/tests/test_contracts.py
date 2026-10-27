"""Synthetic contracts only. Real RGB-D/physics evidence is a separate runner."""
from copy import deepcopy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from graph.persistent_geometry.contracts import (CameraObservation, SurfaceObservation,
                                                CausalClock, synchronized, transform)
from graph.persistent_geometry.self_filter import LinkSurface, classify_robot
from graph.persistent_geometry.registration import rigid_fit, register
from graph.persistent_geometry.map import PersistentMap, current_records
from graph.persistent_geometry.graph import build_graph, surface_edges
from graph.persistent_geometry.fusion import consistent_views
from graph.integration_baseline.graph import pack_graphs


def camera(t=0, name='camera', episode='ep', z=1000):
    return CameraObservation(episode, t, name, t/20, np.zeros((24, 24, 3), np.uint8),
        np.full((24, 24), z, np.int16), np.array([[40., 0, 12], [0, 40, 12], [0, 0, 1]]),
        np.c_[np.eye(3), np.zeros(3)], np.diag([1., -1, -1, 1]), np.array([0., 0, 0, 1, 0, 0, 0]))


def surface(xyz=None, t=0, oid=0, name='camera', episode='ep'):
    if xyz is None:
        xyz = np.random.default_rng(42).uniform(-.02, .02, (50, 3)) + [0, 0, 1]
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    n = len(xyz)
    normals = np.tile([0., 0, -1], (n, 1))
    return SurfaceObservation(episode, t, name, oid, xyz, normals, np.ones(n, bool),
        np.c_[normals, np.zeros((n, 2))], np.ones((n, 3), bool), np.arange(n), np.zeros(3))


def cube(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


class Contracts(unittest.TestCase):
    def test_01_camera_calibration_timing(self):
        a, b = camera(name='main'), camera(name='wrist')
        synchronized([a, b])
        b = camera(t=1, name='wrist')
        with self.assertRaisesRegex(ValueError, 'timestep'):
            synchronized([a, b])

    def test_02_camera_world_ee_rotation_translation(self):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, .8]).as_matrix()
        matrix[:3, 3] = [.3, -.2, .1]
        p = np.array([[.3, -.4, 1.2], [0, 0, 0]])
        np.testing.assert_allclose(transform(transform(p, matrix), np.linalg.inv(matrix)), p, atol=1e-12)

    def test_03_link_local_geometry_pose(self):
        m = np.eye(4); m[:3, 3] = [1, 2, 3]
        link = LinkSurface('finger', cube([0, 0, 0], [1, 1, 1]), m)
        np.testing.assert_array_equal(link.world_vertices.min(0), [1, 2, 3])

    def test_04_robot_depth_surface_removal(self):
        c = camera(z=1000)
        link = LinkSurface('link', cube([-.3, -.3, 1.], [.3, .3, 1.1]), np.eye(4))
        result = classify_robot(c, [link])
        self.assertTrue(result['confirmed_robot'].all())

    def test_05_grasp_gap_and_nearby_object_preserved(self):
        c = camera(z=1000)
        links = [LinkSurface('left', cube([-.15, -.2, 1], [-.03, .2, 1.1]), np.eye(4)),
                 LinkSurface('right', cube([.03, -.2, 1], [.15, .2, 1.1]), np.eye(4))]
        result = classify_robot(c, links)
        self.assertFalse(result['confirmed_robot'][:, 11:13].any())
        self.assertTrue(result['candidate'][:, 11:13].all())
        # A surface in front of a link is measured separately, even very near.
        c = camera(z=992)
        self.assertFalse(classify_robot(c, links)['confirmed_robot'].any())

    def test_robot_visual_surface_inside_collision_link(self):
        c=camera(z=1050)
        link=LinkSurface('link',cube([-.4,-.4,1.],[.4,.4,1.1]),np.eye(4))
        result=classify_robot(c,[link])
        self.assertTrue(result['inside_known_link_volume'].all())
        self.assertTrue(result['confirmed_robot'].all())

    def test_06_identity_sources(self):
        observation = surface().validate()
        self.assertEqual(observation.sources[0], ('ep', 0, 'camera', 0))
        with self.assertRaises(ValueError):
            SurfaceObservation(**{**vars(observation), 'object_id': 7}).validate()

    def test_07_episode_reset(self):
        memory = PersistentMap().update('ep', 0, [surface()])
        old = {n.persistent_node_id for n in memory.objects[0].nodes}
        memory.update('ep2', 0, [surface(episode='ep2')])
        self.assertFalse(old & {n.persistent_node_id for n in memory.objects[0].nodes})
        self.assertEqual(len(memory.events), 2)

    def test_08_persistent_ids(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        ids = {n.persistent_node_id for n in memory.objects[0].nodes}
        memory.update('ep', 1, [surface(o.xyz, t=1)])
        self.assertEqual(ids, {n.persistent_node_id for n in memory.objects[0].nodes})

    def test_09_reobservation_updates(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        memory.update('ep', 1, [surface(o.xyz, t=1)])
        self.assertTrue(all(n.observation_count == 2 for n in memory.objects[0].active()))

    def test_10_novel_surface_addition(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        before = len(memory.objects[0].nodes)
        extra = np.array([[.025, .01, 1], [.027, .01, 1], [.029, .01, 1]])
        memory.update('ep', 1, [surface(np.vstack((o.xyz, extra)), t=1)])
        self.assertGreater(len(memory.objects[0].nodes), before)
        for n in memory.objects[0].nodes:
            self.assertTrue(n.source_observation_ids)

    def test_11_duplicate_fusion(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        before = len(memory.objects[0].nodes)
        memory.update('ep', 1, [surface(o.xyz, t=1)])
        self.assertEqual(len(memory.objects[0].nodes), before)

    def test_12_occlusion_uncertainty(self):
        memory = PersistentMap().update('ep', 0, [surface()])
        before = deepcopy(memory.records())
        memory.update('ep', 1, [])
        after = memory.records()
        self.assertEqual(len(before), len(after))
        self.assertTrue(all(r['estimated_only'] and not r['currently_observed'] for r in after))
        self.assertGreater(after[0]['position_uncertainty'], before[0]['position_uncertainty'])
        # Hidden actual motion is not available to the estimated pipeline.
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a['world_xyz'], b['world_xyz'])

    def test_13_rigid_object_motion(self):
        o = surface(); m = np.eye(4)
        m[:3, :3] = Rotation.from_rotvec([.03, -.02, .04]).as_matrix()
        m[:3, 3] = [.008, -.004, .002]
        fitted, rank = rigid_fit(o.xyz, transform(o.xyz, m))
        np.testing.assert_allclose(fitted, m, atol=1e-12)
        normals=o.xyz-o.xyz.mean(0)
        normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        o=SurfaceObservation(**{**vars(o),'normals':normals,'f5':np.c_[normals,np.zeros((len(normals),2))]})
        memory = PersistentMap().update('ep', 0, [o])
        before = {n.persistent_node_id for n in memory.objects[0].nodes}
        moved=surface(o.xyz+[.004,-.003,.001],t=1)
        moved=SurfaceObservation(**{**vars(moved),'normals':normals,'f5':np.c_[normals,np.zeros((len(normals),2))]})
        memory.update('ep', 1, [moved])
        self.assertTrue(before <= {n.persistent_node_id for n in memory.objects[0].nodes})
        self.assertLess(memory.objects[0].registration['residual_m'], .001)

    def test_partial_plane_motion_is_not_fully_identifiable(self):
        x,y=np.meshgrid(np.linspace(-.02,.02,10),np.linspace(-.02,.02,10))
        xyz=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        memory=PersistentMap().update('ep',0,[surface(xyz)])
        uncertainty=memory.objects[0].pose_uncertainty
        memory.update('ep',1,[surface(xyz+[.008,0,0],t=1)])
        self.assertEqual(memory.objects[0].tracking_state,'PARTIALLY_OBSERVABLE')
        self.assertLess(memory.objects[0].registration['rank'],6)
        self.assertGreater(memory.objects[0].pose_uncertainty,uncertainty)

    def test_14_registration_failure_and_reappearance_conflict(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        count = len(memory.objects[0].nodes)
        memory.update('ep', 1, [surface(o.xyz+[1, 0, 0], t=1)])
        self.assertEqual(len(memory.objects[0].nodes), count)
        self.assertIn(memory.objects[0].tracking_state, ('REGISTRATION_FAILED', 'SYMMETRY_AMBIGUITY'))
        self.assertFalse(any(n.currently_observed for n in memory.objects[0].nodes))

    def test_15_wrong_object_disconnected_not_accumulated(self):
        o = surface(); memory = PersistentMap().update('ep', 0, [o])
        count = len(memory.objects[0].nodes)
        memory.update('ep', 1, [surface(np.vstack((o.xyz, [[.3, .3, 1.]])), t=1)])
        self.assertEqual(len(memory.objects[0].nodes), count)

    def test_16_main_wrist_duplicate(self):
        o = surface()
        records = current_records([o, surface(o.xyz, name='wrist')])
        self.assertEqual(len(records), len(current_records([o])))
        self.assertTrue(all(r['observation_count'] == 1 for r in records))
        self.assertTrue(any(len(r['source_observation_ids']) > 1 for r in records))

    def test_17_wrong_extrinsic_detected(self):
        c = camera()
        wrong = c.extrinsic_cv.copy(); wrong[0, 3] = .1
        with self.assertRaisesRegex(ValueError, 'calibration'):
            CameraObservation(**{**vars(c), 'extrinsic_cv': wrong}).validate()

    def test_18_identity_swap_rejected(self):
        a = surface(); b = surface(a.xyz+[.15, 0, 0], oid=1)
        memory = PersistentMap().update('ep', 0, [a, b])
        memory.update('ep', 1, [surface(b.xyz, t=1, oid=0)])
        self.assertEqual(memory.objects[0].tracking_state, 'IDENTITY_SWITCH_REJECTED')

    def test_19_graph_schema_and_missing_historical_features(self):
        memory = PersistentMap().update('ep', 0, [surface()])
        memory.update('ep', 1, [])
        c = camera(t=1)
        graph = build_graph(memory.records(), np.zeros((3, 3)), c.tcp_pose, 1)
        self.assertIn('task09c_', graph.packet['edge_policy'])
        self.assertFalse(graph.node_input[3:, 9:11].bool().any())
        self.assertFalse(graph.node_input[3:, 12:14].bool().any())

    def test_20_hole_edge_witness(self):
        a = np.array([[-.008, 0, 1], [.008, 0, 1]])
        e, audit = surface_edges(a, np.tile([0, 0, 1.], (2, 1)), np.ones(2, bool), np.zeros(2, int), {0:a, 1:np.empty((0, 3))})
        self.assertEqual(e.shape[1], 0)
        self.assertGreater(audit['rejected_witness'], 0)

    def test_opposing_thin_faces_are_not_surface_neighbours(self):
        points=np.array([[0,0,1.],[0,0,1.001]])
        edges,audit=surface_edges(points,np.array([[0,0,-1.],[0,0,1.]]),np.ones(2,bool),
            np.zeros(2,int),{0:points,1:np.empty((0,3))})
        self.assertEqual(edges.shape[1],0)
        self.assertGreater(audit['rejected_normal'],0)

    def test_21_gt_cannot_enter_and_future_rejected(self):
        with self.assertRaises(TypeError):
            SurfaceObservation(**{**vars(surface()), 'gt_pose': np.eye(4)})
        memory = PersistentMap()
        with self.assertRaisesRegex(ValueError, 'future'):
            memory.update('ep', 0, [surface(t=1)])
        clock = CausalClock(); clock.advance('ep', 0, [surface()])
        with self.assertRaises(ValueError):
            clock.advance('ep', 2, [surface(t=2)])

    def test_22_frozen_inference_weight_identity(self):
        import torch
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        c = camera(); g = build_graph(current_records([surface()]), np.zeros((3, 3)), c.tcp_pose, 0)
        torch.manual_seed(42)
        model = DiagnosticEncoder('E0').eval().requires_grad_(False)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            z, _ = model(pack_graphs([g]))
        self.assertFalse(z.requires_grad)
        self.assertTrue(all(torch.equal(before[k], v) for k,v in model.state_dict().items()))

    def test_23_group_split_reject_overlap(self):
        from graph.persistent_geometry.audit import validate_split
        validate_split({'train':[1, 2], 'validation':[3], 'evaluation':[4]}, {1:'a',2:'b',3:'c',4:'d'})
        with self.assertRaises(ValueError):
            validate_split({'train':[1], 'validation':[2]}, {1:'same',2:'same'})

    def test_24_graph_finite_tensor_shapes(self):
        c = camera(); records = current_records([surface()])
        g = build_graph(records, np.zeros((3, 3)), c.tcp_pose, 0)
        packed = pack_graphs([g, g])
        self.assertEqual(packed.node_input.shape[1], 21)
        self.assertTrue(np.isfinite(packed.node_input.numpy()).all())
        self.assertEqual(packed.edge_index.shape[0], 2)

    def test_25_order_independent_budget_and_edges(self):
        records = current_records([surface()]); c = camera()
        a = build_graph(records, np.zeros((3, 3)), c.tcp_pose, 0, point_cap=12)
        b = build_graph(records[::-1], np.zeros((3, 3)), c.tcp_pose, 0, point_cap=12)
        np.testing.assert_array_equal(a.node_input.numpy(), b.node_input.numpy())
        np.testing.assert_array_equal(a.edge_index.numpy(), b.edge_index.numpy())
        a=build_graph(records,np.zeros((3,3)),c.tcp_pose,0,point_cap=None)
        b=build_graph(records[::-1],np.zeros((3,3)),c.tcp_pose,0,point_cap=None)
        np.testing.assert_array_equal(a.node_input.numpy(),b.node_input.numpy())
        np.testing.assert_array_equal(a.edge_index.numpy(),b.edge_index.numpy())

    def test_26_repeatability(self):
        a = PersistentMap().update('ep', 0, [surface()])
        b = PersistentMap().update('ep', 0, [surface()])
        self.assertEqual([n.persistent_node_id for n in a.objects[0].nodes], [n.persistent_node_id for n in b.objects[0].nodes])
        np.testing.assert_array_equal(a.objects[0].world_points(), b.objects[0].world_points())

    def test_27_protection_detects_mutation(self):
        import tempfile
        from pathlib import Path
        from graph.persistent_geometry.audit import protection_snapshot, verify_protection
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'frozen'; p.write_bytes(b'original')
            before = protection_snapshot([p])
            self.assertEqual(verify_protection(before)['status'], 'PASS')
            p.write_bytes(b'changed')
            self.assertEqual(verify_protection(before)['status'], 'FAIL')

    def test_encoder_permutation_invariance(self):
        import torch
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        torch.set_num_threads(1)
        c = camera(); g = build_graph(current_records([surface()]), np.zeros((3, 3)), c.tcp_pose, 0)
        permutation = np.r_[np.arange(3), np.arange(len(g.node_type)-1, 2, -1)]
        reverse = np.argsort(permutation)
        other = deepcopy(g)
        other.node_input = g.node_input[permutation]
        other.node_type = g.node_type[permutation]
        other.edge_index = torch.from_numpy(reverse[g.edge_index.numpy()])
        for variant in ('E0', 'E1', 'E2'):
            torch.manual_seed(42); model = DiagnosticEncoder(variant).eval()
            with torch.inference_mode():
                a, _ = model(pack_graphs([g])); b, _ = model(pack_graphs([other]))
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_negative_camera_identity_conflict_unknown(self):
        a, b = camera(name='main'), camera(name='wrist')
        p = a.world_points().reshape(-1, 3)[300:320]
        masks = {'main':np.zeros((2, 24, 24), bool), 'wrist':np.zeros((2, 24, 24), bool)}
        masks['wrist'][1] = True
        result, audit = consistent_views([surface(p, name='main')], [a, b], masks)
        self.assertEqual(len(result[0].xyz), 0)
        self.assertEqual(audit[0]['unknown_conflict'], len(p))

    def test_practical_pipeline_has_no_sidecar_dependency(self):
        from graph.persistent_geometry.pipeline import GeometrySystem
        cameras=[camera(name='main'),camera(name='wrist')]
        data=dict(episode=np.array('ep'),timestep=np.array(0),robot_world_geometry=np.zeros((3,3)),
                  tcp_pose=cameras[0].tcp_pose,robot76=np.zeros(76,np.float32),qpos=np.zeros(9),
                  timestamp=np.array(0.),robot_link_world=np.empty((0,4,4)))
        masks={name:np.zeros((2,24,24),bool) for name in ('main','wrist')}
        for value in masks.values():
            value[0,3:21,2:10]=True;value[1,3:21,14:22]=True
        # A hypothetical evaluator can change arbitrary actor labels without
        # the practical module receiving them or observing a global simulator.
        sidecar={'actor_pose':np.eye(4)}
        a=GeometrySystem('main','wrist').process(cameras,data,masks,[])
        sidecar['actor_pose'][:3,3]=1000
        b=GeometrySystem('main','wrist').process(cameras,data,masks,[])
        for variant in ('G0','G1','G2','G3','G4'):
            np.testing.assert_array_equal(np.array([r['world_xyz'] for r in a['records'][variant]]),
                                          np.array([r['world_xyz'] for r in b['records'][variant]]))
        with self.assertRaisesRegex(ValueError,'non-sensor'):
            GeometrySystem('main','wrist').process(cameras,{**data,**sidecar},masks,[])
        from graph.persistent_geometry.oracle import OracleMotionMap
        with self.assertRaisesRegex(ValueError,'sensor-derived'):
            GeometrySystem('main','wrist',main_map=OracleMotionMap())

    def test_evaluation_coverage_counts_missing_prediction_as_zero(self):
        from graph.persistent_geometry.metrics import GeometryEvaluator
        from scipy.spatial import cKDTree
        points=np.array([[0.,0.,0.],[.01,0.,0.]])
        mesh={oid:dict(points=points,normals=np.tile([0.,0.,1.],(2,1)),tree=cKDTree(points)) for oid in (0,1)}
        evaluator=GeometryEvaluator(mesh)
        gt=dict(actor_pose=np.tile([0.,0.,0.,1.,0.,0.,0.],(2,1)))
        refs={name:{oid:points for oid in (0,1)} for name in ('main','both')}
        measured=evaluator.measure('G0',[],None,gt,refs,0)
        self.assertEqual(measured[0]['common_both_visible_coverage_4mm'],0.)
        self.assertIsNone(measured[0]['surface_precision_4mm'])

    def test_gt_reference_excludes_internal_collision_interfaces(self):
        import tempfile
        from pathlib import Path
        from itertools import product
        from scipy.spatial import ConvexHull
        from graph.persistent_geometry.metrics import sample_meshes
        from graph.integration_baseline.io import save_npz
        arrays={}
        for oid in (0,1):
            for index,(lo,hi) in enumerate(((-.1,0.),(0.,.1))):
                v=np.array(list(product((lo,hi),(-.1,.1),(-.1,.1))))
                arrays[f'object{oid}_vertices{index}']=v
                arrays[f'object{oid}_faces{index}']=ConvexHull(v).simplices
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'meshes.npz';save_npz(path,**arrays)
            result=sample_meshes(path,samples_per_object=3000)
            self.assertTrue(all(not np.any(abs(r['points'][:,0])<1e-10) for r in result.values()))
            self.assertTrue(all(len(r['points'])>1000 for r in result.values()))

    def test_final_dataset_uses_frozen_training_statistics(self):
        import tempfile
        from pathlib import Path
        from graph.persistent_geometry.train import Dataset
        from graph.persistent_geometry.evaluate import save_graph
        from graph.integration_baseline.io import save_npz
        c=camera();g=build_graph(current_records([surface()]),np.zeros((3,3)),c.tcp_pose,0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'frame';folder.mkdir()
            save_graph(folder/'G4_graph.npz',g)
            save_npz(folder/'evaluation_targets.npz',target36=np.full(36,1000.),target_valid=np.ones(36,bool),
                robot76=np.zeros(76),timestep=np.array(0))
            frames=[dict(folder='frame',seed=62091,timestep=0,split='evaluation')]
            data=Dataset(root,'G4',frames,target_statistics=(np.zeros(36),np.ones(36)))
            np.testing.assert_array_equal(data.mean,np.zeros(36))
            self.assertFalse(data.train.any());self.assertFalse(data.validation.any())
            with self.assertRaisesRegex(ValueError,'train/validation'):
                Dataset(root,'G4',frames)

    def test_causal_branch_namespace_does_not_change_point_budget(self):
        from graph.persistent_geometry.graph import farthest_rows
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
        parent=[f'ep:{i:08d}' for i in range(4)]
        branch=parent[:2]+[f'ep:branch_repeat:{i:08d}' for i in range(2,4)]
        np.testing.assert_array_equal(farthest_rows(xyz,parent,3),farthest_rows(xyz,branch,3))

    def test_causal_history_fork_matches_direct_prefix_and_rejects_future(self):
        import tempfile
        from pathlib import Path
        from graph.persistent_geometry.history import restore_history
        from graph.persistent_geometry.evaluate import snapshot_records
        from graph.integration_baseline.io import save_npz,write_json
        observed=surface();parent=PersistentMap().update('ep',0,[observed])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'geometry'/'seed123'/'f0000';folder.mkdir(parents=True)
            snapshot_records(folder/'G2_map.npz',parent.records())
            state=dict(episode='ep',timestep=0,next_id=parent.next_id,objects={oid:dict(
                world_from_object=obj.world_from_object,pose_uncertainty=obj.pose_uncertainty,
                pose_confidence=obj.pose_confidence,tracking_state=obj.tracking_state,
                last_tracking_timestep=obj.last_tracking_timestep,registration=obj.registration)
                for oid,obj in parent.objects.items()})
            write_json(folder/'G2_tracker.json',state)
            sources=[(r['persistent_node_id'],s) for r in parent.records() for s in r['source_observation_ids']]
            arrays=dict(persistent_node_id=np.array([r[0] for r in sources]),
                source_episode=np.array([r[1][0] for r in sources]),source_timestep=np.array([r[1][1] for r in sources]),
                source_camera=np.array([r[1][2] for r in sources]),source_pixel=np.array([r[1][3] for r in sources]))
            save_npz(folder/'G2_source_observations.npz',**arrays)
            fork=restore_history(root,dict(seed=123,episode='ep'),'G2',0,'ep:branch_repeat_t1')
            direct=deepcopy(parent).update('ep',1,[surface(observed.xyz,t=1)])
            fork.update('ep:branch_repeat_t1',1,[surface(observed.xyz,t=1,episode='ep:branch_repeat_t1')])
            np.testing.assert_array_equal(direct.objects[0].world_points(),fork.objects[0].world_points())
            self.assertEqual(parent.clock.timestep,0)
            arrays['source_timestep'][:]=1
            save_npz(folder/'G2_source_observations.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'future'):
                restore_history(root,dict(seed=123,episode='ep'),'G2',0,'ep:branch_wrong_t1')


if __name__ == '__main__':
    unittest.main()
