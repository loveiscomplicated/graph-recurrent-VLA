"""Compare connectivity and complete gate decisions with the unchanged builder."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.plan_a_geometry import core
from graph.plan_a_geometry.tests import Contracts

spec = importlib.util.spec_from_file_location('task10i_radius_components',
    Path(__file__).with_name('radius_components.py'))
optimized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized)


class RadiusComponents(unittest.TestCase):
    def compare(self, xyz, radius=.012):
        np.testing.assert_array_equal(optimized.components(xyz, radius),
                                      core.components(xyz, radius))

    def test_empty_single_duplicate_disconnected_and_ties(self):
        for xyz in [np.empty((0, 3)), np.zeros((1, 3)), np.zeros((100, 3)),
                    np.array([[1, 0, 0], [0, 0, 0], [1.001, 0, 0], [.001, 0, 0]])]:
            self.compare(xyz)

    def test_random_clusters_and_permutations(self):
        for seed in range(40):
            rng = np.random.default_rng(seed)
            xyz = rng.normal(0, .005, (250, 3))
            xyz += rng.integers(0, 4, (250, 1))*.04
            for radius in [.001, .006, .012, .04]:
                self.compare(xyz[rng.permutation(len(xyz))], radius)

    def test_radius_boundary_and_cell_boundaries(self):
        r = .012
        for distance in [np.nextafter(r, 0), r, np.nextafter(r, np.inf)]:
            for offset in [0., -.1, .003, 1.]:
                for axis in np.eye(3):
                    self.compare(np.array([axis*offset, axis*(offset+distance)]), r)
        grid = np.array(list(__import__('itertools').product(range(9), repeat=3)))*r/2
        self.compare(grid, r)
        self.compare(np.nextafter(grid, np.inf), r)

    def test_coplanar_and_chain_connected_support(self):
        yy, xx = np.mgrid[:48, :48]
        self.compare(np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))*.001)
        self.compare(np.column_stack((np.arange(1000)*.0119, np.zeros((1000, 2)))))

    def test_complete_gate_decision_and_tied_component_selection(self):
        fixture = Contracts()
        fixture.setUp()
        for mode in ['normal', 'empty', 'conflict', 'fragmented']:
            masks = {k: v.copy() for k, v in fixture.masks.items()}
            if mode == 'empty':
                masks['hand_camera'][:] = False
            if mode == 'conflict':
                masks['base_camera'][1] = masks['hand_camera'][0]
            if mode == 'fragmented':
                masks['hand_camera'][0] = False
                masks['hand_camera'][0, 2:5, 2:5] = True
                masks['hand_camera'][0, 19:22, 19:22] = True
            args = (fixture.wrist, masks['hand_camera'], masks['hand_camera'],
                    fixture.cameras, masks, fixture.cfg['gate'])
            expected, trace = core.wrist_gate(*args)
            with patch.object(core, 'components', optimized.components):
                actual, actual_trace = core.wrist_gate(*args)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_trace, trace)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA server only')
class CUDAComponents(unittest.TestCase):
    def test_cuda_greedy_dedup_matches_reference(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        from fast_graph import gpu_deduplicate, tree_deduplicate
        from graph.plan_a_geometry.core import deduplicate
        rng=np.random.default_rng(211)
        for n in (1,32,200):
            xyz=rng.normal(0,.008,(n,3))
            if n>4:
                xyz[1]=xyz[0];xyz[2]=xyz[0]+[.003,0,0]
                xyz[3]=xyz[0]+[np.nextafter(.003,0),0,0]
            records=[dict(world_xyz=p,object_id=i%2,structural=(i%7==0),
                source_observation_ids=[('episode',0,'base',i)]) for i,p in enumerate(xyz)]
            expected,stats=deduplicate(records)
            for implementation in (gpu_deduplicate,tree_deduplicate):
                actual,actual_stats=implementation(records,fallback=deduplicate)
                self.assertEqual(actual_stats,stats)
                self.assertEqual(len(actual),len(expected))
                for a,b in zip(actual,expected):
                    np.testing.assert_array_equal(a['world_xyz'],b['world_xyz'])
                    self.assertEqual(a['object_id'],b['object_id'])
                    self.assertEqual(a['structural'],b['structural'])
                    self.assertEqual(a['source_observation_ids'],b['source_observation_ids'])

    def test_skipping_unused_edges_preserves_consumed_packet_fields(self):
        from fast_graph import make_packet_without_unused_edges
        from graph.persistent_geometry.fusion import make_packet,observations_from_packet
        fixture=Contracts();fixture.setUp();camera=fixture.wrist;masks=fixture.masks['hand_camera']
        old,_=make_packet(camera,masks);new,_=make_packet_without_unused_edges(camera,masks)
        self.assertEqual(new['edge_status'],'not_implemented')
        for key in ['points','object_id','geometric_features','feature_field_valid']:
            np.testing.assert_array_equal(old[key],new[key])
        for key in ['source_index','surface_selected']:
            np.testing.assert_array_equal(old['provenance'][key],new['provenance'][key])
        for a,b in zip(observations_from_packet(old,camera),observations_from_packet(new,camera)):
            for key in ['xyz','normals','normal_valid','f5','feature_valid','pixels']:
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))

    def test_cuda_local_support_matches_original(self):
        from gpu_graph import OptimizedGraph
        from scipy.spatial import cKDTree
        for seed in range(10):
            rng=np.random.default_rng(seed)
            xyz=np.r_[rng.normal(0,.001,(300,3)),rng.uniform(-.1,.1,(40,3))]
            expected=cKDTree(xyz).query_ball_point(xyz,.02,return_length=True)>=3
            with OptimizedGraph('cuda_fused_v5'):
                core.components(xyz,.012)
                actual=core.cKDTree(xyz).query_ball_point(xyz,.02,return_length=True)>=3
            np.testing.assert_array_equal(actual,expected)

    def test_cuda_fps_matches_cpu_rows(self):
        from gpu_graph import fps_indices_cuda
        from visual.runtime.sampling import fps_indices
        for seed in range(12):
            rng=np.random.default_rng(seed)
            xyz=rng.normal(0,.1,(256,3)).astype(np.float32)
            xyz[100:110]=xyz[:10]
            for initial in [[],[0],[5,0,17]]:
                for fused in [False,True]:
                    np.testing.assert_array_equal(fps_indices_cuda(xyz,50,seed=seed,initial=initial,fused=fused),
                                                  fps_indices(xyz,50,seed=seed,initial=initial))
        symmetric=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],np.float32)
        for fused in [False,True]:
            np.testing.assert_array_equal(fps_indices_cuda(symmetric,5,initial=[0],fused=fused),fps_indices(symmetric,5,initial=[0]))

    def test_cuda_components_match_frozen_cpu(self):
        import gpu_graph
        r=.012
        samples=[np.empty((0,3)),np.zeros((20,3)),
                 np.column_stack((np.arange(100)*.0119,np.zeros((100,2))))]
        for distance in [np.nextafter(r,0),r,np.nextafter(r,np.inf)]:
            for offset in [0.,-.1,.003,1.]:
                samples.append(np.array([[offset,0,0],[offset+distance,0,0]]))
        for seed in range(10):
            rng=np.random.default_rng(seed)
            samples.append(rng.normal(0,.005,(300,3))+rng.integers(0,4,(300,1))*.04)
        for xyz in samples:
            np.testing.assert_array_equal(gpu_graph.radius_components(xyz,r),core.components(xyz,r))

    def test_cuda_complete_gate_and_robot_filter(self):
        from gpu_graph import OptimizedGraph
        from graph.persistent_geometry.self_filter import classify_robot,LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        fixture=Contracts();fixture.setUp()
        args=(fixture.wrist,fixture.masks['hand_camera'],fixture.masks['hand_camera'],
              fixture.cameras,fixture.masks,fixture.cfg['gate'])
        expected,trace=core.wrist_gate(*args)
        surfaces=[LinkSurface('left',cube([-.15,-.2,1],[-.03,.2,1.1]),np.eye(4))]
        old=classify_robot(fixture.wrist,surfaces)
        for backend in ['cuda','cuda_fused']:
            with OptimizedGraph(backend):
                actual,actual_trace=core.wrist_gate(*args)
                np.testing.assert_array_equal(actual,expected);self.assertEqual(trace,actual_trace)
                new=core.classify_robot(fixture.wrist,surfaces)
                for name in ['confirmed_robot','uncertain','candidate','inside_known_link_volume']:
                    np.testing.assert_array_equal(new[name],old[name])


if __name__ == '__main__':
    unittest.main()
