"""Observed-support topology and final-row index contracts, without object GT."""
import copy
from dataclasses import replace
import unittest

import numpy as np

from graph.vision_input import receive_vision_packet
from visual.runtime.intra_object_edges import (
    observed_components, build_intra_object_edges, directed_knn, EDGE_POLICY, edge_diagnostics,
)
from visual.runtime.sampling import extract_sampling_evidence
from visual.runtime.test.test_geometry import make_frame
from visual.runtime.test.test_vision_features import feature_fixture, assert_tree_equal
from visual.runtime.vision_packet import complete_vision_packet


class IntraObjectEdgeTests(unittest.TestCase):
    def test_components_use_full_support_not_selected_adjacency(self):
        frame=make_frame(np.ones((1,9),np.float32))
        evidence=extract_sampling_evidence(frame)
        maps,stats=observed_components(evidence)
        self.assertEqual(stats[0]['components'],1)
        source=np.array([8,0],np.int64)
        xyz=evidence.frame.geometry.xyz.reshape(-1,3)[source]
        edges,comp=build_intra_object_edges(xyz,np.zeros(2,np.int64),source,maps)
        np.testing.assert_array_equal(edges,[[0,1],[1,0]])
        self.assertTrue((comp == 0).all())
        self.assertFalse(evidence.frame.objects[0].normal_valid.any())

    def test_depth_gap_invalid_overlap_and_mask_components(self):
        depth=np.ones((5,17),np.float32);depth[:,8:]=1.4
        peg=np.ones(depth.shape,bool);box=np.zeros_like(peg)
        depth[:,4]=0;box[:,12]=True;peg[:,15]=False
        frame=make_frame(depth,peg,box)
        evidence=extract_sampling_evidence(frame)
        maps,stats=observed_components(evidence)
        self.assertEqual(stats[0]['components'],5)
        self.assertTrue((maps[0][:,4] == -1).all())
        self.assertTrue((maps[0][:,12] == -1).all())
        self.assertTrue((maps[0][:,15] == -1).all())
        self.assertNotEqual(maps[0][0,7],maps[0][0,8])
        source=np.flatnonzero(evidence.frame.geometry.observed_mask(0))[::-1]
        xyz=evidence.frame.geometry.xyz.reshape(-1,3)[source]
        edges,comp=build_intra_object_edges(xyz,np.zeros(len(source),np.int64),source,maps)
        self.assertTrue((comp[edges[0]]==comp[edges[1]]).all())

    def test_exact_tie_coincident_xyz_determinism_and_no_duplicates(self):
        points=np.tile([1,0,1],(11,1)).astype(np.float32);points[0]=[0,0,1]
        oid=np.zeros(11,np.int64);source=np.arange(11,dtype=np.int64)
        maps={0:np.zeros((1,11),np.int64),1:np.full((1,11),-1,np.int64)}
        direct=directed_knn(points,oid,np.zeros(11,np.int64))
        np.testing.assert_array_equal(direct[1,direct[0]==0],np.arange(1,9))
        edges,comp=build_intra_object_edges(points,oid,source,maps)
        again,_=build_intra_object_edges(points,oid,source,maps)
        np.testing.assert_array_equal(edges,again)
        pairs=set(map(tuple,edges.T.tolist()))
        self.assertEqual(len(pairs),edges.shape[1]);self.assertLessEqual(len(pairs),16*len(points))
        self.assertTrue(all(a!=b and (b,a) in pairs for a,b in pairs))

    def test_empty_singleton_cross_object_component_ids_and_invalid_source(self):
        for empty in (False,True):
            source,evidence,frame=feature_fixture(empty=empty)
            before=copy.deepcopy(source)
            maps,_=observed_components(evidence)
            edges,comp=build_intra_object_edges(source['points'],source['object_id'],source['provenance']['source_index'],maps)
            self.assertTrue((source['object_id'][edges[0]]==source['object_id'][edges[1]]).all())
            assert_tree_equal(self,source,before)
            if empty:self.assertEqual(edges.shape,(2,0))
        maps={0:np.array([[0,-1]],np.int64),1:np.array([[0,-1]],np.int64)}
        edges,comp=build_intra_object_edges(np.array([[0,0,1]],np.float32),np.array([0],np.int64),np.array([0],np.int64),maps)
        self.assertEqual(edges.shape,(2,0))
        with self.assertRaises(ValueError):
            build_intra_object_edges(np.array([[0,0,1]],np.float32),np.array([0],np.int64),np.array([1],np.int64),maps)

    def test_receiver_preserves_edges_feature_rows_offset_and_rejects_bad_edges(self):
        source,evidence,frame=feature_fixture()
        packet=complete_vision_packet(source,evidence,frame)
        maps,_=observed_components(evidence)
        edges,comp=build_intra_object_edges(packet['camera_points'],packet['object_id'],packet['provenance']['source_index'],maps)
        packet.update(edge_status='supplied',edge_policy=copy.deepcopy(EDGE_POLICY),
                      intra_object_edge_index=edges,observed_component_id=comp)
        before=copy.deepcopy(packet)
        robots=np.zeros((3,3),np.float32)
        result=receive_vision_packet(packet,robot_positions=robots)
        assert_tree_equal(self,packet,result.packet)
        np.testing.assert_array_equal(result.edge_index,edges+3)
        assert_tree_equal(self,packet,before)
        diagnostic=edge_diagnostics(packet)
        self.assertEqual(diagnostic['cross_component_edges'],0)
        self.assertEqual(diagnostic['cross_object_edges'],0)
        for mutate in (
            lambda p:p.update(intra_object_edge_index=np.concatenate((edges,edges[:,:1]),axis=1)),
            lambda p:p.update(intra_object_edge_index=np.array([[0],[0]],np.int64)),
            lambda p:p.update(intra_object_edge_index=edges[:,:1]),
            lambda p:p['observed_component_id'].__setitem__(edges[0,0],99),
        ):
            bad=copy.deepcopy(packet);mutate(bad)
            with self.assertRaises(ValueError):receive_vision_packet(bad)

    def test_no_hidden_gt_or_geometry_selection_dependencies(self):
        source,evidence,frame=feature_fixture()
        class ObservationOnly:
            def __getattr__(self,name):
                if name in ('estimator','frame','pairs'):return getattr(evidence,name)
                raise AssertionError('unexpected GT/state access: '+name)
        maps,_=observed_components(ObservationOnly())
        build_intra_object_edges(source['points'],source['object_id'],source['provenance']['source_index'],maps)
        bad_pairs=dict(evidence.pairs)
        bad_pairs[0]=replace(bad_pairs[0],supported=~bad_pairs[0].supported)
        with self.assertRaises(ValueError):observed_components(replace(evidence,pairs=bad_pairs))


if __name__=='__main__':unittest.main()
