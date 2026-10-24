"""Direct experimental receiver contracts; not an existing GNN integration test."""
import copy
import inspect
import unittest

import numpy as np

from graph.vision_input import receive_vision_packet, validate_packet
from visual.runtime.test.test_variable_rate_shape_first import packet_fixture
from visual.runtime.test.run_variable_rate_shape_first import packet_digest


class ReceiverTests(unittest.TestCase):
    def test_variable_N_zero_identity_and_no_reordering(self):
        for n in (0,1,7,40,400):
            packet,*_=packet_fixture(n)
            original=packet_digest(packet)
            receiver=receive_vision_packet(packet)
            self.assertEqual(original,packet_digest(receiver.packet))
            self.assertEqual(receiver.packet['source_frame'],0)
            self.assertEqual(receiver.packet['episode'],'ep000')
            np.testing.assert_array_equal(receiver.positions,packet['points'])
            np.testing.assert_array_equal(receiver.point_node_index,np.arange(len(packet['points'])))
            self.assertFalse(receiver.robot_mask.any())
            self.assertEqual(receiver.packet['edge_status'],'not_implemented')

    def test_robot_offset_metadata_supplied_edges_input_unchanged(self):
        packet,*_=packet_fixture(40)
        packet['edge_status']='supplied'
        packet['intra_object_edge_index']=np.array([[0,1],[1,2]],np.int64)
        original=packet_digest(packet)
        robots=np.array([[1,2,3],[4,5,6],[7,8,9]],np.float32)
        receiver=receive_vision_packet(packet,robot_positions=robots)
        np.testing.assert_array_equal(receiver.positions[:3],robots)
        np.testing.assert_array_equal(receiver.positions[3:],packet['points'])
        np.testing.assert_array_equal(receiver.edge_index,packet['intra_object_edge_index']+3)
        np.testing.assert_array_equal(receiver.point_node_index,np.arange(len(packet['points']))+3)
        np.testing.assert_array_equal(receiver.node_source_index[:3],[-1,-1,-1])
        self.assertFalse(receiver.node_candidate_evidence[:3].any())
        np.testing.assert_array_equal(receiver.node_kind[:3],[1,2,3])
        self.assertTrue(receiver.robot_mask[:3].all());self.assertFalse(receiver.robot_mask[3:].any())
        self.assertEqual(original,packet_digest(packet))
        receiver.packet['points'][0]=123
        self.assertEqual(original,packet_digest(packet))

    def test_bad_identity_dtype_row_alignment(self):
        packet,*_=packet_fixture(40)
        mutations=[lambda p:p.update(source_frame=-1),lambda p:p.update(camera=''),
            lambda p:p.update(points=p['points'].astype(np.float64)),
            lambda p:p.update(object_id=p['object_id'].astype(np.int32)),
            lambda p:p['provenance'].update(source_index=p['provenance']['source_index'][:-1]),
            lambda p:p.update(geometric_features=p['geometric_features'][:-1]),
            lambda p:p['provenance'].update(both_evidence=~p['provenance']['both_evidence']),
            lambda p:p['provenance'].update(surface_selected=~p['provenance']['surface_selected'])]
        for mutate in mutations:
            bad=copy.deepcopy(packet);mutate(bad)
            with self.assertRaises(ValueError):receive_vision_packet(bad)

    def test_duplicate_source_and_object_provenance_mismatch(self):
        packet,*_=packet_fixture(40)
        bad=copy.deepcopy(packet);bad['provenance']['source_index'][1]=bad['provenance']['source_index'][0]
        with self.assertRaises(ValueError):receive_vision_packet(bad)
        bad=copy.deepcopy(packet);bad['provenance']['object_id'][0]=1
        with self.assertRaises(ValueError):receive_vision_packet(bad)

    def test_edges_out_of_range_cross_object_and_unimplemented(self):
        packet,*_=packet_fixture(40);n=len(packet['points']);split=int((packet['object_id']==0).sum())
        for edge in (np.array([[0],[n]],np.int64),np.array([[-1],[0]],np.int64),
                     np.array([[0],[split]],np.int64),np.array([[0],[1]],np.int32)):
            bad=copy.deepcopy(packet);bad['edge_status']='supplied';bad['intra_object_edge_index']=edge
            with self.assertRaises(ValueError):receive_vision_packet(bad)
        packet['intra_object_edge_index']=np.array([[0],[1]],np.int64)
        with self.assertRaises(ValueError):receive_vision_packet(packet)

    def test_features_missing_compact_values_positions_and_dtype(self):
        packet,*_=packet_fixture(40)
        for case in ('unavailable_zero','positions','values','dtype','availability'):
            bad=copy.deepcopy(packet);field=bad['compact_features']['normal']
            if case=='unavailable_zero':bad['geometric_features'][~bad['provenance']['feature_available']]=0
            elif case=='positions':field['positions']=field['positions'][::-1]
            elif case=='values':field['values']=np.zeros_like(field['values'])
            elif case=='dtype':field['values']=field['values'].astype(np.float64)
            else:bad['provenance']['feature_available']=~bad['provenance']['feature_available']
            with self.assertRaises(ValueError):receive_vision_packet(bad)

    def test_robot_validation_and_no_encoder_feature_injection(self):
        packet,*_=packet_fixture(40)
        for robot in (np.zeros((2,3),np.float32),np.zeros((3,3),np.float64),np.full((3,3),np.nan,np.float32)):
            with self.assertRaises(ValueError):receive_vision_packet(packet,robot_positions=robot)
        code=inspect.getsource(receive_vision_packet)
        self.assertNotIn('GraphBuilder',code);self.assertNotIn("['x']",code)


if __name__=='__main__':unittest.main()
