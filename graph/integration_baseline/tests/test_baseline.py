"""SYNTHETIC_CONTRACT_ONLY fixtures; real-packet/replay smokes are separate."""
import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import torch

from graph.integration_baseline.graph import (build_graph, pack_graphs, edge_attributes,
    global_peg_box_pairs, nearest_robot_pairs, bidirectional)
from graph.integration_baseline.model import IntegrationPolicy, GraphEncoder, structured_readout, bc_loss, EpisodeStream
from graph.integration_baseline.robot import state_schema, flatten_state, RobotNormalizer, geometry_in_ee
from graph.integration_baseline.vision import packet_from_frame, causal_masks
from graph.integration_baseline.io import packet_digest
from visual.runtime.rgbd import TrackedRGBDFrame


def fixture_frame(empty=()):
    h, w = 16, 16
    y, x = np.mgrid[:h, :w]
    masks = {0: (x >= 2)&(x <= 6)&(y >= 2)&(y <= 13),
             1: (x >= 9)&(x <= 13)&(y >= 2)&(y <= 13)}
    for oid in empty:
        masks[oid][:] = False
    return TrackedRGBDFrame(0, 'base_camera', np.zeros((h,w,3), np.uint8),
        (np.full((h,w), .5)+x*.001).astype(np.float32),
        np.array([[20,0,8],[0,20,8],[0,0,1]], np.float32), np.eye(4, dtype=np.float32),
        np.array([0,0,0,1,0,0,0], np.float32), masks, 0, 0.)


def fixture_packet(empty=()):
    return packet_from_frame(fixture_frame(empty), 'synthetic_contract')[0]


def robot_fixture():
    fields = dict(qpos=np.zeros(9, np.float32), qvel=np.zeros(9, np.float32),
                  gripper_position=np.zeros(2, np.float32), gripper_velocity=np.zeros(2, np.float32),
                  gripper_opening=np.zeros(1, np.float32), gripper_opening_velocity=np.zeros(1, np.float32))
    for name in ('tcp', 'left_finger', 'right_finger', 'robot_base'):
        fields[name+'_pose'] = np.array([0,0,0,1,0,0,0], np.float32)
        fields[name+'_linear_velocity'] = np.zeros(3, np.float32)
        fields[name+'_angular_velocity'] = np.zeros(3, np.float32)
    # Make synthetic pad geometry consistent with the existing link-local pad
    # definition; this is a fixture, not inferred real robot geometry.
    fields['left_finger_pose'][:3] = [0, .02-.00758, .01-.04525]
    fields['right_finger_pose'][:3] = [0, -.02+.00758, .01-.04525]
    return fields, state_schema(fields)


ROBOT = np.array([[0,0,0], [0,.02,.01], [0,-.02,.01]], np.float32)


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.packet = fixture_packet()

    def test_robot_three_rows_source_nan_dtype_and_raw_edges_preserved(self):
        p = self.packet
        before = packet_digest(p)
        graph = build_graph(p, ROBOT)
        self.assertEqual(before, packet_digest(p))
        self.assertEqual(before, packet_digest(graph.packet))
        np.testing.assert_array_equal(graph.node_input[:3,:3], ROBOT)
        np.testing.assert_array_equal(graph.node_input[3:,:3], p['points'])
        np.testing.assert_array_equal(graph.edge_index[:,graph.edge_type==0], p['intra_object_edge_index']+3)
        self.assertEqual(graph.node_type[:3].tolist(), [0,1,2])
        np.testing.assert_array_equal(graph.node_type[3:], p['object_id']+3)
        np.testing.assert_array_equal(graph.node_input[3:,8:11], p['feature_field_available'])
        np.testing.assert_array_equal(graph.node_input[3:,11:14], p['feature_field_valid'])
        self.assertTrue(torch.isfinite(graph.node_input).all())

    def test_exact_global_pair_oracle_random_ties_empty_zero_and_oversize(self):
        rng = np.random.default_rng(12)
        for n in (0,1,2,5,17,30):
            points = rng.integers(-2, 3, (n,3)).astype(np.float32)
            ids = rng.integers(0,2,n)
            src = rng.permutation(n)
            for m in (0,1,4,16,n*n+1):
                pairs = [(i,j) for i in range(n) if ids[i]==0 for j in range(n) if ids[j]==1]
                order = sorted(pairs, key=lambda ij: (sum((float(points[ij[0],a])-float(points[ij[1],a]))**2 for a in range(3)),
                    int(src[ij[0]]), int(src[ij[1]]), *ij))[:m]
                oracle = np.array(order, np.int64).reshape(-1,2)
                for block in (1,3,64):
                    np.testing.assert_array_equal(global_peg_box_pairs(points,ids,src,m,block), oracle)

    def test_knn_oracle_per_robot_and_object(self):
        p = self.packet
        points = np.vstack((ROBOT,p['points']))
        oid, src = p['object_id'], p['provenance']['source_index']
        for k in (1,4,999):
            oracle=[]
            for robot in range(3):
                for obj in (0,1):
                    rows=np.flatnonzero(oid==obj)
                    order=sorted(rows, key=lambda r:(sum((float(points[r+3,a])-float(points[robot,a]))**2 for a in range(3)),int(src[r]),int(r)))[:k]
                    oracle.extend((robot,int(r)+3) for r in order)
            np.testing.assert_array_equal(nearest_robot_pairs(points,oid,src,k),np.array(oracle).reshape(-1,2))

    def test_tie_break_is_source_stable_under_row_permutation(self):
        p=np.zeros((8,3),np.float32);oid=np.array([0]*4+[1]*4);src=np.array([4,2,3,1,10,8,9,7])
        a=global_peg_box_pairs(p,oid,src,6)
        perm=np.array([7,2,0,4,3,5,1,6]);b=global_peg_box_pairs(p[perm],oid[perm],src[perm],6)
        np.testing.assert_array_equal(src[a],src[perm][b])

    def test_reverse_destination_minus_source_and_zero_distance(self):
        pos=np.array([[0,0,0],[1,2,3],[0,0,0]],np.float32)
        edge=bidirectional(np.array([[0,1],[0,2]]));attr=edge_attributes(pos,edge)
        np.testing.assert_array_equal(attr[:2,0],attr[2:,0])
        np.testing.assert_array_equal(attr[:2,1:],-attr[2:,1:])
        reference = pos[1].astype(np.float64)
        np.testing.assert_array_equal(attr[0,1:],(reference/np.linalg.norm(reference)).astype(np.float32))
        self.assertTrue(np.isfinite(attr).all());np.testing.assert_array_equal(attr[1],np.zeros(4))

    def test_consumer_never_retriangulates_or_strict_validates(self):
        with ExitStack() as stack:
            for name in ('candidate_edges','Delaunay','collinear_order','prepare_validation','validate_cover','supercover'):
                stack.enter_context(patch('visual.runtime.extended_delaunay_edges.'+name,side_effect=AssertionError(name)))
            build_graph(self.packet,ROBOT)

    def test_empty_objects_zero_degree_and_readout(self):
        for empty in ((0,),(1,),(0,1)):
            p=fixture_packet(empty);g=pack_graphs([build_graph(p,ROBOT)])
            encoder=GraphEncoder();z,present=encoder(g)
            self.assertEqual(z.shape,(1,320));self.assertTrue(torch.isfinite(z).all())
            for obj in empty:
                self.assertEqual(present[0,obj],0)
                torch.testing.assert_close(z[:,(3+obj)*64:(4+obj)*64],torch.zeros(1,64))
            if len(empty)==2:self.assertEqual(g.edge_index.numel(),0)

    def test_gt_action_future_extra_fields_do_not_change_graph(self):
        a=build_graph(self.packet,ROBOT)
        p=copy.deepcopy(self.packet)
        p.update(peg_pose=np.full(7,999.),hole_pose=np.full(7,-999.),goal=1000.,
                 future_observation=np.zeros(300),expert_action=np.ones(7))
        b=build_graph(p,ROBOT)
        for name in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)

    def test_packed_offsets_mapping_and_separate_equivalence(self):
        graphs=[build_graph(self.packet,ROBOT),build_graph(fixture_packet((1,)),ROBOT)]
        packed=pack_graphs(graphs,[(1,0),(0,0)])
        n=len(graphs[0].node_type);e=graphs[0].edge_index.shape[1]
        torch.testing.assert_close(packed.edge_index[:,e:],graphs[1].edge_index+n)
        self.assertTrue(torch.equal(packed.node_graph[packed.edge_index[0]],packed.node_graph[packed.edge_index[1]]))
        encoder=GraphEncoder();z,_=encoder(packed)
        for i,g in enumerate(graphs):
            single,_=encoder(pack_graphs([g]));torch.testing.assert_close(single,z[i:i+1],rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):pack_graphs(graphs,[(0,0),(0,0)])

    def test_readout_exact_slots_no_projection(self):
        graph=pack_graphs([build_graph(self.packet,ROBOT)])
        h=torch.arange(len(graph.node_type)*2,dtype=torch.float32).reshape(-1,2)
        z,present=structured_readout(h,graph)
        expected=torch.cat((h[:3].flatten(),h[graph.node_type==3].mean(0),h[graph.node_type==4].mean(0)))
        torch.testing.assert_close(z[0],expected)
        self.assertEqual(present.tolist(),[[1,1]])

    def test_bad_config_and_policy_rejected(self):
        for kwargs in ({'k':0},{'M':-1},{'M':1.5},{'block_rows':0}):
            with self.assertRaises(ValueError):build_graph(self.packet,ROBOT,**kwargs)
        p=copy.deepcopy(self.packet);p['edge_policy']['edge_validation']='strict'
        with self.assertRaises(ValueError):build_graph(p,ROBOT)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(3)
        cls.graph=build_graph(fixture_packet(),ROBOT)
        fields,cls.schema=robot_fixture();cls.state=flatten_state(fields,cls.schema)
        cls.normalizer=RobotNormalizer.fit(np.stack([cls.state]*3),[3],{'train':[3],'validation':[4],'test':[5]},cls.schema)

    def test_sequence_step_equivalence_and_nonzero_initial_hidden(self):
        model=IntegrationPolicy(self.schema)
        graphs=pack_graphs([self.graph]*6,[(b,t) for b in range(2) for t in range(3)])
        states=torch.tensor(np.stack([self.state]*6))
        hidden=torch.randn(1,2,144)
        full,hn=model(graphs,states,self.normalizer,hidden)
        outputs=[];hs=hidden.clone()
        for t in range(3):
            action,hs=model.step(pack_graphs([self.graph]*2,[(0,0),(1,0)]),states[:2],self.normalizer,hs)
            outputs.append(action)
        torch.testing.assert_close(full,torch.stack(outputs,1),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(hn,hs,rtol=1e-5,atol=1e-6)

    def test_masks_actually_reach_encoder(self):
        model=GraphEncoder();packed=pack_graphs([self.graph]);z,_=model(packed)
        changed=copy.deepcopy(packed);changed.node_input[:,8:14]=0
        zz,_=model(changed)
        self.assertGreater(float((z-zz).abs().max().detach()),1e-6)

    def test_ragged_padding_preserves_last_valid_hidden_and_resets(self):
        model=IntegrationPolicy(self.schema)
        graphs=pack_graphs([self.graph]*5,[(0,0),(0,1),(0,2),(1,0),(1,1)])
        states=torch.tensor(np.stack([self.state]*5))
        out,h=model(graphs,states,self.normalizer,lengths=[3,2])
        self.assertTrue((out[1,2]==0).all())
        _,short=model(pack_graphs([self.graph]*2,[(0,0),(0,1)]),states[:2],self.normalizer)
        torch.testing.assert_close(h[:,1:2],short,rtol=1e-5,atol=1e-6)
        reset=torch.zeros(2,3,dtype=torch.bool);reset[0,2]=True
        o,_=model(graphs,states,self.normalizer,lengths=[3,2],reset_before=reset)
        single,_=model.step(pack_graphs([self.graph],[(0,0)]),states[:1],self.normalizer)
        torch.testing.assert_close(o[0,2],single[0],rtol=1e-5,atol=1e-6)

    def test_episode_stream_reset_and_gap_rejection(self):
        model=IntegrationPolicy(self.schema);stream=EpisodeStream(model,self.normalizer)
        graph=pack_graphs([self.graph],[(0,0)]);state=torch.tensor(self.state)[None]
        first=stream.step(graph,state,'a',0);stream.step(graph,state,'a',1)
        for kwargs in ({'previous_terminated':True},{'previous_truncated':True},{'reset':True}):
            action=stream.step(graph,state,'a',0,**kwargs);torch.testing.assert_close(action,first)
        new=stream.step(graph,state,'b',0);torch.testing.assert_close(new,first)
        with self.assertRaises(ValueError):stream.step(graph,state,'b',3)

    def test_normalizer_train_only_freeze_and_schema_dimension(self):
        self.assertEqual(self.schema['dimension'],sum(f['dimension'] for f in self.schema['fields']))
        before=self.normalizer.stats.mean.clone()
        self.normalizer.apply(torch.randn(10,self.schema['dimension']))
        torch.testing.assert_close(before,self.normalizer.stats.mean)
        with self.assertRaises(ValueError):RobotNormalizer.fit(np.stack([self.state]),[4],{'train':[3]},self.schema)
        with self.assertRaises(ValueError):self.normalizer.apply(torch.zeros(1,9))

    def test_geometry_current_tcp_transform_and_collision_pad_convention(self):
        pose=np.array([1,2,3,1,0,0,0],np.float32)
        xyz=ROBOT+pose[:3]
        np.testing.assert_allclose(geometry_in_ee(xyz,pose),ROBOT,rtol=0,atol=2e-7)
        with self.assertRaises(ValueError):geometry_in_ee(xyz,pose+np.array([.01,0,0,0,0,0,0]))

    def test_bc_loss_is_existing_seven_dimensional_regression(self):
        raw=torch.arange(14,dtype=torch.float32).reshape(2,7);target=torch.ones_like(raw)
        torch.testing.assert_close(bc_loss(raw,target,'regression'),torch.nn.functional.mse_loss(raw,target))


if __name__=='__main__':
    unittest.main()
