"""Source-aligned measured features, fixed numerical tolerances, no topology."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from graph.vision_input import receive_vision_packet, validate_packet
from visual.runtime.sampling import extract_sampling_evidence
from visual.runtime.test.test_geometry import make_frame
from visual.runtime.vision_packet import (
    complete_vision_packet, camera_to_ee, float32_tolerance, pose_matrix,
)


def feature_fixture(depth=None, empty=False):
    depth = np.ones((9, 15), np.float32) if depth is None else np.asarray(depth, np.float32)
    h, w = depth.shape
    peg = np.zeros((h, w), bool); box = peg.copy()
    if not empty:
        peg[:, :max(1, w//2-1)] = True
        if w > 3: box[:, w//2+1:] = True
    frame = make_frame(depth, peg, box)
    evidence = extract_sampling_evidence(frame)
    g = evidence.frame.geometry
    ids, src = [], []
    for oid in (0, 1):
        s = np.flatnonzero(g.observed_mask(oid))[::-1]
        ids.extend([oid]*len(s)); src.extend(s)
    ids, src = np.asarray(ids, np.int64), np.asarray(src, np.int64)
    n = len(src)
    # Hand-authored selection roles: no sampler or candidate detector is called.
    evidence_bits = np.column_stack((np.isin(np.arange(n)%4,[1,3]), np.arange(n)%4 >= 2))
    chosen = evidence_bits.copy(); chosen[np.arange(n)%3 == 0] = False
    stage = (~chosen.any(1)).astype(np.uint8)
    available = np.arange(n)%2 == 0
    values = np.full((n, 1), np.nan); values[available] = 1
    packet = dict(episode='fixture', camera=frame.camera, source_frame=frame.source_frame,
                  coordinate_frame='CV_camera_metres', points=g.xyz.reshape(-1,3)[src].copy(),
                  object_id=ids, geometric_features=values,
                  feature_fields=[dict(name='old', shape=[], dtype='|b1')],
                  compact_features=dict(old=dict(positions=np.flatnonzero(available),
                                                 values=np.ones(available.sum(), bool))),
                  provenance=dict(source_index=src, object_id=ids.copy(), candidate_evidence=evidence_bits,
                                  both_evidence=evidence_bits.all(1), selected_by_structural_source=chosen,
                                  selection_stage=stage, surface_selected=stage == 1,
                                  feature_available=available, feature_field_available=available[:,None]),
                  edge_status='not_implemented', intra_object_edge_index=np.empty((2,0),np.int64))
    return packet, evidence, frame


def assert_tree_equal(test, a, b):
    if isinstance(a, np.ndarray):
        test.assertEqual(a.dtype, b.dtype)
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: assert_tree_equal(test, a[k], b[k])
    elif isinstance(a, list):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): assert_tree_equal(test,x,y)
    else: test.assertEqual(a,b)


class VisionFeatureTests(unittest.TestCase):
    def test_source_gather_all_roles_and_no_estimator_reentry(self):
        source, evidence, frame = feature_fixture()
        original = copy.deepcopy(source)
        with patch('visual.runtime.sampling.local_plane_optimized', side_effect=AssertionError('re-estimation')):
            packet = complete_vision_packet(source, evidence, frame)
        validate_packet(packet)
        assert_tree_equal(self, source, original)
        assert_tree_equal(self, source['provenance'], packet['provenance'])
        self.assertTrue(packet['feature_field_available'].all())
        self.assertFalse(packet['provenance']['feature_available'].all())
        for oid in (0,1):
            rows = np.flatnonzero(source['object_id'] == oid)
            src = source['provenance']['source_index'][rows]
            f = evidence.frame.objects[oid]
            np.testing.assert_array_equal(packet['geometric_features'][rows,:3], f.normals.reshape(-1,3)[src])
            np.testing.assert_array_equal(packet['geometric_features'][rows,3], f.normal_variation_rad.ravel()[src])
            np.testing.assert_array_equal(packet['geometric_features'][rows,4], f.depth_jump_m.ravel()[src])
        p = packet['provenance']
        self.assertTrue((p['candidate_evidence'][:,1] & ~p['candidate_evidence'][:,0]).any())
        self.assertTrue((p['surface_selected'] & ~p['candidate_evidence'].any(1)).any())
        self.assertTrue(p['both_evidence'].any())

    def test_invalid_partial_measurements_and_empty_shapes(self):
        source, evidence, frame = feature_fixture(np.ones((1,1)))
        packet = complete_vision_packet(source,evidence,frame)
        validate_packet(packet)
        self.assertTrue(packet['feature_available'].all())
        self.assertFalse(packet['feature_field_valid'].any())
        self.assertTrue(np.isnan(packet['geometric_features']).all())
        source,evidence,frame = feature_fixture(np.ones((5,5)))
        # A tiny supported boundary may have depth measurement but no normal.
        packet = complete_vision_packet(source,evidence,frame)
        self.assertTrue((packet['feature_field_valid'][:,2] & ~packet['feature_field_valid'][:,0]).any())
        for depth, empty in ((np.ones((3,3)),True),(np.zeros((3,3)),False)):
            source,evidence,frame = feature_fixture(depth,empty)
            packet = complete_vision_packet(source,evidence,frame); validate_packet(packet)
            self.assertEqual(packet['points'].shape,(0,3))
            self.assertEqual(packet['geometric_features'].shape,(0,5))
            self.assertEqual(packet['feature_field_valid'].shape,(0,3))

    def test_normal_with_no_valid_neighbor_keeps_variation_invalid(self):
        source,evidence,frame=feature_fixture()
        f=evidence.frame.objects[0]
        v,u=np.argwhere(f.normal_valid)[0]
        variation=f.normal_variation_rad.copy();variation[v,u]=np.nan
        valid=f.normal_variation_valid.copy();valid[v,u]=False
        count=f.normal_neighbor_count.copy();count[v,u]=0
        objects=dict(evidence.frame.objects)
        objects[0]=replace(f,normal_variation_rad=variation,normal_variation_valid=valid,normal_neighbor_count=count)
        evidence=replace(evidence,frame=replace(evidence.frame,objects=objects))
        packet=complete_vision_packet(source,evidence,frame);validate_packet(packet)
        self.assertTrue((packet['feature_field_valid'][:,0] & ~packet['feature_field_valid'][:,1]).any())

    def test_known_rotation_translation_wxyz_and_float64_reference(self):
        source,evidence,frame=feature_fixture(np.full((9,15),35.2))
        c2w=np.array([[0,-1,0,2],[1,0,0,-3],[0,0,1,4],[0,0,0,1]],np.float64)
        q=Rotation.from_euler('xyz',[21,-13,37],degrees=True).as_quat()
        pose=np.r_[.2,-.4,.7,q[3],q[:3]]
        frame=replace(frame,cam2world=c2w,tcp_pose=pose)
        packet=complete_vision_packet(source,evidence,frame);validate_packet(packet)
        world=source['points'].astype(float)@c2w[:3,:3].T+c2w[:3,3]
        rot=Rotation.from_quat(q).as_matrix()
        reference=(world-pose[:3])@rot
        t=camera_to_ee(frame)
        computed=source['points'].astype(float)@t[:3,:3].T+t[:3,3]
        np.testing.assert_allclose(computed,reference,rtol=0,atol=64*np.finfo(float).eps*max(1,np.abs(reference).max()))
        self.assertTrue((np.abs(packet['points']-reference)<=float32_tolerance(reference)).all())
        normal=evidence.frame.objects[0].normals.reshape(-1,3)[source['provenance']['source_index'][source['object_id']==0]]
        expected=normal@c2w[:3,:3].T@rot
        np.testing.assert_allclose(packet['geometric_features'][source['object_id']==0,:3],expected,rtol=0,atol=1e-7,equal_nan=True)
        self.assertGreater(float(float32_tolerance([35.2])[0]),1e-6)
        self.assertLess(np.max(np.abs(t@np.linalg.inv(t)-np.eye(4))),1e-12)
        self.assertTrue(np.allclose(pose_matrix(pose)[:3,:3],rot))

    def test_invalid_identity_source_calibration_and_unsupported_modes(self):
        source,evidence,frame=feature_fixture()
        for mode in ('onehot','probability','unknown'):
            with self.assertRaisesRegex(NotImplementedError,'unsupported'):
                complete_vision_packet(source,evidence,frame,mode=mode)
        for mutation in (
            lambda p:p.update(source_frame=8),
            lambda p:p.update(camera='wrong'),
            lambda p:p['points'].__setitem__((0,0),9),
            lambda p:p['provenance']['source_index'].__setitem__(0,frame.depth.size),
            lambda p:p['provenance']['source_index'].__setitem__(0,-1),
            lambda p:p['provenance']['source_index'].__setitem__(0,p['provenance']['source_index'][1]),
        ):
            bad=copy.deepcopy(source);mutation(bad)
            with self.assertRaises(ValueError):complete_vision_packet(bad,evidence,frame)
        for bad in (replace(frame,tcp_pose=None),replace(frame,tcp_pose=np.zeros(7)),
                    replace(frame,cam2world=np.zeros((4,4))),replace(frame,intrinsic=np.eye(3))):
            with self.assertRaises(ValueError):complete_vision_packet(source,evidence,bad)

    def test_receiver_rejects_corrupt_measurements_and_roundoff(self):
        source,evidence,frame=feature_fixture(np.full((9,15),35.2))
        packet=complete_vision_packet(source,evidence,frame)
        for mutate in (
            lambda p:p.update(schema_version=3),
            lambda p:p.update(feature_mode='onehot'),
            lambda p:p['feature_field_valid'].__setitem__((0,0),False),
            lambda p:p['feature_field_available'].__setitem__((0,0),False),
            lambda p:p['points'].__setitem__((0,2),p['points'][0,2]+8*np.spacing(p['points'][0,2])),
            lambda p:p['T_EE_from_camera'].__setitem__((0,3),1),
            lambda p:p['provenance']['both_evidence'].__setitem__(0,~p['provenance']['both_evidence'][0]),
        ):
            bad=copy.deepcopy(packet);mutate(bad)
            with self.assertRaises(ValueError):validate_packet(bad)
        bad=copy.deepcopy(packet);bad['points'][0,2]+=2*np.spacing(bad['points'][0,2])
        validate_packet(bad)
        received=receive_vision_packet(packet)
        assert_tree_equal(self,packet,received.packet)
        received.packet['feature_field_valid'][:]=False
        self.assertTrue(packet['feature_field_valid'].any())


if __name__=='__main__': unittest.main()
