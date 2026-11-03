"""Sparse raw-depth reads preserve units, strict predicates and current data."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from graph.persistent_geometry.contracts import CameraObservation,transform
from graph.persistent_geometry.graph import free_space_violation,surface_edges
from visual.runtime.coordinate_frames import world_from_frame
from .test_contracts import camera


def no_dense_depth(_):
    raise AssertionError('full-image depth getter forbidden')


class SparseFreeSpaceContracts(unittest.TestCase):
    def test_all_signed_sensor_values_match_dense_getter_predicate(self):
        raw=np.arange(-32768,32768,dtype=np.int32).astype(np.int16).reshape(256,256)
        c=replace(camera(),rgb=np.zeros((256,256,3),np.uint8),depth_raw_mm=raw,
            intrinsic=np.array([[128.,0,128],[0,128.,128],[0,0,1.]])).validate()
        dense=c.depth.ravel().copy()  # Independent existing whole-image unit/invalidity oracle.
        v,u=np.indices(raw.shape);u=u.ravel()+.5;v=v.ravel()+.5
        before=raw.copy()
        for z in (.001,.994,1.,32.76):
            points=np.c_[(u-128)*z/128,(v-128)*z/128,np.full(len(u),z)]
            expected=np.isfinite(dense)&(z < dense-.006)
            with patch.object(CameraObservation,'depth',property(no_dense_depth)):
                actual=free_space_violation(points,[c])
            np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(raw,before)

    def test_strict_threshold_outside_behind_and_invalid_depth(self):
        c=camera(z=1000)
        z=np.array([np.nextafter(.994,-np.inf),.994,np.nextafter(.994,np.inf),0.,-1.,1.1])
        points=np.c_[np.zeros(len(z)),np.zeros(len(z)),z]
        points=np.vstack((points,[100,0,.5]))
        with patch.object(CameraObservation,'depth',property(no_dense_depth)):
            np.testing.assert_array_equal(free_space_violation(points,[c]),[True,False,False,False,False,False,False])
            for invalid in (-32768,-1,0):
                bad=replace(c,depth_raw_mm=np.full_like(c.depth_raw_mm,invalid))
                self.assertFalse(free_space_violation(np.array([[0.,0.,.5]]),[bad])[0])
            self.assertEqual(free_space_violation(np.empty((0,3)),[c]).shape,(0,))
            np.testing.assert_array_equal(free_space_violation(points,[]),np.zeros(len(points),bool))

    def test_current_depth_mutation_replacement_and_pose_are_not_cached(self):
        c=camera(z=1000);point=np.array([[0.,0.,.5]])
        with patch.object(CameraObservation,'depth',property(no_dense_depth)):
            self.assertTrue(free_space_violation(point,[c])[0])
            c.depth_raw_mm[:]=400
            self.assertFalse(free_space_violation(point,[c])[0])
            c=replace(c,depth_raw_mm=np.full_like(c.depth_raw_mm,1000))
            self.assertTrue(free_space_violation(point,[c])[0])
            moved=replace(c,extrinsic_cv=np.c_[np.eye(3),[100.,0,0]])
            self.assertFalse(free_space_violation(point,[moved])[0])

    def test_multiview_nonidentity_ee_pose_and_world(self):
        c=replace(camera(),tcp_pose=np.array([.25,-.1,.3,np.cos(.4),0,0,np.sin(.4)]))
        other=replace(c,camera='other',depth_raw_mm=np.zeros_like(c.depth_raw_mm))
        world=np.array([[0.,0.,.5],[0.,0.,1.1],[100.,0.,.5]])
        for frame in ('world','ee'):
            points=transform(world,np.linalg.inv(world_from_frame(frame,c.tcp_pose)))
            with patch.object(CameraObservation,'depth',property(no_dense_depth)):
                result=free_space_violation(points,[other,c],coordinate_frame=frame)
            np.testing.assert_array_equal(result,[True,False,False])

    def test_surface_accept_reject_audit_and_order_without_dense_depth(self):
        points=np.array([[0.,0.,.5],[.005,0.,.5]])
        normals=np.tile([0.,0.,-1.],(2,1));valid=np.ones(2,bool);objects=np.zeros(2,int)
        reference={0:points,1:np.empty((0,3))}
        with patch.object(CameraObservation,'depth',property(no_dense_depth)):
            edges,audit=surface_edges(points,normals,valid,objects,reference,[camera(z=1000)])
            self.assertEqual(edges.shape,(2,0));self.assertEqual(audit['rejected_free_space'],1)
            edges,audit=surface_edges(points,normals,valid,objects,reference,[camera(z=500)])
            np.testing.assert_array_equal(edges,np.array([[0,1],[1,0]],np.int64))
            self.assertEqual(audit,dict(candidates=1,rejected_normal=0,rejected_witness=0,rejected_free_space=0))


if __name__=='__main__':unittest.main()
