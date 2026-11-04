"""Default dispatch, compact storage, CUDA fidelity and final Graph contracts."""
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
import torch

from visual.runtime.geometry import CompactOrganizedGeometry,GeometryParameters
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.graph_perception import process_cuda,to_cpu_evidence,make_packet


def frame(shape=(24,28),mode='normal'):
    h,w=shape
    depth=np.full(shape,.3,np.float32);depth[:,w//2:]=.5
    masks=np.zeros((2,h,w),bool)
    masks[0,1:-1,1:w//2+2]=True;masks[1,3:-2,w//2-1:-1]=True
    if mode=='empty':masks[:]=False
    if mode=='holes':depth[::4,::3]=np.nan;depth[1,1]=0
    if mode=='line':masks[:]=False;masks[0,h//2,:]=True
    k=np.array([[120.,0,w/2],[0,130.,h/2],[0,0,1]],np.float32)
    return TrackedRGBDFrame(0,'base_camera',np.zeros((h,w,3),np.uint8),depth,k,
        np.eye(4),np.array([0.,0.,0.,1.,0.,0.,0.]),dict(enumerate(masks)),0,0.)


class CPUContracts(unittest.TestCase):
    def test_compact_coordinates_and_depth_lookup(self):
        valid=np.ones((2,3),bool);masks={0:valid.copy(),1:np.zeros_like(valid)}
        pixels=np.array([0,2,5]);points=np.arange(9,dtype=np.float32).reshape(3,3)
        g=CompactOrganizedGeometry(None,valid,masks,np.zeros_like(valid),pixels,points,np.ones((2,3),np.float32))
        np.testing.assert_array_equal(g.points_at(np.array([5,0])),points[[2,0]])
        self.assertIsNone(g.xyz)
        with self.assertRaises(ValueError):g.points_at(np.array([1]))

    def test_current_geometry_default_and_explicit_cpu(self):
        from graph.plan_a_bc.geometry import CurrentGeometry
        with patch('graph.plan_a_bc.geometry.RobotFK'):
            default=CurrentGeometry('unused',{})
            cpu=CurrentGeometry('unused',{},perception_backend='cpu')
        expected='triton_fused' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(default.perception_backend,expected)
        self.assertEqual(default.robot_filter_backend,'cuda' if torch.cuda.is_available() else 'cpu')
        self.assertEqual(default.system.packet_builder.keywords['backend'],expected)
        self.assertEqual(cpu.system.packet_builder.keywords['backend'],'cpu')
        self.assertEqual(cpu.robot_filter_backend,'cpu')

    def test_missing_cuda_has_no_silent_cpu_fallback(self):
        from visual.runtime.graph_perception import extract
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'requires CUDA'):extract(frame())


@unittest.skipUnless(torch.cuda.is_available(),'CUDA host required')
class CUDAContracts(unittest.TestCase):
    def test_compact_fused_features_match_existing_dense_fused(self):
        from visual.runtime.test.pca_torch import TorchPCA5
        from visual.runtime.pca import build_geometry
        for mode in ('normal','empty','holes','line'):
            f=frame(mode=mode)
            depth=torch.as_tensor(f.depth,device='cuda')
            masks=torch.as_tensor(np.stack(list(f.masks.values())),device='cuda')
            k=torch.as_tensor(f.intrinsic,device='cuda')
            old=TorchPCA5(f.intrinsic,f.depth.shape,backend='triton_fused')(depth,dict(enumerate(masks)))
            prepared,values=process_cuda(depth,masks,k)
            actual=to_cpu_evidence(f,prepared,values)
            self.assertEqual(prepared['points'].ndim,2)
            self.assertEqual(prepared['points'].shape[1],3)
            self.assertIsNone(actual.frame.geometry.xyz)
            legacy=build_geometry(f)
            source=actual.frame.geometry.source_pixels
            np.testing.assert_array_equal(actual.frame.geometry.points_at(source),legacy.points_at(source))
            for oid in (0,1):
                for name in actual.frame.objects[oid].__dataclass_fields__:
                    a=getattr(actual.frame.objects[oid],name)
                    b=old['objects'][oid][name].cpu().numpy()
                    if a.dtype.kind=='f':np.testing.assert_allclose(a,b,atol=2e-5,rtol=0,equal_nan=True,err_msg=mode+':'+name)
                    else:np.testing.assert_array_equal(a,b,err_msg=mode+':'+name)

    def test_selected_packet_matches_cpu_and_never_builds_dense_xyz(self):
        from graph.persistent_geometry.tests.test_contracts import camera
        c=camera(name='base_camera',z=300)
        masks=np.zeros((2,*c.depth.shape),bool);masks[0,3:20,3:12]=True;masks[1,4:18,14:21]=True
        expected,_=make_packet(c,masks,backend='cpu')
        with patch('visual.runtime.pca.build_geometry',side_effect=AssertionError('dense XYZ forbidden')):
            actual,timing=make_packet(c,masks)
        self.assertEqual(timing['geometry_backend'],'triton_fused')
        for name in ('points','object_id','feature_field_valid'):
            np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)
        np.testing.assert_array_equal(actual['provenance']['source_index'],expected['provenance']['source_index'])
        np.testing.assert_allclose(actual['geometric_features'],expected['geometric_features'],atol=2e-5,rtol=0,equal_nan=True)

    def test_current_geometry_final_graph_matches_cpu_from_prepared_masks(self):
        from graph.plan_a_bc.geometry import CurrentGeometry
        from graph.plan_a_geometry.tests import Contracts
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        fixture=Contracts();fixture.setUp()
        prepared=dict(conditions={'C3':fixture.masks},gate=dict(accepted=True))
        link=LinkSurface('camera_link',cube([0,0,0],[.01,.01,.01]),np.eye(4))
        with patch('graph.plan_a_bc.geometry.RobotFK'):
            gpu=CurrentGeometry('unused',{},robot_filter_backend='cpu')
            cpu=CurrentGeometry('unused',{},perception_backend='cpu')
        # The integration input is already prepared. Robot filtering is outside
        # this change and is neither modified nor invoked by this contract.
        gpu.system.prepare=lambda *a:prepared;cpu.system.prepare=lambda *a:prepared
        a,ar,ag=gpu.build(fixture.cameras,fixture.data,[link],fixture.masks)
        b,br,bg=cpu.build(fixture.cameras,fixture.data,[link],fixture.masks)
        self.assertEqual(a.packet['perception_backend'],'triton_fused')
        for name in ('node_type','edge_index','edge_type','edge_attr'):
            np.testing.assert_array_equal(getattr(a,name).numpy(),getattr(b,name).numpy(),err_msg=name)
        np.testing.assert_allclose(a.node_input.numpy(),b.node_input.numpy(),atol=2e-5,rtol=0)
        self.assertEqual([r['source_observation_ids'] for r in ar],[r['source_observation_ids'] for r in br])
        self.assertEqual(ag,bg)


if __name__=='__main__':unittest.main()
