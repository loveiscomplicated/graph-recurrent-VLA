"""CUDA graph parity, sparse projection boundaries, resident outputs and IDs."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial import cKDTree
from .test_contracts import camera,surface
from ..tensor_fusion import tensor_observations,prepare_tensor_views
from ..tensor_records import current_records_tensor,numpy_records
from ..tensor_graph import build_graph_tensor,numpy_graph,free_space_tensor,surface_edges_tensor,FIELDS
from ..graph import build_graph,free_space_violation,surface_edges
from ..coordinates import observation_in_frame
from ..tensor_knn import neighbor_membership
from graph.plan_a_bc.geometry import validate_graph,CurrentGeometry
from graph.integration_baseline.graph import pack_graphs


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class TensorGraphContracts(unittest.TestCase):
    def setup_case(self,points,objects,frame='world',z=500):
        c=replace(camera(name='base_camera',z=z),tcp_pose=np.array([.02,-.03,.01,np.cos(.2),0,0,np.sin(.2)]))
        obs=[]
        for oid in (0,1):
            o=surface(np.asarray(points)[np.asarray(objects)==oid],oid=oid,name=c.camera)
            # Exercise feature-invalid NaNs and differing identity numeric/string order.
            n=len(o.xyz);o.pixels[:]=np.arange(n)[::-1]*11
            if n>2:
                o.feature_valid[1,1:]=False;o.f5[1,3:]=np.nan
                o.normal_valid[2]=False;o.feature_valid[2,0]=False;o.f5[2,:3]=np.nan
            obs.append(observation_in_frame(o,c,frame))
        ts=tensor_observations(obs,device='cuda')
        ts=[replace(o,structural=torch.zeros(len(o.xyz),device='cuda',dtype=torch.bool)) for o in ts]
        records=current_records_tensor(ts,coordinate_frame=frame)
        views=prepare_tensor_views([c],{c.camera:np.zeros((2,24,24),bool)},coordinate_frame=frame)
        robot=np.array([c.tcp_pose[:3],c.tcp_pose[:3]+[.02,0,0],c.tcp_pose[:3]+[-.02,0,0]])
        return c,records,views,robot

    def test_graph_parity_empty_short_two_object_and_full_budget(self):
        rng=np.random.default_rng(53)
        for counts in ((0,0),(1,0),(0,1),(2,3),(32,32),(128,128)):
            n=sum(counts);points=rng.uniform(-.012,.012,(n,3))+[0,0,.5]
            objects=np.repeat([0,1],counts)
            for frame in ('world','ee'):
                with self.subTest(counts=counts,frame=frame):
                    c,records,views,robot=self.setup_case(points,objects,frame)
                    expected=build_graph(numpy_records(records),robot,c.tcp_pose,0,cameras=[c],point_cap=None,
                        include_observation_status=False,coordinate_frame=frame)
                    def forbidden(*args,**kwargs):raise AssertionError('point-array readback forbidden')
                    with patch.object(torch.Tensor,'cpu',forbidden),patch.object(torch.Tensor,'numpy',forbidden),patch.object(torch.Tensor,'tolist',forbidden):
                        actual=build_graph_tensor(records,robot,c.tcp_pose,0,views=views)
                        validate_graph(actual);packed=pack_graphs([actual,actual])
                    self.assertTrue(packed.node_input.is_cuda);self.assertTrue(packed.edge_index.is_cuda)
                    self.assertTrue(all(getattr(actual,k).is_cuda for k in FIELDS))
                    actual=numpy_graph(actual)
                    for k in ('node_type','edge_type','edge_index'):torch.testing.assert_close(getattr(actual,k),getattr(expected,k),rtol=0,atol=0)
                    for k in ('node_input','edge_attr'):torch.testing.assert_close(getattr(actual,k),getattr(expected,k),rtol=0,atol=1e-6)
                    self.assertEqual(actual.packet['node_ids'],expected.packet['node_ids'])
                    self.assertEqual(actual.packet['surface_edge_audit'],expected.packet['surface_edge_audit'])

    def test_free_space_strict_threshold_invalid_depth_and_frames(self):
        zs=[np.nextafter(.994,-np.inf),.994,np.nextafter(.994,np.inf),1.,0.,-1.]
        c,records,views,robot=self.setup_case(np.empty((0,3)),np.empty(0),z=1000)
        points=np.c_[np.zeros((len(zs),2)),zs]
        for frame in ('ee','world'):
            from visual.runtime.coordinate_frames import world_from_frame
            from ..contracts import transform
            xyz=transform(points,np.linalg.inv(world_from_frame(frame,c.tcp_pose)))
            for raw in (1000,0,-32768):
                current=replace(c,depth_raw_mm=np.full_like(c.depth_raw_mm,raw))
                views=prepare_tensor_views([current],{c.camera:np.zeros((2,24,24),bool)},coordinate_frame=frame)
                expected=free_space_violation(xyz,[current],coordinate_frame=frame)
                result=free_space_tensor(torch.tensor(xyz,device='cuda'),views)
                np.testing.assert_array_equal(result.cpu().numpy(),expected)

    def test_knn_cutoff_ties_duplicates_and_grid(self):
        rng=np.random.default_rng(76)
        for n in (1,2,16,17,33,97,128):
            for xyz in (rng.normal(size=(n,3)),rng.integers(-2,3,size=(n,3)),np.zeros((n,3))):
                xyz=np.asarray(xyz,np.float64);near=cKDTree(xyz).query(xyz,k=min(9,n))[1]
                if n==1:near=near[:,None]
                expected=np.zeros((n,n),bool);expected[np.arange(n)[:,None],near]=True
                np.testing.assert_array_equal(neighbor_membership(torch.tensor(xyz,device='cuda')).cpu(),expected)

    def test_surface_radius_normal_tangency_and_witness_boundaries(self):
        c=camera(name='base_camera',z=500)
        views=prepare_tensor_views([c],{c.camera:np.zeros((2,24,24),bool)},coordinate_frame='world')
        cases=[]
        for threshold in (1e-9,.008,.02):
            for length in (np.nextafter(threshold,0),threshold,np.nextafter(threshold,np.inf)):
                cases.append((np.array([[0.,0.,.5],[length,0.,.5]]),np.tile([0.,0.,-1.],(2,1))))
        for dot in (np.nextafter(.5,0),.5,np.nextafter(.5,1)):
            cases.append((np.array([[0.,0.,.5],[.004,0.,.5]]),np.array([[0.,0.,-1.],[np.sqrt(1-dot*dot),0.,-dot]])))
        for points,normals in cases:
            valid=np.ones(2,bool);objects=np.zeros(2,int)
            expected,audit=surface_edges(points,normals,valid,objects,{0:points,1:np.empty((0,3))},[c],coordinate_frame='world')
            actual,counts=surface_edges_tensor(*(torch.tensor(x,device='cuda') for x in (points,normals,valid,objects)),views)
            np.testing.assert_array_equal(actual.cpu(),expected)
            self.assertEqual({k:int(v) for k,v in counts.items()},audit)

    def test_graph_runs_existing_encoder_on_cuda(self):
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        c,r,v,robot=self.setup_case(np.array([[0.,0.,.5],[.003,0.,.5]]),np.array([0,1]))
        g=build_graph_tensor(r,robot,c.tcp_pose,0,views=v)
        packed=pack_graphs([g]);torch.manual_seed(42)
        model=DiagnosticEncoder('E0').cuda().eval()
        with torch.inference_mode():z,presence=model(packed)
        self.assertTrue(z.is_cuda);self.assertTrue(torch.isfinite(z).all());self.assertTrue(presence.is_cuda)

    def test_stale_frame_identity_and_over_budget_rejected(self):
        c,r,v,robot=self.setup_case(np.array([[0.,0.,.5]]),np.array([0]),'ee')
        for wrong in (replace(r,episode='foreign'),replace(r,timestep=1),replace(r,coordinate_frame='world'),replace(r,reference_ee_pose=None)):
            with self.assertRaises(ValueError):build_graph_tensor(wrong,robot,c.tcp_pose,0,views=v)
        for value in (float('nan'),float('inf'),1e200):
            with self.assertRaisesRegex(ValueError,'finite bounded'):
                neighbor_membership(torch.tensor([[value,0.,0.]],device='cuda',dtype=torch.float64))
        c,r,v,robot=self.setup_case(np.zeros((129,3)),np.zeros(129))
        with self.assertRaisesRegex(ValueError,'128'):build_graph_tensor(r,robot,c.tcp_pose,0,views=v)

    def test_finish_source_gate_checks_resident_provenance(self):
        c,r,v,robot=self.setup_case(np.array([[0.,0.,.5]]),np.array([1]))
        g=build_graph_tensor(r,robot,c.tcp_pose,0,views=v)
        geo=object.__new__(CurrentGeometry);geo.coordinate_frame='world';geo.perception_backend='triton_fused'
        from types import SimpleNamespace
        geo.system=SimpleNamespace(tensor_packet_builder=True)
        data=dict(episode=np.array('ep'),timestep=0,tcp_pose=c.tcp_pose)
        geo._finish(g,r,{'accepted':True},data)
        with self.assertRaisesRegex(ValueError,'Wrist Box'):
            geo._finish(g,replace(r,camera_names=('hand_camera',)),{'accepted':True},data)
        with self.assertRaisesRegex(ValueError,'rejected Wrist'):
            geo._finish(g,replace(r,camera_names=('hand_camera',),object_id=torch.zeros_like(r.object_id)),{'accepted':False},data)


if __name__=='__main__':unittest.main()
