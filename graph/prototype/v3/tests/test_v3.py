import io
import json
import tempfile
import unittest
from dataclasses import replace
import torch
from graph.prototype.tests.fixtures import geometry
from graph.prototype.geometry import transform, rotation
from graph.prototype.common import config
from graph.prototype.v3.interaction_graph import InteractionGraph, Relation, policy
from graph.prototype.v3.geometry_audit import regions, boundary_check, resample
from graph.prototype.v3.edge_diagnostics import diagnose, edge_rows
from pathlib import Path


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_shapes_batch_directions_and_determinism(self):
        g=geometry();builder=InteractionGraph();gr=builder(g);again=builder(g)
        self.assertEqual(gr.x.shape,(4,131,14));self.assertEqual(gr.edge_attr.shape[1],9)
        src,dst=gr.edge_index
        self.assertTrue((src>=0).all() and (dst<524).all())
        self.assertTrue((src//131==dst//131).all())
        pos=torch.cat((g.ee_pose[:,None,:3],torch.stack((g.left_fingertip,g.right_fingertip),1),g.peg_points,g.hole_points),1).flatten(0,1)
        torch.testing.assert_close(gr.edge_attr[:,:3],pos[dst]-pos[src])
        torch.testing.assert_close(gr.edge_attr[:,3],gr.edge_attr[:,:3].norm(dim=-1))
        torch.testing.assert_close(gr.edge_attr[:,4:].argmax(-1),gr.edge_type)
        torch.testing.assert_close(gr.edge_index,again.edge_index)
        torch.testing.assert_close(gr.edge_attr,again.edge_attr)
        rows=edge_rows(gr);self.assertEqual(len(rows),len(src))
        self.assertEqual(set(rows),{(b,a,t) for a,b,t in rows})
        diag=diagnose(gr,[e.name for e in Relation])
        self.assertEqual(diag['components'],[[131]]*4);self.assertEqual(diag['duplicates'],0)
        self.assertEqual(diag['self_loops'],0)
        single=builder(g.select([2]));mask=src//131==2
        torch.testing.assert_close(single.edge_index,gr.edge_index[:,mask]-262)
        torch.testing.assert_close(single.edge_attr,gr.edge_attr[mask])
        for t in range(5):
            self.assertEqual(edge_rows(gr,[t]),[(a,b,k) for a,b,k in rows if k==t])

    def test_no_interaction_retains_global_connectivity(self):
        g=geometry(1);shift=torch.tensor([10.,0.,0.]);pose=g.box_pose.clone();pose[:,:3]+=shift;hole=g.hole_pose.clone();hole[:,:3]+=shift
        g=replace(g,box_pose=pose,hole_pose=hole,hole_points=g.hole_points+shift)
        gr=InteractionGraph(radius=1e-5)(g)
        self.assertEqual(int((gr.edge_type==3).sum()),0)
        self.assertEqual(int((gr.edge_type==4).sum()),4)
        self.assertEqual(diagnose(gr,[e.name for e in Relation])['components'],[[131]])

    def test_regions_radius_and_collision_boundaries(self):
        g=geometry();gr=InteractionGraph()(g);labels=regions(g).flatten()
        src,dst=gr.edge_index
        mask=gr.edge_type==3
        self.assertTrue(((labels[src[mask]]==1)&((labels[dst[mask]]==4)|(labels[dst[mask]]==5)) |
                         (labels[dst[mask]]==1)&((labels[src[mask]]==4)|(labels[src[mask]]==5))).all())
        self.assertTrue((gr.edge_attr[(gr.edge_type==2)|(gr.edge_type==3),3]<=.050001).all())
        for state in (g,resample(g),resample(g,count=80,weights=(1,4,2,1)),resample(g,entry_band=.02)):
            result=boundary_check(state)
            self.assertTrue(result['hole_inward_inside']);self.assertTrue(result['hole_outward_outside'])
            self.assertTrue((regions(state)[:,3+64:]>=4).all())
        torch.testing.assert_close(resample(g,weights=(1,1,1,1)).hole_local,g.hole_local)
        torch.testing.assert_close(resample(g).hole_local,resample(g).hole_local)
        self.assertEqual(resample(g,count=80).hole_local.shape,(4,80,3))
        targeted=resample(g,entry_band=.02)
        label=regions(targeted)[:,67:]
        distance=(targeted.hole_local-targeted.hole_offset[:,None])[...,1:].abs().amax(-1)-targeted.hole_radius
        self.assertTrue((((label==4)&(distance<=.020001)).sum(-1)>=18).all())

    def test_common_rigid_transform(self):
        g=geometry(1);q=torch.tensor([[.8,0.,0.,.6]]);r=rotation(q);t=torch.tensor([[1.,2.,3.]])
        def pose(p):
            # Fixture has identity rotations.
            return torch.cat((p[:,:3]@r[0].T+t,q),-1)
        def point(p):return p@r[0].T+t
        moved=replace(g,ee_pose=pose(g.ee_pose),root_pose=pose(g.root_pose),peg_pose=pose(g.peg_pose),
                      box_pose=pose(g.box_pose),hole_pose=pose(g.hole_pose),left_fingertip=point(g.left_fingertip),
                      right_fingertip=point(g.right_fingertip),peg_points=point(g.peg_points),hole_points=point(g.hole_points),
                      peg_normals=g.peg_normals@r[0].T,hole_normals=g.hole_normals@r[0].T)
        builder=InteractionGraph();a,b=builder(g),builder(moved)
        torch.testing.assert_close(a.edge_index,b.edge_index)
        torch.testing.assert_close(b.edge_attr[:,:3],a.edge_attr[:,:3]@r[0].T,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a.edge_attr[:,3],b.edge_attr[:,3],atol=1e-6,rtol=1e-5)
        self.assertLess(boundary_check(moved)['hole_world_transform_max_error_m'],1e-6)

    def test_forward_backward_checkpoint(self):
        cfg=config(Path('graph/prototype/v2/config.toml'));model=policy(cfg);gr=InteractionGraph()(geometry(2))
        prediction=model(gr,torch.zeros(2,31));self.assertEqual(prediction.shape,(2,7))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.encoder.node_encoder[0].weight.grad.norm()),0)
        for layer in model.encoder.layers:
            self.assertGreater(float(layer.message[0].weight.grad.norm()),0)
        self.assertGreater(float(model.encoder.readout[0].weight.grad.norm()),0)
        self.assertGreater(float(model.action_head[0].weight.grad.norm()),0)
        buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0);other=policy(cfg)
        other.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(prediction,other(gr,torch.zeros(2,31)))

    def test_rendered_edge_export_matches_tensor(self):
        from graph.prototype.v3.visualization import draw
        g=geometry(1);gr=InteractionGraph()(g)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'edges.png'
            draw(g,gr,path,selected=[2,4],zoom=True)
            rows=json.loads(path.with_suffix('.edges.json').read_text())
            self.assertEqual(rows,[list(row) for row in edge_rows(gr,[2,4])])
            self.assertTrue(path.stat().st_size>0)


if __name__=='__main__':unittest.main()
