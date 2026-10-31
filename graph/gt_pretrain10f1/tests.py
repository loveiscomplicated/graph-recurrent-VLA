"""Semantic contracts: world axes, measured robot, schema/GT exclusion, E0 slots."""
import unittest
import numpy as np
import torch
from torch import nn

from recurrent.features import build_features,quat_rotate
from graph.integration_baseline.graph import Graph,pack_graphs
from graph.integration_baseline.robot import state_schema,flatten_state
from graph.integration_baseline.model import structured_readout
from graph.prototype.models import mlp
from graph.plan_a_bc.common import tensor_identity
from .contract import (assemble_gt38,object_target,split_fields,measured_robot,
    normalize_gt38,quaternion_geodesic,feature_contract)
from .model import StateEncoder,absolute_loss,displacement_loss


class Contracts(unittest.TestCase):
    def raw(self):
        rng=np.random.default_rng(71)
        raw=dict(tcp_pose=np.r_[.1,-.2,.3,.5,.5,.5,.5].astype('f'),
            left_tip=np.array([.12,-.22,.34],dtype='f'),right_tip=np.array([.08,-.18,.34],dtype='f'),
            gripper=np.array([.03,.035],dtype='f'),peg_pose=np.r_[.2,-.1,.1,1.,0,0,0].astype('f'),
            hole_pose=np.r_[-.1,.3,.15,.5,.5,.5,.5].astype('f'),peg_half=np.array([.11,.022,.022],dtype='f'),
            hole_radius=np.array([.025],dtype='f'))
        return raw

    def test_world_displacements_and_derived_head_match_official(self):
        raw=self.raw();o=object_target(raw)
        robot={k:raw[k] for k in ('tcp_pose','left_tip','right_tip','gripper')}
        state=assemble_gt38(split_fields(o),robot)
        np.testing.assert_allclose(state,build_features(raw,np.float32(1),np.float32(0)),atol=1e-7,rtol=0)
        np.testing.assert_array_equal(state[15:18],raw['peg_pose'][:3]-raw['tcp_pose'][:3])
        np.testing.assert_allclose(state[22:25],raw['peg_pose'][:3]+[.11,0,0]-raw['hole_pose'][:3],atol=1e-7)
        self.assertEqual(list(state[-2:]),[1,0])
        self.assertEqual(len(feature_contract()['dimensions']),38)

    def test_assembly_rejects_sidecar_and_missing_field(self):
        raw=self.raw();objects=split_fields(object_target(raw))
        robot={k:raw[k] for k in ('tcp_pose','left_tip','right_tip','gripper')}
        with self.assertRaises(ValueError):assemble_gt38(objects,{**robot,'peg_pose':raw['peg_pose']})
        with self.assertRaises(ValueError):assemble_gt38({**objects,'gt38':np.zeros(38)},robot)
        with self.assertRaises(ValueError):assemble_gt38({k:v for k,v in objects.items() if k!='hole_radius'},robot)

    def test_robot76_reconstructs_pad_offsets(self):
        raw=self.raw();f=dict(qpos=np.zeros(9,'f'),qvel=np.zeros(9,'f'),
            gripper_position=raw['gripper'],gripper_velocity=np.zeros(2,'f'),
            gripper_opening=np.array([.065],'f'),gripper_opening_velocity=np.zeros(1,'f'))
        for n in ('tcp','left_finger','right_finger','robot_base'):
            f[n+'_pose']=raw['tcp_pose'].copy();f[n+'_linear_velocity']=np.zeros(3,'f');f[n+'_angular_velocity']=np.zeros(3,'f')
        schema=state_schema(f);self.assertEqual(schema['dimension'],76)
        measured=measured_robot(flatten_state(f,schema),schema)
        np.testing.assert_array_equal(measured['gripper'],raw['gripper'])
        np.testing.assert_allclose(measured['left_tip'],raw['tcp_pose'][:3]+quat_rotate(raw['tcp_pose'][3:],np.array([0,.00758,.04525],'f')))

    def test_quaternion_equivalence_and_physical_units(self):
        q=torch.tensor([[1.,0,0,0],[2**-.5,0,0,2**-.5]])
        self.assertTrue(torch.equal(quaternion_geodesic(q,-q),quaternion_geodesic(q,q)))
        self.assertAlmostEqual(float(quaternion_geodesic(q[:1],q[1:])),np.pi/2,places=5)
        gt=torch.from_numpy(object_target(self.raw())[None])
        pred=gt.clone();pred[:,3:7]*=-1;pred[:,10:14]*=-1
        self.assertAlmostEqual(float(absolute_loss(pred,gt)[0]),0,places=6)
        self.assertAlmostEqual(float(displacement_loss(gt,gt,gt,gt)),0)

    def test_displacement_metrics_zero_and_signed_response(self):
        from .evaluate import delta_metrics
        gt=np.array([-.005,0,0])
        zero=delta_metrics(gt,np.zeros(3));correct=delta_metrics(gt,gt)
        self.assertIsNone(zero['direction_cosine']);self.assertEqual(zero['signed_gain'],0)
        self.assertAlmostEqual(zero['delta_vector_error_mm'],5)
        self.assertAlmostEqual(correct['direction_cosine'],1);self.assertAlmostEqual(correct['signed_gain'],1)
        self.assertAlmostEqual(correct['magnitude_ratio'],1);self.assertAlmostEqual(correct['delta_vector_error_mm'],0)

    def test_five_slot_order_and_empty_object(self):
        def graph(missing=False):
            typ=torch.tensor([0,1,2] if missing else [0,1,2,3,3,4])
            n=len(typ)
            return Graph(torch.zeros(n,21),typ,torch.empty(2,0,dtype=torch.long),
                torch.empty(0,4),torch.empty(0,dtype=torch.long),{}, {})
        g=pack_graphs([graph(),graph(True)])
        h=torch.arange(len(g.node_type)*64).reshape(-1,64).float()
        z,pres=structured_readout(h,g)
        torch.testing.assert_close(z[0,:192],h[:3].flatten())
        torch.testing.assert_close(z[0,192:256],h[3:5].mean(0))
        torch.testing.assert_close(z[0,256:],h[5])
        self.assertTrue(torch.equal(pres,torch.tensor([[1.,1.],[0,0]])))
        self.assertTrue(torch.equal(z[1,192:],torch.zeros(128)))
        model=StateEncoder(np.zeros(76),np.ones(76),np.r_[np.zeros(14),.1,.02,.02,.023]).freeze()
        self.assertFalse(any(isinstance(m,(nn.GRU,nn.LSTM,nn.RNN)) for m in model.modules()))
        before=tensor_identity(model.state_dict())
        with torch.inference_mode():
            a=model(g,torch.zeros(2,76));b=model(g,torch.zeros(2,76))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.isfinite(a[0]).all());self.assertEqual(a[1].shape,(2,320))
        self.assertEqual(before,tensor_identity(model.state_dict()))


if __name__=='__main__':unittest.main()
