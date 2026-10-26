"""Real stored t27 fixture and negative causal-boundary contracts for Task07-F."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from graph.integration_baseline.autonomous_reach_report import load_arrays
from graph.integration_baseline.io import read_packet, save_packet, save_npz
from graph.integration_baseline.vision_oracle_adapter import (
    CurrentSegmentation, POLICIES, apply_masks, build_input, rgbd_hash,
    validate_input, OwnedHistory)


def fixture_contract(folder, output=None, require_saved_exact=False):
    folder=Path(folder);t=27;identity='stored_t27_fixture'
    obs=load_arrays(folder/'observations.npz');row={k:v[t] for k,v in obs.items()}
    pred=load_arrays(folder/'masks.npz')['masks'][t]
    sidecar=load_arrays(folder/'gt_segmentation_diagnostic_only.npz')
    gt=CurrentSegmentation(identity,t,rgbd_hash(row),sidecar['segmentation'][t][None]==sidecar['actor_ids'].reshape(2,1,1))
    built={};counts={}
    if output is not None:Path(output).mkdir(parents=True,exist_ok=False)
    for condition in POLICIES:
        masks,packet,graph,_=build_input(row,pred,gt,condition=condition,episode=identity,
            timestep=t,graph_config=dict(k=4,M=16,block_rows=64))
        built[condition]=(masks,packet,graph)
        counts[condition]=dict(points=len(packet['points']),edges=graph.edge_index.shape[1],
            raw_ede=packet['intra_object_edge_index'].shape[1])
        if condition!='A':
            source=packet['provenance']['source_index'];obj=packet['object_id']
            if not np.all(gt.masks.reshape(2,-1)[obj,source]):raise AssertionError('oracle left false source pixels')
        if output is not None:
            save_packet(Path(output)/(condition+'_packet.npz'),packet)
            save_npz(Path(output)/(condition+'_graph.npz'),**{k:getattr(graph,k).numpy() for k in
                ('node_input','node_type','edge_index','edge_attr','edge_type')})
    # Same-runtime A must be exact; cross-platform stored FP32 F5 may differ.
    from visual.runtime.rgbd import TrackedRGBDFrame
    from graph.integration_baseline.vision import packet_from_frame
    direct,_=packet_from_frame(TrackedRGBDFrame(t,'base_camera',row['rgb'],row['depth'],
        row['intrinsic'],row['cam2world'],row['tcp_pose'],{0:pred[0],1:pred[1]},t,t/20),identity,.5)
    old=read_packet(folder/'packet_0027.npz')
    saved_exact=True;maximum_difference=0.
    for k in ('points','camera_points','geometric_features','feature_field_available','feature_field_valid','intra_object_edge_index'):
        actual=built['A'][1][k]
        np.testing.assert_array_equal(actual,direct[k])
        if actual.dtype.kind=='f':
            np.testing.assert_allclose(actual,old[k],atol=1e-6,rtol=1e-5,equal_nan=True)
            saved_exact &= np.array_equal(actual,old[k],equal_nan=True)
            finite=np.isfinite(actual)&np.isfinite(old[k])
            maximum_difference=max(maximum_difference,float(np.max(np.abs(actual[finite]-old[k][finite]),initial=0)))
        else:
            saved_exact &= np.array_equal(actual,old[k])
            if require_saved_exact:np.testing.assert_array_equal(actual,old[k])
    oldgraph=load_arrays(folder/'graph_0027.npz')
    for k in ('node_input','node_type','edge_index','edge_attr','edge_type'):
        actual=getattr(built['A'][2],k).numpy()
        if require_saved_exact:np.testing.assert_array_equal(actual,oldgraph[k])
        saved_exact &= np.array_equal(actual,oldgraph[k])
    if require_saved_exact and not saved_exact:raise AssertionError('same-host stored fixture changed')
    # Even spoofing the packet stamp cannot make the original A graph pass B/C.
    rejected=[]
    for condition in ('B','C'):
        try:
            validate_input(built[condition][1],built['A'][2],built[condition][0],row,
                identity,t,POLICIES[condition])
        except (ValueError,AssertionError):rejected.append(condition)
        else:raise AssertionError('A graph reuse accepted for changed oracle mask')
    return dict(status='PASS',timestep=t,conditions=counts,A_same_runtime_packet_exact=True,
        A_existing_packet_graph_exact=bool(saved_exact),maximum_saved_packet_difference=maximum_difference,
        B_C_no_non_target_sources=True,negative_A_graph_reuse_rejected=rejected,
        rebuilt_with_original_shape_first_f5_raw_ede=True)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(rgb=np.zeros((3,4,3),np.uint8),depth_raw_mm=np.ones((3,4),np.int16),
            depth=np.ones((3,4),np.float32),intrinsic=np.eye(3),cam2world=np.eye(4))
        self.p=np.zeros((2,3,4),bool);self.p[0,0,:3]=True;self.p[1,2,:2]=True
        masks=np.zeros_like(self.p);masks[0,0,1:]=True;masks[1,2,1:]=True
        self.gt=CurrentSegmentation('ep',4,rgbd_hash(self.row),masks)

    def apply(self,c,gt=None):
        return apply_masks(self.p,gt or self.gt,condition=c,episode='ep',timestep=4,row=self.row)

    def test_policies_and_no_mutation(self):
        p=self.p.copy();g=self.gt.masks.copy()
        np.testing.assert_array_equal(self.apply('A'),p)
        b=self.apply('B');np.testing.assert_array_equal(b,p&g)
        self.assertFalse(np.any(b&~p));self.assertFalse(np.any(b&~g))
        np.testing.assert_array_equal(self.apply('C'),g)
        b[:]=False
        np.testing.assert_array_equal(self.p,p);np.testing.assert_array_equal(self.gt.masks,g)

    def test_future_stale_foreign_and_wrong_rgbd_rejected(self):
        for g in (replace(self.gt,timestep=5),replace(self.gt,timestep=3),
                  replace(self.gt,episode='other'),replace(self.gt,rgbd_hash='wrong')):
            with self.assertRaises(ValueError):self.apply('B',g)

    def test_alignment_dtype_identity_rejected(self):
        for m in (self.gt.masks.astype(np.uint8),self.gt.masks[:,:,:3],np.ones_like(self.p)):
            with self.assertRaises(ValueError):self.apply('C',replace(self.gt,masks=m))

    def test_hidden_transplant_and_gap_rejected(self):
        class Stream:
            def reset(s):s.hidden=s.identity=s.timestep=None
        s=Stream();h=OwnedHistory(s,'own');h.before(0)
        s.hidden=torch.ones(1,1,144);s.identity='own';s.timestep=0;h.after(0);h.before(1)
        with self.assertRaises(ValueError):h.before(2)
        s.hidden=torch.zeros_like(s.hidden)
        with self.assertRaises(ValueError):h.before(1)
        s.hidden=torch.ones_like(s.hidden);s.identity='other'
        with self.assertRaises(ValueError):h.before(1)

    def test_fixed_protocol(self):
        from graph.integration_baseline.vision_oracle_reach import validate_config
        from graph.integration_baseline.direct_runtime import read
        c=read(Path(__file__).parents[1]/'vision_oracle_reach_config.json');validate_config(c)
        for k,v in [('seed',41006),('keep_ratio',.6),('horizon',249),('conditions',['C','B','A'])]:
            bad=deepcopy(c);bad[k]=v
            with self.assertRaises(ValueError):validate_config(bad)

    def test_real_frame_rebuild_and_negative_graph_reuse(self):
        explicit=os.environ.get('TASK07F_FIXTURE_DIR')
        folder=Path(explicit or 'experiments/c76d3cf8f8354c7199e13f3068d2d1c3/trials/C')
        if not folder.is_dir():self.skipTest('07E fixture passed explicitly on CUDA host')
        self.assertEqual(fixture_contract(folder,require_saved_exact=bool(explicit))['status'],'PASS')


if __name__=='__main__':unittest.main()
