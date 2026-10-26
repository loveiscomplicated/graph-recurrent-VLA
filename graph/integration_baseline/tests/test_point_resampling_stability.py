"""Task07-G contracts and failure injection; synthetic cases are not research data."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from graph.integration_baseline import point_resampling_adapter as a
from graph.integration_baseline.point_resampling_stability import (controller,rotation_distance,action_metrics,
    recurrent,encode,recovery,tensor_controls)
from graph.integration_baseline.graph import build_graph,pack_graphs
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.robot import RobotNormalizer
from graph.integration_baseline.tests.test_baseline import fixture_frame,fixture_packet,robot_fixture,ROBOT
from graph.integration_baseline.vision import packet_from_frame
from graph.prototype.v2.state import StateNormalizer


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_phase0_and_valid_alternatives_including_empty_support(self):
        for empty in ((),(0,),(1,),(0,1)):
            frame=fixture_frame(empty);e,p,c,m=a.prepare(frame)
            baseline=packet_from_frame(frame,'synthetic')[0]
            built,_=a.build(frame,'synthetic',e,p,c,m,'phase0');a.assert_packet_equal(baseline,built)
            ga=build_graph(built,ROBOT)
            for condition in ('phase1','local01','local05'):
                b,_=a.build(frame,'synthetic',e,p,c,m,condition)
                g=build_graph(b,ROBOT);metrics=a.sampling_metrics(ga,g,p)
                self.assertTrue(all(o['presence_equal'] for o in metrics['objects']))

    def test_replacement_keeps_structural_count_support_component_and_is_deterministic(self):
        x=np.arange(100,dtype=np.int64)
        xyz=np.c_[x%10,x//10,np.ones(100)].astype(np.float32)*.001
        prep=dict(X=x,xyz=xyz,shape=(10,10),scale=dict(a_m=.01))
        original=dict(P=np.array([0,9,90,99]),S=np.arange(20,80,2),T={'pca':np.array([0,9,90,99]),'sil':np.array([],int)})
        original['F']=np.r_[original['P'],original['S']];original['xyz']=xyz[original['F']]
        components=np.zeros((10,10),int);components[:,5:]=1
        for f in (.01,.05):
            b,meta=a.local_replace(prep,original,components,f)
            b2,meta2=a.local_replace(prep,original,components,f)
            a.assert_packet_equal(b,b2);self.assertEqual(meta,meta2)
            np.testing.assert_array_equal(b['P'],original['P'])
            self.assertEqual(len(b['F']),len(original['F']));self.assertEqual(meta['actual'],meta['requested'])
            for old,new in meta['pairs']:
                self.assertNotIn(new,original['F']);self.assertEqual(components.ravel()[old],components.ravel()[new])
                np.testing.assert_array_equal(b['xyz'][np.flatnonzero(b['F']==new)[0]],xyz[new])
        full=deepcopy(original);full['S']=x[~np.isin(x,full['P'])];full['F']=np.r_[full['P'],full['S']]
        _,meta=a.local_replace(prep,full,components,.05);self.assertEqual(meta['actual'],0)

    def test_corrupted_source_feature_and_edge_identity_are_rejected(self):
        frame=fixture_frame();e,p,c,m=a.prepare(frame);baseline,_=a.build(frame,'s',e,p,c,m,'phase0')
        for field in ('points','geometric_features'):
            bad=deepcopy(baseline);bad[field][0,0]+=1
            with self.assertRaises(AssertionError):a.assert_packet_equal(baseline,bad)
        bad=deepcopy(baseline);bad['provenance']['source_index'][0]=0
        with self.assertRaises(AssertionError):a.assert_packet_equal(baseline,bad)
        bad=deepcopy(baseline);bad['intra_object_edge_index'][0,0]=len(bad['points'])+1
        with self.assertRaises(ValueError):build_graph(bad,ROBOT)


class RecurrentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(70)
        self.schema=robot_fixture()[1];self.model=IntegrationPolicy(self.schema).eval()
        self.norm=RobotNormalizer(self.schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[17])
        self.graphs=[build_graph(fixture_packet((0,) if t==2 else ()),ROBOT) for t in range(4)]
        self.states=np.random.default_rng(70).normal(size=(4,76)).astype(np.float32)

    def test_repetition_and_permutation_with_edge_remapping(self):
        with torch.inference_mode():
            c=tensor_controls(self.model,self.norm,self.graphs,self.states,'cpu',dict(seed=70,atol=1e-6,rtol=1e-5))
        self.assertEqual(c['status'],'PASS')

    def test_bad_permutation_without_remapping_fails(self):
        graph=self.graphs[0];rng=np.random.default_rng(4);bad=a.permute_graph(graph,rng)
        bad.edge_index=graph.edge_index
        with torch.inference_mode():
            za,_=self.model.graph_encoder(pack_graphs([graph]));zb,_=self.model.graph_encoder(pack_graphs([bad]))
        self.assertGreater(float((za-zb).abs().max()),1e-5)

    def test_pre_observation_hidden_and_independent_carry_are_distinct(self):
        with torch.inference_mode():
            _,_,fa=encode(self.model,self.norm,self.graphs,self.states,'cpu')
            aa,ha=recurrent(self.model,fa);pre=np.r_[np.zeros((1,144),np.float32),ha[:-1]]
            fb=fa.clone();fb[1,:320]+=1
            instantaneous,_=recurrent(self.model,fb,pre);history,hb=recurrent(self.model,fb)
            np.testing.assert_array_equal(instantaneous[0],aa[0]);np.testing.assert_array_equal(history[0],aa[0])
            np.testing.assert_allclose(instantaneous[1],history[1],atol=1e-7)
            self.assertGreater(np.linalg.norm(history[2]-instantaneous[2]),1e-5)
            # Feeding current post-hidden instead of previous hidden is detectable.
            wrong,_=recurrent(self.model,fa,ha)
            self.assertGreater(np.linalg.norm(wrong-aa),1e-5)
            reset,_=recurrent(self.model,fb,np.zeros_like(pre))
            self.assertGreater(np.linalg.norm(reset-history),1e-5)

    def test_controller_clipping_negative_rotation_scale_and_xyz_distance(self):
        command=controller(np.array([[2,-2,.5,0,0,2,2]],np.float32))
        np.testing.assert_allclose(command,[[.1,-.1,.05,0,0,-.1,.04]],atol=1e-8,rtol=0)
        zero=np.zeros((1,7),np.float32)
        self.assertAlmostEqual(float(rotation_distance(zero,command)[0]),.1,places=6)
        metrics=action_metrics(zero,zero,zero)
        self.assertEqual(metrics['command_translation_mm'][0],0)
        self.assertEqual(metrics['command_rotation_rad'][0],0)

    def test_recovery_requires_whole_remaining_tail_and_records_censoring(self):
        result=recovery([1,.01,.2,.01,0],0,.05,1e-6)
        self.assertEqual(result['stable_below_after_steps'],3)
        self.assertTrue(recovery([1,.5,.2],0,.05,1e-6)['censored'])


if __name__=='__main__':unittest.main()
