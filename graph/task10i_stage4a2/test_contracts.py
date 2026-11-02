import unittest
import tempfile
from pathlib import Path
import numpy as np
from .dinov2_matching_audit import crop_input
from .development_inference import WristPrompt
from .benchmark_protocol import integer_values
from .dino_proposal_audit import pixel_boxes
from .reanchor_state_machine import ReliabilitySlot,Evidence,State
from .evaluation_metrics import pixel_metrics,correct_source_coverage,hierarchical_paired
from .tri_model_integration import CausalCorrection
from types import SimpleNamespace
from .stress_capture import slab_pose,occlusion_active,active_cameras
from .final_capture import freeze_barrier
from graph.task10i_stage4a.run_pipeline import write,sha
from .reference_pair_audit import rectangle_identity


class ActorLabelContracts(unittest.TestCase):
    def test_rectangular_background_does_not_change_localized_actor_identity(self):
        seg=np.zeros((16,16),int);seg[np.arange(4,12),np.arange(4,12)]=1
        mask=np.zeros_like(seg,bool);mask[4:12,4:12]=True
        rr=dict(object_ids=dict(peg=[1],box=[2]),robot_ids=[3])
        label,detail=rectangle_identity(mask,seg,rr)
        self.assertEqual(label,'peg');self.assertEqual(detail['actor_coverage'],1.)
        self.assertLess((mask&(seg==1)).sum()/mask.sum(),.8)
        label,_=rectangle_identity(np.ones_like(mask),seg,rr)
        self.assertIsNone(label)

    def test_mixed_actors_are_unknown_and_robot_crop_is_negative(self):
        rr=dict(object_ids=dict(peg=[1],box=[2]),robot_ids=[3])
        mask=np.zeros((16,16),bool);mask[:4,:4]=True
        seg=np.zeros((16,16),int);seg[:2,:4]=1;seg[2:4,:4]=3
        self.assertIsNone(rectangle_identity(mask,seg,rr)[0])
        seg[:4,:4]=3
        self.assertEqual(rectangle_identity(mask,seg,rr)[0],'robot')


class FinalCaptureContracts(unittest.TestCase):
    def test_occluder_camera_alignment_and_fixed_timing(self):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('xyz',[.4,-.7,.1]).as_matrix()
        cv=np.column_stack((rotation,[.3,.2,-.5]))
        p,q=slab_pose(cv)
        np.testing.assert_allclose(rotation@p+cv[:,3],[0,0,.1],atol=1e-12)
        slab_rotation=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
        np.testing.assert_allclose(rotation@slab_rotation,np.eye(3),atol=1e-12)
        self.assertEqual([t for t in range(26) if occlusion_active(t)],list(range(8,20)))
        self.assertEqual(len(active_cameras('both')),2)
        with self.assertRaises(ValueError):slab_pose(np.eye(4))

    def test_final_capture_rejects_missing_or_modified_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/'protocol.json';design=root/'design.json'
            write(protocol,dict(seed_queue='not opened'));write(design,dict(horizon=120))
            with self.assertRaises(FileNotFoundError):freeze_barrier(root,protocol,design)
            source=root/'source.py';source.write_text('frozen')
            write(root/'candidate_source_frozen.json',dict(status='FROZEN',files={str(source):sha(source)}))
            write(root/'adoption_criteria_frozen.json',dict(status='FROZEN',primary='registered before test'))
            write(root/'candidate_config_frozen.json',dict(status='FROZEN',allow_independent_final_capture=True,
                new_final_benchmark_protocol_sha256=sha(protocol),final_collection_design_sha256=sha(design),
                adoption_criteria_sha256=sha(root/'adoption_criteria_frozen.json')))
            self.assertEqual(len(freeze_barrier(root,protocol,design)),3)
            source.write_text('changed after freeze')
            with self.assertRaises(AssertionError):freeze_barrier(root,protocol,design)


class CropContracts(unittest.TestCase):
    def test_empty_is_missing(self):
        self.assertIsNone(crop_input(np.zeros((8,8,3),np.uint8),np.zeros((8,8),bool),False))

    def test_boundary_and_aspect_preservation(self):
        rgb=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        mask=np.zeros((8,8),bool)
        mask[0:2,0:8]=True
        image,support,box=crop_input(rgb,mask,False)
        self.assertEqual(box,[0,0,8,2])
        np.testing.assert_array_equal(image,rgb[:2])
        image,support,box=crop_input(rgb,mask,True)
        self.assertEqual(image.shape,(8,8,3))
        self.assertEqual(int(support.sum()),16)
        np.testing.assert_array_equal(image[3:5],rgb[:2])
        self.assertFalse(support[:3].any())


class InferenceIsolation(unittest.TestCase):
    def test_camera_wrapper_only_rewrites_peg(self):
        calls=[]
        def detector(rgb,prompt):
            calls.append((rgb,prompt));return [[0,0,2,2]],[.4]
        wrapper=WristPrompt(detector);rgb=np.zeros((4,4,3),np.uint8)
        self.assertEqual(wrapper(rgb,'red and white block'),([[0,0,2,2]],[.4]))
        wrapper(rgb,'cardboard box')
        self.assertEqual([x[1] for x in calls],['red and white peg','cardboard box'])
        self.assertIs(calls[0][0],rgb)

    def test_gt_and_future_sidecars_not_in_detector_contract(self):
        wrapper=WristPrompt(lambda rgb,prompt:([],[]))
        with self.assertRaises(TypeError):
            wrapper(np.zeros((4,4,3),np.uint8),'peg',gt_mask=np.ones((4,4),bool))
        with self.assertRaises(TypeError):
            wrapper(np.zeros((4,4,3),np.uint8),'peg',future_rgb=np.zeros((4,4,3),np.uint8))

    def test_query_coordinate_mapping_keeps_axis_and_index(self):
        boxes=np.array([[.5,.25,.25,.125],[.2,.3,.1,.2]],np.float32)
        expected=np.array([[48,24,80,40],[19.2,25.6,32,51.2]])
        np.testing.assert_allclose(pixel_boxes(boxes),expected,atol=1e-5)

    def test_conservative_seed_scan_excludes_booleans(self):
        self.assertEqual(list(integer_values(dict(seed=123,flag=True,nested=[456,False,-1,2**31]))),[123,456])


class StateContracts(unittest.TestCase):
    good=Evidence(30,1.,1.,30,3.)
    bad=Evidence(0,0.,0.,0,-2.)

    def test_ever_found_does_not_prevent_causal_loss(self):
        slot=ReliabilitySlot('e','base_camera',0)
        self.assertFalse(slot.observe('e','base_camera',0,0,self.good,initialized=True))
        self.assertFalse(slot.observe('e','base_camera',0,1,self.bad))
        self.assertTrue(slot.observe('e','base_camera',0,2,self.bad))
        self.assertTrue(slot.ever_found);self.assertFalse(slot.currently_valid)
        self.assertEqual(slot.state,State.REDETECTION_REQUESTED)

    def test_missing_reference_and_rejected_memory_transaction(self):
        slot=ReliabilitySlot('e','hand_camera',0)
        slot.observe('e','hand_camera',0,0,self.bad)
        slot.observe('e','hand_camera',0,1,self.bad)
        self.assertTrue(slot.assess_candidate(1,.4,self.good,True,False))
        with self.assertRaises(ValueError):slot.observe('e','hand_camera',0,2,self.good)
        self.assertFalse(slot.commit(1,False))
        self.assertFalse(slot.ever_found)
        self.assertEqual(slot.state,State.REANCHOR_REJECTED)

    def test_same_frame_candidate_needs_past_confirmation(self):
        slot=ReliabilitySlot('e','base_camera',0)
        slot.observe('e','base_camera',0,0,self.bad)
        slot.observe('e','base_camera',0,1,self.bad)
        self.assertFalse(slot.assess_candidate(1,.9,self.good,False,False))

    def test_camera_episode_and_future_isolation(self):
        slot=ReliabilitySlot('e','base_camera',0)
        for episode,camera,t in [('other','base_camera',0),('e','hand_camera',0),('e','base_camera',2)]:
            with self.assertRaises(ValueError):slot.observe(episode,camera,0,t,self.good)
        with self.assertRaises(ValueError):ReliabilitySlot('e','hand_camera',1)

    def test_healthy_sequence_cannot_request_reanchor(self):
        slot=ReliabilitySlot('e','base_camera',0)
        for t in range(20):
            self.assertFalse(slot.observe('e','base_camera',0,t,self.good,initialized=t==0))
        with self.assertRaises(ValueError):slot.assess_candidate(19,.9,self.good,True,True,.9,.2)


class EvaluationContracts(unittest.TestCase):
    def test_named_challenger_does_not_relabel_A_or_reuse_B0(self):
        rows=[dict(parent_id='p',episode='e',timestep=0,camera='base_camera',object_id=0,
            condition=c,score=v) for c,v in [('B0',.2),('A',.9),('P',.4)]]
        result=hierarchical_paired(rows,'score',candidate='P',bootstrap=20)[0]
        self.assertAlmostEqual(result['delta'],.2)
        with self.assertRaises(ValueError):hierarchical_paired(rows,'score',candidate='B0')

    def test_empty_precision_remains_undefined(self):
        result=pixel_metrics(np.zeros((4,4),bool),np.ones((4,4),bool))
        self.assertIsNone(result['precision'])
        self.assertEqual(result['recall'],0.)
        self.assertEqual(result['correct_mask_availability'],0.)

    def test_nearby_wrong_source_never_receives_coverage_credit(self):
        point=np.array([[0.,0.,0.]])
        self.assertEqual(correct_source_coverage(point,[False],point),0.)
        self.assertEqual(correct_source_coverage(point,[True],point),1.)
        self.assertIsNone(correct_source_coverage(point,[True],np.empty((0,3))))

    def test_parent_macro_and_unsupported_exclusion(self):
        rows=[]
        for parent,nframes in [('p1',100),('p2',1)]:
            for t in range(nframes):
                for condition in ('B0','A'):
                    rows.append(dict(parent_id=parent,episode=parent,timestep=t,condition=condition,
                        camera='base_camera',object_id=0,score=float(condition=='A' and parent=='p1'),
                        reference_kind='EXACT_GT'))
        for condition in ('B0','A'):
            rows.append(dict(parent_id='unsupported',episode='u',timestep=0,condition=condition,
                camera='base_camera',object_id=0,score=1000.,reference_kind='APPROXIMATE_REFERENCE'))
        result=hierarchical_paired(rows,'score',bootstrap=100)[0]
        self.assertEqual(result['paired_parents'],2)
        self.assertEqual(result['delta'],.5)
        with self.assertRaises(ValueError):hierarchical_paired(rows+[rows[0]],'score')


class PracticalIntegrationContracts(unittest.TestCase):
    def test_late_initialization_handoff_uses_observed_edge_and_overlap(self):
        for event_t in (8,11):
            policy,_,_=self.policy(correction_slots=[('hand_camera',0),('base_camera',1)],
                uncertain_output_policy='preserve_current',late_baseline_handoff=True)
            try:
                for t in range(event_t+2):
                    masks=self.masks(False)
                    if t>=event_t-1:masks['hand_camera'][0,:6,:3]=True
                    found={('base_camera',1):True,('hand_camera',0):t>=event_t}
                    out=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],masks,found)
                    if t==event_t-1:self.assertEqual(out['hand_camera'][0].sum(),9)
                    if t>=event_t:np.testing.assert_array_equal(out['hand_camera'],masks['hand_camera'])
                self.assertEqual(len(policy.handoffs),1);self.assertTrue(policy.handoffs[0]['accepted'])
                self.assertNotIn(('hand_camera',0),policy.active)
            finally:policy.close()
        policy,_,_=self.policy(correction_slots=[('hand_camera',0)],
            uncertain_output_policy='preserve_current',late_baseline_handoff=True)
        try:
            for t in range(9):
                masks=self.masks(False)
                if t==8:masks['hand_camera'][0,8:14,:3]=True
                out=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],masks,
                    {('base_camera',1):True,('hand_camera',0):t==8})
            self.assertEqual(out['hand_camera'][0].sum(),9)
            self.assertFalse(policy.handoffs[0]['accepted'])
        finally:policy.close()

    def test_main_box_scope_never_redetects_or_changes_peg_slots(self):
        policy,calls,states=self.policy(correction_slots=[('base_camera',1)],uncertain_output_policy='preserve_current')
        try:
            for t in range(12):
                masks=self.masks(False)
                current=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],masks,{('base_camera',1):True})
                for camera in masks:np.testing.assert_array_equal(current[camera],masks[camera])
            self.assertEqual(calls,[])
            self.assertTrue(any(e['redetection_due_but_scope_disabled'] for e in policy.events))
            self.assertFalse(states)
        finally:policy.close()
        with self.assertRaises(ValueError):self.policy(correction_slots=[('hand_camera',1)])

    def test_lazy_matching_skips_unused_features_and_preserves_causal_masks(self):
        class Features:
            def __init__(self):self.calls=[]
            def reset(self,episode):self.calls=[]
            def observe(self,*args):pass
            def embed(self,camera,t,mask):
                self.calls.append((camera,t,int(mask.sum())))
                return np.array([1.,0.]) if mask.any() else None
        results=[]
        for lazy in (False,True):
            feature=Features()
            policy,_,_=self.policy(features=feature,lazy_matching=lazy,uncertain_output_policy='preserve_current')
            try:
                frames=[]
                for t in range(7):
                    masks=self.masks(False)
                    current=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],masks,
                        {('base_camera',1):True})
                    frames.append(np.stack(list(current.values())))
                results.append((np.stack(frames),feature.calls.copy(),policy.reference_provenance.copy(),
                                [e['accepted'] for e in policy.events]))
            finally:policy.close()
        np.testing.assert_array_equal(results[0][0],results[1][0])
        self.assertEqual(results[0][2:],results[1][2:])
        self.assertLess(len(results[1][1]),len(results[0][1]))

    def test_actual_geometry_firewall_rejects_GT_phase_and_future_fields(self):
        from graph.plan_a_geometry.core import PlanA
        for forbidden in ('gt_mask','actor_ids','expert_phase','future_rgb','peg_pose'):
            policy,_,_=self.policy()
            policy.geometry.system=PlanA('base_camera','hand_camera',{})
            try:
                with self.assertRaisesRegex(ValueError,'forbidden non-sensor input'):
                    policy.process(self.cameras(0),dict(episode='e',timestep=0,**{forbidden:None}),[],self.masks(True),{})
            finally:policy.close()

    def cameras(self,t):
        result=[]
        yy,xx=np.indices((16,16));world=np.stack([xx*.001,yy*.001,np.ones_like(xx)],axis=-1)
        for name in ('base_camera','hand_camera'):
            c=SimpleNamespace(episode='e',camera=name,timestep=t,timestamp=t/20,
                tcp_pose=np.array([0,0,0,1,0,0,0]),rgb=np.zeros((16,16,3),np.uint8),depth=np.ones((16,16)))
            c.validate=lambda c=c:c;c.world_points=lambda:world
            result.append(c)
        return result

    def policy(self,**kwargs):
        calls=[];states=[]
        def detector(rgb,prompt):
            calls.append(prompt);return [[0.,0.,3.,3.]],[.25]
        class Anchor:
            def __init__(self,tracker,rgb,box,t):
                self.t=t;self.anchor_t=t;self.closed=False;states.append(self)
            def current(self):
                mask=np.zeros((16,16),bool);mask[:3,:3]=True;return mask
            def append(self,rgb,t):
                if t!=self.t+1:raise ValueError('anchor time')
                self.t=t
            def close(self):self.closed=True
        def prepare(cameras,data,masks,surfaces):
            zero=np.zeros((16,16),bool)
            return dict(clean={c:m.copy() for c,m in masks.items()},filters={c:dict(confirmed_robot=zero,uncertain=zero) for c in masks})
        policy=CausalCorrection(detector,SimpleNamespace(),SimpleNamespace(system=SimpleNamespace(prepare=prepare)),anchor_factory=Anchor,**kwargs)
        policy.reset('e');return policy,calls,states

    def masks(self,peg):
        masks={c:np.zeros((2,16,16),bool) for c in ('base_camera','hand_camera')}
        for m in masks.values():
            m[1,8:12,8:12]=True
            if peg:m[0,:3,:3]=True
        return masks

    def test_no_reference_bootstrap_waits_for_past_confirmation(self):
        policy,calls,states=self.policy()
        try:
            for t in range(7):
                original=self.masks(False)
                out=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],original,
                    {('base_camera',1):True})
                self.assertEqual(bool(out['base_camera'][0].any()),t==6)
                self.assertFalse(original['base_camera'][0].any())
            self.assertEqual(len(calls),4)
            self.assertTrue(states[0].closed)
            self.assertTrue(policy.slots['base_camera',0].ever_found)
        finally:policy.close()
        self.assertTrue(all(s.closed for s in states))

    def test_healthy_masks_do_not_call_detector_or_change_output(self):
        policy,calls,states=self.policy()
        try:
            for t in range(12):
                original=self.masks(True)
                out=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],original,
                    {('base_camera',0):True,('hand_camera',0):True,('base_camera',1):True})
                for camera in original:np.testing.assert_array_equal(out[camera],original[camera])
            self.assertEqual(calls,[])
        finally:policy.close()

    def test_unverified_uncertainty_keeps_current_mask_until_replacement(self):
        policy,calls,states=self.policy(uncertain_output_policy='preserve_current')
        original_prepare=policy.geometry.system.prepare
        def unsupported(cameras,data,masks,surfaces):
            result=original_prepare(cameras,data,masks,surfaces)
            result['clean']['base_camera'][1]=False
            return result
        policy.geometry.system.prepare=unsupported
        try:
            for t in range(8):
                baseline=self.masks(True)
                out=policy.process(self.cameras(t),dict(episode='e',timestep=t),[],baseline,
                    {('base_camera',0):True,('hand_camera',0):True,('base_camera',1):True})
                for camera in baseline:np.testing.assert_array_equal(out[camera],baseline[camera])
            self.assertFalse(policy.slots['base_camera',1].currently_valid)
            self.assertTrue(policy.slots['base_camera',1].ever_found)
            self.assertTrue(calls)
            self.assertTrue(all(s.closed for s in states))
        finally:policy.close()


if __name__=='__main__':
    unittest.main()
