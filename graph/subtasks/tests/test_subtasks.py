import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from graph.subtasks.metrics import rotation,rotation_error,relative_pose,compose,compute_metrics
from graph.subtasks.specs import SubtaskSpec,load_config,NAMES
from graph.subtasks.conditions import ConditionContext
from graph.subtasks.segmentation import segment_episode,evaluate,runs
from graph.subtasks.dataset import STATE_KEYS,load_audit,state_row,save_segments,load_segments,segment_arrays


def fixture():
    T=15;half=np.array([.1,.02,.02]);local=np.array([-.06,0,.05,0,1,0,0]);grasp_local=local.copy();grasp_local[2]=0
    data={k:[] for k in STATE_KEYS}
    for t in range(T+1):
        peg=np.array([0.,0.,.02,1,0,0,0])
        if t>=7:peg[:3]=[-.4,.1,.12]
        if t>=8:peg[:3]=[-.25,.4,.12]
        if t>=10:peg[:3]=[-.21,.4,.12]
        if t>=12:peg[:3]=[-.19,.4,.12]
        if t>=14:peg[:3]=[-.1,.4,.12]
        ee=compose(peg,local if t<2 else grasp_local)
        if t==0:ee[2]+=.1
        if t==2:ee[2]+=.02
        live=t>=4
        row=dict(ee_pose=ee,peg_pose=peg,hole_pose=np.array([0.,.4,.12,1,0,0,0]),box_pose=np.array([0.,.4,.12,1,0,0,0]),
            peg_half_sizes=half,hole_radius=.023,gripper_qpos=np.array([.02,.02]) if live else np.array([.04,.04]),
            qpos=np.zeros(9),qvel=np.zeros(9),success=t>=14,grasp=live,contact_valid=t>0,
            finger_forces=np.ones((2,3)) if live else np.zeros((2,3)),peg_box_force=np.zeros(3))
        for k in STATE_KEYS:data[k].append(row[k])
    data={k:np.asarray(v) for k,v in data.items()};actions=np.zeros((T,7),np.float32);actions[:,-1]=1;actions[3:,-1]=-1
    data.update(actions=actions,reach_local=local,grasp_local=grasp_local,record=dict(episode_id=0,replay_valid=True))
    return data


class MetricsTests(unittest.TestCase):
    def test_relative_frames_and_wxyz(self):
        a=np.array([1,2,3,np.sqrt(.5),0,0,np.sqrt(.5)])
        b=compose(a,np.array([1,0,0,1,0,0,0]))
        np.testing.assert_allclose(b[:3],[1,3,3],atol=1e-7)
        p,r=relative_pose(a,b);np.testing.assert_allclose(p,[1,0,0],atol=1e-7);np.testing.assert_allclose(r,np.eye(3),atol=1e-7)
        self.assertAlmostEqual(rotation_error(rotation(a)),np.pi/2)
        b[3:]*=-1;np.testing.assert_allclose(relative_pose(a,b)[1],np.eye(3),atol=1e-7)
    def test_global_transform_invariance(self):
        d=fixture();s=state_row(d,10);c=load_config();m=compute_metrics(s,c)
        g=np.r_[.3,-.2,.1,Rotation.from_euler('z',.7).as_quat()[[3,0,1,2]]]
        for key in ('peg_pose','ee_pose','hole_pose','initial_peg_pose'):s[key]=compose(g,s[key])
        mm=compute_metrics(s,c)
        for key in ('position_error','rotation_error','transport_error','head_lateral','center_lateral','surface_clearance','insertion_depth'):
            self.assertAlmostEqual(m[key],mm[key],places=7)
    def test_head_and_entry_reference(self):
        m=compute_metrics(state_row(fixture(),10),load_config())
        np.testing.assert_allclose(m['head_in_hole'],[-.11,0,0],atol=1e-7)
        self.assertAlmostEqual(m['insertion_depth'],-.01);self.assertAlmostEqual(m['surface_clearance'],.003)
    def test_invalid_quaternion(self):
        with self.assertRaises(ValueError):rotation(np.zeros(7))
    def test_center_near_does_not_mean_align(self):
        s=state_row(fixture(),10);s['peg_pose']=s['peg_pose'].copy();s['peg_pose'][3:]=[np.cos(.2),0,0,np.sin(.2)]
        self.assertFalse(SubtaskSpec('align').align_raw(compute_metrics(s,load_config())))


class ConditionTests(unittest.TestCase):
    def setUp(self):self.d=fixture();self.c=load_config();self.ms,self.starts,self.suc,self.fail=evaluate(self.d,self.c)
    def test_every_start_success_failure(self):
        points={'reach':(0,1),'grasp':(1,6),'transport':(6,8),'align':(8,11),'insert':(11,14)}
        for name,(i,j) in points.items():
            self.assertTrue(self.starts[i][name],name);self.assertTrue(self.suc[j][name],name)
            spec=SubtaskSpec(name,self.c);ctx=ConditionContext(t=250)
            self.assertTrue(spec.check_failure(self.ms[i],ctx),name)
            self.assertFalse(spec.check_failure(self.ms[i],ConditionContext(t=0)),name)
    def test_contacts_not_proximity(self):
        self.assertFalse(self.suc[3]['grasp']);self.assertFalse(self.suc[4]['grasp']);self.assertFalse(self.suc[5]['grasp']);self.assertTrue(self.suc[6]['grasp'])
    def test_grasp_relative_pose_stability(self):
        d=fixture();d['ee_pose'][5,0]+=.01
        _,_,success,_=evaluate(d,self.c);self.assertFalse(success[6]['grasp'])
    def test_consecutive_and_episode_context(self):
        ctx=ConditionContext();ctx.update(self.ms[0],0,self.c)
        with self.assertRaises(ValueError):ctx.update(self.ms[2],2,self.c)
        with self.assertRaises(ValueError):ctx.update(self.ms[0],0,self.c)
        ctx=ConditionContext();ctx.update(self.ms[0],0,self.c);self.assertFalse(ctx.acquired)
    def test_drop_and_temporal_reset(self):
        d=fixture();d['grasp'][8:11]=False
        _,_,success,failure=evaluate(d,self.c)
        self.assertFalse(success[8]['grasp']);self.assertIn('grasp_lost',failure[10]['transport'])
        ctx=ConditionContext(acquired=True)
        unknown=dict(self.ms[0],contact_valid=False,grasp=False)
        for t in range(3):ctx.update(unknown,t,self.c)
        self.assertEqual(ctx.drop_count,0)
    def test_no_reach_after_acquisition(self):
        m=copy.deepcopy(self.ms[1]);self.assertFalse(SubtaskSpec('reach').check_success(m,ConditionContext(ever_grasped=True)))
        m['contact_valid']=False;self.assertFalse(SubtaskSpec('reach').check_success(m))
    def test_align_hold_and_config(self):
        self.assertFalse(self.suc[10]['align']);self.assertTrue(self.suc[11]['align'])
        c=dict(self.c,align_steps=1);_,_,s,_=evaluate(self.d,c);self.assertTrue(s[10]['align'])
    def test_displacement_and_workspace_failure(self):
        m=dict(self.ms[0],peg_motion=.1);self.assertTrue(SubtaskSpec('reach').check_failure(m))
        m=dict(self.ms[0],workspace_valid=False)
        for name in NAMES:self.assertTrue(SubtaskSpec(name).check_failure(m))


class SegmentationTests(unittest.TestCase):
    def test_order_boundaries_determinism(self):
        d=fixture();a,_,_=segment_episode(d,load_config());b,_,_=segment_episode(d,load_config());self.assertEqual(a,b)
        self.assertEqual([(s['start_t'],s['end_t']) for s in a],[(0,1),(1,6),(6,8),(8,11),(11,14)])
        self.assertTrue(all(s['status']=='detected' for s in a));self.assertTrue(a[0]['bc_eligible'])
    def test_missing_phase_no_forced_insert(self):
        d=fixture();d['peg_pose'][10:12,1]+=.02
        s,_,_=segment_episode(d,load_config())
        self.assertEqual(s[3]['status'],'unresolved');self.assertEqual(s[4]['status'],'unresolved')
        self.assertTrue(any(d['success']))
    def test_multiple_candidates(self):
        self.assertEqual(runs([False,True,True,False,True]),[[1,3],[4,5]])
        d=fixture();d['grasp'][7]=False;s,_,_=segment_episode(d,load_config());self.assertTrue(s[1]['multiple_candidates'])
    def test_no_close(self):
        d=fixture();d['actions'][:,-1]=1;s,_,_=segment_episode(d,load_config());self.assertTrue(all(v['status']=='unresolved' for v in s))
    def test_invalid_replay(self):
        d=fixture();d['record']['replay_valid']=False;s,_,_=segment_episode(d,load_config());self.assertTrue(all(v['status']=='unresolved' for v in s))
    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'index.json';payload={'episodes':[],'splits':{'train':[0]}}
            save_segments(p,payload);self.assertEqual(load_segments(p,verify_sources=False),payload)
    def test_audit_alignment(self):
        d=fixture();d['grasp']=d['grasp'][:-1];d.pop('record');d.update(replay_record=json.dumps({'episode_id':0}))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.npz';np.savez(p,**d)
            with self.assertRaises(ValueError):load_audit(p)
    def test_segment_cannot_cross_episode_boundary(self):
        d=fixture();d.pop('record');d['replay_record']=json.dumps({'episode_id':0})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audit.npz';np.savez(p,**d)
            dataset={'episodes':[{'episode_id':0,'audit_file':str(p)}]}
            for i,j in ((-1,2),(2,16),(4,4)):
                segment={'status':'detected','source_episode_id':0,'start_t':i,'end_t':j}
                with self.assertRaises(ValueError):segment_arrays(dataset,segment)


@unittest.skipUnless(os.environ.get('SUBTASK_INTEGRATION')=='1','set SUBTASK_INTEGRATION=1 after real replay/segmentation')
class ExpertIntegrationTests(unittest.TestCase):
    def test_fresh_live_replay_and_conditions(self):
        from graph.subtasks.audit import collect
        with tempfile.TemporaryDirectory() as tmp:
            collect('outputs/graph/dataset.json',tmp,[0])
            fresh=load_audit(Path(tmp)/'episode_0.npz')
            cached=load_audit('outputs/subtasks/audit/episode_0.npz')
            self.assertEqual(fresh['record']['max_observation_error'],0.)
            np.testing.assert_array_equal(fresh['grasp'],cached['grasp'])
            np.testing.assert_allclose(fresh['reach_local'],cached['reach_local'],atol=1e-7)
            self.assertEqual(segment_episode(fresh,load_config())[0],segment_episode(cached,load_config())[0])

    def test_real_all_episode_alignment_and_replay(self):
        data=load_segments('outputs/subtasks/segments/index.json')
        self.assertEqual({k:len(v) for k,v in data['splits'].items()},{'train':73,'validation':9,'test':9})
        self.assertEqual(len(data['episodes']),91)
        for ep in data['episodes']:
            self.assertTrue(ep['replay']['replay_valid']);self.assertTrue(ep['replay']['official_success']);self.assertLessEqual(ep['replay']['max_observation_error'],.001)
            d=load_audit(ep['audit_file']);ss,_,_=segment_episode(d,data['thresholds']);ss2,_,_=segment_episode(d,data['thresholds']);self.assertEqual(ss,ss2)
            for s in ep['segments']:
                if s['status']=='detected':
                    arrays=segment_arrays(data,s);i,j=s['start_t'],s['end_t']
                    self.assertEqual(len(arrays['states']),len(arrays['actions'])+1)
                    np.testing.assert_array_equal(arrays['actions'],d['actions'][i:j])
                    np.testing.assert_array_equal(arrays['geometry']['peg_pose'],d['peg_pose'][i:j+1])
    def test_rendered_prefix_boundary_replay(self):
        result=json.loads(Path('outputs/subtasks/figures/boundary_replay.json').read_text())
        self.assertGreaterEqual(len(result['checks']),5)
        for check in result['checks']:
            self.assertTrue(check['contact_matches']);self.assertLessEqual(check['max_observation_error'],.001)

if __name__=='__main__':unittest.main()
