"""Default current-geometry scheduling and the actual cache/live entrypoints."""
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch,Mock
import json,tempfile,unittest
import h5py
import numpy as np
import torch
from graph.integration_baseline.graph import Graph
from graph.plan_a_geometry import tests as geometry_fixtures
from .geometry import CurrentGeometry


def empty_graph():
    return Graph(torch.zeros((3,21)),torch.arange(3),torch.empty((2,0),dtype=torch.int64),
        torch.empty((0,4)),torch.empty(0,dtype=torch.int64),dict(node_ids=[],coordinate_frame='EE_metres'),{})


class DefaultRobotOverlap(unittest.TestCase):
    def test_cache_identity_records_schedule_and_rejects_old_cache(self):
        from .cache import input_identity,CachedDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'config.json').write_text(json.dumps(dict(geometry={},source_sha256={})))
            (root/'dataset_audit.json').write_text(json.dumps(dict(dataset_sha256='dataset',split_sha256='split')))
            ident=input_identity(root)
            self.assertEqual(ident['robot_filter_backend'],'cuda');self.assertEqual(ident['robot_preparation'],'before_vision')
            self.assertEqual(ident['grounding']['primary'],'gpt-6.1-terra')
            self.assertEqual(ident['cache_version'],'graphs_common_coordinate_frame_v4_terra_primary')
            for version in ('graphs_common_coordinate_frame_v2','graphs_common_coordinate_frame_v3_robot_overlap'):
                old={**ident,'cache_version':version}
                (root/'cache_manifest.json').write_text(json.dumps(dict(identity=old,episodes=[])))
                with self.assertRaisesRegex(ValueError,'identity mismatch'):CachedDataset(root)

    def test_default_submits_before_vision_and_discards_on_vision_error(self):
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation import RobotFilterPreparation
        fixture=geometry_fixtures.Contracts();fixture.setUp();started=Event()
        class Filter:
            def prepare_surfaces(self,*_):started.set();return None
        with patch('graph.plan_a_bc.geometry.RobotFK'):
            geometry=CurrentGeometry('unused',{})
        self.assertEqual(geometry.robot_filter_backend,'cuda')
        geometry._check_mount=lambda *args:None
        geometry.robot_preparation=RobotFilterPreparation(Filter())
        def vision():
            self.assertTrue(started.wait(3));raise RuntimeError('vision failure')
        with self.assertRaisesRegex(RuntimeError,'vision failure'):
            geometry.build(fixture.cameras,fixture.data,[],vision)
        self.assertIsNone(geometry.robot_preparation._pending)
        self.assertEqual(geometry.last_robot_schedule,'before_vision');geometry.close()

    def test_preclean_path_preserves_raw_gate_and_never_filters_twice(self):
        fixture=geometry_fixtures.Contracts();fixture.setUp();system=fixture.system
        reference=system.prepare(fixture.cameras,fixture.data,fixture.masks,[])
        with patch('graph.plan_a_geometry.core.classify_robot',side_effect=AssertionError('second filtering')):
            actual=system.prepare(fixture.cameras,fixture.data,fixture.masks,[],clean_masks=reference['clean'])
        self.assertEqual(actual['gate'],reference['gate']);self.assertEqual(actual['gate']['raw_pixels'],36)
        self.assertTrue(all(v is None for v in actual['filters'].values()))
        for condition in actual['conditions']:
            for name in fixture.masks:np.testing.assert_array_equal(actual['conditions'][condition][name],reference['conditions'][condition][name])
        bad={k:v.copy() for k,v in reference['clean'].items()};bad['base_camera'][0,0,0]=True
        with self.assertRaisesRegex(ValueError,'subset'):
            system.prepare(fixture.cameras,fixture.data,fixture.masks,[],clean_masks=bad)

    def test_real_cache_entrypoint_defers_vision_and_uses_cuda_masks(self):
        from . import cache
        events=[];test=self
        class Adapter:
            def __init__(self,*_):self.states={};self.found={};self.rgb=[];self.records=[]
            def reset(self,*_):pass
            def process(self,rgb,identity,t,*,mask_device):
                test.assertEqual(events[-1],'inside_build');test.assertEqual(mask_device,'cuda')
                return np.zeros((2,4,4),bool),{}
            def close(self):pass
        class Geometry:
            coordinate_frame='ee';robot_filter_backend='cuda';last_robot_schedule='before_vision'
            def __init__(self,*args,**kwargs):pass
            def from_h5(self,f,t,identity):return [SimpleNamespace(camera=n,rgb=np.zeros((4,4,3),np.uint8)) for n in ('base_camera','hand_camera')],dict(robot76=np.zeros(76)),[]
            def build(self,cameras,data,surfaces,provider):
                test.assertTrue(callable(provider));events.append('inside_build')
                self.last_raw_masks=provider();return empty_graph(),[],dict(accepted=False)
            def close(self):events.append('closed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset=root/'dataset';dataset.mkdir()
            with h5py.File(dataset/'ep.h5','w') as f:f['actions']=np.zeros((1,7),np.float32)
            (root/'normalizer.json').write_text('{"schema":{}}')
            entry=dict(episode_id=1,path='ep.h5',sha256=cache.sha(dataset/'ep.h5'),actions=1,seed=1,split='train')
            with patch.object(cache,'CurrentGeometry',Geometry),patch.object(cache,'OnlineVisionAdapter',Adapter), \
                    patch.object(cache,'input_identity',return_value=dict(coordinate_frame='ee')):
                result=cache.cache_episode(root,dataset,entry,None,None,Mock())
            self.assertEqual(result['robot_preparation'],'before_vision');self.assertEqual(events,['inside_build','closed'])

    def test_real_live_entrypoint_defers_vision_and_uses_cuda_masks(self):
        from . import rollout
        test=self;events=[];fixture=geometry_fixtures.Contracts();fixture.setUp()
        data={**fixture.data,'robot76':np.zeros(76,np.float32)}
        class Adapter:
            def __init__(self,*_):self.records=[]
            def reset(self,*_):pass
            def process(self,rgb,identity,t,*,mask_device):
                test.assertEqual(events[-1],'inside_build');test.assertEqual(mask_device,'cuda')
                return np.zeros((2,*rgb.shape[:2]),bool),{}
            def close(self):pass
        class Geometry:
            robot_filter_backend='cuda';last_robot_schedule='before_vision'
            def __init__(self,*args,**kwargs):pass
            def build(self,cameras,data,surfaces,provider):
                test.assertTrue(callable(provider));events.append('inside_build')
                self.last_raw_masks=provider();return empty_graph(),[],dict(accepted=False)
            def close(self):events.append('closed')
        model=Mock(variant='fixture');model.parameters.side_effect=lambda:iter([torch.zeros(1)])
        model.step.return_value=(torch.zeros((1,7)),torch.zeros((1,1)))
        env=Mock();env.unwrapped.device=torch.device('cpu');env.reset.return_value=({}, {})
        env.step.return_value=({},None,torch.tensor([False]),torch.tensor([False]),dict(success=torch.tensor([True])))
        metric=dict(reach=True,grasp=False,lift_candidate=False,lift_height_m=0.,alignment=False,success=True,
            peg_pose=[0]*7,box_pose=[0]*7,hole_pose=[0]*7,gripper_qpos=[0]*2)
        codec=Mock();codec.decode.return_value=torch.zeros((1,7));writer=Mock();writer.isOpened.return_value=True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config.json').write_text('{"geometry":{},"evaluation":{"horizon":1,"smoke_horizon":1,"lift_consecutive_steps":2}}')
            def video(path,*args):Path(path).write_bytes(b'fixture');return writer
            with patch.object(rollout,'CurrentGeometry',Geometry),patch.object(rollout,'OnlineVisionAdapter',Adapter), \
                    patch.object(rollout,'environment',return_value=env),patch.object(rollout,'reset_diagnostics',return_value={}), \
                    patch.object(rollout,'sensors',return_value=(fixture.cameras,data,[])), \
                    patch.object(rollout,'PhysicsAudit',return_value=Mock(count=1)),patch.object(rollout,'controller_audit'), \
                    patch.object(rollout,'get_gt_state',return_value=SimpleNamespace(peg_pose=torch.zeros((1,7)))), \
                    patch.object(rollout,'validate_controller_command'),patch.object(rollout,'evaluation_measure',return_value=metric), \
                    patch.object(rollout,'video_frame',return_value=np.zeros((4,4,3),np.uint8)),patch.object(rollout.cv2,'VideoWriter',side_effect=video):
                rollout.run_episode(root,root/'episode',model,SimpleNamespace(schema={}),codec,None,None,dict(seed=1,group='fixture'),0,smoke=True)
            self.assertEqual(events,['inside_build','closed'])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA host required')
class CudaDefaultRobotOverlap(unittest.TestCase):
    def test_default_geometry_matches_serial_robot_and_keeps_raw_masks(self):
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        fixture=geometry_fixtures.Contracts();fixture.setUp();started=Event()
        surfaces=[LinkSurface('camera_link',cube([-.02,-.1,.099],[-.006,.1,.105]),np.eye(4))]
        raw=fixture.masks
        for frame in ('world','ee'):
            with patch('graph.plan_a_bc.geometry.RobotFK'):
                candidate=CurrentGeometry('unused',{},coordinate_frame=frame)
                reference=CurrentGeometry('unused',{},coordinate_frame=frame,robot_filter_backend='cpu')
            try:
                expected,er,eg=reference.build(fixture.cameras,fixture.data,surfaces,raw)
                candidate.build(fixture.cameras,fixture.data,surfaces,raw)
                original=candidate.robot_preparation.robot_filter.prepare_surfaces
                def prepare(*args):started.set();return original(*args)
                device_raw={k:torch.as_tensor(v,device='cuda') for k,v in raw.items()}
                def sam2():
                    self.assertTrue(started.wait(3));return device_raw
                with patch.object(candidate.robot_preparation.robot_filter,'prepare_surfaces',side_effect=prepare), \
                        patch('graph.plan_a_geometry.core.classify_robot',side_effect=AssertionError('second robot filter')):
                    actual,ar,ag=candidate.build(fixture.cameras,fixture.data,surfaces,sam2)
                self.assertEqual(candidate.last_robot_schedule,'before_vision');self.assertEqual(ag,eg)
                for name in raw:np.testing.assert_array_equal(candidate.last_raw_masks[name],raw[name])
                for key in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                    a,b=getattr(actual,key),getattr(expected,key)
                    if key=='node_input':
                        torch.testing.assert_close(a[:,:3],b[:,:3],rtol=0,atol=1e-6)
                        torch.testing.assert_close(a[:,3:8],b[:,3:8],rtol=0,atol=2e-5)
                        torch.testing.assert_close(a[:,8:],b[:,8:],rtol=0,atol=0)
                    else:torch.testing.assert_close(a,b,rtol=0,atol=1e-6 if a.is_floating_point() else 0)
                from graph.persistent_geometry.tensor_records import numpy_records
                self.assertTrue(actual.node_input.is_cuda)
                ar=numpy_records(ar);er=numpy_records(er)
                self.assertEqual([r['source_observation_ids'] for r in ar],[r['source_observation_ids'] for r in er])
                from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.graph_boundary import GpuMaskBoundary
                def twice(*args):raise AssertionError('legacy scope filtered a second time')
                with GpuMaskBoundary(twice) as capture:
                    again,_,_=candidate.build(fixture.cameras,fixture.data,surfaces,device_raw)
                    self.assertEqual(set(capture.clean_cuda),set(raw))
                torch.testing.assert_close(again.node_input,actual.node_input,rtol=0,atol=2e-5)
            finally:candidate.close();reference.close();started.clear()
