"""Thread ordering/lifecycle on CPU, exact CUDA filter results when available."""
from dataclasses import replace
from threading import Event,current_thread
from unittest.mock import patch
import unittest
import numpy as np
import torch
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.tests.test_contracts import cube
from .preparation import RobotFilterPreparation


class ControlledFilter:
    def __init__(self):self.started=Event();self.release=Event();self.completed=Event();self.thread=None;self.calls=0
    def prepare_surfaces(self,surfaces,cameras):
        self.thread=current_thread().name;self.started.set()
        if not self.release.wait(3):raise TimeoutError('test worker release')
        result=dict(vertices=[s.world_vertices.copy() for s in surfaces],k=cameras[0].intrinsic.copy())
        self.completed.set();return result
    def filter_cameras(self,cameras,surfaces,masks,*,prepared):
        self.calls+=1;self.prepared=prepared;return masks


class PreparationContracts(unittest.TestCase):
    def setUp(self):
        from graph.plan_a_geometry.tests import Contracts
        f=Contracts();f.setUp();self.camera=f.wrist
        self.surfaces=[LinkSurface('robot',cube([-.15,-.2,.1],[.1,.2,.12]),np.eye(4))]

    def test_snapshot_bounded_queue_single_use_and_close(self):
        filt=ControlledFilter()
        with RobotFilterPreparation(filt) as prep:
            c=self.camera;original_k=c.intrinsic.copy();vertices=self.surfaces[0].world_vertices.copy()
            job=prep.submit((c,),self.surfaces);self.assertTrue(filt.started.wait(3))
            self.assertNotEqual(filt.thread,current_thread().name)
            with self.assertRaises(RuntimeError):prep.submit((c,),self.surfaces)
            c.intrinsic[0,0]*=2;c.depth_raw_mm[:]=-32768;self.surfaces[0].world_from_link[0,3]=10
            filt.release.set();masks={c.camera:np.zeros((2,24,24),bool)}
            self.assertIs(prep.filter(job,masks),masks)
            np.testing.assert_array_equal(filt.prepared['k'],original_k)
            np.testing.assert_array_equal(filt.prepared['vertices'][0],vertices)
            self.assertTrue((job.cameras[0].depth_raw_mm>0).all())
            with self.assertRaises(ValueError):prep.filter(job,masks)
        prep.close()
        with self.assertRaises(RuntimeError):prep.submit((c,),self.surfaces)

    def test_image_bridge_starts_preparation_before_sam2_and_recovers_errors(self):
        from .sam2_bridge import image_clean_masks_cuda
        filt=ControlledFilter();c=self.camera;raw=np.zeros((2,24,24),bool)
        with RobotFilterPreparation(filt) as prep:
            def sam(*_):
                self.assertTrue(filt.started.wait(3));self.assertFalse(filt.release.is_set())
                filt.release.set();self.assertTrue(filt.completed.wait(3));return raw,[None,None]
            with patch(__package__+'.sam2_bridge.image_masks_cuda',side_effect=sam):
                actual,_=image_clean_masks_cuda(None,c.rgb,[None,None],c,self.surfaces,preparation=prep)
            self.assertIs(actual,raw);self.assertEqual(filt.calls,1)
            def fail(*_):raise RuntimeError('SAM2 failed')
            with patch(__package__+'.sam2_bridge.image_masks_cuda',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'SAM2 failed'):
                    image_clean_masks_cuda(None,c.rgb,[None,None],c,self.surfaces,preparation=prep)
            # Failed inference does not leave a stale job for the next frame.
            job=prep.submit((replace(c,timestep=1),),self.surfaces);prep.discard(job)

    def test_preparation_error_reaches_caller_and_next_job_is_allowed(self):
        filt=ControlledFilter();filt.release.set()
        with RobotFilterPreparation(filt) as prep:
            with patch.object(filt,'prepare_surfaces',side_effect=ValueError('hull failed')):
                job=prep.submit((self.camera,),self.surfaces)
                with self.assertRaisesRegex(ValueError,'hull failed'):prep.filter(job,{})
            job=prep.submit((self.camera,),self.surfaces);prep.discard(job)

    def test_tracking_prepares_before_each_forward_frame_and_cleans_up(self):
        from visual.runtime.tracking import SAM2Tracker
        events=[]
        class Hook:
            def prepare_frame(self,i):events.append(('prepare',i))
            def __call__(self,i,m):events.append(('filter',i));return m
            def cancel_prepared(self):events.append(('cancel',None))
        class Predictor:
            def propagate_in_video(self,state,start_frame_idx):
                try:
                    for i in range(start_frame_idx,state['num_frames']):
                        self_check=events[-1]==('prepare',i)
                        if not self_check:raise AssertionError('inference before preparation submission')
                        events.append(('sam2',i));yield i,[1],torch.ones((1,1,2,2))
                finally:events.append(('predictor_closed',None))
        tracker=SAM2Tracker(predictor=Predictor());hook=Hook()
        self.assertEqual(len(list(tracker.track({'num_frames':2},mask_filter=hook))),2)
        self.assertEqual(events[:6],[('prepare',0),('sam2',0),('filter',0),('prepare',1),('sam2',1),('filter',1)])
        self.assertIn(('cancel',None),events)
        events.clear();it=tracker.track({'num_frames':2},mask_filter=hook);next(it);it.close()
        self.assertIn(('cancel',None),events);self.assertIn(('predictor_closed',None),events)
        class WrongFrame:
            def propagate_in_video(self,state,start_frame_idx):yield 1,[1],torch.ones((1,1,2,2))
        with self.assertRaisesRegex(ValueError,'consecutive'):
            list(SAM2Tracker(predictor=WrongFrame()).track({'num_frames':2},mask_filter=hook))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on current host')
class CudaPreparationContracts(unittest.TestCase):
    setUp=PreparationContracts.setUp
    def test_prepared_batch_matches_serial_every_field_and_pose_refresh(self):
        from .gpu_filter import CudaRobotFilter
        base=self.camera;filt=CudaRobotFilter()
        with RobotFilterPreparation(filt) as prep:
            for h,w in ((24,24),(129,131)):
                k=base.intrinsic.copy();k[:2,2]=[w/2,h/2]
                c=replace(base,rgb=np.zeros((h,w,3),np.uint8),depth_raw_mm=np.full((h,w),100,np.int16),intrinsic=k)
                for offset in (0.,10.,0.):
                    self.surfaces[0].world_from_link[0,3]=offset
                    cameras=(c,replace(c,camera='base_camera'))
                    raw={x.camera:torch.ones((2,h,w),device='cuda',dtype=torch.bool) for x in cameras}
                    job=prep.submit(cameras,self.surfaces)
                    serial=filt.prepare_surfaces(self.surfaces,cameras)
                    for (_,a),(_,b) in zip(job.future.result().hulls,serial.hulls):np.testing.assert_array_equal(a,b)
                    actual=prep.filter(job,raw,diagnostics=True)
                    expected=filt.filter_cameras(cameras,self.surfaces,raw,diagnostics=True)
                    for name in raw:
                        for key in actual[name]:self.assertTrue(torch.equal(actual[name][key],expected[name][key]),key)

    def test_real_cuda_filter_image_and_tracking_bridge(self):
        from .sam2_bridge import images_clean_masks_cuda,SAM2RobotFilter
        from .gpu_filter import CudaRobotFilter
        from visual.runtime.tracking import SAM2Tracker
        c=self.camera;h,w=c.depth_raw_mm.shape;filt=CudaRobotFilter()
        class Predictor:
            device=torch.device('cuda:0')
            def set_image(self,rgb):pass
            def _prep_prompts(self,*args):return None,None,None,args[2]
            def _predict(self,*args,**kwargs):return torch.ones((1,1,h,w),device='cuda'),torch.ones((1,1),device='cuda'),None
        cameras=(c,replace(c,camera='base_camera'));names=[x.camera for x in cameras]
        with RobotFilterPreparation(filt) as prep:
            actual,scores=images_clean_masks_cuda({n:Predictor() for n in names},{n:c.rgb for n in names},
                {n:[[0,0,w,h],None] for n in names},cameras,self.surfaces,preparation=prep)
            raw={n:torch.stack((torch.ones((h,w),device='cuda',dtype=torch.bool),torch.zeros((h,w),device='cuda',dtype=torch.bool))) for n in names}
            expected=filt.filter_cameras(cameras,self.surfaces,raw)
            for name in names:self.assertTrue(torch.equal(actual[name],expected[name]))
        seen=[]
        class Video:
            def propagate_in_video(self,state,start_frame_idx):
                for i in range(state['num_frames']):
                    if seen!=list(range(i+1)):raise AssertionError('observation prepared too late')
                    yield i,[1],torch.ones((1,1,h,w),device='cuda')
        def current(i):seen.append(i);return replace(c,timestep=i),self.surfaces
        with SAM2RobotFilter(current,filt) as callback:
            result=list(SAM2Tracker(predictor=Video()).track({'num_frames':2},mask_filter=callback))
        self.assertEqual([i for i,_ in result],[0,1]);self.assertEqual(seen,[0,1])
