"""GPU mask ownership, unchanged decisions, no host readback, SAM2 connections."""
import unittest
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NoReadback(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        name=str(func)
        if name in ('aten._local_scalar_dense.default','aten.nonzero.default'):
            raise AssertionError('device scalar synchronization: '+name)
        if name=='aten._to_copy.default' and str(kwargs.get('device','')).startswith('cpu'):
            raise AssertionError('unexpected D2H copy')
        return func(*args,**kwargs)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on this host')
class GPUContracts(unittest.TestCase):
    def setUp(self):
        from graph.plan_a_geometry.tests import Contracts
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        f=Contracts();f.setUp();self.camera=f.wrist
        self.surfaces=[LinkSurface('a',cube([-.15,-.2,1],[-.03,.2,1.1]),np.eye(4)),
            LinkSurface('b',cube([-.08,-.2,.9],[.1,.2,1.]),np.eye(4))]

    def compare(self,camera,surfaces,raw):
        from .gpu_filter import CudaRobotFilter
        from graph.persistent_geometry.self_filter import classify_robot
        from graph.persistent_geometry.fusion import filtered_masks
        from graph.experiments.resolution_perception_graph_20261103_v1.gpu_graph import OptimizedGraph
        from graph.experiments.graph_latency_10i_20261103_v2 import robot
        context=OptimizedGraph('cuda_fused_v5')
        expected=(robot.classify(context,camera,surfaces) if camera.depth_raw_mm.size<=128*128 else context.classify_robot(camera,surfaces))
        source=torch.as_tensor(raw,device='cuda');saved=source.clone()
        filt=CudaRobotFilter()
        actual=filt(camera,surfaces,source,diagnostics=True)
        self.assertTrue(actual['clean_masks'].is_cuda)
        np.testing.assert_array_equal(actual['clean_masks'].cpu().numpy(),filtered_masks(raw,expected)[0])
        valid=actual['evaluated_pixels'].cpu().numpy()
        for k in ('confirmed_robot','uncertain','candidate','link_owner','inside_known_link_volume'):
            np.testing.assert_array_equal(actual[k].cpu().numpy()[valid],expected[k][valid],err_msg=k)
        np.testing.assert_allclose(actual['predicted_robot_depth'].cpu().numpy()[valid],expected['predicted_robot_depth'][valid],atol=1e-12,rtol=0)
        # Warm all kernels before asserting there is no tensor readback/sync.
        filt(camera,surfaces,source)
        with NoReadback(),patch.object(torch.cuda,'synchronize',side_effect=AssertionError('host sync')):
            clean=filt(camera,surfaces,source)
        self.assertTrue(clean.is_cuda);self.assertEqual(clean.dtype,torch.bool)
        self.assertNotEqual(clean.data_ptr(),source.data_ptr())
        self.assertTrue(torch.equal(source,saved));self.assertTrue(torch.equal(clean,actual['clean_masks']))

    def test_empty_full_overlap_invalid_depth_and_link_order(self):
        camera=self.camera;raw=np.zeros((2,*camera.depth_raw_mm.shape),bool)
        for ss in ([],self.surfaces,self.surfaces[::-1],self.surfaces*2):
            for mode in ('empty','full','overlap','edges'):
                m=raw.copy()
                if mode=='full':m[0]=True
                if mode=='overlap':m[:]=True
                if mode=='edges':m[:,[0,-1],:]=True;m[:,:,[0,-1]]=True
                self.compare(camera,ss,m)
        mm=camera.depth_raw_mm.copy();mm.ravel()[::3]=-32768;mm.ravel()[1::3]=0
        self.compare(replace(camera,depth_raw_mm=mm),self.surfaces,np.ones_like(raw))

    def test_high_resolution_tolerance_interior_and_camera_crossing(self):
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        h,w=129,131
        values=[np.nan,1.,1.-.0025,1.-.006,1.+.0005,1.1-.0005]
        c=SimpleNamespace(camera='boundary',depth_raw_mm=np.ones((h,w),np.int16),
            intrinsic=np.array([[500,0,w/2],[0,500,h/2],[0,0,1]],float),world_from_camera=np.eye(4),depth=np.resize(values,(h,w)))
        ss=[LinkSurface('tie',cube([-1,-1,1],[1,1,1.1]),np.eye(4))]
        for ss in (ss,ss*2,ss+[LinkSurface('cross',cube([-.1,-.1,-.1],[.1,.1,.1]),np.eye(4))]):
            self.compare(c,ss,np.ones((2,h,w),bool))
        # Current pose/extrinsic is recomputed on every call.
        moved=c.world_from_camera.copy();moved[0,3]=.03
        self.compare(SimpleNamespace(**{**vars(c),'world_from_camera':moved}),ss,np.stack((np.ones((h,w),bool),np.zeros((h,w),bool))))

    def test_compaction_order_capacity_and_tiles_preserve_dense_decisions(self):
        from .gpu_filter import CudaRobotFilter
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        h,w=129,131
        c=SimpleNamespace(camera='compact',depth_raw_mm=np.ones((h,w),np.int16),
            intrinsic=np.array([[500,0,w/2],[0,500,h/2],[0,0,1]],float),world_from_camera=np.eye(4),
            depth=np.resize([np.nan,1.,1.-.0025,1.-.006,1.+.0005,1.1-.0005],(h,w)))
        surface=LinkSurface('tie',cube([-1,-1,1],[1,1,1.1]),np.eye(4))
        surfaces=[surface,surface,LinkSurface('cross',cube([-.1,-.1,-.1],[.1,.1,.1]),np.eye(4))]
        rng=np.random.default_rng(7)
        for mode in ('empty','sparse','full'):
            mask=(rng.random((2,h,w))<.025 if mode=='sparse' else np.full((2,h,w),mode=='full'))
            mask[1]=False;raw=torch.as_tensor(mask,device='cuda')
            expected=CudaRobotFilter(compact=False)(c,surfaces,raw,diagnostics=True)
            for pixels in (8,16,32,64):
                filt=CudaRobotFilter(clip_pixels=pixels)
                for _ in range(3):
                    with NoReadback():actual=filt(c,surfaces,raw,diagnostics=True)
                    for field in expected:self.assertTrue(torch.equal(actual[field],expected[field]),field)

    def test_gpu_depth_input_noncontiguous_masks_and_bad_contracts(self):
        from .gpu_filter import CudaRobotFilter
        f=CudaRobotFilter();c=self.camera;h,w=c.depth_raw_mm.shape
        raw=torch.ones((2,w,h),device='cuda',dtype=torch.bool).transpose(1,2)
        a=f(c,self.surfaces,raw)
        b=f(c,self.surfaces,raw,depth_cuda=torch.as_tensor(c.depth,device='cuda'))
        self.assertTrue(torch.equal(a,b))
        for masks in (raw.cpu(),raw.float(),raw[:1]):
            with self.assertRaisesRegex(ValueError,'CUDA bool'):f(c,self.surfaces,masks)
        with self.assertRaisesRegex(ValueError,'float64'):f(c,self.surfaces,raw,depth_cuda=b[0].float())

    def test_signed_mm_gpu_conversion_exhaustive_and_sensor_refresh(self):
        from .gpu_filter import CudaRobotFilter,_depth_mm
        # Every signed int16 value, including -32768 overflow and zero.
        mm=np.arange(-32768,32768,dtype=np.int32).astype(np.int16).reshape(256,256)
        expected=np.where(mm>0,mm.astype(np.float64)/1000,np.nan)
        raw=torch.as_tensor(mm,device='cuda');filt=CudaRobotFilter();table=filt._depth_table(raw.device)
        actual=torch.empty((256,256),device='cuda',dtype=torch.float64)
        with NoReadback():_depth_mm[(256,)](raw,table,actual,65536)
        host=actual.cpu().numpy();valid=mm>0
        np.testing.assert_array_equal(np.isnan(host),np.isnan(expected))
        np.testing.assert_array_equal(host[valid].view(np.uint64),expected[valid].view(np.uint64))
        self.assertEqual(table.data_ptr(),filt._depth_table(raw.device).data_ptr())
        # Live signed sensor data changes every call; no converted-depth cache.
        c=self.camera;source=torch.as_tensor(np.stack((np.ones(c.depth_raw_mm.shape,bool),np.zeros(c.depth_raw_mm.shape,bool))),device='cuda')
        fast=CudaRobotFilter();cpu=CudaRobotFilter(depth_from_mm=False)
        for value in (100,1000,0,-32768,1100):
            current=replace(c,depth_raw_mm=np.full(c.depth_raw_mm.shape,value,np.int16))
            a=fast(current,self.surfaces,source,diagnostics=True)
            b=cpu(current,self.surfaces,source,diagnostics=True)
            for field in a:self.assertTrue(torch.equal(a[field],b[field]),field)
        def fail_cpu_depth(_):raise AssertionError('CPU float64 depth conversion used')
        with patch.object(type(c),'depth',property(fail_cpu_depth)),NoReadback():
            fast(c,self.surfaces,source)

    def test_batch_prepares_current_pose_once_and_refreshes_next_frame(self):
        from . import gpu_filter
        from graph.persistent_geometry.self_filter import LinkSurface
        from graph.persistent_geometry.tests.test_contracts import cube
        f=gpu_filter.CudaRobotFilter();c=self.camera
        second=replace(c,camera='second')
        raw={x.camera:torch.ones((2,*x.depth_raw_mm.shape),device='cuda',dtype=torch.bool) for x in (c,second)}
        # These matrices may be edited in place by a live FK producer. The batch
        # must share only within this invocation, then prepare the updated pose.
        surfaces=[LinkSurface('moving',cube([-.15,-.2,1],[.1,.2,1.1]),np.eye(4))]
        with patch.object(gpu_filter,'ConvexHull',wraps=gpu_filter.ConvexHull) as hull:
            for offset in (0.,10.,0.):
                surfaces[0].world_from_link[0,3]=offset
                count=hull.call_count
                with NoReadback():actual=f.filter_cameras((c,second),surfaces,raw,diagnostics=True)
                self.assertEqual(hull.call_count-count,len(surfaces) if offset==0. else 0)
                for cam in (c,second):
                    expected=f(cam,surfaces,raw[cam.camera],diagnostics=True)
                    for key in ('clean_masks','predicted_robot_depth','link_owner','inside_known_link_volume'):
                        self.assertTrue(torch.equal(actual[cam.camera][key],expected[key]))
        with self.assertRaisesRegex(ValueError,'camera identity'):f.filter_cameras((c,c),surfaces,raw)
        # Cached CV rays must also refresh when calibration changes in place.
        changed=replace(c,intrinsic=c.intrinsic.copy())
        for focal_scale in (1.,2.,1.):
            changed.intrinsic[:2,:2]=c.intrinsic[:2,:2]*focal_scale
            actual=f(changed,surfaces,raw[c.camera],diagnostics=True)
            expected=gpu_filter.CudaRobotFilter()(changed,surfaces,raw[c.camera],diagnostics=True)
            for key in ('clean_masks','predicted_robot_depth','link_owner'):
                self.assertTrue(torch.equal(actual[key],expected[key]))

    def test_sam2_image_and_tracking_deliver_gpu_clean_masks(self):
        from .sam2_bridge import image_masks_cuda,image_clean_masks_cuda,SAM2RobotFilter
        from .gpu_filter import CudaRobotFilter
        from visual.runtime.tracking import SAM2Tracker
        c=self.camera;h,w=c.depth_raw_mm.shape
        class ImagePredictor:
            device=torch.device('cuda:0')
            def set_image(self,rgb):self.rgb=rgb
            def _prep_prompts(self,*args):
                assert args[-1] is True
                return None,None,None,args[2]
            def _predict(self,coords,labels,bb,mi,multi,return_logits):
                assert multi is False and return_logits is True
                return torch.ones((1,1,h,w),device='cuda'),torch.ones((1,1),device='cuda'),None
            def predict(self,*a,**kw):raise AssertionError('CPU public SAM2 predict called')
        pred=ImagePredictor();boxes=[[0,0,w,h],None]
        raw,scores=image_masks_cuda(pred,c.rgb,boxes)
        self.assertTrue(raw.is_cuda);self.assertTrue(scores[0].is_cuda);self.assertIsNone(scores[1])
        self.assertFalse(raw[1].any())
        expected=CudaRobotFilter()(c,self.surfaces,raw)
        clean,_=image_clean_masks_cuda(pred,c.rgb,boxes,c,self.surfaces)
        self.assertTrue(torch.equal(clean,expected))
        class VideoPredictor:
            def propagate_in_video(self,state,start_frame_idx):
                yield start_frame_idx,[2,1],torch.ones((2,1,h,w),device='cuda')
        tracker=SAM2Tracker(predictor=VideoPredictor());seen=[]
        def observation(i):seen.append(i);return c,self.surfaces
        with NoReadback():
            index,tracked=next(tracker.track({},start_frame=7,mask_filter=SAM2RobotFilter(observation)))
        self.assertEqual(index,7);self.assertEqual(seen,[7]);self.assertEqual(set(tracked),{1,2})
        self.assertTrue(all(m.is_cuda and m.dtype==torch.bool for m in tracked.values()))


if __name__=='__main__':unittest.main()
