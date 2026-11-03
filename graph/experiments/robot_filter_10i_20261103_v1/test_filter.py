"""Boundary, scope, provenance and moving-frame cache regression contracts."""
import os,sys,unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
for env,fallback in [('TASK10I_REFERENCE_CODE',HERE.parent/'resolution_perception_graph_20261103_v1'),('TASK10I_LATENCY_CODE',HERE.parent/'graph_latency_10i_20261103_v2')]:
    sys.path.insert(0,os.environ.get(env,str(fallback)))
sys.path.insert(0,str(HERE))


@unittest.skipUnless(torch.cuda.is_available(),'CUDA server only; no CUDA execution on Mac')
class FilterContracts(unittest.TestCase):
    def setUp(self):
        from graph.plan_a_geometry.tests import Contracts
        fixture=Contracts();fixture.setUp();self.camera=fixture.wrist
        from graph.persistent_geometry.tests.test_contracts import cube
        from graph.persistent_geometry.self_filter import LinkSurface
        self.make=lambda a,b:LinkSurface('shape',cube(a,b),np.eye(4))
        self.surfaces=[self.make([-.15,-.2,1],[-.03,.2,1.1]),self.make([-.08,-.2,.9],[.1,.2,1.])]

    def check(self,camera,surfaces,masks,cache=None):
        from roi_filter import RobotFilter,filtered_masks
        from candidates import Candidate
        import graph.plan_a_geometry.core as core
        with Candidate(9) as ctx:expected=core.classify_robot(camera,surfaces)
        for variant in ('roi','hulls','packed','rays','atomic','strict'):
            f=RobotFilter(variant,cache);out=f.classify(camera,surfaces,masks,diagnostics=True)
            p=out.pixels
            for key,value in (('confirmed_robot',out.confirmed),('uncertain',out.uncertain)):
                np.testing.assert_array_equal(value,expected[key].ravel()[p],err_msg=variant+'/'+key)
            for key in ('candidate','link_owner','inside_known_link_volume'):
                np.testing.assert_array_equal(out.diagnostics[key],expected[key].ravel()[p],err_msg=variant+'/'+key)
            np.testing.assert_allclose(out.diagnostics['predicted_robot_depth'],expected['predicted_robot_depth'].ravel()[p],atol=1e-12,rtol=0)
            from graph.persistent_geometry.fusion import filtered_masks as original
            np.testing.assert_array_equal(filtered_masks(masks,out)[0],original(masks,expected)[0])
        return out

    def test_empty_full_overlap_image_edges_and_link_ties(self):
        h,w=self.camera.depth_raw_mm.shape
        rng=np.random.default_rng(581)
        for ss in ([],self.surfaces,self.surfaces[::-1],self.surfaces+self.surfaces):
            for mode in ('empty','full','overlap','sparse','edge'):
                masks=np.zeros((2,h,w),bool)
                if mode=='full':masks[0]=True
                if mode=='overlap':masks[:]=True
                if mode=='sparse':masks=rng.random(masks.shape)<.1
                if mode=='edge':masks[:,[0,-1],:]=True;masks[:,:,[0,-1]]=True
                self.check(self.camera,ss,masks)

    def test_depth_tolerance_uncertainty_and_half_mm_interior_boundaries(self):
        h,w=9,11;k=np.array([[50,0,w/2],[0,50,h/2],[0,0,1]],float)
        values=[np.nan,1.,1.+.0025,np.nextafter(1.+.0025,0),np.nextafter(1.+.0025,np.inf),
            1.-.006,np.nextafter(1.-.006,0),np.nextafter(1.-.006,np.inf),
            1.+.0005,np.nextafter(1.+.0005,0),np.nextafter(1.+.0005,np.inf),
            1.1-.0005,np.nextafter(1.1-.0005,0),np.nextafter(1.1-.0005,np.inf)]
        camera=SimpleNamespace(camera='boundary',depth_raw_mm=np.ones((h,w),np.int16),intrinsic=k,world_from_camera=np.eye(4))
        masks=np.zeros((2,h,w),bool);masks[0]=True
        surfaces=[self.make([-1,-1,1],[1,1,1.1])]
        for value in values:
            camera.depth=np.full((h,w),value)
            self.check(camera,surfaces,masks)

    def test_strict_high_resolution_atomic_ties_and_boundaries(self):
        h,w=129,131
        values=np.array([np.nan,1.,1.-.0025,1.-.006,1.+.0005,1.1-.0005])
        camera=SimpleNamespace(camera='high_boundary',depth_raw_mm=np.ones((h,w),np.int16),
            intrinsic=np.array([[500,0,w/2],[0,500,h/2],[0,0,1]],float),world_from_camera=np.eye(4),
            depth=np.resize(values,(h,w)))
        masks=np.zeros((2,h,w),bool);masks[0]=True
        first=self.make([-1,-1,1],[1,1,1.1])
        self.check(camera,[first,first,self.make([-.1,-.1,-.1],[.1,.1,.1])],masks)

    def test_invalid_raw_depth_and_front_back_crossing(self):
        raw=self.camera.depth_raw_mm.copy();raw.ravel()[::7]=-32768;raw.ravel()[1::7]=0;raw.ravel()[2::7]=-1
        camera=replace(self.camera,depth_raw_mm=raw)
        masks=np.zeros((2,*raw.shape),bool);masks[0]=True
        ss=[self.make([-.3,-.3,-.2],[.3,.3,.2]),self.make([-.2,-.2,-2],[.2,.2,-1])]
        self.check(camera,ss,masks)

    def test_cache_keys_intrinsics_resolution_pose_geometry_and_eviction(self):
        from roi_filter import RayCache
        cache=RayCache();camera=self.camera
        masks=np.ones((2,*camera.depth_raw_mm.shape),bool)
        for shift in (0.,.03,-.02):
            cv=camera.extrinsic_cv.copy();cv[0,3]+=shift
            c=replace(camera,extrinsic_cv=cv)
            self.check(c,self.surfaces,masks,cache)
        k=camera.intrinsic.copy()
        for scale in np.linspace(.8,1.3,11):
            k[0,0]=camera.intrinsic[0,0]*scale
            self.check(replace(camera,intrinsic=k),self.surfaces,masks,cache)
        self.assertLessEqual(len(cache.entries),8)
        smaller=replace(camera,rgb=camera.rgb[:-1,:-1],depth_raw_mm=camera.depth_raw_mm[:-1,:-1])
        self.check(smaller,self.surfaces,np.ones((2,*smaller.depth_raw_mm.shape),bool),cache)
        changed=[]
        for surface in self.surfaces:
            pose=surface.world_from_link.copy();pose[0,3]+=.1
            changed.append(replace(surface,world_from_link=pose))
        self.check(camera,changed,masks,cache)
        self.assertGreater(cache.hits,0);self.assertGreater(cache.misses,8)

    def test_private_scope_rejects_wrong_mask_union_and_restores_on_exception(self):
        from roi_filter import filtered_masks,RobotFilter
        from adapter import SparseGraph
        import graph.plan_a_geometry.core as core
        names=[(core.PlanA,'prepare'),(core,'classify_robot'),(core,'filtered_masks'),(core,'deduplicate')]
        before=[getattr(o,n) for o,n in names]
        with self.assertRaisesRegex(RuntimeError,'scope'):
            with SparseGraph('packed'):raise RuntimeError('scope')
        for (o,n),v in zip(names,before):self.assertIs(getattr(o,n),v)
        masks=np.zeros((2,*self.camera.depth_raw_mm.shape),bool);masks[0,0,0]=True
        result=RobotFilter().classify(self.camera,self.surfaces,masks)
        masks[0,0,1]=True
        with self.assertRaisesRegex(ValueError,'union'):filtered_masks(masks,result)


if __name__=='__main__':unittest.main()
