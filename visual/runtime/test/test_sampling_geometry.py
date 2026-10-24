"""Cuboid edge and clustered-candidate spatial-dispersion tests."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from visual.runtime.sampling import SamplingConfig, extract_sampling_evidence, sample_points, fps_indices
from visual.runtime.test.normal_diagnostics import synthetic_frame, oracle, real_regions


class SamplingGeometryTests(unittest.TestCase):
    def test_cuboid_visible_seam_is_preserved_by_priority(self):
        k=np.array([[160.,0,48],[0,160.,48],[0,0,1]])
        frame=synthetic_frame(np.ones((96,96),np.float32),k,np.ones((96,96),bool))
        q=Rotation.from_euler('xy',[15,40],degrees=True).as_quat()
        pose=np.r_[0.,0.,.5,q[3],q[:3]];half=np.array([.08,.06,.05])
        truth=oracle(frame,pose,[(np.zeros(3),half)])
        visible=np.isfinite(truth['depth'])
        frame=replace(frame,depth=np.where(visible,truth['depth'],0).astype(np.float32),masks={0:visible})
        evidence=extract_sampling_evidence(frame)
        # Pose/size are used only by the synthetic renderer and expected metrics.
        _,_,maps=real_regions(frame,evidence.frame,visible,truth,0,half)
        cfg=SamplingConfig(num_points=64,structural_ratio=.5)
        a=sample_points(evidence,cfg);b=sample_points(evidence,cfg,method='S0')
        reference=np.column_stack(np.nonzero(maps['seam']))
        self.assertGreater(len(reference),0);self.assertGreater(a.priority_selected.sum(),0)
        priority=a.pixels_vu[a.priority_selected]
        self.assertGreater(np.mean(cKDTree(reference).query(priority)[0]<=3),.8)
        def coverage(result):return np.mean(cKDTree(result.pixels_vu).query(reference)[0]<=2)
        self.assertGreater(coverage(a),coverage(b))

    def test_clustered_candidates_fps_covers_small_remote_region(self):
        rng=np.random.default_rng(7)
        dense=rng.normal(0,.001,(900,3));remote=rng.normal([.1,0,0],.001,(100,3))
        candidates=np.concatenate([dense,remote])
        rows=fps_indices(candidates,16,seed=0)
        self.assertTrue((rows<900).any());self.assertTrue((rows>=900).any())
        distances=cKDTree(candidates[rows]).query(candidates)[0]
        # Test a coverage outcome, not only the implementation's output length.
        self.assertLess(np.percentile(distances[900:],95),.003)


if __name__=='__main__':unittest.main()
