"""Same-estimator equivalence including disconnected and degenerate patches."""
from dataclasses import replace
import unittest
import numpy as np

from visual.runtime.geometry import build_organized_geometry, compute_geometric_features, GeometryParameters
from visual.runtime.test.normal_diagnostics import angular_error, quantize, synthetic_case, synthetic_frame
from visual.runtime.test.normal_estimators import local_plane
from visual.runtime.test.pca_optimized import build_geometry, local_plane_optimized, process_optimized, smallest_eigenvector

K = np.array([[256., 0, 32.], [0, 256., 32.], [0, 0, 1]])


class PCAOptimizationTests(unittest.TestCase):
    def assert_equivalent(self, frame):
        g = build_organized_geometry(frame)
        p = GeometryParameters()
        reference = {i: compute_geometric_features(g, i, *local_plane(g, i, 5), p) for i in g.object_masks}
        for solver in ('eigh', 'analytic'):
            actual = process_optimized(frame, solver=solver)
            np.testing.assert_array_equal(g.xyz, actual.geometry.xyz)
            for oid, expected in reference.items():
                f = actual.objects[oid]
                for name in expected.__dataclass_fields__:
                    if name == 'normals':
                        if f.normal_valid.any():
                            self.assertLess(float(angular_error(f.normals[f.normal_valid], expected.normals[f.normal_valid]).max()), .001)
                    elif name == 'normal_variation_rad':
                        np.testing.assert_allclose(f.normal_variation_rad, expected.normal_variation_rad, atol=2e-5, rtol=0)
                    else:
                        np.testing.assert_array_equal(getattr(f, name), getattr(expected, name), err_msg=name)

    def test_all_synthetic_geometries_float_and_quantized(self):
        for kind in ('front', 'tilted', 'crease', 'step'):
            frame, _, _ = synthetic_case(kind, K, shape=(64, 64), phase_m=.00025)
            for q in (False, True):
                with self.subTest(kind=kind, quantized=q):
                    self.assert_equivalent(replace(frame, depth=quantize(frame.depth)) if q else frame)

    def test_image_borders_invalid_overlap_two_objects_and_background_jumps(self):
        rng = np.random.default_rng(21)
        depth = np.full((31, 35), .25, np.float32)
        depth[:, 18:] = .45
        depth[8, 8], depth[11, 10], depth[3, 3] = np.nan, 0, np.inf
        peg = rng.random(depth.shape) > .18
        box = np.zeros_like(peg); box[10:22, 14:26] = True
        frame = replace(synthetic_frame(depth, K, peg), masks={0: peg, 1: box})
        self.assert_equivalent(frame)

    def test_disconnected_depth_surface_and_patch_limited_connectivity(self):
        depth = np.full((25, 25), .25, np.float32)
        mask = np.zeros_like(depth, bool); mask[3:22, 3:22] = True
        mask[5:20, 12] = False  # Far away connection cannot bypass the local gap.
        depth[:, 13:] = .26
        self.assert_equivalent(synthetic_frame(depth, K, mask))

    def test_sparse_empty_collinear_tiny_and_invalid_only(self):
        for h, w in ((1, 1), (2, 2), (7, 9), (15, 17)):
            for kind in ('empty', 'line', 'all', 'invalid'):
                mask = np.zeros((h, w), bool)
                if kind=='line': mask[h//2] = True
                if kind in ('all', 'invalid'): mask[:] = True
                depth = np.full((h, w), 0 if kind=='invalid' else .25, np.float32)
                with self.subTest(h=h, w=w, kind=kind): self.assert_equivalent(synthetic_frame(depth, K, mask))

    def test_camera_backprojection_preserves_arithmetic_and_inputs(self):
        rng = np.random.default_rng(0)
        depth = rng.uniform(.1, 2, (23, 29)).astype(np.float32); depth[0, :5] = [0, -1, np.nan, np.inf, -np.inf]
        intrinsic = np.array([[173.7, 0, 14.1], [0, 231.3, 11.8], [0, 0, 1]])
        frame = synthetic_frame(depth, intrinsic, rng.random(depth.shape)>.1)
        before = frame.depth.copy(); a, b = build_organized_geometry(frame), build_geometry(frame)
        np.testing.assert_array_equal(a.xyz, b.xyz); np.testing.assert_array_equal(before, frame.depth)
        np.testing.assert_array_equal(a.valid_depth, b.valid_depth)

    def test_analytic_eigensolver_rotations_and_degeneracy(self):
        rng = np.random.default_rng(5)
        basis, _ = np.linalg.qr(rng.normal(size=(200, 3, 3)))
        values = np.tile([1e-8, 2e-6, 8e-6], (200, 1))
        values[:4] = [[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1+1e-12, 2]]
        covariance = np.einsum('nij,nj,nkj->nik', basis, values, basis)
        actual, vectors, fallback = smallest_eigenvector(covariance)
        expected, truth = np.linalg.eigh(covariance)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-7)
        self.assertGreaterEqual(fallback, 4)
        good = np.arange(len(values)) >= 4
        dot = np.abs(np.sum(vectors[good]*truth[good, :, 0], axis=1))
        self.assertLess(float(np.max(1-dot)), 1e-12)

    def test_large_coordinate_translation_does_not_destroy_local_covariance(self):
        frame, _, _ = synthetic_case('tilted', K, shape=(64, 64), z0=10, phase_m=.00025)
        self.assert_equivalent(frame)

    def test_parameters_reject_invalid_solver_and_unknown_mask(self):
        frame = synthetic_frame(np.ones((9, 9), np.float32), K, np.ones((9, 9), bool))
        with self.assertRaises(ValueError): local_plane_optimized(build_geometry(frame), 0, solver='bad')
        with self.assertRaises(ValueError): build_geometry(replace(frame, masks={3:frame.masks[0]}))

    def test_two_object_threads_preserve_every_output(self):
        frame,_,_=synthetic_case('crease',K,shape=(64,64),phase_m=.00025)
        box=np.roll(frame.masks[0],17,axis=1)
        frame=replace(frame,depth=quantize(frame.depth),masks={0:frame.masks[0],1:box})
        serial=process_optimized(frame,solver='analytic')
        parallel=process_optimized(frame,solver='analytic',parallel=True)
        for oid in serial.objects:
            for key in serial.objects[oid].__dataclass_fields__:
                np.testing.assert_array_equal(getattr(serial.objects[oid],key),getattr(parallel.objects[oid],key))

    def check_torch_device(self, device, backend='eigh'):
        import torch
        from visual.runtime.test.pca_torch import TorchPCA5
        for kind in ('front', 'tilted', 'crease', 'step'):
            frame, _, _ = synthetic_case(kind, K, shape=(32, 32), phase_m=.00025)
            depth = quantize(frame.depth); depth[15, 15] = 0
            frame = replace(frame, depth=depth)
            g = build_organized_geometry(frame)
            processor = TorchPCA5(K, frame.depth.shape, device=device, backend=backend)
            td=torch.as_tensor(depth,device=device).T.contiguous().T
            tm={i:torch.as_tensor(m,device=device).T.contiguous().T for i,m in frame.masks.items()}
            actual = processor(td,tm)
            np.testing.assert_array_equal(actual['xyz'].cpu().numpy(), g.xyz)
            for oid in g.object_masks:
                expected = compute_geometric_features(g, oid, *local_plane(g, oid, 5), GeometryParameters())
                for key in expected.__dataclass_fields__:
                    a, b = actual['objects'][oid][key].cpu().numpy(), getattr(expected, key)
                    if key in ('normals', 'normal_variation_rad'):
                        np.testing.assert_allclose(a, b, atol=2e-5, rtol=0, err_msg=key)
                    else: np.testing.assert_array_equal(a, b, err_msg=key)

    def test_torch_float64_contract_on_cpu(self):
        self.check_torch_device('cpu')

    def test_torch_cuda_contract_when_available(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable on current host')
        self.check_torch_device('cuda')

    def test_triton_cuda_contract_when_available(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable on current host')
        self.check_torch_device('cuda',backend='triton')

    def test_fused_features_cuda_contract_when_available(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable on current host')
        self.check_torch_device('cuda',backend='triton_fused')


if __name__ == '__main__': unittest.main()
