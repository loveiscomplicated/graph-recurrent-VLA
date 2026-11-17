"""Device/geometry/visibility contracts for the Torch migration."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from visual.corner_detection.backends import torch_ops as K
from visual.corner_detection.detector import TorchCornerDetector
from visual.corner_detection.geometry import _line, Parameters


class TorchDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = K.resolve_device(getattr(cls, "backend", "auto"))
        cls.model = TorchCornerDetector(cls.device)
        cls.k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]], np.float64)
        cls.ext = np.c_[np.eye(3), [0.1, 0.2, 0.3]]

    def rectangle(self):
        rgb = np.zeros((512, 512, 3), np.uint8)
        depth = np.full((512, 512), 2000, np.int16)
        depth[100:200, 120:240] = 1000
        masks = np.zeros((2, 512, 512), bool)
        masks[0, 100:200, 120:240] = True
        return rgb, depth, masks

    def test_resident_tensor_output_and_world_transform(self):
        rgb, depth, masks = self.rectangle()
        args = [torch.as_tensor(v, device=self.device) for v in (rgb, depth)]
        # Float64 host calibration is accepted by MPS by casting before upload.
        r = self.model(
            *args, self.k, self.ext, masks=torch.as_tensor(masks, device=self.device)
        )
        self.assertEqual(r.xyz.device, self.device)
        self.assertEqual(r.uv.device, self.device)
        self.assertEqual(r.masks.device, self.device)
        self.assertEqual(r.object_id.dtype, torch.int64)
        self.assertEqual(len(r.xyz), 4)
        np.testing.assert_allclose(
            r.xyz.cpu(), r.camera_xyz.cpu().numpy() - self.ext[:, 3], atol=1e-6
        )
        for u, v in r.uv.cpu().tolist():
            self.assertLess(min(abs(u - 119.5), abs(u - 239.5)), 0.001)
            self.assertLess(min(abs(v - 99.5), abs(v - 199.5)), 0.001)

    def test_no_legacy_detector_fallback(self):
        rgb, depth, masks = self.rectangle()
        with patch(
            "visual.corner_detection.numpy_reference.detect",
            side_effect=AssertionError("CPU detector invoked"),
        ):
            r = self.model(rgb, depth, self.k, self.ext, masks=masks)
        self.assertEqual(len(r.xyz), 4)

    def test_invalid_depth_and_explicit_empty_mask(self):
        rgb, depth, masks = self.rectangle()
        self.assertEqual(
            len(
                self.model(
                    rgb, np.full_like(depth, -32768), self.k, self.ext, masks
                ).xyz
            ),
            0,
        )
        self.assertEqual(
            len(self.model(rgb, depth, self.k, self.ext, np.zeros_like(masks)).xyz), 0
        )

    def test_no_hidden_corner_support_or_near_aperture(self):
        device = self.device
        depth = torch.full((512, 512), 0.5, device=device)
        k = torch.as_tensor(self.k, dtype=torch.float32, device=device)
        points = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        uv = torch.tensor([[256.0, 256.0]], device=device)
        support = torch.ones((512, 512), dtype=torch.bool, device=device)
        self.assertFalse(bool(K.visible(points, uv, support, K.unproject(depth, k))[0]))
        depth.fill_(1.0)
        depth[256, 256:258] = 0.8
        self.assertFalse(bool(K.unoccluded(points, uv, depth, support)[0]))

    def test_batch_line_fit_preserves_robust_geometry(self):
        rng = np.random.default_rng(4)
        parts = []
        for angle in [0.0, 0.3, 1.2, 1.57]:
            x = np.arange(30.0)
            p = (
                np.stack((x, x * 0.0), -1)
                @ np.array(
                    [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
                )
                + 100
            )
            p += rng.normal(0, 0.03, p.shape)
            p[4] += [0, 3]
            parts.append(p)
        fits = K.fit_lines(parts, self.device)
        for points, row in zip(parts, fits):
            n, d, res = _line(points)
            sign = 1 if n @ row[:2] > 0 else -1
            np.testing.assert_allclose(row[:3] * sign, np.r_[n, d], atol=5e-4)
            self.assertLess(abs(row[3] - res), 1e-4)

    def test_plane_fit_stays_on_device_and_recovers_metric_plane(self):
        points = torch.rand((700, 3), device=self.device) * 0.2
        points[:, 2] = 1.0 + 0.1 * points[:, 0] - 0.2 * points[:, 1]
        planes = K.fit_planes(points, Parameters(), 42, 4)
        self.assertEqual(planes.device, self.device)
        self.assertEqual(len(planes), 1)
        residual = (points @ planes[0, :3] + planes[0, 3]).abs().max()
        self.assertLess(float(residual), 2e-6)

    def test_unavailable_explicit_gpu_does_not_silently_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                K.resolve_device("cuda")


"""Numerical/visibility contracts against the portable Torch CPU reference."""
import unittest
from types import SimpleNamespace


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS required")
class MetalKernelTests(unittest.TestCase):
    def setUp(self):
        self.k = torch.tensor(
            [[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]]
        )

    def test_plane_masks_and_morphology_match_cpu(self):
        depth = torch.full((512, 512), 2.0)
        depth[80:280, 110:310] = 1.0
        mask = torch.zeros((512, 512), dtype=torch.bool)
        mask[80:280, 110:310] = True
        mask[120, 140] = False
        xyz = K.unproject(depth, self.k)
        planes = torch.tensor([[0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, -0.2]])
        expected = K.plane_masks(xyz, mask, planes, 0.001)
        actual = K.plane_masks(xyz.to("mps"), mask.to("mps"), planes.to("mps"), 0.001)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a.cpu(), b))

    def test_snap_to_observed_planes_matches_cpu(self):
        generator = torch.Generator().manual_seed(9)
        points = torch.rand((100, 3), generator=generator) * 0.007 - 0.0035
        points[:, 2] += 1.0
        planes = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
        )
        primary = torch.full((100,), 2, dtype=torch.int64)
        eligible = torch.tensor([True, True, True])
        expected, counts = K.snap(points, primary, planes, eligible)
        actual, actual_counts = K.snap(
            points.to("mps"), primary.to("mps"), planes.to("mps"), eligible.to("mps")
        )
        np.testing.assert_allclose(actual.cpu(), expected, atol=2e-7)
        self.assertTrue(torch.equal(actual_counts.cpu(), counts))

    def test_physical_edges_match_cpu_including_occlusion(self):
        for foreground in (0.8, 2.0):
            plane_mask = torch.zeros((1, 512, 512), dtype=torch.bool)
            plane_mask[:, 256:, 256:] = True
            depth = torch.where(plane_mask[0], 1.0, foreground)
            uv = torch.tensor([[255.5, 255.5], [280.1, 255.4], [2.0, 2.0]])
            adjacent = uv[:, None] + torch.tensor([[[25.0, 0.0], [0.0, 25.0]]])
            args = (
                uv,
                adjacent,
                torch.tensor([[0.0, 0.0, 1.0, -1.0]]),
                torch.zeros(3, dtype=torch.int64),
                plane_mask,
                torch.ones((512, 512), dtype=torch.bool),
                depth,
                torch.ones((512, 512), dtype=torch.bool),
                K.unproject(depth, self.k),
                self.k,
                torch.zeros((512, 512), dtype=torch.bool),
            )
            expected, evidence = K.physical_edges(*args)
            actual, actual_evidence = K.physical_edges(*(x.to("mps") for x in args))
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertTrue(torch.equal(actual_evidence.cpu(), evidence))

    def test_ransac_recovers_two_planes_with_outliers(self):
        generator = torch.Generator().manual_seed(23)
        a = torch.rand((900, 3), generator=generator) * 0.2
        b = torch.rand((700, 3), generator=generator) * 0.2
        a[:, 2] = 1.0 + 0.1 * a[:, 0] - 0.2 * a[:, 1]
        b[:, 0] = 0.3 + 0.2 * b[:, 1]
        b[:, 2] += 0.8
        outliers = torch.rand((30, 3), generator=generator)
        points = torch.cat((a, b, outliers))
        expected = K.fit_planes(points, Parameters(), 42, 4)
        actual = K.fit_planes(points.to("mps"), Parameters(), 42, 4).cpu()
        self.assertEqual(len(actual), len(expected))
        for plane, reference in zip(actual, expected):
            sign = 1 if plane[:3] @ reference[:3] > 0 else -1
            np.testing.assert_allclose(plane * sign, reference, atol=2e-6)

    def stage_outputs(self, device, stage):
        """Same synthetic buffers, portable Torch CPU vs candidate-batch MPS."""
        m = TorchCornerDetector(device)
        bottom = 220 if stage == "box_cube" else 200
        target = torch.zeros((512, 512), dtype=torch.bool)
        target[100:bottom, 120:240] = True
        masks = torch.stack((target, target.clone(), target.clone()))
        masks[1, :, :235] = False
        masks[2, : bottom - 5] = False
        cores = K.erode(masks)
        planes = torch.tensor(
            [
                [0.0, 0.0, 1.0, -1.0],
                [1.0, 0.0, 0.0, 0.033],
                [0.0, 1.0, 0.0, (256 - bottom + 0.5) / 500],
            ]
        )
        depth = torch.where(target, 1.0, 2.0)
        m.k = self.k.to(device)
        m.k_cpu = self.k.numpy().astype(float)
        m.depth = depth.to(device)
        m.valid = torch.ones_like(target, device=device)
        m.xyz = K.unproject(m.depth, m.k)
        m.neutral = torch.zeros_like(target, device=device)
        m.current_masks = torch.stack((target, target)).to(device)
        m.blocks = []
        m.rows = []
        proposals = []
        for uv, adjacent in [
            ([119.5, 99.5], [[239.5, 99.5], [119.5, 199.5]]),
            ([239.5, 199.5], [[119.5, 199.5], [239.5, 99.5]]),
            ([1.0, 1.0], [[20.0, 1.0], [1.0, 20.0]]),
        ]:
            for method in ["boundary", "local_boundary", "short_boundary"]:
                proposals.append(
                    dict(
                        uv=np.array(uv),
                        adjacent=np.array(adjacent),
                        plane=0,
                        method=method,
                        line_residual_px=0.1,
                        contour_hole=False,
                    )
                )
        data = SimpleNamespace(
            oid=1 if stage in ("box_rgb", "box_cube") else 0,
            support=target.to(device),
            support_cpu=target.numpy(),
            planes=planes.to(device),
            masks=masks.to(device),
            cores=cores.to(device),
            eligible=torch.ones(3, dtype=torch.bool, device=device),
            eligible_cpu={0, 1, 2},
            planes_cpu=[(p[:3].numpy().astype(float), float(p[3])) for p in planes],
            masks_cpu=masks.numpy(),
            proposals=proposals,
            core=True,
            recovery=True,
        )
        if stage in ("box_rgb", "box_cube"):
            features = [
                (np.array(uv), "mask")
                for uv in [
                    [119.5, 99.5],
                    [239.5, 99.5],
                    [119.5, bottom - 0.5],
                    [239.5, bottom - 0.5],
                    [180.0, 150.0],
                ]
            ]
            getattr(m, stage)(data, features)
        else:
            getattr(m, stage)(data)
        return [
            (b[0].cpu(), b[1].cpu(), b[3].cpu(), None if b[4] is None else b[4].cpu())
            for b in m.blocks
        ]

    def compare_stage(self, stage):
        reference = self.stage_outputs("cpu", stage)
        actual = self.stage_outputs("mps", stage)
        self.assertEqual(len(actual), len(reference))
        self.assertGreater(sum(int(x[2].sum()) for x in reference), 0)
        for (p, uv, keep, count), (rp, ru, rk, rc) in zip(actual, reference):
            self.assertTrue(torch.equal(keep, rk), stage)
            np.testing.assert_allclose(p[keep], rp[rk], atol=1e-6)
            np.testing.assert_allclose(uv[keep], ru[rk], atol=1e-3)
            if count is not None:
                self.assertTrue(torch.equal(count[keep], rc[rk]))

    def test_fused_boundary_matches_portable_stage(self):
        self.compare_stage("boundary")

    def test_fused_junction_matches_portable_stage(self):
        self.compare_stage("junctions")

    def test_fused_envelope_matches_portable_stage(self):
        self.compare_stage("envelopes")

    def test_fused_box_rgb_matches_portable_stage(self):
        self.compare_stage("box_rgb")

    def test_fused_box_cube_matches_portable_stage(self):
        self.compare_stage("box_cube")


class CpuDetectorTests(TorchDetectorTests):
    backend = "cpu"


class MetalCapabilityTests(unittest.TestCase):
    def test_missing_compiler_has_explicit_error(self):
        from visual.corner_detection.backends import metal_ops

        with patch.object(torch.mps, "compile_shader", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "torch.mps.compile_shader"):
                metal_ops.library.__wrapped__()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on current host")
class CudaDetectorTests(TorchDetectorTests):
    backend = "cuda"


if __name__ == "__main__":
    unittest.main()
