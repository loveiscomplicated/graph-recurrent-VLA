"""Device/geometry/visibility contracts for the Torch migration."""

import unittest
from unittest.mock import patch
import numpy as np
import torch
from visual.corner_detection import detect_tensors
from visual.corner_detection.backends import material_torch, torch_ops as K
from visual.corner_detection.candidates import filter_material_components
from visual.corner_detection.backends.rear_end_torch import box_cut_vertices
from visual.corner_detection.detector import TorchCornerDetector, default_detector
from visual.corner_detection.geometry import _line, Parameters


class MaterialMaskTests(unittest.TestCase):
    def test_cropped_components_preserve_connectivity_area_and_image_coordinates(self):
        empty = np.zeros((512, 512), dtype=bool)
        peg = empty.copy()
        np.fill_diagonal(peg[477:512, 90:125], True)
        np.fill_diagonal(peg[463:497, 180:214], True)
        box = empty.copy()
        box[302, 191:199] = True
        box[312, 210:217] = True
        box[328:335, 226:231] = True

        for colors in ((peg, box), (empty, box), (peg, empty), (empty, empty)):
            expected = filter_material_components(*colors, optimized=True)
            actual = filter_material_components(*colors, optimized=True, crop=True)
            for before, after in zip(expected, actual, strict=True):
                self.assertEqual(after.dtype, np.bool_)
                np.testing.assert_array_equal(after, before)


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

    def test_gpu_material_components_preserve_areas_and_connectivity(self):
        filter_components = material_torch.component_filter(self.device.type)
        if filter_components is None:
            self.skipTest("GPU connected-components backend unavailable")

        colors = np.zeros((2, 512, 512), dtype=bool)
        for mask in colors:
            # Diagonal neighbors belong to one component under 8-connectivity.
            np.fill_diagonal(mask[:35, :35], True)
            np.fill_diagonal(mask[:34, 80:114], True)
            mask[100, 20:28] = True
            mask[120, 20:27] = True
            mask[200, 511] = mask[201, 0] = True
        colors[0, 511, :] = True
        colors[1, 0, :] = True

        expected = filter_material_components(*colors, optimized=True)
        actual = filter_components(torch.as_tensor(colors, device=self.device))
        for before, after in zip(expected, actual, strict=True):
            self.assertEqual(after.device, self.device)
            np.testing.assert_array_equal(after.cpu().numpy(), before)

    def test_grid_plane_fit_keeps_parallel_depth_layers_separate(self):
        from visual.corner_detection.backends import grid_planes

        y, x = torch.meshgrid(
            torch.arange(64, dtype=torch.float32),
            torch.arange(64, dtype=torch.float32),
            indexing="ij",
        )
        depth = torch.where(x < 32, 0.9, 0.92)
        xyz = torch.stack((x * 0.002, y * 0.002, depth), dim=-1)
        support = (x > 2) & (x < 61) & (y > 2) & (y < 61)

        planes = grid_planes.fit_planes(xyz, support, xyz[support], Parameters(), 4)

        self.assertEqual(planes.shape, (2, 4))
        torch.testing.assert_close(planes[:, 2].abs(), torch.ones(2))
        torch.testing.assert_close(
            planes[:, 3].abs().sort().values, torch.tensor([0.9, 0.92])
        )

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

    def test_public_detector_uses_main_preset(self):
        rgb, depth, masks = self.rectangle()
        detector = default_detector(str(self.device))
        self.assertTrue(detector.recover_hole_rim)
        self.assertTrue(detector.recover_peg_rear_end)
        self.assertTrue(detector.reject_peg_box_cut)
        result = detect_tensors(
            rgb, depth, self.k, self.ext, masks=masks, device=self.device
        )
        expected = TorchCornerDetector.main(device=self.device)(
            rgb, depth, self.k, self.ext, masks=masks
        )
        torch.testing.assert_close(result.xyz, expected.xyz)
        self.assertEqual(result.metadata, expected.metadata)

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

    def test_peg_prism_box_cut_requires_occlusion_evidence(self):
        device = self.device
        points = torch.tensor(
            [[-0.08, -0.02, 1], [-0.08, 0.02, 1],
             [0.08, -0.02, 1], [0.08, 0.02, 1]], device=device
        )
        planes = torch.tensor([[0., 1., 0., 0.02], [0., 0., 1., -1.]], device=device)
        basis = torch.eye(3, device=device)
        box = torch.tensor([[0.2, 0., 0.98]], device=device)
        depth = torch.full((512, 512), 0.98, device=device)
        box_support = torch.zeros_like(depth, dtype=torch.bool)
        peg_support = torch.zeros_like(box_support)
        k = torch.tensor([[1000., 0., 256.], [0., 1000., 256.], [0., 0., 1.]], device=device)

        def reject(observed_planes):
            return box_cut_vertices(
                points, observed_planes, basis, box, depth, peg_support, box_support, k
            ).tolist()

        box_support[236, 339] = box_support[276, 339] = True
        self.assertEqual(reject(planes), [False, False, True, True])
        end_plane = torch.tensor([[1., 0., 0., -0.08]], device=device)
        self.assertEqual(reject(torch.cat((planes, end_plane))), [False] * 4)

        # One vertex still sees the Peg just beyond the estimated shared bound.
        peg_support[276, 337] = True
        depth[276, 337] = 1.0
        self.assertEqual(reject(planes), [False] * 4)

        # The other vertex is covered on both sides and can be rejected alone.
        box_support[236, 334:340] = True
        self.assertEqual(reject(planes), [False, False, True, False])

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
        optimized = K.fit_planes(points, Parameters(), 42, 4, optimized=True)
        self.assertTrue(torch.equal(planes, optimized))
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

    def test_public_mps_detector_uses_frame_validation_by_default(self):
        from visual.corner_detection.backends.metal_candidates import FrameCandidates

        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        depth = np.full((512, 512), 2000, dtype=np.int16)
        masks = np.zeros((2, 512, 512), dtype=bool)
        depth[100:200, 120:240] = 1000
        masks[0, 100:200, 120:240] = True
        extrinsic = np.eye(4, dtype=np.float32)[:3]

        self.assertTrue(default_detector("mps").optimized)
        with patch.object(
            FrameCandidates, "validate", autospec=True, side_effect=FrameCandidates.validate
        ) as validate:
            result = detect_tensors(rgb, depth, self.k, extrinsic, masks=masks, device="mps")

        validate.assert_called_once()
        self.assertGreater(len(result.xyz), 0)

    def test_plane_masks_and_morphology_match_cpu(self):
        depth = torch.full((512, 512), 2.0)
        depth[80:280, 110:310] = 1.0
        depth[:12, :18] = 1.0
        depth[500:, 500:] = 1.0
        mask = torch.zeros((512, 512), dtype=torch.bool)
        mask[80:280, 110:310] = True
        mask[120, 140] = False
        mask[:12, :18] = True
        mask[500:, 500:] = True
        xyz = K.unproject(depth, self.k)
        planes = torch.tensor([[0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, -0.2]])
        expected = K.plane_masks(xyz, mask, planes, 0.001)
        actual = K.plane_masks(xyz.to("mps"), mask.to("mps"), planes.to("mps"), 0.001)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a.cpu(), b))

    def test_box_tail_quantiles_match_mps_sort_and_interpolation_exactly(self):
        from visual.corner_detection.backends import metal_ops

        generator = torch.Generator().manual_seed(21)
        quantiles = torch.tensor([0.001, 0.999], device="mps")
        for size in (2, 80, 999, 1001, 4097):
            points = torch.randn((size, 3), generator=generator)
            points[0] *= -1000
            points[-1] *= 1000
            cases = {"outliers": points, "repeated": points.round()}
            exceptional = points.clone()
            exceptional[:, 0] = -0.0
            exceptional[0, 0] = 0.0
            exceptional[0, 1] = float("nan")
            exceptional[0, 2] = float("inf")
            cases["exceptional"] = exceptional
            for kind, coords in cases.items():
                with self.subTest(size=size, kind=kind):
                    coords = coords.to("mps")
                    expected = torch.quantile(coords, quantiles, dim=0).cpu()
                    actual = metal_ops.box_tail_quantiles(coords)
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_packed_plane_descriptors_preserve_coefficients_and_mask_bits(self):
        from visual.corner_detection.backends import metal_ops

        planes = np.array([[0.0, -0.0, 1.0, -0.9], [0.0, 1.0, 0.0, 0.4]], np.float32)
        eligible = np.array([False, True])
        masks = np.zeros((2, 512, 512), bool)
        masks[0, 0, [0, 7, 8, 511]] = True
        masks[1, 511, [0, 6, 9, 510]] = True
        support = masks.any(0)
        inputs = [
            torch.as_tensor(value, device="mps") for value in (planes, eligible, masks, support)
        ]

        for host_support in (None, support):
            rows, ids, host_masks, actual_support = metal_ops.read_plane_descriptors(
                *inputs, host_support
            )
            self.assertEqual(rows.astype(np.float32).tobytes(), planes.tobytes())
            self.assertEqual(ids, {1})
            np.testing.assert_array_equal(host_masks, masks)
            np.testing.assert_array_equal(actual_support, support)

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
        # Leave enough outliers for a final RANSAC pass that must reject the fit.
        outliers = torch.rand((70, 3), generator=generator)
        points = torch.cat((a, b, outliers))
        expected = K.fit_planes(points, Parameters(), 42, 4)
        self.assertEqual(len(expected), 2)
        device_points = points.to("mps")
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                actual = K.fit_planes(
                    device_points, Parameters(), 42, 4, optimized=optimized
                ).cpu()
                self.assertEqual(len(actual), len(expected))
                for plane, reference in zip(actual, expected):
                    sign = 1 if plane[:3] @ reference[:3] > 0 else -1
                    np.testing.assert_allclose(plane * sign, reference, atol=2e-6)

    def stage_outputs(self, device, stage, *, frame_batch=False):
        """Same synthetic buffers, portable Torch CPU vs candidate-batch MPS."""
        m = TorchCornerDetector(device)
        bottom = 220 if stage in ("box_cube", "frame") else 200
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
        if stage in ("box_rgb", "box_cube", "frame"):
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
            if stage == "frame":
                from visual.corner_detection.backends.metal_candidates import FrameCandidates

                # Use all three slots, including distinct Box core and recovery planes.
                box_core = SimpleNamespace(**(vars(data) | {"oid": 1, "recovery": False}))
                box_recovery = SimpleNamespace(**(vars(data) | {"oid": 1, "core": False}))
                box_core.masks = masks.flip(0).to(device)
                groups = [data, box_core, box_recovery]
                core = m.current_masks
                recovery = core.clone()
                recovery[1, 110:130, 130:150] = False
                batch = FrameCandidates(groups, core, recovery) if frame_batch else None
                m.optimized = True
                for group in groups:
                    m.current_masks = core if group.core else recovery
                    m.boundary(group, batch)
                    m.junctions(group, batch)
                    m.envelopes(group, batch)
                m.current_masks = recovery
                m.box_cube(box_recovery, features, batch)
                m.box_rgb(box_recovery, features, batch)
                if batch is not None:
                    m._merge_frame_candidates(batch)
            else:
                getattr(m, stage)(data, features)
        else:
            getattr(m, stage)(data)
        outputs = [
            (b[0].cpu(), b[1].cpu(), b[3].cpu(), None if b[4] is None else b[4].cpu())
            for b in m.blocks
        ]
        if stage == "frame":
            rgb_block, holes = m.box_rgb_holes
            return outputs, m.rows, rgb_block, holes.cpu()
        return outputs

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

    def test_frame_batch_preserves_order_and_outputs_across_plane_data_slots(self):
        reference, reference_rows, reference_rgb, reference_holes = self.stage_outputs("mps", "frame")
        actual, actual_rows, actual_rgb, actual_holes = self.stage_outputs(
            "mps", "frame", frame_batch=True
        )

        self.assertEqual(actual_rows, reference_rows)
        self.assertEqual(actual_rgb, reference_rgb)
        self.assertTrue(torch.equal(actual_holes, reference_holes))
        self.assertEqual(len(actual), len(reference))
        self.assertGreater(sum(int(block[2].sum()) for block in reference), 0)
        for (points, uv, selected, counts), (ref_points, ref_uv, ref_selected, ref_counts) in zip(
            actual, reference, strict=True
        ):
            self.assertTrue(torch.equal(selected, ref_selected))
            self.assertTrue(torch.equal(points[selected].view(torch.int32), ref_points[selected].view(torch.int32)))
            self.assertTrue(torch.equal(uv[selected].view(torch.int32), ref_uv[selected].view(torch.int32)))
            if counts is not None:
                self.assertTrue(torch.equal(counts[selected], ref_counts[selected]))


class CpuDetectorTests(TorchDetectorTests):
    backend = "cpu"


class MetalCapabilityTests(unittest.TestCase):
    def test_main_auto_falls_back_if_metal_compiler_is_missing(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.mps, "compile_shader", None, create=True),
        ):
            detector = TorchCornerDetector.main()

        self.assertEqual(detector.device.type, "cpu")
        self.assertTrue(detector.recover_peg_rear_end)
        self.assertTrue(detector.reject_peg_box_cut)

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
