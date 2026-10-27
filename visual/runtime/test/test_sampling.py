"""Task03 observation provenance, budgets, geometry and regression contracts."""
from dataclasses import replace, fields
import unittest
import numpy as np

from visual.runtime.geometry import build_organized_geometry, compute_geometric_features, process_geometry, GeometryParameters
from visual.runtime.rgbd import PEG, BOX
from visual.runtime.sampling import (SamplingConfig, allocate_budget, fps_indices,
    extract_sampling_evidence, extract_candidates, sample_points)
from visual.runtime.test.normal_diagnostics import synthetic_frame, synthetic_case, quantize
from visual.runtime.test.normal_estimators import local_plane
from visual.runtime.test.structural_edges import compare

K = np.array([[256., 0, 32], [0, 256., 32], [0, 0, 1]])


class SamplingTests(unittest.TestCase):
    def frame(self, n=64):
        depth = np.full((n, n), .25, np.float32)
        peg = np.zeros((n, n), bool); peg[2:n//2-1, 2:-2] = True
        box = np.zeros_like(peg); box[n//2+1:-2, 2:-2] = True
        return replace(synthetic_frame(depth, K, peg), masks={PEG: peg, BOX: box})

    def evidence(self, frame=None, method='pca5'):
        return extract_sampling_evidence(self.frame() if frame is None else frame, method)

    def test_optional_fps_backend_preserves_seed_prefix_and_source_rows(self):
        frame, _, _ = synthetic_case('crease', K, shape=(64, 64))
        evidence = self.evidence(frame)
        reference = sample_points(evidence)
        calls = []
        def backend(points, count, **kwargs):
            calls.append((count, kwargs))
            return fps_indices(points, count, **kwargs)
        candidate = sample_points(evidence, fps_fn=backend)
        self.assertEqual(len(calls), 6)
        self.assertTrue(any(len(k.get('initial', ())) for _, k in calls))
        np.testing.assert_array_equal(candidate.flat_index, reference.flat_index)
        np.testing.assert_array_equal(candidate.object_id, reference.object_id)

    def test_budget_ratio_observed_uniqueness_alignment_and_reproducibility(self):
        f, _, _ = synthetic_case('crease', K, shape=(64, 64))
        e = self.evidence(f)
        for budget in (64, 128, 256, 512):
            for ratio in (0., .125, .25, .5, 1.):
                cfg = SamplingConfig(num_points=budget, structural_ratio=ratio)
                a = sample_points(e, cfg); b = sample_points(e, cfg)
                self.assertLessEqual(len(a.points), budget)
                self.assertLessEqual(int(a.priority_selected.sum()), int(budget*ratio))
                self.assertEqual(len(np.unique(a.flat_index)), len(a.points))
                np.testing.assert_array_equal(a.flat_index, b.flat_index)
                v, u = a.pixels_vu.T
                np.testing.assert_array_equal(a.flat_index, v*f.depth.shape[1]+u)
                np.testing.assert_array_equal(a.points, e.frame.geometry.xyz[v, u])
                c = extract_candidates(e, cfg)
                for oid in (PEG, BOX):
                    chosen = a.object_id == oid
                    rows = np.searchsorted(c[oid].flat_index, a.flat_index[chosen])
                    self.assertTrue(e.frame.geometry.observed_mask(oid)[v[chosen], u[chosen]].all())
                    for name, values in a.metadata.items():
                        np.testing.assert_array_equal(values[chosen], c[oid].metadata[name][rows], err_msg=name)
                for field in fields(a):
                    if field.name not in ('timings_ms', 'metadata'):
                        if isinstance(getattr(a, field.name), np.ndarray):
                            np.testing.assert_array_equal(getattr(a, field.name), getattr(b, field.name))

    def test_zero_ratio_matches_general_fps_for_every_method(self):
        f, _, _ = synthetic_case('crease', K, shape=(64, 64))
        cfg = SamplingConfig(structural_ratio=0)
        baseline = sample_points(self.evidence(f), cfg, method='S0')
        for method, est in [('S1', 'baseline'), ('S2', 'fd2'), ('S3', 'pca5'), ('S4', 'pca5')]:
            a = sample_points(self.evidence(f, est), cfg, method=method)
            np.testing.assert_array_equal(a.flat_index, baseline.flat_index)
            self.assertFalse(a.priority_selected.any())

    def test_plane_has_no_invented_structural_and_equals_fps(self):
        e = self.evidence()
        for method in ('S3', 'S4'):
            a = sample_points(e, method=method); b = sample_points(e, method='S0')
            self.assertFalse(a.priority_selected.any()); self.assertFalse(a.metadata['structural_candidate'].any())
            np.testing.assert_array_equal(a.flat_index, b.flat_index)

    def test_minimums_and_configurable_observation_allocation(self):
        for counts in ((1000, 1000), (3, 1000), (1000, 3), (0, 1000), (1000, 0), (0, 0), (3, 4)):
            a = allocate_budget(64, counts, (16, 16))
            self.assertEqual(int(a.sum()), min(64, sum(counts)))
            self.assertTrue((a <= counts).all())
            self.assertTrue((a >= np.minimum(counts, [16, 16])).all())
        np.testing.assert_array_equal(allocate_budget(64, (1000, 1000), (16, 16)), [32, 32])
        a = allocate_budget(64, (100, 900), (8, 8))
        np.testing.assert_array_equal(a, [13, 51])
        e = self.evidence(); cfg = SamplingConfig(allocation='sqrt_observed_points', min_points=(8, 8))
        self.assertEqual(len(sample_points(e, cfg).points), 256)

    def test_one_object_empty_current_observation_no_cache(self):
        for absent in (PEG, BOX):
            f = self.frame(); masks = {i:m.copy() for i,m in f.masks.items()}; masks[absent][:] = False
            a = sample_points(self.evidence(replace(f, masks=masks)))
            self.assertEqual(a.counts[absent]['selected'], 0)
            self.assertEqual(a.counts[1-absent]['selected'], 256)

    def test_insufficient_both_empty_and_zero_budget(self):
        f = self.frame(); masks = {i:np.zeros_like(m) for i,m in f.masks.items()}
        for empty in (True, False):
            if not empty: masks[PEG][4, 4] = True; masks[BOX][40, 40] = True
            e = self.evidence(replace(f, masks=masks)); a = sample_points(e)
            self.assertEqual(len(a.points), 0 if empty else 2)
            self.assertEqual(a.points.shape[1:], (3,)); self.assertEqual(a.pixels_vu.shape[1:], (2,))
            for values in a.metadata.values(): self.assertEqual(len(values), len(a.points))
        a = sample_points(self.evidence(), SamplingConfig(num_points=0, min_points=(0, 0)))
        self.assertEqual(len(a.points), 0)

    def test_invalid_config_is_explicit(self):
        cases = [dict(num_points=-1), dict(num_points=1.5), dict(num_points=True), dict(num_points=16),
                 dict(structural_ratio=-.1), dict(structural_ratio=1.1), dict(structural_ratio=np.nan),
                 dict(depth_priority_fraction=np.inf), dict(min_points=(-1, 1)), dict(min_points=(1,)),
                 dict(orientation_threshold_deg=-1), dict(orientation_threshold_deg=181), dict(seed=-1),
                 dict(allocation='random'), dict(sampling='random')]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError): SamplingConfig(**case)
        with self.assertRaises(ValueError): allocate_budget(5, [4, 4], [3, 3])
        with self.assertRaises(ValueError): allocate_budget(5, [-1, 4])

    def test_sparse_candidates_surface_fills_without_duplicates(self):
        f, _, _ = synthetic_case('crease', K, shape=(64, 64))
        a = sample_points(self.evidence(f), SamplingConfig(structural_ratio=1., orientation_threshold_deg=40))
        self.assertEqual(len(a.points), min(256, int(f.masks[PEG].sum())))
        self.assertEqual(a.counts[PEG]['priority'], a.counts[PEG]['candidates'])
        self.assertGreater(a.counts[PEG]['surface_fps'], 0)

    def test_real_crease_priority_hits_visible_edge_and_spreads(self):
        f, _, _ = synthetic_case('crease', K, shape=(64, 64))
        e = self.evidence(f); a = sample_points(e, SamplingConfig(num_points=64, structural_ratio=.5))
        self.assertGreater(a.priority_selected.sum(), 0)
        vu = a.pixels_vu[a.priority_selected]
        # Compare to the actual two-face transition in observed synthetic geometry.
        self.assertLessEqual(np.max(np.abs(vu[:, 1]-31.5)), 4)
        candidates = extract_candidates(e, SamplingConfig())[PEG]
        cp = candidates.points[candidates.metadata['orientation_candidate']]
        chosen = a.points[a.priority_selected]
        self.assertGreater(np.ptp(chosen[:, 1]), .8*np.ptp(cp[:, 1]))

    def test_mask_cut_and_invalid_hole_not_structural_but_depth_step_separate(self):
        f = self.frame(); mask = f.masks[PEG].copy(); mask[10:17, 20:28] = False
        e = self.evidence(replace(f, masks={PEG:mask})); a = sample_points(e, method='S4')
        self.assertTrue(a.metadata['mask_evidence'].any()); self.assertFalse(a.priority_selected.any())
        d = f.depth.copy(); d[10:17, 20:28] = 0
        e = self.evidence(replace(f, depth=d)); c = extract_candidates(e, SamplingConfig(), 'S4')
        self.assertTrue(c[PEG].metadata['invalid_evidence'].any())
        self.assertFalse(c[PEG].metadata['depth_evidence'].any()); self.assertFalse(sample_points(e, method='S4').priority_selected.any())
        d = f.depth.copy(); d[:, 32:] = .5
        e = self.evidence(replace(f, depth=d)); c = extract_candidates(e, SamplingConfig(), 'S4')
        self.assertTrue(c[PEG].metadata['depth_evidence'].any())
        self.assertTrue(c[PEG].metadata['occluding_evidence'].any()); self.assertTrue(c[PEG].metadata['occluded_evidence'].any())
        self.assertFalse(sample_points(e, method='S3').priority_selected.any())
        a = sample_points(e, method='S4'); self.assertGreater(a.depth_priority_selected.sum(), 0)
        self.assertLessEqual(a.depth_priority_selected.sum(), int(256*.25*.25))

    def test_all_false_candidates_still_respect_contract_and_roles(self):
        f, _, _ = synthetic_case('tilted', K, shape=(64, 64), phase_m=.00025)
        e = self.evidence(replace(f, depth=quantize(f.depth)), 'baseline')
        cfg = SamplingConfig(num_points=64, structural_ratio=.125, orientation_threshold_deg=0)
        a = sample_points(e, cfg, method='S1')
        self.assertLessEqual(a.priority_selected.sum(), 8)
        self.assertEqual(len(a.flat_index), len(np.unique(a.flat_index)))
        self.assertGreater(a.metadata['structural_candidate'].sum(), a.priority_selected.sum())
        self.assertTrue(a.metadata['structural_candidate'][a.priority_selected].all())

    def test_overlap_invalid_depth_and_input_immutability(self):
        f = self.frame(); d = f.depth.copy(); d[5, 5:10] = [0, -1, np.nan, np.inf, -np.inf]
        box = f.masks[BOX].copy(); box[4:9, 4:9] = True
        f = replace(f, depth=d, masks={PEG:f.masks[PEG], BOX:box})
        before = d.copy(); masks = {i:m.copy() for i,m in f.masks.items()}
        a = sample_points(self.evidence(f))
        self.assertTrue(np.isfinite(a.points).all()); v,u = a.pixels_vu.T
        self.assertFalse((f.masks[PEG]&f.masks[BOX])[v,u].any())
        np.testing.assert_array_equal(d, before)
        for i in masks: np.testing.assert_array_equal(masks[i], f.masks[i])

    def test_fps_seed_ties_initial_and_coincident_rows(self):
        points = np.zeros((20, 3)); points[:, 0] = np.arange(20)
        a = fps_indices(points, 10, seed=12)
        np.testing.assert_array_equal(a, fps_indices(points, 10, seed=12))
        b = fps_indices(points, 20, initial=a)
        np.testing.assert_array_equal(b[:10], a); self.assertEqual(len(np.unique(b)), 20)
        self.assertEqual(len(np.unique(fps_indices(np.zeros((20, 3)), 20))), 20)
        for bad in ([1, 1], [-1], [20]):
            with self.assertRaises(ValueError): fps_indices(points, 5, initial=bad)

    def test_normal_and_candidate_regression_against_task02_all_fields(self):
        for kind in ('front', 'tilted', 'crease', 'step'):
            f, _, _ = synthetic_case(kind, K, shape=(64, 64), phase_m=.00025)
            f = replace(f, depth=quantize(f.depth)); original, pairs = compare(f)
            for estimator in ('baseline', 'fd2', 'pca5'):
                actual = self.evidence(f, estimator)
                for oid in (PEG, BOX):
                    ef = original[estimator].objects[oid]; af = actual.frame.objects[oid]
                    for name in ef.__dataclass_fields__:
                        if name in ('normals', 'normal_variation_rad'):
                            np.testing.assert_allclose(getattr(af,name), getattr(ef,name), atol=2e-5, rtol=0)
                        else: np.testing.assert_array_equal(getattr(af,name),getattr(ef,name))
                    for t in (10, 20, 30, 40):
                        for name, arr in pairs[estimator][oid].candidates(t).items():
                            np.testing.assert_array_equal(actual.pairs[oid].candidates(t)[name], arr)
        f = self.frame(); baseline = process_geometry(f)
        actual = self.evidence(f, 'baseline')
        for oid in (PEG,BOX):
            for name in baseline.objects[oid].__dataclass_fields__:
                np.testing.assert_array_equal(getattr(baseline.objects[oid],name),getattr(actual.frame.objects[oid],name))

    def test_gt_not_accepted_and_rgb_cannot_change_selection(self):
        f = self.frame(); a = sample_points(self.evidence(f))
        altered = replace(f, rgb=255-f.rgb, cam2world=np.full((4,4),np.nan), tcp_pose=np.full(7,np.nan))
        b = sample_points(self.evidence(altered))
        np.testing.assert_array_equal(a.flat_index,b.flat_index)
        with self.assertRaises(TypeError): sample_points(self.evidence(), gt={})
        with self.assertRaises(ValueError): sample_points(self.evidence(), method='S1')


if __name__ == '__main__': unittest.main()
