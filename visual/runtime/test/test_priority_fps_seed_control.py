"""Phase3 contracts. All FPS calls here use synthetic points, never Task03 data."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.rgbd import PEG, BOX
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_priority_fps_seed_control as control
from visual.runtime.test.sampling_evaluation import CoverageMatcher


class Archive(dict):
    @property
    def files(self):
        return list(self)


def synthetic_candidates():
    # Common source pixels always point at exactly the same source XYZ.
    def method(flat):
        xyz = np.column_stack((np.sin(flat*.3), np.cos(flat*.2), flat*.01)).astype(np.float32)
        return dict(flat=flat, points=xyz)
    return dict(S2=method(np.arange(100, 148)), S3=method(np.arange(108, 156)))


def saved_fixture():
    """Synthetic 256-row archives with the real shape/quota/field contracts."""
    shape = (512, 512)
    flat = np.arange(np.prod(shape))
    xyz = np.column_stack((flat*.001, flat*.002, np.ones(len(flat)))).astype(np.float32).reshape(*shape, 3)
    geometry = SimpleNamespace(xyz=xyz,
        observed_mask=lambda oid: (flat < 90000 if oid == PEG else flat >= 90000).reshape(shape))
    frame = SimpleNamespace(source_frame=0, depth=np.ones(shape, np.float32))
    archive = Archive(source_frame=np.array(0), shape=np.array(shape))
    config, metrics, phase2 = dict(conditions={}), {}, dict(methods={})
    field_names = ['object_mask', 'valid_points', 'normals', 'normal_valid', 'depth_jump_m',
        'depth_pair_valid', 'depth_discontinuity', 'mask_boundary', 'depth_validity_boundary',
        'image_boundary', 'normal_variation_rad', 'normal_variation_valid', 'normal_neighbor_count',
        'orientation_evidence', 'depth_evidence', 'occluding_evidence', 'occluded_evidence',
        'invalid_evidence', 'mask_evidence', 'orientation_score_deg', 'orientation_candidate',
        'depth_only_candidate', 'structural_candidate']
    for method, (estimator, _, count, _, _) in control.METHODS.items():
        candidates = np.sort(np.r_[np.arange(90000, 90000+count-3), control.FIRST[method]])
        archive[f'evidence/{estimator}/1/20/orientation_flat'] = candidates
        phase2['methods'][method] = dict(candidate_flat_indices=candidates.tolist())
        for seed in (0, 1, 2):
            key = f'{method}_n256_r0.25_t20_s{seed}'
            config['conditions'][key] = dict(method=method, num_points=256, structural_ratio=.25,
                                             orientation_threshold_deg=20, seed=seed)
            first = control.FIRST[method][seed]
            selected = np.r_[np.arange(129), first, candidates[candidates != first][:126]]
            box = np.arange(256) >= 129
            archive[key+'/flat_index'] = selected
            archive[key+'/points'] = xyz.reshape(-1, 3)[selected]
            archive[key+'/pixels_vu'] = np.column_stack(np.unravel_index(selected, shape))
            archive[key+'/object_id'] = box.astype(np.int64)
            archive[key+'/priority_selected'] = np.r_[np.arange(129) < 32, np.arange(127) < 32]
            archive[key+'/depth_priority_selected'] = np.zeros(256, bool)
            for name in field_names:
                value = np.zeros((256, 3), np.float32) if name == 'normals' else np.zeros(256, bool)
                if name in ('object_mask', 'valid_points'):
                    value[:] = True
                if name in ('orientation_candidate', 'structural_candidate'):
                    value[box] = True
                archive[key+'/metadata/'+name] = value
            objects = {}
            for name, budget in [('peg', 129), ('box', 127)]:
                objects[name] = dict(counts=dict(budget=budget, selected=budget, priority=32,
                    orientation_priority=32, depth_priority=0, surface_fps=budget-32))
            n = control.FINAL[method][seed]
            objects['box']['structural'] = dict(tolerance={'2px_3mm': dict(gt_visible=103, sampled=n, gt_coverage=n/103)})
            metrics[key] = dict(objects=objects)
    return archive, config, metrics, frame, geometry, phase2


class SharedInitialTests(unittest.TestCase):
    def select(self, candidates=None):
        return control.select_control(synthetic_candidates() if candidates is None else candidates,
                                      expected_shared=40, expected_initial=128)

    def test_initial_included_total_32_exact_local_global_no_rng(self):
        candidates = synthetic_candidates()
        before = copy.deepcopy(candidates)
        with patch.object(control, 'fps_indices', wraps=fps_indices) as calls:
            orders, info = self.select(candidates)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(info['candidate_local_index'], {'S2': 28, 'S3': 20})
        self.assertEqual(info['initial_rng_draws'], 0)
        for method, row in orders.items():
            self.assertEqual(len(row['flat']), 32)
            self.assertEqual(len(set(row['flat'])), 32)
            self.assertEqual(row['flat'][0], 128)
            np.testing.assert_array_equal(candidates[method]['flat'][row['local']], row['flat'])
            np.testing.assert_array_equal(candidates[method]['points'][row['local']], row['xyz'])
            for field in ('flat', 'points'):
                np.testing.assert_array_equal(candidates[method][field], before[method][field])
        np.testing.assert_array_equal(orders['S2']['xyz'][0], orders['S3']['xyz'][0])

    def test_initial_path_never_constructs_rng_even_when_seed_changes(self):
        points = synthetic_candidates()['S2']['points']
        with patch('visual.runtime.sampling.np.random.default_rng', side_effect=AssertionError('RNG draw')):
            a = fps_indices(points, 32, seed=0, initial=[28])
            b = fps_indices(points, 32, seed=98765, initial=[28])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[0], 28)

    def test_intersection_and_initial_mismatch_stop_before_fps(self):
        for kwargs in (dict(expected_shared=39, expected_initial=128), dict(expected_shared=40, expected_initial=127)):
            with patch.object(control, 'fps_indices', side_effect=AssertionError('must stop before FPS')):
                with self.assertRaises(ValueError):
                    control.select_control(synthetic_candidates(), **kwargs)

    def test_order_xyz_and_short_count_mismatch_stop_before_fps(self):
        for mutation in ('order', 'xyz', 'count'):
            c = synthetic_candidates()
            if mutation == 'order':
                c['S2']['flat'] = c['S2']['flat'][::-1]
            elif mutation == 'xyz':
                c['S3']['points'][20, 0] += .001
            else:
                c['S2'] = {k: v[:31] for k, v in c['S2'].items()}
            with patch.object(control, 'fps_indices', side_effect=RuntimeError('must stop before FPS')):
                with self.assertRaises((ValueError, AssertionError)):
                    self.select(c)

    def test_invalid_fps_order_is_rejected(self):
        for order in (np.zeros(32, np.int64), np.arange(31), np.r_[28, np.arange(30), 999]):
            with patch.object(control, 'fps_indices', return_value=order):
                with self.assertRaises(ValueError):
                    self.select()
        with patch.object(control, 'fps_indices', return_value=np.arange(32)):
            with self.assertRaisesRegex(ValueError, 'initial included'):
                self.select()

    def test_gt_annotation_change_cannot_change_control_selection(self):
        candidates = synthetic_candidates()
        orders_a, _ = self.select(candidates)
        xyz = np.zeros((1, 200, 3), np.float32)
        geometry = SimpleNamespace(valid_depth=np.ones((1, 200), bool), xyz=xyz,
                                   observed_mask=lambda oid: np.ones((1, 200), bool))
        frame = SimpleNamespace(depth=np.ones((1, 200), np.float32))
        annotations = [((np.array([[128., 0.]]), np.zeros((1, 3))), np.zeros((1, 200), bool)),
                       ((np.array([[999., 999.]]), np.ones((1, 3))*100), np.ones((1, 200), bool))]
        scores = []
        for refs, negative in annotations:
            matcher = CoverageMatcher(geometry, BOX, refs)
            scores.append(control.score_control(orders_a, matcher, frame, {'negative': negative}))
        orders_b, _ = self.select(candidates)  # synthetic repeat only
        for method in control.METHODS:
            np.testing.assert_array_equal(orders_a[method]['flat'], orders_b[method]['flat'])
            self.assertNotEqual(scores[0][method]['negative_roi'], scores[1][method]['negative_roi'])
        with self.assertRaises(TypeError):
            control.select_control(candidates, gt={})


class SavedAnalysisTests(unittest.TestCase):
    def test_source_feature_role_alignment_without_fps(self):
        fixture = saved_fixture()
        with control.forbid_sampling():
            candidates, selected, checks = control.load_saved(*fixture)
        self.assertEqual(len(checks), 6)
        for method in control.METHODS:
            self.assertEqual(len(candidates[method]['flat']), control.METHODS[method][2])
            for seed in (0, 1, 2):
                self.assertEqual(selected[method][seed]['flat'][0], control.FIRST[method][seed])
        self.assertTrue(all(c['feature_and_role_alignment'] for c in checks.values()))

    def test_corrupt_source_feature_role_or_order_fails(self):
        key = 'S2_n256_r0.25_t20_s1'
        for field in ('points', 'metadata/normals', 'metadata/orientation_candidate', 'priority_selected', 'pixels_vu'):
            fixture = saved_fixture()
            array = fixture[0][key+'/'+field]
            fixture[0][key+'/'+field] = array.copy()
            array = fixture[0][key+'/'+field]
            if field == 'metadata/normals':
                array[0, 0] = 1  # same source pixel, inconsistent saved feature
            elif field == 'priority_selected':
                array[129] = False
            elif field == 'metadata/orientation_candidate':
                array[129] = False
            else:
                array[0, 0] += 1
            with control.forbid_sampling(), self.assertRaises((ValueError, AssertionError)):
                control.load_saved(*fixture)

    def test_seed0_reused_and_only_seed1_2_matched_without_fps(self):
        flat = np.arange(32, dtype=np.int64)
        xyz = np.zeros((32, 3), np.float32)
        frame = SimpleNamespace(depth=np.ones((1, 32), np.float32))
        negatives = dict(negative=np.zeros((1, 32), bool))
        rows = {m: {s: dict(flat=flat, xyz=xyz, final=dict(sampled=10, gt_coverage=10/103))
                    for s in (0, 1, 2)} for m in control.METHODS}
        trace = [dict(xyz_m=p.tolist(), flat_index=int(f), depth_m=1., rim_reference_matched=i == 1,
                      matched_rim_reference_ids=[2, 3] if i == 1 else [],
                      selected_negative_roi=dict(negative=False, known_negative_union=False))
                 for i, (p, f) in enumerate(zip(xyz, flat))]
        phase2 = dict(methods={m: dict(stored_priority_flat_indices=flat.tolist(), trace=trace,
                    priority_reference_ids=[2, 3], priority_reference_count=2) for m in control.METHODS})
        phase1 = dict(frames=[dict(conditions={m: dict(reference_ids=dict(P=[2, 3])) for m in control.METHODS})])
        calls = []
        def match(selected, **kwargs):
            calls.append(selected.copy())
            self.assertEqual(kwargs, dict(px=2., m=.003))
            return np.array([2, 3]), selected[1:2]
        with control.forbid_sampling():
            result = control.summarize_saved(rows, phase1, phase2, SimpleNamespace(reference_hits=match), frame, negatives)
        self.assertEqual(len(calls), 4)
        for method in control.METHODS:
            self.assertEqual(result[method]['0']['priority_reference_ids'], [2, 3])
            self.assertEqual(result[method]['0']['rim_matching_node_count'], 1)
            self.assertEqual(result[method]['0']['final_sampling']['numerator'], 10)
        phase2['methods']['S2']['priority_reference_ids'] = [2, 4]
        with control.forbid_sampling(), self.assertRaisesRegex(ValueError, 'Phase1/2 IDs'):
            control.summarize_saved(rows, phase1, phase2, SimpleNamespace(reference_hits=match), frame, negatives)

    def test_joint_same_pair_and_reference_dedup(self):
        shape = (1, 10)
        xyz = np.zeros((*shape, 3), np.float32)
        xyz[0, 3] = [.1, 0, 0]
        geometry = SimpleNamespace(valid_depth=np.ones(shape, bool), xyz=xyz,
                                   observed_mask=lambda oid: np.ones(shape, bool))
        # Selected nodes: 0 is near ref 0 only in 2D, 3 only in XYZ.
        refs = (np.array([[0., 0.], [7., 0.]]), np.array([[.1, 0, 0], [0, 0, 0]]))
        matcher = CoverageMatcher(geometry, BOX, refs)
        ids, hits = matcher.reference_hits(np.array([0, 3], np.int64))
        self.assertEqual(ids.tolist(), [])
        self.assertEqual(hits.tolist(), [])
        ids, hits = matcher.reference_hits(np.array([6, 7, 8], np.int64))
        self.assertEqual(ids.tolist(), [1])
        self.assertEqual(len(hits), 3)
        frame = SimpleNamespace(depth=np.ones(shape, np.float32))
        a = np.zeros(shape, bool); b = a.copy(); a[0, [6, 7]] = True; b[0, [7, 8]] = True
        summary = control.point_summary(np.array([6, 7, 8]), frame, dict(a=a, b=b), ids, hits, xyz[0, [6, 7, 8]])
        self.assertEqual(summary['negative_roi']['known_negative_union']['count'], 3)
        self.assertEqual(summary['priority_reference_count'], 1)
        with self.assertRaisesRegex(ValueError, 'deduplicated'):
            control.point_summary(np.array([6, 7, 8]), frame, dict(a=a), [1, 1], hits, xyz[0, [6, 7, 8]])


if __name__ == '__main__':
    unittest.main()
