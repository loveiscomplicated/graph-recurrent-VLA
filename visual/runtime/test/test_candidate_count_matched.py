"""Task03-H synthetic contracts; no actual episode, FPS or detector execution."""
import ast
from fractions import Fraction
import inspect
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.rgbd import BOX
from visual.runtime.test.sampling_evaluation import CoverageMatcher
from visual.runtime.test.run_candidate_count_matched import (
    SEEDS, STREAMS, array_sha, checked_rows, coverage_probability, describe,
    hash_files, preserved, probabilities, summary, uniform_rows,
)


class ExpectedCoverageTests(unittest.TestCase):
    def test_all_small_scalar_hypergeom_cases_against_rational(self):
        for n in range(9):
            for h in range(n+1):
                for k in range(n+1):
                    exact = Fraction(1)-Fraction(math.comb(n-h, k) if k <= n-h else 0, math.comb(n, k))
                    self.assertAlmostEqual(coverage_probability(n, h, k), float(exact), places=14)

    def test_expected_unique_exhaustive_shared_nodes_and_duplicate_references(self):
        a = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0],
                      [0, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]], bool)
        h = a.sum(axis=0)
        for k in range(len(a)+1):
            totals = [int(a[list(rows)].any(axis=0).sum()) for rows in itertools.combinations(range(len(a)), k)]
            expected = Fraction(sum(totals), len(totals))
            p = probabilities(len(a), h, k)
            self.assertAlmostEqual(math.fsum(p), float(expected), places=14)
        self.assertEqual(math.fsum(probabilities(len(a), h, len(a))), 3.)

    def test_stable_tiny_coverage_probability(self):
        # Subtracting exp(logq) from 1 loses this probability entirely.
        self.assertAlmostEqual(coverage_probability(10**18, 1, 1)/1e-18, 1., places=14)
        self.assertEqual(coverage_probability(0, 0, 0), 0.)
        self.assertEqual(coverage_probability(5, 0, 5), 0.)
        self.assertEqual(coverage_probability(5, 5, 1), 1.)
        self.assertEqual(coverage_probability(5, 3, 3), 1.)

    def test_invalid_ranges_are_rejected_without_clipping(self):
        for args in [(-1, 0, 0), (5, -1, 2), (5, 6, 2), (5, 1, -1), (5, 1, 6)]:
            with self.assertRaises(ValueError): coverage_probability(*args)
        with self.assertRaises(TypeError): coverage_probability(5, 1.5, 2)

    def test_monotone_saturating_curve_and_full_set(self):
        h = [0, 1, 2, 5, 10, 20]
        curve = np.array([math.fsum(probabilities(20, h, k)) for k in range(21)])
        self.assertTrue((np.diff(curve) >= -1e-14).all())
        self.assertTrue((np.diff(curve, n=2) <= 1e-14).all())
        self.assertEqual(curve[0], 0.)
        self.assertEqual(curve[-1], 5.)


class UniformSubsetTests(unittest.TestCase):
    def test_all_preregistered_draws_unique_fixed_k_and_membership(self):
        for stream, n in [(0, 599), (1, 11665)]:
            population = np.arange(n)*7+3
            before = population.copy()
            for seed in SEEDS:
                rows = uniform_rows(n, 297, seed, stream)
                self.assertEqual(len(rows), 297)
                self.assertEqual(len(np.unique(rows)), 297)
                self.assertTrue(np.all(rows[:-1] < rows[1:]))
                np.testing.assert_array_equal(checked_rows(population, population[rows]), rows)
                np.testing.assert_array_equal(rows, uniform_rows(n, 297, seed, stream))
            np.testing.assert_array_equal(population, before)
        self.assertEqual(SEEDS, tuple(range(20261024, 20261040)))
        self.assertEqual(STREAMS, {'pca5_matched': 0, 'observed_random': 1})

    def test_exact_pcg64_seedsequence_rule_and_independent_order(self):
        rows = uniform_rows(599, 297, SEEDS[0], 0)
        expected = np.sort(np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[0], 0]))).choice(599, 297, replace=False))
        np.testing.assert_array_equal(rows, expected)
        uniform_rows(11665, 297, SEEDS[-1], 1)
        np.random.seed(999); np.random.random(100)
        np.testing.assert_array_equal(rows, uniform_rows(599, 297, SEEDS[0], 0))
        self.assertFalse(np.array_equal(rows, uniform_rows(599, 297, SEEDS[0], 1)))

    def test_selector_has_no_gt_geometry_roi_score_or_result_access(self):
        self.assertEqual(list(inspect.signature(uniform_rows).parameters), ['n', 'k', 'seed', 'stream'])
        tree = ast.parse(inspect.getsource(uniform_rows))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {'gt', 'roi', 'xyz', 'score', 'matcher', 'labels', 'reference', 'result', 'open'})
        self.assertTrue(names <= {'n', 'k', 'seed', 'stream', 'map', 'index', 'min', 'ValueError', 'np', 'rng'})
        annotations = dict(gt=np.zeros(599), roi=np.zeros(599), xyz=np.zeros((599, 3)), score=np.zeros(599))
        before = uniform_rows(599, 297, SEEDS[0], 0)
        for key in annotations:
            annotations[key] = annotations[key][::-1]+1
        np.testing.assert_array_equal(before, uniform_rows(599, 297, SEEDS[0], 0))
        with self.assertRaises(TypeError): uniform_rows(599, 297, SEEDS[0], 0, gt=annotations['gt'])

    def test_empty_full_and_invalid_count(self):
        np.testing.assert_array_equal(uniform_rows(0, 0, 0, 0), np.array([], int))
        np.testing.assert_array_equal(uniform_rows(5, 5, 0, 0), np.arange(5))
        for args in [(5, 6, 0, 0), (5, -1, 0, 0), (5, 2, -1, 0), (5, 2, 0, -1)]:
            with self.assertRaises(ValueError): uniform_rows(*args)

    def test_source_membership_and_xyz_preserved(self):
        support = np.array([5, 9, 13, 17, 28], np.int64)
        xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
        before = array_sha(xyz)
        rows = uniform_rows(5, 3, 9, 0)
        selected = support[rows]
        np.testing.assert_array_equal(xyz[checked_rows(support, selected)], xyz[rows])
        self.assertEqual(before, array_sha(xyz))
        for bad in [np.array([5, 5]), np.array([-1]), np.array([100]), np.array([6]), np.array([5.])]:
            with self.assertRaises(ValueError): checked_rows(support, bad)
        with self.assertRaises(ValueError): checked_rows(np.array([9, 5]), np.array([5]))


class MatchingAndPreservationTests(unittest.TestCase):
    def geometry(self, xyz):
        valid = np.ones(xyz.shape[:2], bool)
        return OrganizedGeometry(xyz, valid, {BOX: valid}, np.zeros_like(valid))

    def test_same_pair_different_nodes_cannot_jointly_match_reference(self):
        xyz = np.full((1, 5, 3), 10.)
        xyz[0, 0] = [0, 0, 1.1]  # UV passes, XYZ fails
        xyz[0, 4] = [0, 0, 1.]   # XYZ passes, UV fails
        g = self.geometry(xyz)
        m = CoverageMatcher(g, BOX, (np.array([[0., 0.]]), np.array([[0., 0., 1.]])), ((2., .003),))
        ids, nodes = m.reference_hits(np.array([0, 4]))
        self.assertEqual((len(ids), len(nodes)), (0, 0))

    def test_inclusive_pair_boundaries_and_node_reference_distinction(self):
        xyz = np.zeros((1, 3, 3), np.float64)
        xyz[0, :, 0] = [.003, .003000001, 0]
        g = self.geometry(xyz)
        ref = (np.array([[0., 0.], [0., 0.], [10., 0.]]), np.zeros((3, 3)))
        m = CoverageMatcher(g, BOX, ref, ((2., .003),))
        a = m.adjacencies['2px_3mm']
        r = describe(a)
        self.assertEqual(r['node_count'], 3)
        self.assertEqual(r['matching_node_count'], 2)
        self.assertEqual(r['unique_reference_count'], 2)
        self.assertEqual(r['pair_count'], 4)
        self.assertEqual(r['reference_multiplicity'], [2, 2, 0])
        self.assertEqual(r['node_reference_degree'], [2, 0, 2])
        self.assertEqual(r['reference_denominator'], 3)
        self.assertEqual(r['rim_matching_node_fraction'], 2/3)
        ids, nodes = m.reference_hits(np.array([0]))
        self.assertEqual((len(nodes), len(ids)), (1, 2))

    def test_duplicate_pairs_do_not_count_as_multiple_nodes(self):
        a = csr_matrix((np.ones(3, bool), ([0, 0, 1], [0, 0, 0])), shape=(2, 2))
        r = describe(a)
        self.assertEqual(r['reference_multiplicity'], [2, 0])
        self.assertEqual(r['pair_count'], 2)
        self.assertEqual(r['unique_reference_count'], 1)

    def test_summary_uses_sample_standard_deviation(self):
        self.assertEqual(summary([1, 3]), dict(mean=2., sample_sd=math.sqrt(2), min=1., max=3.))

    def test_preservation_failure_detects_changed_file(self):
        # Keep the fixture under REPO so the same relative-path hash logic is tested.
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as folder:
            p = Path(folder)/'input.bin'; p.write_bytes(b'original')
            hashes = hash_files([p]); self.assertEqual(preserved(hashes)['status'], 'PASS')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): preserved(hashes)


if __name__ == '__main__': unittest.main()
