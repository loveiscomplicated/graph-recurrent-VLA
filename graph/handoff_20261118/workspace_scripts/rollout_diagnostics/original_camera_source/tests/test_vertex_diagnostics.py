"""Identity, clustered uncertainty and oracle-definition regression tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.gnn.vertex_diagnostics import MODELS, endpoint_components, oracle_replacements
from evaluation.gnn.vertex_metrics import cluster_interval, load_aligned_predictions


class VertexDiagnosticsTests(unittest.TestCase):
    def test_shared_helpers_keep_legacy_imports(self):
        from evaluation.gnn import vertex_diagnostics, vertex_metrics

        self.assertIs(vertex_diagnostics.load_aligned_predictions, vertex_metrics.load_aligned_predictions)
        self.assertIs(vertex_diagnostics.cluster_interval, vertex_metrics.cluster_interval)

    def fixture(self):
        target = np.zeros((3, 21), dtype=np.float32)
        target[:, 3] = target[:, 13] = 1
        target[:, 17] = .1
        target[:, 7] = .1
        prediction = target.copy()
        prediction[:, 0] += .01
        prediction[:, 10] += .03
        geometric = target[:, 7:10].copy()
        geometric[:, 0] -= .02
        residual = np.zeros((3, 3), dtype=np.float32)
        residual[:, 0] = .005
        prediction[:, 7:10] = geometric + residual
        return dict(sample_ids=np.array(['episode_b/frame_0', 'episode_a/frame_1', 'episode_a/frame_0']),
                    counts=np.ones((3, 3), dtype=np.int64), target=target,
                    object21=prediction, endpoint_geom=geometric, endpoint_residual=residual,
                    peg_half_size_x_error_mm=np.zeros(3))

    def test_alignment_joins_rows_and_rejects_changed_labels(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in MODELS:
                order = [0, 1, 2] if name == 'baseline' else [2, 0, 1]
                np.savez(directory / f'{name}_predictions.npz', **{k: v[order] for k, v in data.items()})
            predictions, hashes = load_aligned_predictions(directory)
            self.assertEqual(len(hashes), 4)
            np.testing.assert_array_equal(predictions['combined']['sample_ids'], data['sample_ids'])
            changed = {k: v.copy() for k, v in data.items()}
            changed['target'][1, 17] += .01
            np.savez(directory / 'combined_predictions.npz', **changed)
            with self.assertRaisesRegex(ValueError, 'target differs'):
                load_aligned_predictions(directory)

    def test_oracle_hole_is_corrected_endpoint_without_double_subtraction(self):
        data = self.fixture()
        components = endpoint_components(data)
        np.testing.assert_allclose(components['geometric_relative_error'][:, 0], -.02, atol=1e-8)
        np.testing.assert_allclose(components['final_relative_error'][:, 0], -.015, atol=1e-8)
        scores = oracle_replacements(data)
        self.assertAlmostEqual(scores['original']['relative_3d_mm']['mean'], 15, places=4)
        self.assertAlmostEqual(scores['gt_hole_position']['relative_3d_mm']['mean'], 15, places=4)
        self.assertAlmostEqual(scores['gt_peg_endpoint']['relative_3d_mm']['mean'], 25, places=4)
        self.assertAlmostEqual(scores['no_residual']['relative_3d_mm']['mean'], 20, places=4)

    def test_bootstrap_resamples_whole_episodes(self):
        values = np.array([-2., -2., 5.])
        result = cluster_interval(values, np.array(['a', 'a', 'b']), repetitions=1000)
        self.assertAlmostEqual(result['mean'], 1 / 3)
        self.assertEqual(result['low'], -2)
        self.assertEqual(result['high'], 5)
        self.assertEqual(result['clusters'], 2)
        self.assertIsNone(cluster_interval(np.array([]), np.array([])))


if __name__ == '__main__':
    unittest.main()
