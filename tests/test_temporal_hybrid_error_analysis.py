"""Analytic regression cases for diagnostic identities, oracle coupling and axes."""

import unittest

import numpy as np
import torch
import yaml

from evaluation.gnn.temporal.hybrid_diagnostics import paired_bootstrap
from evaluation.gnn.temporal.hybrid_error_analysis import (
    cancellation_changes,
    prepare_geometry,
)
from evaluation.gnn.temporal.hybrid_error_geometry import (
    endpoint_factor_vectors,
    endpoint_interventions,
    endpoint_position,
    factor_squared_terms,
    geometry_groups,
    local_vectors,
    ordered_oracle_reductions,
    squared_decomposition,
)
from evaluation.gnn.temporal.insertion import insertion_errors
from graph.pose_geometry import geometric_endpoint, rotvec_quaternion


def objects(count=1):
    value = np.zeros((count, 21), dtype=np.float64)
    value[:, (3, 13)] = 1
    value[:, 17:21] = [0.05, 0.01, 0.01, 0.013]
    value[:, 7:10] = geometric_endpoint(torch.tensor(value)).numpy()
    return value


class HybridErrorDiagnosticsTests(unittest.TestCase):
    def test_canonical_geometry_tolerates_quaternion_renormalization_without_changing_scores(self):
        target = objects(2).astype(np.float32)
        mamba = target.copy()
        mamba[:, 3:7] = rotvec_quaternion(torch.tensor([[0.1, 0.2, 0.7]])).numpy()
        mamba[:, 7:10] = geometric_endpoint(torch.from_numpy(mamba)).numpy()
        hybrid = mamba.copy()
        norm = np.linalg.norm(hybrid[:, 3:7], axis=1, keepdims=True)
        hybrid[:, 3:7] /= norm
        hybrid[:, 10:13] += 0.002
        hybrid[:, 7:10] = geometric_endpoint(torch.from_numpy(hybrid)).numpy()
        predictions = {"mamba": mamba, "hybrid_h1_a": hybrid}
        summary = {"methods": {}}
        for name, value in predictions.items():
            metrics = {
                metric: {"mean_deg" if "rotation" in metric else "mean_mm": float(values.mean())}
                for metric, values in insertion_errors(value, target).items()
            }
            summary["methods"][name] = {"groups": {"all": metrics}}
        _, canonical, _, residuals = prepare_geometry({"target": target}, predictions, summary)
        np.testing.assert_array_equal(
            endpoint_position(canonical["mamba"]), endpoint_position(canonical["hybrid_h1_a"])
        )
        self.assertLess(residuals["shared_quaternion_max_component_difference"], 1e-7)

    def test_squared_identity_cancellation_amplification_and_undefined_cosine(self):
        endpoint = np.array([[0.003, 0, 0], [0.003, 0, 0], [0, 0, 0]])
        hole = np.array([[0.002, 0, 0], [-0.002, 0, 0], [0.001, 0, 0]])
        terms = squared_decomposition(endpoint, hole)
        np.testing.assert_allclose(terms["relative_squared_mm2"], [1, 25, 1])
        np.testing.assert_allclose(terms["cross_term_mm2"], [-12, 12, 0])
        np.testing.assert_allclose(terms["cosine"][:2], [1, -1])
        self.assertTrue(np.isnan(terms["cosine"][2]))

    def test_endpoint_oracles_cover_all_subsets_and_size_yz_is_irrelevant(self):
        target = objects()
        prediction = target.copy()
        prediction[0, 0] += 0.003
        prediction[0, 17] += 0.004
        values = endpoint_interventions(prediction, target)
        np.testing.assert_allclose(values[0], [[0.007, 0, 0]], atol=1e-14)
        np.testing.assert_allclose(values[1], [[0.004, 0, 0]], atol=1e-14)
        np.testing.assert_allclose(values[4], [[0.003, 0, 0]], atol=1e-14)
        np.testing.assert_array_equal(values[7], 0)
        prediction[:, 18:20] *= 100
        np.testing.assert_allclose(endpoint_interventions(prediction, target)[0], values[0])
        # Shared TCP translation leaves vector errors invariant.
        prediction[:, :3] += [1, -2, 3]
        target[:, :3] += [1, -2, 3]
        np.testing.assert_allclose(
            endpoint_interventions(prediction, target)[0], values[0], atol=1e-14
        )

    def test_rotation_size_coupling_and_all_squared_cross_terms_are_exact(self):
        target = objects()
        prediction = target.copy()
        prediction[:, :3] += [0.003, -0.001, 0.002]
        prediction[:, 3:7] = rotvec_quaternion(torch.tensor([[0.0, 0.0, 0.3]])).numpy()
        prediction[:, 17] += 0.007
        factors = endpoint_factor_vectors(prediction, target)
        error = endpoint_position(prediction) - endpoint_position(target)
        np.testing.assert_allclose(sum(factors.values()), error, atol=1e-14)
        self.assertGreater(np.linalg.norm(factors["rotation_size_interaction"]), 0.001)
        np.testing.assert_allclose(
            sum(factor_squared_terms(factors).values()), np.sum(error**2, axis=1) * 1e6, atol=1e-10
        )

    def test_oracle_order_effects_are_not_independent_norm_contributions(self):
        target = objects()
        prediction = target.copy()
        prediction[:, 0] += 0.003
        prediction[:, 17] += 0.004
        values = endpoint_interventions(prediction, target)
        result = ordered_oracle_reductions(values, np.array([True]))
        allocation = result["order_averaged_mse_reduction_mm2"]
        self.assertEqual(len(result["orders"]), 6)
        self.assertAlmostEqual(sum(row["mean"] for row in allocation.values()), 49)
        self.assertLess(allocation["translation"]["minimum"], allocation["translation"]["maximum"])
        # Replacing translation first reduces 49 to 16; replacing it last reduces 9 to 0.
        self.assertAlmostEqual(allocation["translation"]["minimum"], 9)
        self.assertAlmostEqual(allocation["translation"]["maximum"], 33)

    def test_rotated_hole_axes_and_quaternion_sign_preserve_projection_energy(self):
        target = objects(2)
        target[:, 13:17] = rotvec_quaternion(
            torch.tensor([[0.0, 0.0, np.pi / 2]], dtype=torch.float64)
        ).numpy()
        target[1, 13:17] *= -1
        vectors = np.array([[0.002, 0.003, 0.004], [0.002, 0.003, 0.004]])
        local = local_vectors(vectors, target)
        np.testing.assert_allclose(local, [[0.003, -0.002, 0.004]] * 2, atol=1e-14)
        np.testing.assert_allclose(np.sum(local**2, axis=1), np.sum(vectors**2, axis=1), atol=1e-14)

    def test_geometry_proxy_priority_partition_and_boundaries(self):
        with open("graph/configs/temporal/hybrid/analyze_errors.yaml") as source:
            settings = yaml.safe_load(source)["geometry_proxies"]
        target = objects(5)
        target[:, 7:10] = [
            [-0.5, 0, 0],
            [-0.18, 0.04, 0],
            [-0.10, 0.02, 0],
            [-0.03, 0, 0],
            [0, 0, 0],
        ]
        groups, _ = geometry_groups(target, settings)
        for index, name in enumerate(
            ("distant", "pre_alignment", "alignment", "insertion_approach", "head_in_region")
        ):
            self.assertEqual(np.flatnonzero(groups[name + "_geometry_proxy"]).tolist(), [index])
        self.assertEqual(int(groups["historical_alignment_geometry_proxy"].sum()), 2)
        settings["approach_min_x_m"] = settings["head_min_x_m"]
        with self.assertRaisesRegex(ValueError, "approach_min_x"):
            geometry_groups(target, settings)

    def test_four_cancellation_quadrants_and_axis_change_identity(self):
        # Each endpoint error is 3mm. Rows exercise all four strict outcome combinations.
        target = objects(4)
        endpoint = np.tile([0.003, 0, 0], (4, 1))
        old = np.array([[0.010, 0, 0], [0.003, 0, 0], [0, 0, 0], [0.001, 0, 0]])
        new = np.array([[0.005, 0, 0], [0, 0, 0], [0.002, 0, 0], [-0.005, 0, 0]])
        errors = {}
        for name, hole in (("mamba", old), ("hybrid_h1_a", new)):
            errors[name] = {
                "hole_position": np.linalg.norm(hole, axis=1) * 1000,
                "relative_3d": np.linalg.norm(endpoint - hole, axis=1) * 1000,
            }
        result, arrays = cancellation_changes(
            {},
            endpoint,
            {"mamba": old, "hybrid_h1_a": new},
            target,
            errors,
            {"all": np.ones(4, dtype=bool)},
        )
        for name, row in result["quadrants"].items():
            self.assertEqual(row["frames"], 0 if name == "tie_in_either_metric" else 1)
        axis = result["axis_change_by_group"]["all"]["x_axial"]
        self.assertAlmostEqual(
            axis["relative_squared_delta_mm2"],
            axis["lost_cancellation_mm2"] - axis["hole_squared_gain_mm2"],
        )
        self.assertEqual(len(arrays["quadrant"]), 4)

    def test_parent_bootstrap_does_not_count_sibling_episodes_as_independent(self):
        data = {"episode": np.array(["first"] * 20 + ["second"] * 2)}
        manifest = {"parent_by_episode": {"first": "one_parent", "second": "one_parent"}}
        result = paired_bootstrap(data, np.array([-1.0] * 20 + [3.0] * 2), manifest, 0, 100)
        self.assertEqual(result["parents"], 1)
        self.assertEqual(result["episode_equal_percentile_ci_mm"], [1.0, 1.0])
        self.assertNotEqual(
            result["frame_weighted_mean_delta_mm"], result["episode_equal_mean_delta_mm"]
        )


if __name__ == "__main__":
    unittest.main()
