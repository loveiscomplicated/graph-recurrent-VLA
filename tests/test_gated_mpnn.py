"""Regression checks for quaternion safety and model input validation."""

import unittest
from types import SimpleNamespace

from graph.build_graph import GraphBuilder

import torch

from graph.gated_mpnn import (
    VertexEdgeGatedMPNN,
    normalize_quat_wxyz,
)


class QuaternionNormalizationTests(unittest.TestCase):
    def test_degenerate_predictions_use_identity_with_finite_gradients(self):
        raw = torch.tensor(
            [[-1.0, 0.0, 0.0, 0.0], [-1.0, 1e-10, 0.0, 0.0]],
            requires_grad=True,
        )

        quaternion = normalize_quat_wxyz(raw)

        torch.testing.assert_close(
            quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
        )
        quaternion.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all().item())

    def test_regular_predictions_preserve_rotation_and_have_unit_norm(self):
        raw = torch.tensor([[0.0, 1.0, 2.0, 3.0], [-2.0, 1.0, 0.0, 0.0]])
        expected = torch.nn.functional.normalize(
            raw + torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=-1
        )
        expected = torch.where(expected[:, :1] < 0, -expected, expected)

        quaternion = normalize_quat_wxyz(raw)

        torch.testing.assert_close(quaternion, expected)
        torch.testing.assert_close(quaternion.norm(dim=-1), torch.ones(2))


class ModelValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = VertexEdgeGatedMPNN(2, hidden=16, layers=1)
        result = SimpleNamespace(
            xyz=torch.zeros(2, 3),
            object_id=torch.tensor([0, 1]),
            metadata=[{}, {}],
        )
        self.batch = GraphBuilder()(
            [result], torch.zeros(1, 3), semantic_ids=[torch.tensor([0, 1])],
        )

    def test_valid_batch_runs_without_explicit_validation(self):
        output = self.model(self.batch)

        self.assertEqual(output["object21"].shape, (1, 21))
        self.assertTrue(torch.isfinite(output["object21"]).all().item())

    def test_explicit_validation_rejects_invalid_semantic_id(self):
        self.batch.semantic_id[0, 0] = 2

        with self.assertRaisesRegex(ValueError, "Invalid semantic IDs"):
            self.batch.validate(num_semantic_ids=self.model.num_semantic_ids)

    def test_explicit_validation_rejects_edge_with_wrong_object_type(self):
        self.batch.edge_type[0] = 0

        with self.assertRaisesRegex(ValueError, "Edge type disagrees"):
            self.batch.validate(num_semantic_ids=self.model.num_semantic_ids)


if __name__ == "__main__":
    unittest.main()
