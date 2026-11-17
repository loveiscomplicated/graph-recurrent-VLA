"""Corner-to-GNN contracts, padding isolation, and trainable detector handoff."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from graph.basic_gnn import BasicGNN
from graph.build_graph import (
    ConfidenceSettings, EdgeType, GraphBuilder, GraphSettings, NodeType,
    compute_confidence,
)
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss


def corners(points=None, objects=None, metadata=None):
    """Create a detector-shaped observation with distinct peg and box points."""
    if points is None:
        points = [[0.0, 0, 0], [0.03, 0, 0], [0.1, 0.01, 0], [0.1, 0.03, 0]]
    if objects is None:
        objects = [0, 0, 1, 1]
    return SimpleNamespace(
        xyz=torch.tensor(points, dtype=torch.float32).reshape(-1, 3),
        object_id=torch.tensor(objects, dtype=torch.long),
        metadata=[{} for _ in objects] if metadata is None else metadata,
    )


def models():
    """Both models consume the same GraphBatch, with different prediction tasks."""
    return (
        BasicGNN(2, 8, 16, 4, 8, dropout=0),
        VertexEdgeGatedMPNN(4, hidden=16, layers=1,
                           use_confidence=True, use_semantic_ids=True),
    )


def prediction(model, graph):
    output = model(graph)
    return output['object21'] if isinstance(output, dict) else output


class ConfidenceTests(unittest.TestCase):
    def test_missing_evidence_and_zero_quality_are_distinct(self):
        scores, valid = compute_confidence([{}, {'supporting_planes': 0}])
        torch.testing.assert_close(scores, torch.zeros(2))
        self.assertEqual(valid.tolist(), [False, True])

    def test_available_evidence_formula_and_monotonicity(self):
        scores, valid = compute_confidence([
            {'line_residual_px': 0}, {'line_residual_px': 2},
            {'supporting_planes': 1}, {'supporting_planes': 3},
            {'sources': 1}, {'sources': 3},
            {'line_residual_px': 0, 'supporting_planes': 3, 'sources': 3},
        ])
        self.assertTrue(valid.all())
        self.assertAlmostEqual(scores[1].item(), np.exp(-1), places=6)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[3], scores[2])
        self.assertGreater(scores[5], scores[4])
        self.assertAlmostEqual(scores[-1].item(), 1)
        # Missing residual is excluded rather than treated as a perfect fit.
        score, _ = compute_confidence([{'supporting_planes': 1}])
        self.assertAlmostEqual(score.item(), 1 / 3, places=6)

    def test_invalid_settings_and_evidence_fail_at_input(self):
        for kwargs in ({'residual_scale_px': 0}, {'source_weight': -1},
                       {'plane_reference': float('nan')},
                       {'residual_weight': 0, 'plane_weight': 0, 'source_weight': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConfidenceSettings(**kwargs)
        for value in (-1, float('nan'), True, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_confidence([{'line_residual_px': value}])


class GraphBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = GraphBuilder()
        self.result = corners()
        self.tcp = torch.tensor([[0.02, 0.01, 0]])

    def test_box_is_preserved_and_semantics_are_unknown(self):
        graph = self.builder([self.result], self.tcp)
        self.assertEqual(graph.object_type.tolist(), [[0, 0, 1, 1]])
        self.assertFalse(graph.semantic_valid.any())
        self.assertTrue((graph.semantic_id == -1).all())
        self.assertFalse(graph.confidence_valid.any())
        self.assertEqual(graph.edge_index.shape, (2, 12))
        self.assertEqual(set(graph.edge_type.tolist()), {EdgeType.PEG, EdgeType.BOX, EdgeType.CROSS})
        torch.testing.assert_close(graph.xyz_world[0], self.result.xyz)
        src, dst = graph.edge_index
        torch.testing.assert_close(graph.relative_xyz, self.result.xyz[src] - self.result.xyz[dst])
        self.assertFalse(graph.xyz_world.is_inference())

    def test_explicit_hole_and_semantic_labels(self):
        graph = self.builder([self.result], self.tcp,
                             object_types=[torch.tensor([0, 0, 1, 2])],
                             semantic_ids=[torch.tensor([0, -1, 2, -1])])
        self.assertEqual(graph.object_type[0, -1], NodeType.HOLE)
        self.assertEqual(graph.semantic_valid.tolist(), [[True, False, True, False]])
        graph.validate(num_semantic_ids=3)

    def test_batch_offsets_padding_and_all_empty(self):
        empty = corners([], [])
        single = corners([[0.1, 0, 0]], [1])
        graph = self.builder([self.result, single, empty], self.tcp.expand(3, -1), pad_to=7)
        self.assertEqual(graph.node_valid.sum(1).tolist(), [4, 1, 0])
        self.assertTrue((graph.object_type[~graph.node_valid] == NodeType.PAD).all())
        graph.validate()
        # Repeat nonempty graphs so the second sample has real offset edges.
        graph = self.builder([self.result, self.result], self.tcp.expand(2, -1), pad_to=7)
        self.assertEqual((graph.edge_index[0] // 7).bincount().tolist(), [12, 12])
        empty_graph = self.builder([empty, empty], torch.zeros(2, 3))
        self.assertEqual(empty_graph.xyz_world.shape, (2, 1, 3))
        self.assertEqual(empty_graph.edge_index.shape, (2, 0))
        empty_graph.to('cpu').validate()

    def test_knn_selects_nearest_senders_without_self_edges(self):
        result = corners([[0, 0, 0], [0.01, 0, 0], [0.04, 0, 0], [0.1, 0, 0]], [0, 0, 0, 1])
        graph = GraphBuilder(GraphSettings(intra_k=1, cross_k=1))([result], self.tcp)
        senders = graph.edge_index[0, graph.edge_index[1] == 0].tolist()
        self.assertEqual(senders, [1, 3])
        graph.validate()

    def test_invalid_detector_inputs_and_options(self):
        bad = deepcopy(self.result)
        bad.object_id[0] = 2
        with self.assertRaisesRegex(ValueError, 'Detector object_id'):
            self.builder([bad], self.tcp)
        bad = deepcopy(self.result)
        bad.metadata.pop()
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.builder([bad], self.tcp)
        bad = deepcopy(self.result)
        bad.xyz[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.builder([bad], self.tcp)
        with self.assertRaisesRegex(ValueError, 'pad_to'):
            self.builder([self.result], self.tcp, pad_to=2)
        with self.assertRaises(ValueError):
            self.builder([self.result], self.tcp, semantic_ids=[torch.tensor([-2, 0, 0, 0])])
        with self.assertRaises(ValueError):
            GraphSettings(cross_k=0)

    def test_validation_rejects_padding_cross_sample_and_wrong_geometry(self):
        graph = self.builder([self.result, self.result], self.tcp.expand(2, -1), pad_to=5)
        for name, change, message in (
            ('edge_index', lambda t: t.__setitem__((0, 0), 4), 'padding'),
            ('edge_index', lambda t: t.__setitem__((0, 0), 5), 'cross batch'),
            ('relative_xyz', lambda t: t.__setitem__((0, 0), 123), 'sender minus'),
            ('distance', lambda t: t.__setitem__((0, 0), 123), 'edge length'),
            ('node_valid', lambda t: t.__setitem__((0, 0), False), 'Padding'),
        ):
            bad = deepcopy(graph)
            change(getattr(bad, name))
            with self.subTest(name=name, message=message), self.assertRaisesRegex(ValueError, message):
                bad.validate()


class GraphModelContractTests(unittest.TestCase):
    def test_compiled_gated_calls_match_eager_across_padding_and_empty_graphs(self):
        """Trace GraphBatch calls on CPU without running CUDA/Inductor."""
        reference = VertexEdgeGatedMPNN(4, hidden=16, layers=1)
        compiled = deepcopy(reference)
        compiled.compile(backend='eager', dynamic=True)
        builder = GraphBuilder()
        for observation, padding in ((corners(), 4), (corners(), 9), (corners([], []), 1)):
            with self.subTest(padding=padding):
                graph = builder([observation], torch.zeros(1, 3), pad_to=padding)
                reference.zero_grad(set_to_none=True)
                compiled.zero_grad(set_to_none=True)
                expected = reference(graph)
                actual = compiled(graph)
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name])
                expected['object21'].sum().backward()
                actual['object21'].sum().backward()
                for parameter, other in zip(compiled.parameters(), reference.parameters()):
                    if other.grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, other.grad)

    def test_absent_edge_groups_keep_none_gradients(self):
        for object_type in (0, 1):
            with self.subTest(object_type=object_type):
                graph = GraphBuilder()(
                    [corners([[0, 0, 0], [0.03, 0, 0]], [object_type, object_type])],
                    torch.zeros(1, 3), pad_to=5,
                )
                model = VertexEdgeGatedMPNN(4, hidden=16, layers=1)
                model(graph)['object21'].sum().backward()
                inactive_layers = model.hole_layers if object_type == 0 else model.peg_layers
                for layer in [*inactive_layers, model.cross_layer]:
                    self.assertTrue(all(parameter.grad is None for parameter in layer.parameters()))

    def test_padding_batch_composition_and_permutation_preserve_predictions(self):
        result = corners(metadata=[{'supporting_planes': 2}] * 4)
        order = torch.tensor([2, 0, 3, 1])
        permuted = corners(result.xyz[order].tolist(), result.object_id[order].tolist(),
                           [result.metadata[i] for i in order])
        builder = GraphBuilder()
        tcp = torch.tensor([[0.02, 0.01, 0]])
        graph = builder([result], tcp)
        padded = builder([result], tcp, pad_to=9)
        reordered = builder([permuted], tcp)
        mixed = builder([result, corners([], [])], tcp.expand(2, -1), pad_to=9)
        for model in models():
            model.eval()
            with self.subTest(model=type(model).__name__):
                expected = prediction(model, graph)
                actual = prediction(model, padded)
                batched = prediction(model, mixed)
                permuted_prediction = prediction(model, reordered)
                if isinstance(model, BasicGNN):
                    actual = actual[:, :4]
                    batched = batched[:1, :4]
                    permuted_prediction = permuted_prediction[:, order.argsort()]
                    self.assertTrue((prediction(model, padded)[:, 4:] == 0).all())
                else:
                    batched = batched[:1]
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(batched, expected, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(permuted_prediction, expected, atol=1e-6, rtol=1e-5)

    def test_padding_does_not_change_parameter_gradients(self):
        builder = GraphBuilder()
        graph = builder([corners()], torch.zeros(1, 3))
        padded = builder([corners()], torch.zeros(1, 3), pad_to=11)
        for model in models():
            original = deepcopy(model)
            prediction(model, graph).sum().backward()
            prediction(original, padded).sum().backward()
            for name, parameter in model.named_parameters():
                other = dict(original.named_parameters())[name]
                with self.subTest(model=type(model).__name__, parameter=name):
                    if parameter.grad is None:
                        self.assertIsNone(other.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-5, rtol=1e-5)

    def test_empty_graphs_are_finite_and_backward_is_safe(self):
        graph = GraphBuilder()([corners([], [])], torch.zeros(1, 3))
        for model in models():
            output = prediction(model, graph)
            self.assertTrue(torch.isfinite(output).all())
            output.sum().backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_box_support_does_not_claim_observed_hole(self):
        graph = GraphBuilder()([corners()], torch.zeros(1, 3))
        output = models()[1](graph)
        self.assertTrue(output['box_observed'].all())
        self.assertTrue(output['target_support_observed'].all())
        self.assertFalse(output['hole_observed'].any())

    def test_optional_features_can_be_disabled_and_unknown_is_neutral(self):
        builder = GraphBuilder()
        unknown = builder([corners()], torch.zeros(1, 3))
        known = builder([corners(metadata=[{'supporting_planes': 1}] * 4)], torch.zeros(1, 3),
                        semantic_ids=[torch.tensor([0, 1, 2, 3])])
        model = VertexEdgeGatedMPNN(4, hidden=16, layers=1)
        torch.testing.assert_close(model(unknown)['object21'], model(known)['object21'])
        model.use_confidence = True
        model.use_semantic_ids = True
        self.assertTrue(torch.isfinite(model(unknown)['object21']).all())
        self.assertTrue(model(unknown)['peg_observed'].all())

    def test_actual_visual_result_supports_loss_backward_and_optimizer_step(self):
        from visual.corner_detection import detect_tensors

        rgb = np.zeros((512, 512, 3), np.uint8)
        depth = np.full((512, 512), 2000, np.int16)
        depth[100:200, 120:240] = 1000
        masks = np.zeros((2, 512, 512), bool)
        masks[0, 100:200, 120:240] = True
        intrinsic = np.array([[500., 0, 256], [0, 500, 256], [0, 0, 1.]])
        extrinsic = np.c_[np.eye(3), np.array([0.1, 0.2, 0.3])]
        result = detect_tensors(rgb, depth, intrinsic, extrinsic, masks=masks, device='cpu')
        self.assertEqual(len(result.xyz), 4)
        self.assertTrue(result.xyz.is_inference())
        graph = GraphBuilder()([result], torch.zeros(1, 3), pad_to=7)
        for value in asdict(graph).values():
            self.assertFalse(value.is_inference())
        # Avoid random all-inactive ReLU outputs in the BasicGNN fixture, while
        # preserving the caller's RNG state for the other contract tests.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            test_models = models()
        for model in test_models:
            with self.subTest(model=type(model).__name__):
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                before = model.layers[0].update_layer.weight.detach().clone() if isinstance(model, BasicGNN) else model.peg_head[0].weight.detach().clone()
                output = model(graph)
                if isinstance(model, BasicGNN):
                    loss = output[graph.node_valid].square().mean()
                else:
                    target = torch.zeros(1, 21)
                    target[:, 3] = target[:, 13] = 1
                    target[:, 17:21] = torch.tensor([0.04, 0.01, 0.01, 0.02])
                    loss = object21_loss(output, target)['total']
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                optimizer.step()
                after = model.layers[0].update_layer.weight if isinstance(model, BasicGNN) else model.peg_head[0].weight
                self.assertFalse(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
