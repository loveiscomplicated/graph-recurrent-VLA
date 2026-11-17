"""Contracts for GNN inputs from detector-only online rollouts."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from graph.gated_mpnn import VertexEdgeGatedMPNN
from evaluation.detected_corner_gnn_comparison import comparable_config, evaluate_run
from training.detected_corner_gnn import graph_batch, rollout_episode
from training.gnn.loss import object21_loss


class ZeroPredictor(nn.Module):
    def forward(self, inputs, context):
        return torch.zeros((*inputs.shape[:2], 3), device=inputs.device)


class DetectedCornerGNNTests(unittest.TestCase):
    def test_rollout_keeps_all_unknown_detections_and_never_reads_gt(self):
        source = dict(
            xyz=np.array([[[0., 0., 0.], [.1, 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32),
            valid=np.array([[True, True], [False, False]]), kind=np.array([[0, 1], [0, 1]], dtype=np.int64),
            robot=np.zeros((2, 9), dtype=np.float32),
        )
        frames, counts = rollout_episode(ZeroPredictor(), source, .03, "cpu")
        self.assertEqual(counts["final_tracks"], 2)
        np.testing.assert_array_equal(frames[0]["xyz"], source["xyz"][0])
        self.assertEqual(frames[0]["kinds"].tolist(), [0, 1])
        self.assertFalse(frames[1]["observed"].any())

    def test_gnn_uses_detector_type_and_detection_flag_only(self):
        xyz = torch.tensor([[[.01, 0., 0.], [.02, 0., 0.], [0., 0., 0.]]])
        kinds = torch.tensor([[0, 1, -1]])
        valid = torch.tensor([[True, True, False]])
        observed = torch.tensor([[True, False, False]])
        graph, feature = graph_batch(xyz, kinds, valid, observed, torch.zeros(1, 3))
        graph.validate()
        self.assertFalse(graph.semantic_valid.any())
        self.assertEqual(graph.object_type.tolist(), [[0, 1, -1]])
        self.assertEqual(feature[0, :, 0].tolist(), [1., 0., 0.])
        gnn = VertexEdgeGatedMPNN(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1)
        target = torch.zeros(1, 21)
        target[:, 3] = target[:, 13] = 1
        target[:, 17:21] = .01
        loss = object21_loss(gnn(graph, feature), target)["total"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in gnn.parameters()))

    def test_comparison_replays_aligned_frames_and_rejects_wrong_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.pt"
            model_config = dict(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1)
            model = VertexEdgeGatedMPNN(**model_config)
            target = np.zeros((2, 21), dtype=np.float32)
            target[:, 3] = target[:, 13] = 1
            target[:, 17:21] = .01
            arrays = dict(xyz=np.zeros((2, 2, 3), dtype=np.float32),
                          kinds=np.array([[0, 1], [0, 1]], dtype=np.int64),
                          valid=np.ones((2, 2), dtype=bool), observed=np.ones((2, 2), dtype=bool),
                          tcp=np.zeros((2, 3), dtype=np.float32), target=target,
                          split=np.array([False, True]), sample_ids=np.array(["train", "validation"]))
            torch.save(dict(arrays=arrays), path)
            run = dict(cache_path=path, checkpoint=dict(model=model.state_dict(),
                       config=dict(gnn=model_config, training=dict(batch_size=2))))
            prediction, errors = evaluate_run(run, np.array(["validation"]), target[1:], torch.device("cpu"))
            self.assertEqual(prediction.shape, (1, 21))
            self.assertEqual(errors["hole_position_error_mm"].shape, (1,))
            with self.assertRaisesRegex(ValueError, "GT differ"):
                evaluate_run(run, np.array(["validation"]), target[1:] + 1, torch.device("cpu"))

    def test_comparison_ignores_only_per_model_sources(self):
        left = dict(seed=1, data="same", temporal="mamba", source_validation="a", output="a")
        right = dict(seed=1, data="same", temporal="transformer", source_validation="b", output="b")
        self.assertEqual(comparable_config(left), comparable_config(right))


if __name__ == "__main__":
    unittest.main()
