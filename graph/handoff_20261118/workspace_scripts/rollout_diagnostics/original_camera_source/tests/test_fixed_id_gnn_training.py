"""Check frame reconstruction, frozen forecasts and batched GNN graph semantics."""

import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.temporal.fixed_id_corner_sequences import pack_episode
from data_collection.temporal.fixed_id_gnn_data import episode_arrays
from evaluation.fixed_id_gnn_comparison import compare
from evaluation.fixed_id_gnn_leaderboard import leaderboard_statistics
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from graph.temporal.fixed_id_gnn import build_corner_graph
from training.fixed_id_gnn import make_graph, predict_frozen


class FixedIdGNNTrainingTests(unittest.TestCase):
    def test_terminal_only_sighting_is_kept_as_actual_detection(self):
        source = dict(
            xyz=np.array([[[1., 2., 3.]], [[4., 5., 6.]], [[7., 8., 9.]]], dtype=np.float32),
            valid=np.ones((3, 1), dtype=bool), physical_ids=np.array([[0], [0], [9]], dtype=np.int64),
            reference_xyz=np.zeros((3, 24, 3), dtype=np.float32),
            robot=np.zeros((3, 9), dtype=np.float32), sample_ids=np.array(["a", "b", "c"]),
        )
        packed, _ = pack_episode(source)
        measured, last, observed, seen, sequences = episode_arrays(
            packed, dict(episode="e", parent="p"), source)
        self.assertEqual(len(sequences), 1)
        self.assertFalse(seen[1, 9])
        self.assertTrue(seen[2, 9] and observed[2, 9])
        np.testing.assert_array_equal(measured[2, 9], [7, 8, 9])
        np.testing.assert_array_equal(last[2, 0], [4, 5, 6])

    def test_batched_graph_matches_individual_graph_and_provenance(self):
        xyz = torch.zeros(2, 24, 3)
        xyz[0, 0], xyz[0, 16], xyz[1, 8] = torch.tensor([1., 2., 3.]), torch.tensor([3., 2., 1.]), torch.ones(3)
        seen = torch.zeros(2, 24, dtype=torch.bool)
        seen[0, [0, 16]] = True
        seen[1, 8] = True
        observed = seen.clone()
        observed[0, 16] = False
        tcp = torch.zeros(2, 3)
        graph, flag = make_graph(xyz, seen, observed, tcp)
        graph.validate()
        single, single_flag = build_corner_graph(xyz[0, seen[0]], torch.tensor([0, 2]),
                                                 observed[0, seen[0]], tcp[0])
        self.assertTrue(torch.equal(graph.relative_xyz, single.relative_xyz))
        torch.testing.assert_close(flag[0, seen[0]], single_flag[0])
        model = VertexEdgeGatedMPNN(num_semantic_ids=24, hidden=16, layers=1, extra_node_dim=1).eval()
        with torch.no_grad():
            batch_output = model(graph, flag)["object21"][0]
            single_output = model(single, single_flag)["object21"][0]
        torch.testing.assert_close(batch_output, single_output, atol=1e-6, rtol=1e-5)

    def test_frozen_full_history_matches_causal_prefix(self):
        torch.manual_seed(5)
        model = FixedIdCornerPredictor(architecture="transformer", width=16, layers=1, heads=4).eval()
        model.requires_grad_(False)
        values = np.random.default_rng(1).normal(size=(4, 13)).astype(np.float32)
        sequences = [dict(inputs=values, frame_offset=1, vertex_id=2)]
        prediction = predict_frozen(model, sequences, 5, torch.device("cpu"), 2)
        for step in range(4):
            prefix = torch.from_numpy(values[:step + 1])[None]
            with torch.no_grad():
                expected = model(prefix, torch.ones((1, step + 1), dtype=torch.bool))[0, -1]
            torch.testing.assert_close(prediction[step + 1, 2], expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_comparison_rejects_different_initial_gnn_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for architecture, initial_hash in (("mamba", "same"), ("transformer", "different")):
                run = root / "runs" / architecture / "seed_0" / "run"
                run.mkdir(parents=True)
                config = dict(seed=0, temporal=dict(architecture=architecture), output={})
                torch.save(dict(epoch=1, config=config, cache_sha256="data",
                                gnn_initial_sha256=initial_hash), run / "best.pt")
                validation = dict(loss=1.0, frames=2)
                (run / "completed.json").write_text(json.dumps(dict(best_epoch=1,
                    best_validation_loss=1.0, validation=validation)))
                (run / "history.json").write_text(json.dumps([dict(epoch=1, validation=validation)]))
            with self.assertRaisesRegex(ValueError, "initial GNN weights"):
                compare(dict(seeds=[0], runs_root=str(root / "runs"), output=dict(root=str(root / "out"))))

    def test_leaderboard_averages_per_frame_before_percentiles(self):
        errors = {}
        for name in ("mamba", "transformer"):
            errors[(name, 0)] = {"metric": np.array([0., 100.])}
            errors[(name, 1)] = {"metric": np.array([100., 0.])}
            errors[(name, 2)] = {"metric": np.array([50., 50.])}
        result = leaderboard_statistics(errors, [0, 1, 2])
        self.assertEqual(result["mamba"]["metric"]["three_seed_frame_mean"]["p90"], 50.)


if __name__ == "__main__":
    unittest.main()
