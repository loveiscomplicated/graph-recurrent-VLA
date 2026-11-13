"""Mask, permutation, physical-metric and matched four-model workflow contracts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from evaluation.gnn.vertex_metrics import (
    observation_groups, paired_improvements, per_frame_errors, statistics,
)
from data_collection.episode_dataset import file_sha256
from graph.build_graph import GraphBuilder
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.pose_geometry import geometric_endpoint
from graph.vertex_aggregation import (
    AGGREGATION_DEFAULTS, GlobalGeometryAggregation, MultiSeedAttentionPool,
)
from main import load_config
from tests.test_temporal_hole_gru import write_smoke_fixture
from training.gnn.checkpoint import canonical_resume_signature, resume_signature
from training.gnn.trainer import resolve_config
from training.gnn.vertex_ablation import VARIANTS, train_variants


def observed_graph(permuted=False, pad_to=None):
    """Mix PEG/BOX/HOLE support, one-sided observations and an empty graph."""
    generator = torch.Generator().manual_seed(19)
    results, objects, semantics = [], [], []
    for labels in ([0, 0, 0, 0, 1, 1, 1, 2, 2], [0], [2], []):
        labels = torch.tensor(labels, dtype=torch.long)
        points = torch.randn(len(labels), 3, generator=generator) * 0.05
        ids = torch.arange(len(labels))
        if permuted:
            order = torch.arange(len(labels) - 1, -1, -1)
            points, labels, ids = points[order], labels[order], ids[order]
        results.append(SimpleNamespace(
            xyz=points, object_id=(labels != 0).long(), metadata=[{} for _ in labels],
        ))
        objects.append(labels)
        semantics.append(ids)
    graph = GraphBuilder()(results, torch.zeros(4, 3), object_types=objects,
                           semantic_ids=semantics, pad_to=pad_to)
    graph.confidence = torch.where(graph.node_valid, graph.xyz_world[..., 0].sigmoid(), 0)
    graph.confidence_valid = graph.node_valid.clone()
    graph.validate(num_semantic_ids=16)
    return graph


class VertexAggregationTests(unittest.TestCase):
    def test_all_variants_masks_permutations_quaternions_and_gradients(self):
        graph = observed_graph()
        for name, flags in VARIANTS.items():
            for quality in (False, True):
                with self.subTest(variant=name, confidence=quality):
                    torch.manual_seed(3)
                    model = VertexEdgeGatedMPNN(
                        16, hidden=16, layers=1, use_semantic_ids=True,
                        use_confidence=quality, **flags,
                    )
                    output = model(graph)
                    self.assertEqual(output["object21"].shape, (4, 21))
                    self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
                    for other in (observed_graph(permuted=True), observed_graph(pad_to=17)):
                        compared = model(other)
                        for field in output:
                            torch.testing.assert_close(output[field], compared[field], atol=2e-6, rtol=2e-5)
                    for start in (3, 13):
                        torch.testing.assert_close(output["object21"][:, start:start + 4].norm(dim=-1),
                                                   torch.ones(4))
                    torch.testing.assert_close(output["endpoint_geom"], geometric_endpoint(output["object21"]))
                    output["object21"].square().sum().backward()
                    for module in (model.global_geometry, model.peg_pool, model.hole_pool):
                        if module is None:
                            continue
                        gradients = [parameter.grad for parameter in module.parameters()]
                        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                            for gradient in gradients))
                        self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_global_geometry_ignores_other_group_and_padded_values(self):
        module = GlobalGeometryAggregation(16, 8)
        h = torch.randn(2, 5, 16)
        xyz = torch.randn(2, 5, 3)
        valid = torch.tensor([[True, True, False, False, False], [False] * 5])
        expected = module(h, xyz, valid)
        h[~valid] = 1000
        xyz[~valid] = -1000
        torch.testing.assert_close(module(h, xyz, valid), expected)
        self.assertTrue(torch.equal(expected[~valid], torch.zeros_like(expected[~valid])))

    def test_multiseed_empty_zero_confidence_and_padding_are_safe(self):
        module = MultiSeedAttentionPool(16, 8, seeds=4, heads=2)
        h = torch.randn(3, 5, 16)
        valid = torch.tensor([[True, True, False, False, False], [False] * 5, [True] * 5])
        confidence = torch.zeros(3, 5)
        expected, summary = module(h, confidence, valid)
        self.assertTrue(torch.isfinite(expected).all())
        torch.testing.assert_close(expected[1], torch.zeros(16))
        torch.testing.assert_close(summary[:, 0], torch.tensor([1., 0, 1.]))
        h[~valid] = 1e4
        confidence[~valid] = 1
        actual, actual_summary = module(h, confidence, valid)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_summary, summary)
        expected.sum().backward()
        self.assertTrue(torch.isfinite(module.seeds.grad).all())

    def test_default_baseline_preserves_keys_initialization_and_heads(self):
        torch.manual_seed(23)
        legacy = VertexEdgeGatedMPNN(16, hidden=64, layers=2)
        expected_rng = torch.get_rng_state()
        variants = {}
        for name, flags in VARIANTS.items():
            torch.manual_seed(23)
            model = VertexEdgeGatedMPNN(16, hidden=64, layers=2, **flags)
            variants[name] = model
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            for key, value in legacy.state_dict().items():
                if not key.startswith(("peg_pool.", "hole_pool.")):
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for field, name in (("global_geometry", "geometry"), ("peg_pool", "multiseed"),
                            ("hole_pool", "multiseed")):
            expected_module = getattr(variants[name], field)
            actual_module = getattr(variants["combined"], field)
            for key, value in expected_module.state_dict().items():
                torch.testing.assert_close(actual_module.state_dict()[key], value, rtol=0, atol=0)
        baseline = VertexEdgeGatedMPNN(16, hidden=64, layers=2, **AGGREGATION_DEFAULTS)
        baseline.load_state_dict(legacy.state_dict(), strict=True)
        for name, value in legacy(observed_graph()).items():
            torch.testing.assert_close(baseline(observed_graph())[name], value, rtol=0, atol=0)

    def test_bad_aggregation_config_rejected_before_data_access(self):
        for changed in ({"global_geometry": "true"}, {"readout": "gmt"},
                        {"readout_seeds": 0}, {"geometry_dim": True},
                        {"readout_heads": 3}, {"readout_dim": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                config = load_config("graph/configs/train_gated_mpnn_cached.yaml")
                config["model"].update(changed)
                resolve_config(config)

    def test_legacy_resume_signature_normalizes_only_missing_defaults(self):
        config = resolve_config(load_config("graph/configs/train_gated_mpnn_cached.yaml"))
        current = resume_signature(config)
        legacy = deepcopy(current)
        for name in AGGREGATION_DEFAULTS:
            legacy["model"].pop(name)
        self.assertEqual(canonical_resume_signature(legacy), canonical_resume_signature(current))
        current["model"]["global_geometry"] = True
        self.assertNotEqual(canonical_resume_signature(legacy), canonical_resume_signature(current))


class VertexMetricTests(unittest.TestCase):
    def test_endpoint_position_relative_lateral_axial_and_residual_are_distinct(self):
        target = torch.zeros(1, 21)
        target[:, 3] = target[:, 13] = 1
        prediction = target.clone()
        prediction[:, 10] = 0.003
        prediction[:, 7:10] = torch.tensor([[0.004, 0.012, 0]])
        output = {"object21": prediction, "endpoint_geom": torch.tensor([[0.001, 0.012, 0]]),
                  "endpoint_residual": torch.tensor([[0.003, 0, 0]])}
        errors = per_frame_errors(output, target)
        self.assertAlmostEqual(errors["relative_3d_error_mm"][0], np.hypot(4, 12), places=5)
        self.assertAlmostEqual(errors["endpoint_position_error_mm"][0], np.hypot(7, 12), places=5)
        self.assertAlmostEqual(errors["lateral_error_mm"][0], 12, places=5)
        self.assertAlmostEqual(errors["axial_error_mm"][0], 4, places=5)
        self.assertAlmostEqual(errors["endpoint_residual_magnitude_mm"][0], 3, places=5)
        self.assertLess(errors["endpoint_residual_gain_mm"][0], 0)

    def test_visibility_counts_paired_size_gain_and_empty_statistics(self):
        groups = observation_groups(np.array([[4, 4, 0], [1, 0, 2], [0, 0, 0]]), 4, 4)
        self.assertEqual(groups["high_support_proxy"].tolist(), [True, False, False])
        self.assertEqual(groups["hole_vertices_2"].tolist(), [False, True, False])
        baseline = {name: np.array([3., 4., 5.]) for name in (
            "peg_half_size_x_error_mm", "endpoint_position_error_mm", "relative_3d_error_mm",
            "endpoint_geom_relative_error_mm",
        )}
        candidate = {name: np.array([1., 5., 2.]) for name in baseline}
        paired = paired_improvements(baseline, candidate, groups)["all"]
        self.assertEqual(paired["size_endpoint_change"]["relative_3d_error_mm"]["both_improved_frames"], 2)
        self.assertIsNone(statistics(np.array([]))["mean"])


class VertexWorkflowTests(unittest.TestCase):
    def test_spawned_parallel_training_preserves_separate_runs_and_logs(self):
        """Exercise the same trainer through separate processes on a CPU fixture."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = {}
            for name in ("baseline", "geometry"):
                config = load_config("graph/configs/train_gated_mpnn_cached.yaml")
                config["device"] = "cpu"
                config["model"].update(hidden=8, layers=1, **VARIANTS[name])
                config["data"] = {"factory": "tests.test_gnn_training:fixture_data"}
                config["training"] = {"epochs": 1, "batch_size": 1}
                config["output"] = {"root": str(root / name)}
                configs[name] = config
            with patch.dict(os.environ, WANDB_MODE="disabled", EXPERIMENTS_DIR=str(root / "experiments")):
                completed = dict(train_variants(configs, 2, root))
            self.assertEqual(set(completed), set(configs))
            for name, directory in completed.items():
                self.assertTrue((directory / "best.pt").is_file())
                self.assertTrue((root / "model_logs" / f"{name}.log").is_file())

    def test_root_cli_trains_and_evaluates_all_four_on_synthetic_cache(self):
        """Execute real optimizer updates/checkpoints, never infer scientific results."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = write_smoke_fixture(root, pose_targets=True)
            cache = Path(fixture["data"]["cache_dir"])
            source = Path(fixture["checkpoint"])
            payload = torch.load(source, map_location="cpu", weights_only=False)
            base = payload["config"]
            base["device"] = "cpu"
            base["training"].update(epochs=1, batch_size=4)
            base["training"]["compile"]["enabled"] = False
            base_path = root / "base.yaml"
            base_path.write_text(yaml.safe_dump(base))
            train_config = {
                "trainer": "training.gnn.vertex_ablation:train", "base_config": str(base_path),
                "reference_checkpoint": str(source), "device": "cpu",
                "data": {"cache_dir": str(cache)}, "output": {"root": str(root / "training")},
            }
            environment = dict(os.environ, WANDB_MODE="disabled", EXPERIMENTS_DIR=str(root / "experiments"))
            self.run_config(root / "train.yaml", train_config, environment)
            manifest_path, = (root / "training").glob("*/ablation_manifest.json")
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual(set(manifest["variants"]), set(VARIANTS))
            eval_config = {
                "trainer": "evaluation.gnn.vertex_ablation:evaluate", "manifest": str(manifest_path),
                "device": "cpu", "output": {"root": str(root / "evaluation")},
            }
            self.run_config(root / "evaluate.yaml", eval_config, environment)
            summary_path, = (root / "evaluation").glob("*/summary.json")
            summary = json.loads(summary_path.read_text())
            self.assertEqual(set(summary["models"]), set(VARIANTS))
            for result in summary["models"].values():
                self.assertEqual(result["groups"]["all"]["frames"], 5)
                self.assertIn("p95", result["groups"]["all"]["metrics"]["peg_half_size_x_error_mm"])
            self.assertEqual(len(list((root / "experiments").glob("*/metadata.json"))), 6)
            # Incomplete matrices and changed training conditions cannot be scored.
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest))
            completed = self.run_config(root / "evaluate.yaml", eval_config, environment, expected_success=False)
            self.assertIn("all four newly trained", completed.stderr)
            manifest["complete"] = True
            changed = manifest["variants"]["geometry"]
            changed_path = Path(changed["checkpoint"])
            payload = torch.load(changed_path, map_location="cpu", weights_only=False)
            payload["config"]["training"]["epochs"] = 2
            torch.save(payload, changed_path)
            changed["sha256"] = file_sha256(changed_path)
            manifest_path.write_text(json.dumps(manifest))
            completed = self.run_config(root / "evaluate.yaml", eval_config, environment, expected_success=False)
            self.assertIn("training protocol mismatch", completed.stderr)

    def run_config(self, path, config, environment, expected_success=True):
        """Exercise the public entrypoint in a separate, bounded CPU process."""
        path.write_text(yaml.safe_dump(config))
        completed = subprocess.run(
            [sys.executable, "main.py", "--config", str(path)],
            env=environment, capture_output=True, text=True, timeout=60,
        )
        if expected_success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0)
        return completed


if __name__ == "__main__":
    unittest.main()
