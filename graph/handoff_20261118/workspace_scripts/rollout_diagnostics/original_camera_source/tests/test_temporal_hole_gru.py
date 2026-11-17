"""Causal sequence, streaming, freeze and real cache/CLI workflow checks on CPU."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.frozen_episodes import (
    FrozenEpisodeDataset,
    infer_observations,
)
from data_collection.temporal_gate_data import make_gate_manifest
from evaluation.gnn.temporal.hole_gru import (
    absence_diagnostics,
    replay_gru,
    score_predictions,
)
from evaluation.gnn.temporal_memory import replay_memory
from graph.build_graph import GraphBuilder
from graph.temporal.hole_gru import StreamingHoleEstimator, TemporalHoleGRU
from graph.temporal.observations import OBSERVATION_NAMES, temporal_inputs
from graph.temporal_gate import LearnedTemporalGate
from graph.utils.hole_memory import MemorySettings
from main import load_config, run
from training.gnn.cached_data import build_data
from training.gnn.data import TrainingBatch, canonical_hash, save_cached_sample
from training.gnn.temporal.config import resolve_gru_config
from training.gnn.temporal.frozen_source import prepare_sequences
from training.gnn.temporal.hole_gru import run_gru_epoch
from training.gnn.trainer import build_model, resolve_config


def synthetic_sequences():
    """Distinct episodes with moving TCP, bad initialization and an absent interval."""
    count = 10
    tcp = np.zeros((count, 3), dtype=np.float32)
    tcp[:, 0] = np.tile(np.arange(5) * 0.01, 2)
    world = np.repeat(np.asarray([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]], dtype=np.float32), 5, axis=0)
    target = np.zeros((count, 21), dtype=np.float32)
    target[:, 10:13] = world - tcp
    prediction = target.copy()
    prediction[[0, 2, 3, 5, 7, 8], 10] += 0.1
    features = np.zeros((count, 17), dtype=np.float32)
    features[[0, 1, 4, 5, 6, 9], :8] = [6, 1, 0, 1, 1, 0.9, 0.9, 0.9]
    features[[0, 1, 4, 5, 6, 9], 8:16] = [2, 1, 0, 1, 1, 0.9, 0.9, 0.9]
    missing = features[:, 0] == 0
    features[missing, 2] = 1
    features[missing, 10] = 1
    features[missing, 16] = 1
    quality = np.zeros((count, 3), dtype=np.float32)
    quality[~missing] = [8, 0.9, 1]
    episodes = np.repeat(["a", "b"], 5)
    frames = np.tile(np.arange(5), 2)
    return {
        "prediction": prediction,
        "target": target,
        "tcp_world": tcp,
        "features": features,
        "quality": quality,
        "episode": episodes,
        "frame": frames,
        "sample_ids": np.asarray(
            [f"{ep}/frame_{frame:06d}.npz" for ep, frame in zip(episodes, frames)]
        ),
    }


def write_smoke_fixture(root, pose_targets=False, raw_proprioception=False):
    """Write a tiny checksummed cache with real graph schema and reduced frozen GNN.

    Fixture parents are explicitly independent; this is not production provenance.
    Frames include moving TCP and entirely missing vertices to exercise cold start
    and absence supervision. The reserved GNN test episode is never materialized.
    """
    cache = root / "cache"
    (cache / "metadata").mkdir(parents=True)
    source_config = resolve_config(load_config("graph/configs/train_gated_mpnn_cached.yaml"))
    source_config["model"].update(hidden=8, layers=1, edge_embedding_dim=4)
    source_config["data"]["cache_dir"] = str(cache)
    source_config["data"]["num_workers"] = 0
    metadata = {
        "preprocessing": {
            "format_version": 1,
            "camera": "base_camera",
            "graph": source_config["graph"],
            "confidence": source_config["confidence"],
        },
        "splits": {"train": ["a", "b"], "validation": ["v"], "test": ["t"]},
        "frames": [],
    }
    namespace = canonical_hash(metadata["preprocessing"])
    inventory = {"preprocessing_sha256": namespace, "files": []}
    for episode in ("a", "b", "v"):
        for frame in range(5):
            tcp = torch.tensor([[frame * 0.01, 0, 0]], dtype=torch.float32)
            present = frame not in (2, 3)
            observation = SimpleNamespace(
                xyz=torch.tensor([[0.1, 0.2, 0.3]]) if present else torch.empty(0, 3),
                object_id=torch.tensor([1]) if present else torch.empty(0, dtype=torch.long),
                metadata=[{}] if present else [],
            )
            graph = GraphBuilder()([observation], tcp)
            target = torch.zeros(1, 21)
            target[:, 10:13] = torch.tensor([0.1, 0.2, 0.3]) - tcp
            sample_id = f"{episode}/frame_{frame:06d}.npz"
            digest = canonical_hash({"id": sample_id})
            if pose_targets:
                from graph.pose_geometry import rotvec_quaternion
                from graph.utils.object21_targets import object21_from_gt

                peg_rotation = rotvec_quaternion(torch.tensor([[0.0, 0.0, frame * 0.05]]))
                target = object21_from_gt(
                    {
                        "ee_pose": torch.cat((tcp, torch.tensor([[1.0, 0.0, 0.0, 0.0]])), dim=1),
                        "peg_pose": torch.cat(
                            (torch.tensor([[frame * 0.003, 0.1, 0.25]]), peg_rotation), dim=1
                        ),
                        "hole_pose": torch.tensor([[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]]),
                        "peg_half_sizes": torch.tensor([[0.05, 0.01, 0.01]]),
                        "hole_radius": torch.tensor([0.02]),
                    }
                )
            if raw_proprioception:
                raw_path = root / "raw" / sample_id
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    raw_path,
                    **{
                        "observation/extra/tcp_pose": np.r_[tcp.numpy()[0], [1.0, 0.0, 0.0, 0.0]][
                            None
                        ].astype(np.float32),
                        "observation/agent/qpos": np.array(
                            [[0.0] * 7 + [0.02, 0.02]], dtype=np.float32
                        ),
                    },
                )
                digest = file_sha256(raw_path)
            metadata["frames"].append({"id": sample_id, "sha256": digest})
            identity = {
                "frame_sha256": digest,
                "sample_id": sample_id,
                "preprocessing_sha256": namespace,
            }
            relative = f"{namespace}/{canonical_hash(identity)}.npz"
            path = cache / relative
            save_cached_sample(
                path, TrainingBatch(graph, {"object21": target}, [sample_id]), identity
            )
            inventory["files"].append(
                {
                    "sample_id": sample_id,
                    "file": relative,
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
            )
    metadata["dataset_sha256"] = canonical_hash(metadata)
    metadata["sample_counts"] = {"train": 10, "validation": 5, "test": 0}
    (cache / "metadata/data_identity.json").write_text(json.dumps(metadata))
    (cache / "metadata/cache_inventory.json").write_text(json.dumps(inventory))
    bundle = build_data(source_config, torch.Generator().manual_seed(0))
    source = root / "frozen_gnn.pt"
    frozen = build_model(source_config["model"], torch.device("cpu"))
    torch.save(
        {
            "kind": "best_weights",
            "model": frozen.state_dict(),
            "config": source_config,
            "dataset_sha256": bundle.identity["dataset_sha256"],
        },
        source,
    )
    manifest = root / "splits.json"
    manifest.write_text(
        json.dumps(make_gate_manifest(metadata["splits"], {name: name for name in "abvt"}))
    )
    return {
        "trainer": "training.gnn.temporal.hole_gru:train",
        "device": "cpu",
        "checkpoint": str(source),
        "split_manifest": str(manifest),
        "data": {"cache_dir": str(cache)},
        "training": {"epochs": 1},
        "evaluation": {"latency_steps": 3, "latency_warmup_steps": 1, "long_occlusion_frames": 2},
        "output": {"root": str(root / "training")},
    }


class TemporalGRUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = TemporalHoleGRU()
        self.data = synthetic_sequences()

    def test_model_size_direct_hidden_head_and_validation(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 17283)
        larger = TemporalHoleGRU(hidden=128)
        self.assertEqual(sum(p.numel() for p in larger.parameters()), 59139)
        self.assertFalse(self.model.gru.bidirectional)
        self.assertEqual(self.model.head.in_features, 64)
        for settings in ({"hidden": 0}, {"layers": 2}, {"position_scale_m": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TemporalHoleGRU(**settings)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 0, 23))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 2, 23), torch.zeros(1, 2, 64))

    def test_raw_features_world_coordinates_and_target_isolation(self):
        dataset = FrozenEpisodeDataset(self.data)
        inputs, target = dataset[0]
        self.assertEqual(len(OBSERVATION_NAMES), 23)
        torch.testing.assert_close(
            inputs[:, :3],
            torch.tensor(self.data["prediction"][:5, 10:13] + self.data["tcp_world"][:5]),
        )
        torch.testing.assert_close(target, torch.tensor([[0.1, 0.2, 0.3]]).expand(5, 3))
        before = inputs.clone()
        self.data["target"][:] = 1234
        torch.testing.assert_close(dataset[0][0], before)
        with self.assertRaises(ValueError):
            temporal_inputs(torch.zeros(2, 21), torch.zeros(1, 3), torch.zeros(2, 17))

    def test_learned_gate_uses_identical_external_observation_contract(self):
        prediction = torch.tensor(self.data["prediction"][0])
        tcp = torch.tensor(self.data["tcp_world"][0])
        quality = torch.tensor(self.data["features"][0])
        gate = LearnedTemporalGate()
        recorded = []
        handle = gate.network.register_forward_pre_hook(
            lambda module, args: recorded.append(args[0])
        )
        gate.step(prediction[10:13], tcp, quality)
        handle.remove()
        observations = temporal_inputs(prediction, tcp, quality)
        external = torch.cat(
            (
                recorded[0][:3] * gate.position_scale_m,
                recorded[0][9:12] * gate.position_scale_m,
                recorded[0][12:29],
            )
        )
        torch.testing.assert_close(external, observations)

    def test_full_sequence_streaming_and_reset(self):
        observations, _ = FrozenEpisodeDataset(self.data)[0]
        observations = observations.unsqueeze(0).expand(2, -1, -1)
        full, last = self.model(observations)
        hidden, steps = None, []
        for index in range(observations.shape[1]):
            output, hidden = self.model.step(observations[:, index], hidden)
            steps.append(output)
        torch.testing.assert_close(torch.stack(steps, 1), full, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(hidden, last)
        result = replay_gru(self.model, self.data, "cpu")
        self.assertLess(result["streaming_max_difference_m"], 1e-6)
        second = {key: value[5:] for key, value in self.data.items()}
        np.testing.assert_allclose(
            replay_gru(self.model, second, "cpu")["world"], result["world"][5:], atol=1e-7
        )

    def test_causal_prefix_and_future_or_gt_changes(self):
        before = replay_gru(self.model, self.data, "cpu")["world"]
        changed = deepcopy(self.data)
        changed["prediction"][4] = 1000
        changed["features"][4] = 1000
        changed["target"][:] = -999
        after = replay_gru(self.model, changed, "cpu")["world"]
        np.testing.assert_array_equal(before[:4], after[:4])
        np.testing.assert_array_equal(before[5:], after[5:])
        observations, _ = FrozenEpisodeDataset(self.data)[0]
        prefix, _ = self.model(observations[:3].unsqueeze(0))
        full, _ = self.model(observations.unsqueeze(0))
        torch.testing.assert_close(prefix, full[:, :3])

    def test_missing_frames_backpropagate_to_past_and_initial_frame(self):
        inputs, target = FrozenEpisodeDataset(self.data)[0]
        inputs.requires_grad_(True)
        output, _ = self.model(inputs.unsqueeze(0))
        # Frame 3 has no vertices; its loss must reach the earlier recurrent inputs.
        loss = (output[0, 3] - target[3]).square().sum()
        loss.backward()
        self.assertGreater(float(inputs.grad[0, :3].abs().sum()), 0)
        self.model.zero_grad(set_to_none=True)
        first, _ = self.model(inputs[:1].unsqueeze(0))
        (first[0, 0] - target[0]).square().sum().backward()
        self.assertGreater(float(self.model.head.weight.grad.abs().sum()), 0)
        self.assertFalse(torch.equal(first[0, 0].detach(), inputs[0, :3].detach()))

    def test_alignment_rejects_gaps_offset_duplicates_and_shifted_metadata(self):
        for failure in ("gap", "offset", "duplicate", "metadata", "unsorted", "shape", "nan"):
            broken = deepcopy(self.data)
            if failure == "gap":
                broken = {key: np.delete(value, 2, axis=0) for key, value in broken.items()}
            elif failure == "offset":
                broken = {key: np.delete(value, 0, axis=0) for key, value in broken.items()}
            elif failure == "duplicate":
                broken["sample_ids"][1] = broken["sample_ids"][0]
            elif failure == "metadata":
                broken["frame"][0] = 1
            elif failure == "unsorted":
                broken = {key: value[::-1] for key, value in broken.items()}
            elif failure == "shape":
                broken["target"] = broken["target"][:-1]
            else:
                broken["target"][0, 0] = np.nan
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                FrozenEpisodeDataset(broken)

    def test_shuffled_frame_inference_keeps_graph_target_tcp_ids_aligned(self):
        batches = []
        for frame in (2, 0, 1):
            observation = SimpleNamespace(
                xyz=torch.tensor([[0.1, 0.2, 0.3]]), object_id=torch.tensor([1]), metadata=[{}]
            )
            graph = GraphBuilder()([observation], torch.tensor([[frame * 0.01, 0, 0]]))
            target = torch.full((1, 21), float(frame))
            batches.append(TrainingBatch(graph, {"object21": target}, [f"a/frame_{frame:06d}.npz"]))

        def model(graph):
            return {"object21": graph.tcp_xyz_world[:, :1].expand(-1, 21)}

        result = infer_observations(model, batches, "cpu")
        np.testing.assert_allclose(result["tcp_world"][:, 0], [0, 0.01, 0.02])
        np.testing.assert_allclose(result["prediction"][:, 0], [0, 0.01, 0.02])
        np.testing.assert_array_equal(result["target"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(result["frame"], [0, 1, 2])
        FrozenEpisodeDataset(result)

    def test_all_frame_loss_includes_initial_and_missing_frames(self):
        dataset = FrozenEpisodeDataset(self.data)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        expected = np.mean([target.abs().mean().item() for _, target in dataset])
        loss = run_gru_epoch(self.model, dataset, "cpu", {"loss": "l1"})
        self.assertAlmostEqual(loss, expected, places=7)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        settings = {"loss": "huber", "huber_delta_m": 0.01, "gradient_clip_norm": 1}
        before = run_gru_epoch(self.model, dataset, "cpu", settings)
        for _ in range(3):
            run_gru_epoch(self.model, dataset, "cpu", settings, optimizer, np.random.default_rng(0))
        self.assertLess(run_gru_epoch(self.model, dataset, "cpu", settings), before)

    def test_streaming_geometry_state38_and_non_hole_fields(self):
        estimator = StreamingHoleEstimator(self.model)
        prediction = torch.arange(42, dtype=torch.float32).reshape(2, 21)
        tcp = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.1, 0.1]])
        robot, freshness = torch.randn(2, 15), torch.randn(2, 2)
        result = estimator.step(prediction, tcp, torch.zeros(2, 17), robot, freshness)
        unchanged = list(range(10)) + list(range(13, 21))
        torch.testing.assert_close(result["object21"][:, unchanged], prediction[:, unchanged])
        consistent = result["consistent_object21"]
        torch.testing.assert_close(
            consistent[:, 7:10] + consistent[:, 10:13], prediction[:, 7:10] + prediction[:, 10:13]
        )
        torch.testing.assert_close(result["state38"][:, :15], robot)
        torch.testing.assert_close(result["state38"][:, 36:], freshness)
        self.assertEqual(result["state38"].shape, (2, 38))
        first = result["hole_world"].clone()
        estimator.reset()
        torch.testing.assert_close(
            estimator.step(prediction, tcp, torch.zeros(2, 17))["hole_world"], first
        )

    def test_absence_age_resets_at_episode_boundary_and_scores_all_methods(self):
        data = self.data
        age, long_runs, events = absence_diagnostics(data, data["quality"][:, 0] == 0, 2)
        np.testing.assert_array_equal(age, [0, 0, 1, 2, 0, 0, 0, 1, 2, 0])
        self.assertEqual(long_runs.sum(), 4)
        self.assertEqual(len(events), 2)
        learned = replay_gru(self.model, data, "cpu")
        rule = replay_memory(
            data["prediction"],
            data["tcp_world"],
            data["quality"],
            data["episode"],
            data["frame"],
            MemorySettings(),
        )
        summary, errors, events = score_predictions(
            data,
            learned,
            rule,
            {"visibility_labels": None, "long_occlusion_frames": 2, "bad_error_mm": 10},
        )
        self.assertEqual(set(summary["methods"]), {"gnn", "rule", "gru"})
        self.assertEqual(
            summary["methods"]["gru"]["groups"]["no_hole_vertices_long_age"]["frames"], 2
        )
        self.assertEqual(len(events["recovery"]["gru"]["no_hole_vertices"]), 2)
        self.assertEqual(
            len(summary["methods"]["gru"]["temporal"]["bad_gnn_initialization_recovery"]), 2
        )

    def test_real_cache_root_cli_train_evaluate_and_frozen_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_smoke_fixture(root)
            source = Path(config["checkpoint"])
            original_hash = file_sha256(source)
            effective = resolve_gru_config(config)
            frozen, loaders, identity = prepare_sequences(
                effective, torch.device("cpu"), ("train", "validation")
            )
            self.assertFalse(frozen.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in frozen.parameters()))
            original = {name: value.clone() for name, value in frozen.state_dict().items()}
            dataset = FrozenEpisodeDataset(infer_observations(frozen, loaders["train"], "cpu"))
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            run_gru_epoch(
                self.model,
                dataset,
                "cpu",
                effective["training"],
                optimizer,
                np.random.default_rng(0),
            )
            self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
            for name, value in frozen.state_dict().items():
                torch.testing.assert_close(value, original[name])
            config_path = root / "config.json"
            with patch.dict(
                "os.environ",
                {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")},
            ):
                config_path.write_text(json.dumps(config))
                run(config_path)
                best = next((root / "training").glob("*/best.pt"))
                config.update(
                    trainer="evaluation.gnn.temporal.hole_gru:evaluate", gru_checkpoint=str(best)
                )
                config["output"]["root"] = str(root / "evaluation")
                config_path.write_text(json.dumps(config))
                run(config_path)
            output = next(path.parent for path in (root / "evaluation").glob("*/summary.json"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["performance"]["gru_parameters"], 17283)
            self.assertIsNone(
                summary["performance"]["measurements"]["gru_only"]["cuda_peak_allocated_bytes"]
            )
            self.assertTrue((output / "episode_trace_000.png").is_file())
            self.assertEqual(file_sha256(source), original_hash)
            self.assertEqual(len(list((root / "experiments").iterdir())), 2)
            from utils.experiment_artifacts import plan_upload

            plans = plan_upload(root / "experiments" / output.name, {"metrics"})
            self.assertIn("checkpoint", plans)
            # A different source checkpoint cannot be substituted at evaluation.
            payload = torch.load(best, weights_only=True)
            self.assertEqual(payload["config"]["observation_contract"], "hole_world_tcp_quality_v1")
            payload["config"]["source_checkpoint_sha256"] = "different_source"
            incompatible = root / "incompatible.pt"
            torch.save(payload, incompatible)
            config["gru_checkpoint"] = str(incompatible)
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "source_checkpoint_sha256"):
                run(config_path)


if __name__ == "__main__":
    unittest.main()
