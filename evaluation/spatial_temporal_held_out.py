"""Evaluate fixed validation-selected tracker and Object21 weights on test only."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import file_sha256
from evaluation.spatial_temporal_corner_tracking import evaluate_online
from evaluation.spatial_temporal_object21 import evaluate_arrays, save_predictions
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.spatial_temporal_corner_tracker import load_tracker_checkpoint
from training.gnn.spatial_temporal_corner_data import (
    CornerTrackingDataset, CornerTrackingEpisode, collate_corner_tracking,
)
from training.spatial_temporal_corner_tracker import run_epoch
from utils.experiment_logger import ExperimentLogger, write


@dataclass
class HeldOutEpisode(CornerTrackingEpisode):
    """Keep Object21 labels separate from observed graphs and tracker inputs."""

    object21: np.ndarray
    source_episode_id: str


class HeldOutDataset(CornerTrackingDataset):
    """Load only checksummed test episodes with offline corner and Object21 labels."""

    def __getitem__(self, index):
        episode = super().__getitem__(index)
        path = self.root / self.records[index]["file"]
        with np.load(path, allow_pickle=False) as archive:
            target = archive["object21"].copy()
        if target.shape != (episode.length, 21) or not np.isfinite(target).all():
            raise ValueError(f"Invalid test Object21 labels: {episode.episode_id}")
        record = self.records[index]
        source_episode_id = f"{record['seed']}:{record['policy_mode']}"
        return HeldOutEpisode(
            **vars(episode), object21=target, source_episode_id=source_episode_id
        )


def validate_test_manifest(manifest, release, expected):
    """Require complete test coverage and seed separation before model evaluation."""
    episodes = manifest["episodes"]
    records = [row for row in release["records"] if row["split"] == "test"]
    episode_keys = [(row["seed"], row["policy_mode"]) for row in episodes]
    source_keys = [(row["seed"], row["policy_mode"]) for row in records]
    if any(row["split"] != "test" for row in episodes):
        raise ValueError("Held-out evaluation accepts only test episodes")
    if len(set(episode_keys)) != len(episode_keys) or episode_keys != source_keys:
        raise ValueError("Prepared test episodes must exactly match release order and inventory")
    if any(
        episode["frames"] != record["frames"] or episode["batch"] != record["batch"]
        for episode, record in zip(episodes, records, strict=True)
    ):
        raise ValueError("Prepared test frame counts or batches differ from the release")
    test_seeds = {row["seed"] for row in episodes}
    development_seeds = {row["seed"] for row in release["records"] if row["split"] != "test"}
    if test_seeds & development_seeds:
        raise ValueError("Test seeds overlap train/validation")
    frames = sum(row["frames"] for row in episodes)
    if len(episodes) != expected["episodes"] or frames != expected["frames"]:
        raise ValueError(f"Unexpected test inventory: {len(episodes)} episodes / {frames} frames")
    if manifest.get("condition_filter") is not None:
        raise ValueError("Held-out evaluation must include all test frames")


class RolloutCollector:
    """Collect unchanged online outputs without re-running association or reading GT inputs."""

    def __init__(self, maximum_tracks):
        self.maximum_tracks = maximum_tracks
        self.offsets = [0]
        self.nodes = {name: [] for name in ("xyz", "kinds", "observed", "uncertainty", "age")}
        self.tcp = []
        self.targets = []
        self.sample_ids = []

    def __call__(self, episode, frame_index, result):
        count = len(result["positions"])
        if count > self.maximum_tracks:
            raise ValueError(
                f"Test tracker produced {count} tracks at {episode.episode_id}:{frame_index}; "
                f"fixed limit is {self.maximum_tracks}"
            )
        self.offsets.append(self.offsets[-1] + count)
        for name in self.nodes:
            result_name = "positions" if name == "xyz" else name
            self.nodes[name].append(result[result_name].detach().cpu().numpy().copy())
        self.tcp.append(episode.tcp[frame_index])
        self.targets.append(episode.object21[frame_index])
        self.sample_ids.append(f"{episode.source_episode_id}:{frame_index}")

    def arrays(self):
        return {
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            **{name: np.concatenate(values) for name, values in self.nodes.items()},
            "tcp": np.asarray(self.tcp, dtype=np.float32),
            "target": np.asarray(self.targets, dtype=np.float32),
            "sample_ids": np.asarray(self.sample_ids),
        }


def evaluate(config):
    """Run test once using pinned weights/settings, with no optimizer or calibration."""
    device = torch.device(config["device"])
    torch.set_num_threads(config["evaluation"].get("cpu_threads", 4))
    tracker_path = Path(config["tracker"]["checkpoint"]).resolve()
    object_path = Path(config["object21"]["checkpoint"]).resolve()
    for path, expected in (
        (tracker_path, config["tracker"]["sha256"]),
        (object_path, config["object21"]["sha256"]),
    ):
        if not expected or file_sha256(path) != expected:
            raise ValueError(f"Held-out checkpoint must match its validation-selected SHA-256: {path}")
    tracker, tracker_checkpoint = load_tracker_checkpoint(tracker_path, config["tracker"]["sha256"], device)
    object_checkpoint = torch.load(object_path, map_location="cpu", weights_only=False)
    if object_checkpoint.get("format") != "spatial_temporal_object21_v1":
        raise ValueError("Expected a spatial-temporal Object21 checkpoint")
    identity = object_checkpoint["identity"]
    gnn_settings = object_checkpoint["config"]["gnn"]
    if gnn_settings["use_semantic_ids"] or gnn_settings["extra_node_dim"] != 3:
        raise ValueError("Held-out Object21 needs the frozen ID-free reliability-feature model")
    if identity["tracker_checkpoint_sha256"] != config["tracker"]["sha256"]:
        raise ValueError("Object21 weights were trained with another tracker")
    if config["association"] != identity["association"]:
        raise ValueError("Test association settings differ from validation-selected settings")
    manifest_path = Path(config["data"]["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    release_path = Path(config["data"]["release_manifest"]).resolve()
    if file_sha256(release_path) != identity["manifest_sha256"]:
        raise ValueError("Test source release differs from the frozen training release")
    if manifest["source"]["manifest_sha256"] != identity["manifest_sha256"]:
        raise ValueError("Prepared test dataset has a different source release")
    validate_test_manifest(manifest, json.loads(release_path.read_text()), config["expected"])
    dataset = HeldOutDataset(
        manifest_path, "test", max_edge_m=tracker_checkpoint["identity"]["max_edge_m"],
        graph_cache_root=config["data"].get("graph_cache_root"),
    )
    with ExperimentLogger(
        team="graph", model="spatial_temporal_held_out", seed=tracker_checkpoint["config"]["seed"],
        config=config, job_type="evaluation",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        write(output / "progress.json", {"phase": "teacher_bound", "split": "test"})
        loader = DataLoader(
            dataset, batch_size=config["evaluation"].get("episode_batch_size", 1),
            shuffle=False, collate_fn=collate_corner_tracking,
        )
        teacher = run_epoch(tracker, loader, tracker_checkpoint["config"]["training"], device)
        write(output / "progress.json", {"phase": "online_rollout", "split": "test"})
        collector = RolloutCollector(object_checkpoint["config"]["training"]["max_tracks_per_frame"])
        online = evaluate_online(
            tracker, dataset, {**config["association"], "log_every_episodes": 25}, device,
            frame_callback=collector,
        )
        arrays = collector.arrays()
        if len(arrays["sample_ids"]) != config["expected"]["frames"]:
            raise ValueError("Frozen test rollout is incomplete")
        cache_path = output / "frozen_test_rollouts.pt"
        torch.save({"format": "spatial_temporal_test_rollouts_v1", "split": "test",
                    "arrays": arrays, "tracker_checkpoint_sha256": config["tracker"]["sha256"],
                    "association": config["association"]}, cache_path)
        cache_sha256 = file_sha256(cache_path)
        write(output / "progress.json", {"phase": "object21", "split": "test"})
        model = VertexEdgeGatedMPNN(**object_checkpoint["config"]["gnn"]).to(device).eval()
        model.load_state_dict(object_checkpoint["model"], strict=True)
        model.requires_grad_(False)
        indices = np.arange(len(arrays["sample_ids"]))
        prediction, errors, metrics = evaluate_arrays(
            model, arrays, indices, object_checkpoint["config"], device,
            batch_size=config["evaluation"]["frame_batch_size"],
        )
        predictions_path = output / "predictions.npz"
        save_predictions(predictions_path, arrays, indices, prediction, errors)
        for path, expected in (
            (tracker_path, config["tracker"]["sha256"]), (object_path, config["object21"]["sha256"]),
        ):
            if file_sha256(path) != expected:
                raise ValueError(f"Pinned checkpoint changed during test: {path}")
        result = {
            "split": "test", "episodes": len(dataset), "frames": len(indices),
            "tracker_checkpoint_sha256": config["tracker"]["sha256"],
            "object21_checkpoint_sha256": config["object21"]["sha256"],
            "prepared_manifest_sha256": file_sha256(manifest_path),
            "release_manifest_sha256": file_sha256(release_path),
            "rollout_cache_sha256": cache_sha256,
            "predictions_sha256": file_sha256(predictions_path),
            "association": config["association"], "teacher_bound": teacher,
            "detector_only_online": online, "object21_metrics": metrics,
            "weights_frozen": True, "test_used_for_selection": False,
        }
        write(output / "results.json", result)
        write(output / "progress.json", {"phase": "complete", "status": "completed", "split": "test"})
        for path in (tracker_path, object_path):
            logger.link_output(path, checkpoint=True)
        for path in (cache_path, predictions_path, output / "results.json", output / "effective_config.yaml"):
            logger.link_output(path)
        logger.log({"teacher_bound": teacher, "object21": metrics})
        logger.record(split="test", episodes=len(dataset), frames=len(indices),
                      tracker_checkpoint_sha256=config["tracker"]["sha256"],
                      object21_checkpoint_sha256=config["object21"]["sha256"],
                      rollout_cache_sha256=cache_sha256)
    return str(output)
