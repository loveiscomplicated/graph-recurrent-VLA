"""Align Object21 supervision and cache outputs of a hash-pinned frozen tracker."""

import gc
import json
import tarfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import (
    load_training_episodes, read_member, sequence_arrays,
)
from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker
from utils.experiment_logger import write


def prepare_targets(config):
    """Export train/validation labels in the existing tracker episode order."""
    release = Path(config["release"])
    if file_sha256(release / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Five-mode release manifest checksum mismatch")
    source = json.loads((release / "manifest.json").read_text())
    # This loader verifies the existing split, payload checksums and reset seeds.
    episodes, tracking_manifest = load_training_episodes(config["data_root"])
    records = {
        (record["seed"], record["policy_mode"]): record
        for record in source["records"] if record["split"] in episodes
    }
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "five_mode_object21_targets_v1",
        "tracking_manifest_sha256": file_sha256(Path(config["data_root"]) / "manifest.json"),
        "release_manifest_sha256": config["manifest_sha256"],
        "source_hf": tracking_manifest["source_hf"],
        "test_payloads_read": False, "splits": {},
    }
    with ExitStack() as stack:
        archives = {}
        for split, group in episodes.items():
            labels, identities, receipts = [], [], []
            for episode in group:
                identity = (episode["seed"], episode["mode"])
                record = records[identity]
                if record["split"] != split:
                    raise ValueError(f"Split mismatch for {identity}")
                receipt = record["files"]["targets"]
                archive_name = receipt["archive"]
                if archive_name not in archives:
                    archives[archive_name] = stack.enter_context(
                        tarfile.open(release / archive_name)
                    )
                target = read_member(archives[archive_name], receipt)
                length = len(episode["tcp"])
                if not np.array_equal(target["frame_indices"], np.arange(length)):
                    raise ValueError(f"Target frame alignment mismatch: {identity}")
                np.testing.assert_allclose(target["tcp"], episode["tcp"], atol=1e-6, rtol=0)
                labels_array = target["object21"].astype(np.float32)
                if labels_array.shape != (length, 21) or not np.isfinite(labels_array).all():
                    raise ValueError(f"Expected finite Object21 labels [T,21]: {identity}")
                # Object21 stores TCP-relative centers in world axes.
                np.testing.assert_allclose(
                    labels_array[:, :3], episode["gt"][:, :8].mean(axis=1) - episode["tcp"],
                    atol=2e-6, rtol=0,
                )
                labels.append(labels_array)
                identities.append([*identity, length])
                receipts.append(receipt["sha256"])
            path = output / f"{split}.npz"
            np.savez_compressed(path, target=np.concatenate(labels))
            result["splits"][split] = {
                "path": path.name, "sha256": file_sha256(path),
                "episodes": identities, "frames": sum(len(value) for value in labels),
                "target_member_sha256": receipts,
            }
            print(f"TARGETS {split}: {len(group)} episodes", flush=True)
    write(output / "manifest.json", result)


def cache_frozen_outputs(config):
    """Run FP32 causal inference once; cache no GT-derived graph coordinates."""
    tracker_path = Path(config["tracker"]["checkpoint"])
    tracker_hash = file_sha256(tracker_path)
    if tracker_hash != config["tracker"]["sha256"]:
        raise ValueError("Frozen tracker checkpoint checksum mismatch")
    targets_root = Path(config["targets_root"])
    targets = json.loads((targets_root / "manifest.json").read_text())
    tracking_hash = file_sha256(Path(config["data_root"]) / "manifest.json")
    if targets["tracking_manifest_sha256"] != tracking_hash or targets["test_payloads_read"]:
        raise ValueError("Target sidecar must match the train/validation-only tracker dataset")
    identity = {
        "tracker_sha256": tracker_hash, "tracking_manifest_sha256": tracking_hash,
        "targets_manifest_sha256": file_sha256(targets_root / "manifest.json"),
        "precision": "FP32", "validity": "cumulative actual observations",
        "postprocessing": "none; direct frozen tracker output",
    }
    output = Path(config["cache_root"])
    output.mkdir(parents=True, exist_ok=False)
    episodes, _ = load_training_episodes(config["data_root"])
    checkpoint = torch.load(tracker_path, map_location="cpu", weights_only=True)
    tracker = SpatialTemporalCornerTracker(**checkpoint["config"]["model"]).cuda()
    tracker.load_state_dict(checkpoint["model"], strict=True)
    tracker.eval().requires_grad_(False)
    fallback = checkpoint["fallback"].cpu().numpy()
    modes = sorted({episode["mode"] for episode in episodes["train"]})
    manifest = {"identity": identity, "modes": modes, "splits": {}}
    batch_size = config["tracker"]["inference_batch_size"]
    for split, group in episodes.items():
        record = targets["splits"][split]
        if record["episodes"] != [[e["seed"], e["mode"], len(e["tcp"])] for e in group]:
            raise ValueError(f"Target episode order differs for {split}")
        target_path = targets_root / record["path"]
        if file_sha256(target_path) != record["sha256"]:
            raise ValueError(f"Target sidecar checksum mismatch: {target_path}")
        collected = {key: [] for key in ("xyz", "seen", "observed", "tcp", "mode")}
        for start in range(0, len(group), batch_size):
            selected = group[start:start + batch_size]
            arrays, lengths, _ = sequence_arrays(selected, fallback)
            keys = ("features", "anchor", "detected", "observed", "valid")
            inputs = [torch.as_tensor(arrays[key], device="cuda") for key in keys]
            with torch.inference_mode():
                prediction = tracker(*inputs).cpu()
            for index, (episode, length) in enumerate(zip(selected, lengths, strict=True)):
                observed = torch.from_numpy(episode["observed"])
                seen = observed.cumsum(dim=0) > 0
                points = prediction[index, :length].clone()
                torch.testing.assert_close(
                    points[observed], torch.from_numpy(episode["detected"])[observed],
                    rtol=0, atol=0,
                )
                if not torch.isfinite(points).all():
                    raise RuntimeError("Nonfinite frozen tracker output")
                collected["xyz"].append(torch.where(seen[..., None], points, 0))
                collected["seen"].append(seen)
                collected["observed"].append(observed)
                collected["tcp"].append(torch.from_numpy(episode["tcp"]))
                collected["mode"].append(torch.full((int(length),), modes.index(episode["mode"])))
            del inputs, prediction, arrays
            if start % (batch_size * 20) == 0:
                print(f"CACHE {split}: {min(start + batch_size, len(group))}/{len(group)}", flush=True)
        tensors = {key: torch.cat(values) for key, values in collected.items()}
        with np.load(target_path, allow_pickle=False) as source:
            tensors["target"] = torch.from_numpy(source["target"].copy())
        path = output / f"{split}.pt"
        torch.save(tensors, path)
        manifest["splits"][split] = {
            "path": path.name, "sha256": file_sha256(path),
            "episodes": len(group), "frames": len(tensors["target"]),
            "observed_points": int(tensors["observed"].sum()),
            "previously_seen_missing_points": int((tensors["seen"] & ~tensors["observed"]).sum()),
            "never_seen_points": int((~tensors["seen"]).sum()),
        }
        del collected, tensors
        gc.collect()
    for name, value in tracker.state_dict().items():
        if not torch.equal(value.cpu(), checkpoint["model"][name]):
            raise RuntimeError(f"Frozen tracker parameter changed: {name}")
    if any(parameter.grad is not None or parameter.requires_grad for parameter in tracker.parameters()):
        raise RuntimeError("Frozen tracker must have no gradients or trainable parameters")
    if file_sha256(tracker_path) != tracker_hash:
        raise RuntimeError("Frozen tracker checkpoint was modified")
    manifest["tracker_parameters_unchanged"] = True
    write(output / "manifest.json", manifest)
    return manifest
