"""Adapt frozen Five-mode tracker frames to the existing GNN training lifecycle."""

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import load_training_episodes
from data_collection.five_mode_surface_context import attach_surface_context, surface_sequence_arrays
from graph.frozen_tracker_graph import build_tracker_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker
from training.gnn.data import DataBundle
from training.gnn.loss import object21_loss
from training.gnn.tasks import Object21Task, StepResult, geometry_errors
from utils.experiment_logger import write


def state_digest(model):
    """Fingerprint parameter and buffer bytes to verify the frozen forward pass."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def attach_targets(groups, root, source_hash):
    """Validate episode membership, labels and TCP alignment before any rollout."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source_manifest_sha256"] != source_hash or manifest["test_payloads_read"]:
        raise ValueError("Object21 targets must match the train/validation release")
    records = {(r["seed"], r["mode"]): r for r in manifest["records"]}
    for split, episodes in groups.items():
        for episode in episodes:
            row = records.pop((episode["seed"], episode["mode"]))
            path = root / row["path"]
            if row["split"] != split or row["frames"] != len(episode["tcp"]) or file_sha256(path) != row["sha256"]:
                raise ValueError(f"Object21 episode identity/checksum differs: {path}")
            with np.load(path, allow_pickle=False) as archive:
                target, tcp = archive["object21"], archive["tcp"]
            if (target.shape != (len(tcp), 21) or not np.isfinite(target).all()
                    or not np.allclose(tcp, episode["tcp"], atol=1e-6, rtol=0)):
                raise ValueError(f"Object21/TCP frame alignment differs: {path}")
            episode["target"] = target.astype(np.float32)
    if records:
        raise ValueError("Object21 sidecar contains unmatched episodes")
    return file_sha256(root / "manifest.json")


def frozen_rollouts(config):
    """Run the verified tracker once; cache only causal, previously observed slots."""
    settings = config["data"]
    path = Path(settings["tracker_checkpoint"])
    if file_sha256(path) != settings["tracker_sha256"]:
        raise ValueError("Frozen surface tracker checkpoint checksum differs")
    cache = Path(settings["rollout_cache"])
    receipt = cache.with_suffix(".json")
    if cache.exists():
        identity = json.loads(receipt.read_text())
        expected = {
            "dataset_sha256": file_sha256(Path(settings["corner_root"]) / "manifest.json"),
            "surface_manifest_sha256": file_sha256(Path(settings["surface_root"]) / "manifest.json"),
            "target_manifest_sha256": file_sha256(Path(settings["target_root"]) / "manifest.json"),
            "tracker_sha256": settings["tracker_sha256"],
            "rollout_cache_sha256": file_sha256(cache),
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing frozen rollout cache identity differs")
        if expected["dataset_sha256"] != settings["corner_manifest_sha256"]:
            raise ValueError("Cached corner manifest checksum differs")
        return torch.load(cache, map_location="cpu", weights_only=False)["arrays"], identity
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["config"]["training"]["supervision"] != "after_first_observation":
        raise ValueError("Expected the corrected observed-only tracker")
    tracker = SurfaceSpatialCornerTracker(**payload["config"]["model"]).cuda().eval()
    tracker.load_state_dict(payload["model"], strict=True)
    tracker.requires_grad_(False)
    before = state_digest(tracker)
    fallback = np.asarray(payload["fallback"], dtype=np.float32)
    groups, manifest = load_training_episodes(settings["corner_root"])
    base_hash = file_sha256(Path(settings["corner_root"]) / "manifest.json")
    if base_hash != settings["corner_manifest_sha256"]:
        raise ValueError("Prepared corner manifest checksum differs")
    surface_hash = attach_surface_context(groups, settings["surface_root"], settings["corner_root"])
    targets_hash = attach_targets(groups, settings["target_root"], manifest["source_manifest_sha256"])
    arrays = {}
    inputs = ("features", "anchor", "detected", "observed", "valid", "surface_features", "surface_valid")
    for split, episodes in groups.items():
        collected = {key: [] for key in ("xyz", "observed", "valid", "tcp", "target")}
        for left in range(0, len(episodes), settings["rollout_batch_size"]):
            subset = episodes[left:left + settings["rollout_batch_size"]]
            sequences, lengths, _ = surface_sequence_arrays(subset, fallback)
            tensors = {key: torch.from_numpy(sequences[key]).cuda() for key in inputs}
            with torch.inference_mode():
                prediction = tracker(*(tensors[key] for key in inputs)).cpu().numpy()
            for index, (episode, length) in enumerate(zip(subset, lengths)):
                observed = episode["observed"]
                valid = np.logical_or.accumulate(observed, axis=0)
                xyz = prediction[index, :length]
                if not np.isfinite(xyz).all():
                    raise ValueError("Frozen tracker returned nonfinite coordinates")
                if not np.array_equal(xyz[observed], episode["detected"][observed]):
                    raise ValueError("Frozen tracker changed measured coordinates")
                collected["xyz"].append(np.where(valid[..., None], xyz, 0))
                for key, value in (("valid", valid), ("observed", observed),
                                   ("tcp", episode["tcp"]), ("target", episode["target"])):
                    collected[key].append(value)
            del tensors, sequences, prediction
            if left % (settings["rollout_batch_size"] * 20) == 0:
                print(f"frozen rollout {split}: {min(left + len(subset), len(episodes))}/{len(episodes)}", flush=True)
        arrays[split] = {key: np.concatenate(values) for key, values in collected.items()}
    if state_digest(tracker) != before or any(p.grad is not None for p in tracker.parameters()):
        raise RuntimeError("Frozen tracker state or gradients changed")
    identity = dict(dataset_sha256=base_hash, surface_manifest_sha256=surface_hash,
                    target_manifest_sha256=targets_hash, source_manifest_sha256=manifest["source_manifest_sha256"],
                    tracker_sha256=settings["tracker_sha256"], tracker_state_sha256=before,
                    tracker_epoch=payload["epoch"], tracker_frozen=True, tracker_precision="float32",
                    test_payloads_read=False, never_observed_slots="excluded from GNN nodes and edges",
                    splits={split: dict(episodes=len(groups[split]), frames=len(values["xyz"]))
                            for split, values in arrays.items()})
    del tracker, groups
    gc.collect()
    torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        raise FileExistsError(f"Preserving existing rollout cache: {cache}")
    torch.save(dict(arrays=arrays, identity=identity), cache)
    identity["rollout_cache_sha256"] = file_sha256(cache)
    write(cache.with_suffix(".json"), identity)
    return arrays, identity


@dataclass
class TrackerBatch:
    graph: object
    flags: torch.Tensor
    target: torch.Tensor

    def to(self, device):
        return TrackerBatch(self.graph.to(device), self.flags.to(device), self.target.to(device))


def make_batch(arrays, indices):
    graph, flags = build_tracker_graph(*(arrays[key][indices] for key in ("xyz", "observed", "valid", "tcp")))
    return TrackerBatch(graph, flags, arrays["target"][indices])


class TrackerObject21Task(Object21Task):
    """Keep the established loss while adding measured/tracked provenance flags."""

    def compute(self, model, batch):
        output = model(batch.graph, batch.flags)
        terms = object21_loss(output, batch.target, **self.loss_settings)
        metrics = {name: value.detach() for name, value in terms.items()}
        metrics["loss"] = metrics.pop("total")
        metrics.update({name: error.mean() for name, error in geometry_errors(output["object21"], batch.target).items()})
        return StepResult(terms["total"], metrics, {name: len(batch.target) for name in metrics})


class FrameLoader:
    """Shuffle whole frames on the GPU, preserving the release episode split."""

    def __init__(self, arrays, batch_size, shuffle, generator):
        self.arrays, self.batch_size = arrays, batch_size
        self.shuffle, self.generator = shuffle, generator

    def __len__(self):
        return math.ceil(len(self.arrays["xyz"]) / self.batch_size)

    def __iter__(self):
        count = len(self.arrays["xyz"])
        indices = torch.randperm(count, generator=self.generator) if self.shuffle else torch.arange(count)
        indices = indices.cuda()
        for left in range(0, count, self.batch_size):
            yield make_batch(self.arrays, indices[left:left + self.batch_size])


def tune_batch(config, arrays):
    """Measure real forward/backward/AdamW steps and retain training memory headroom."""
    from training.gnn.trainer import build_model

    settings = config["batch_tuning"]
    model = build_model(config["model"], torch.device("cuda"))
    task = TrackerObject21Task(**config["task"]["settings"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"])
    # Include the maximum edge count, even if no real batch has every slot present.
    worst = {key: value[:max(settings["candidates"])].clone() for key, value in arrays.items()}
    worst["valid"].fill_(True)
    reports = []
    for size in settings["candidates"]:
        try:
            torch.cuda.reset_peak_memory_stats()
            batch = make_batch(worst, torch.arange(size, device="cuda"))
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                task.compute(model, batch).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            started = perf_counter()
            for _ in range(settings["timed_steps"]):
                optimizer.zero_grad(set_to_none=True)
                batch = make_batch(worst, torch.arange(size, device="cuda"))
                task.compute(model, batch).loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
            torch.cuda.synchronize()
            rate = size * settings["timed_steps"] / (perf_counter() - started)
            memory = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            reports.append(dict(batch_size=size, frames_per_second=rate, memory_fraction=memory,
                                accepted=memory <= settings["memory_fraction"]))
            print(f"batch probe {reports[-1]}", flush=True)
            del batch
        except torch.cuda.OutOfMemoryError:
            reports.append(dict(batch_size=size, accepted=False, error="CUDA out of memory"))
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            break
    eligible = [row for row in reports if row["accepted"]]
    if not eligible:
        raise RuntimeError("No batch candidate fits the configured memory budget")
    peak = max(row["frames_per_second"] for row in eligible)
    selected = max(row["batch_size"] for row in eligible
                   if row["frames_per_second"] >= peak * settings["throughput_fraction"])
    del model, optimizer, worst
    gc.collect()
    torch.cuda.empty_cache()
    return selected, reports


def build_data(config, generator):
    """Supply existing trainer plugins; no tracker or GNN weights are resumed."""
    if config["resume"]:
        raise ValueError("This recipe requires GNN training from random initialization")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    arrays, identity = frozen_rollouts(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    arrays = {split: {key: torch.from_numpy(value).cuda() for key, value in values.items()}
              for split, values in arrays.items()}
    batch_size, reports = tune_batch(config, arrays["train"])
    config["training"]["batch_size"] = batch_size
    identity.update(selected_batch_size=batch_size, batch_tuning=reports, gnn_initialization="random")
    write(Path(config["data"]["rollout_cache"]).with_name("batch_tuning.json"), identity)
    print(f"GNN batch selected={batch_size}; beginning fresh GNN training", flush=True)
    return DataBundle(FrameLoader(arrays["train"], batch_size, True, generator),
                      FrameLoader(arrays["validation"], batch_size, False, generator), identity)
