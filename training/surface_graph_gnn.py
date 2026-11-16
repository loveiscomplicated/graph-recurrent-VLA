"""Prepare, benchmark and train the leaderboard GNN on observed surface graphs."""

from math import isfinite
from pathlib import Path

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.surface_graph_frames import (
    SurfaceReplay,
    load_graphs,
    save_graphs,
    select_episodes,
)
from evaluation.surface_graph_gnn import benchmark
from graph.build_graph.surface_graph import collate_surface_graphs
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss
from training.gnn.tasks import geometry_errors
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


TRACKER_DEFAULTS = {
    "tap_input_resolution": 256,
    "tap_cuda_autocast": False,
    "tap_cache_cuda_autocast": False,
    "tap_mps_fp16_vit": False,
    "tap_mps_fp16_recurrent_mlp": False,
    "tap_mps_skip_past_query_embeddings": False,
    "tap_mps_compile_recurrent": False,
}


def prepare(config, output, logger):
    """Build and benchmark a checksum-linked archive from the selected trajectories."""
    from evaluation.tapnextpp_surface_tracking import load_tracker

    data = config["data"]
    source, episodes = select_episodes(data)
    tracker_config = config["tracker"]
    tracker_hash = file_sha256(tracker_config["checkpoint"])
    detector = TorchCornerDetector.main(device=config["detector_device"])
    tracker = load_tracker(tracker_config)
    logger.record(detector_device=str(detector.device), tracker_device=str(tracker.device),
                  source_sha256=file_sha256(source), tracker_sha256=tracker_hash)
    replay = SurfaceReplay(detector, tracker, tracker_config)

    graphs, timings, receipts = [], [], []
    for episode in episodes:
        samples = list(replay.frames(episode, data["camera"], data["stride"]))
        path = output / f"{episode.name}.npz"
        save_graphs(path, samples)
        _, episode_graphs, episode_timings = zip(*samples)
        graphs.extend(episode_graphs)
        timings.extend(episode_timings)
        receipts.append(dict(episode=episode.name, seed=episode.seed, split=episode.split,
                             file=path.name, sha256=file_sha256(path),
                             source_frames=len(episode.frame_hashes), graphs=len(samples)))
        print(f"{episode.split} {episode.name}: {len(samples)} graphs", flush=True)

    manifest = output / "manifest.json"
    write_json(manifest, dict(source=str(source), source_sha256=file_sha256(source),
                              tracker_sha256=tracker_hash, episodes=receipts))
    logger.link_output(manifest)
    result = benchmark(graphs, timings, config["gnn"], config["benchmark"])
    result["corner_nodes"] = sum(int(frame.is_corner.sum()) for frame in graphs)
    result["surface_nodes"] = sum(int((~frame.is_corner).sum()) for frame in graphs)
    result["surface_nodes_by_object"] = {
        name: sum(int(((~frame.is_corner) & (frame.kinds == kind)).sum()) for frame in graphs)
        for kind, name in enumerate(("peg", "box"))
    }
    result["empty_graphs"] = sum(len(frame.xyz) == 0 for frame in graphs)
    metrics = output / "benchmark.json"
    write_json(metrics, result)
    logger.log(result, prefix="graph_benchmark")
    logger.link_output(metrics)
    print(f"Benchmark: {metrics}", flush=True)


def run_epoch(model, batches, config, optimizer=None, *, on_update=None, score_geometry=False):
    """Compute frame-weighted Object21 metrics and optionally optimize each batch."""
    model.train(optimizer is not None)
    device = torch.device(config["device"])
    loss_sum, count = 0.0, 0
    error_sums = {}
    with torch.set_grad_enabled(optimizer is not None):
        for frames in batches:
            graph, features, target = collate_surface_graphs(frames)
            target = target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(graph.to(device), features.to(device))
            loss = object21_loss(prediction, target, **config["loss"])["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Object21 loss")
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True,
                )
                optimizer.step()
                if on_update is not None:
                    on_update()
            loss_sum += float(loss.detach()) * len(frames)
            count += len(frames)
            if score_geometry:
                errors = geometry_errors(prediction["object21"].float(), target)
                for name, values in errors.items():
                    if name.endswith("_m"):
                        name = name.removesuffix("_m") + "_mm"
                        values = values * 1000
                    total = values.sum()
                    error_sums[name] = error_sums.get(name, 0.0) + total
    metrics = {"loss": loss_sum / count}
    metrics.update({name: float(value) / count for name, value in error_sums.items()})
    if not all(isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite Object21 metrics")
    return metrics


def make_batches(frames, batch_size):
    """Group frame graphs for collation."""
    return [frames[left:left + batch_size] for left in range(0, len(frames), batch_size)]


def log_validation(model, batches, config, logger, epoch, updates):
    """Score the full holdout at this optimizer update and log physical errors."""
    metrics = run_epoch(model, batches, config, score_geometry=True)
    metrics["frames"] = sum(len(frames) for frames in batches)
    logger.log({"optimizer_updates": updates, "val": metrics}, step=epoch)
    print(f"Update {updates} (epoch {epoch}): validation loss {metrics['loss']:.6f}, "
          f"Hole center {metrics['hole_position_error_mm']:.2f} mm, "
          f"Peg center {metrics['peg_position_error_mm']:.2f} mm, "
          f"endpoint relative to Hole {metrics['endpoint_error_mm']:.2f} mm", flush=True)
    return metrics


def train(config, output, logger):
    """Train from random GNN weights with a fresh AdamW optimizer."""
    manifest = Path(config["graph_manifest"])
    split = load_graphs(manifest)
    logger.record(graph_manifest_sha256=file_sha256(manifest), initialization="random",
                  train_frames=len(split["train"]), validation_frames=len(split["validation"]))
    settings = config["training"]
    model = VertexEdgeGatedMPNN(**config["gnn"]).to(config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
    )
    train_batches = make_batches(split["train"], settings["batch_size"])
    val_batches = make_batches(split["validation"], settings["batch_size"])
    validation_interval = settings.get("validation_every_updates", len(train_batches))
    total_updates = settings["epochs"] * len(train_batches)
    logger.record(validation_every_updates=validation_interval)

    generator = np.random.default_rng(config["seed"])
    history = []
    updates = 0
    last_validation_update = 0
    validation = log_validation(model, val_batches, config, logger, epoch=0, updates=0)

    def after_update():
        nonlocal updates, last_validation_update, validation
        updates += 1
        if updates % validation_interval == 0 or updates == total_updates:
            validation = log_validation(model, val_batches, config, logger, epoch, updates)
            last_validation_update = updates
            model.train()

    for epoch in range(settings["epochs"] + 1):
        row = {"epoch": epoch}
        if epoch:
            shuffled = [split["train"][index] for index in generator.permutation(len(split["train"]))]
            row["optimization_loss"] = run_epoch(
                model, make_batches(shuffled, settings["batch_size"]), config, optimizer,
                on_update=after_update,
            )["loss"]
        row["optimizer_updates"] = updates
        row["train_loss"] = run_epoch(model, train_batches, config)["loss"]
        if last_validation_update == updates:
            row["validation_loss"] = validation["loss"]
            row["validation_metrics"] = validation
        history.append(row)
        metrics = {"optimizer_updates": updates, "train/loss": row["train_loss"]}
        if epoch:
            metrics["train/optimization_loss"] = row["optimization_loss"]
        logger.log(metrics, step=epoch)
        print(f"Epoch {epoch} (update {updates}): train loss {row['train_loss']:.6f}", flush=True)

    history_path = output / "history.json"
    write_json(history_path, history)
    logger.link_output(history_path)
    checkpoint = output / "last.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=settings["epochs"], config=config), checkpoint)
    logger.link_output(checkpoint)


def run(config):
    """Dispatch to graph preparation and timing, or train on prepared graphs."""
    phase = config["phase"]
    if phase not in ("benchmark", "train"):
        raise ValueError("phase must be benchmark or train")
    config = {**config, "tracker": {**TRACKER_DEFAULTS, **config["tracker"]}}
    if any(type(value) is not int or value < 1 for value in (
        config["data"]["stride"], config["tracker"]["surface_queries_per_object"],
        config["benchmark"]["repeats"], config["training"]["epochs"],
        config["training"]["batch_size"], config["cpu_threads"],
    )):
        raise ValueError("Stride, query count, repeats, epochs, batch size and threads must be positive integers")
    if phase == "train" and not config.get("graph_manifest"):
        raise ValueError("Set graph_manifest to the benchmark run's output manifest.json")
    if "validation_every_updates" in config["training"]:
        interval = config["training"]["validation_every_updates"]
        if type(interval) is not int or interval < 1:
            raise ValueError("training.validation_every_updates must be a positive integer")

    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    cv2.setNumThreads(1)
    with ExperimentLogger(team="graph", config=config, seed=config["seed"],
                          model="surface_graph_gnn", job_type="train" if phase == "train" else "evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        logger.record(torch_version=str(torch.__version__))
        print(f"Output directory: {output}", flush=True)
        if phase == "benchmark":
            prepare(config, output, logger)
        else:
            train(config, output, logger)
