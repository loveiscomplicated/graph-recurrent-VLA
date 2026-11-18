"""Train a randomly initialized GNN on immutable causal tracker outputs."""

import gc
import json
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.frozen_tracker_gnn import cache_frozen_outputs
from graph.frozen_tracker_gnn import completed_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss
from utils.experiment_logger import ExperimentLogger, write


def take_frames(data, indices):
    """Select one shuffled frame batch, retaining only forward/loss inputs."""
    return {key: value[indices] for key, value in data.items() if key != "mode"}


def predict(model, batch):
    """Construct GT-free graph inputs and run the unmodified FP32 GNN."""
    graph, provenance = completed_graph(
        batch["xyz"], batch["seen"], batch["observed"], batch["tcp"]
    )
    output = model(graph, extra_node_features=provenance)
    return {
        "object21": output["object21"].float(),
        "endpoint_residual": output["endpoint_residual"].float(),
    }


def train_step(model, optimizer, batch, config):
    """Perform a complete finite-gradient optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    output = predict(model, batch)
    loss = object21_loss(output, batch["target"], **config["loss"])["total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return loss.detach()


def make_optimizer(model, config, learning_rate):
    """Create the optimizer over GNN parameters only."""
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=config["training"]["weight_decay"], fused=True,
    )


def autotune_batch(model, data, config):
    """Time full training steps with all 20 nodes present and reserve VRAM headroom."""
    initial = deepcopy(model.state_dict())
    options = config["batch_tuning"]
    capacity = torch.cuda.get_device_properties(0).total_memory
    # Stress the dense graph even when most real frames have fewer known slots.
    example = take_frames(data, torch.tensor([0], device="cuda"))
    records = []
    for size in options["candidates"]:
        batch = optimizer = loss = None
        try:
            model.load_state_dict(initial)
            optimizer = make_optimizer(model, config, config["training"]["learning_rates"][0])
            batch = {key: value.expand(size, *value.shape[1:]) for key, value in example.items()}
            batch["seen"] = torch.ones((size, 20), device="cuda", dtype=torch.bool)
            torch.cuda.reset_peak_memory_stats()
            for step in range(options["timed_steps"] + 2):
                if step == 2:
                    torch.cuda.synchronize()
                    started = monotonic()
                loss = train_step(model, optimizer, batch, config)
            torch.cuda.synchronize()
            elapsed = monotonic() - started
            peak = torch.cuda.max_memory_reserved()
            row = {
                "batch_size": size, "frames_per_second": size * options["timed_steps"] / elapsed,
                "peak_reserved_bytes": peak,
                "within_memory_budget": peak <= capacity * options["memory_fraction"],
                "stress_nodes_per_frame": 20,
            }
            records.append(row)
            print("BATCH_PROBE " + json.dumps(row), flush=True)
        except torch.OutOfMemoryError:
            records.append({"batch_size": size, "out_of_memory": True})
            print(f"BATCH_PROBE batch_size={size} out_of_memory", flush=True)
            break
        finally:
            batch = optimizer = loss = None
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    model.load_state_dict(initial)
    eligible = [row for row in records if row.get("within_memory_budget")]
    if not eligible:
        raise RuntimeError("No training batch fits the configured GPU memory budget")
    fastest = max(row["frames_per_second"] for row in eligible)
    selected = max(
        row["batch_size"] for row in eligible
        if row["frames_per_second"] >= fastest * options["throughput_fraction"]
    )
    return selected, records


@torch.inference_mode()
def validate(model, data, modes, batch_size, loss_config):
    """Average each mode's frame losses, then weight the five modes equally."""
    model.eval()
    result = {}
    for mode_index, name in enumerate(modes):
        indices = (data["mode"] == mode_index).nonzero(as_tuple=True)[0]
        sums = torch.zeros(4, device="cuda", dtype=torch.float64)
        for selected in indices.split(batch_size):
            batch = take_frames(data, selected)
            output = predict(model, batch)
            loss = object21_loss(output, batch["target"], **loss_config)["total"]
            errors = [
                torch.linalg.vector_norm(
                    output["object21"][:, start:start + 3] - batch["target"][:, start:start + 3],
                    dim=-1,
                ).mean() * 1000
                for start in (0, 7, 10)
            ]
            sums += torch.stack([loss, *errors]).double() * len(selected)
        values = (sums / len(indices)).cpu().tolist()
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite validation metrics for {name}")
        result[name] = dict(zip(("loss", "peg_position_mm", "endpoint_mm", "hole_position_mm"), values))
    score = float(np.mean([row["loss"] for row in result.values()]))
    return {"mode_balanced_loss": score, "modes": result}


def save_checkpoint(path, state):
    """Atomically retain a recoverable complete epoch checkpoint."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_cached_data(config):
    """Verify cached provenance and content before loading train/validation to CUDA."""
    root = Path(config["cache_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = cache_frozen_outputs(config)
    else:
        manifest = json.loads(manifest_path.read_text())
    expected = {
        "tracker_sha256": config["tracker"]["sha256"],
        "tracking_manifest_sha256": file_sha256(Path(config["data_root"]) / "manifest.json"),
        "targets_manifest_sha256": file_sha256(Path(config["targets_root"]) / "manifest.json"),
        "precision": "FP32", "validity": "cumulative actual observations",
        "postprocessing": "none; direct frozen tracker output",
    }
    if manifest["identity"] != expected or not manifest["tracker_parameters_unchanged"]:
        raise ValueError("Cached frozen outputs have a different source or validity contract")
    datasets = {}
    for split in ("train", "validation"):
        receipt = manifest["splits"][split]
        path = root / receipt["path"]
        if file_sha256(path) != receipt["sha256"]:
            raise ValueError(f"Cached graph checksum mismatch: {path}")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data["observed"] & ~data["seen"]).any():
            raise ValueError("Current observations must be included in cumulative visibility")
        if not torch.isfinite(data["target"]).all():
            raise ValueError("Nonfinite Object21 supervision")
        datasets[split] = {key: value.cuda() for key, value in data.items()}
        print(f"LOADED {split}: {receipt['episodes']} episodes, {receipt['frames']} frames", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return datasets, manifest


def train(config):
    """Freeze tracker inference, tune GNN throughput and train until validation plateaus."""
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen-tracker GNN training requires CUDA")
    source_path = Path(config["source"]["source_hash_manifest"])
    hashes = json.loads(source_path.read_text())["files"]
    repository = Path(__file__).resolve().parents[1]
    for relative, expected in hashes.items():
        if file_sha256(repository / relative) != expected:
            raise ValueError(f"Deployed source checksum mismatch: {relative}")
    if file_sha256(config["tracker"]["checkpoint"]) != config["tracker"]["sha256"]:
        raise ValueError("Frozen tracker checkpoint checksum mismatch")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["cpu_threads"])
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Cache inference keeps the tracker's strict FP32 execution. Enable TF32
    # only afterward for the new GNN's matrix multiplications. The shared GNN
    # uses FP32 scatter accumulators and is intentionally left unchanged.
    torch.backends.cuda.matmul.allow_tf32 = False
    datasets, cache_manifest = load_cached_data(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    # No GNN checkpoint is loaded, including during the batch-tuning reset.
    torch.manual_seed(seed)
    model = VertexEdgeGatedMPNN(**config["model"]).cuda()
    initial = deepcopy(model.state_dict())
    batch_size, probes = autotune_batch(model, datasets["train"], config)
    model.load_state_dict(initial)
    del initial
    torch.manual_seed(seed)
    effective = deepcopy(config)
    effective.update(
        batch_size=batch_size, initialization="random; no GNN checkpoint loaded",
        cache_manifest=cache_manifest, source_files=hashes,
    )
    write(output / "effective_config.json", effective)
    write(output / "batch_tuning.json", {"selected_batch_size": batch_size, "probes": probes})
    print(f"SELECTED_BATCH_SIZE={batch_size}", flush=True)
    training = config["training"]
    rates = training["learning_rates"]
    optimizer = make_optimizer(model, config, rates[0])
    best = threshold = float("inf")
    best_epoch = stale = rate_index = 0
    history = []
    started = monotonic()
    stopping = "maximum_epochs"
    with ExperimentLogger(
        team="graph", model="frozen_tracker_gnn", config=effective, seed=seed,
        root=output / "experiments",
    ) as logger:
        logger.record(
            runtime={"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                     "cuda": torch.version.cuda, "precision": "FP32 tensors; TF32 GNN matmul"},
            frozen_tracker=cache_manifest["identity"], test_payloads_read=False,
        )
        logger.link_output(config["tracker"]["checkpoint"], checkpoint=True)
        for path in (output / "effective_config.json", output / "batch_tuning.json"):
            logger.link_output(path)
        initial_validation = validate(
            model, datasets["validation"], cache_manifest["modes"], batch_size, config["loss"]
        )
        write(output / "initial_validation.json", initial_validation)
        print("INITIAL_VALIDATION " + json.dumps(initial_validation), flush=True)
        for epoch in range(1, training["maximum_epochs"] + 1):
            epoch_started = monotonic()
            model.train()
            count = len(datasets["train"]["target"])
            order = torch.randperm(count, device="cuda")
            accumulated = torch.zeros((), device="cuda", dtype=torch.float64)
            for selected in order.split(batch_size):
                loss = train_step(model, optimizer, take_frames(datasets["train"], selected), config)
                accumulated += loss.double() * len(selected)
            metrics = validate(
                model, datasets["validation"], cache_manifest["modes"], batch_size, config["loss"]
            )
            score = metrics["mode_balanced_loss"]
            improved = score < best
            if improved:
                best, best_epoch = score, epoch
            if score < threshold * (1 - training["minimum_relative_improvement"]):
                threshold, stale = score, 0
            else:
                stale += 1
            used_rate = optimizer.param_groups[0]["lr"]
            stop = False
            if stale >= training["patience"]:
                if rate_index + 1 < len(rates):
                    rate_index += 1
                    for group in optimizer.param_groups:
                        group["lr"] = rates[rate_index]
                    threshold, stale = best, 0
                    print(f"LEARNING_RATE={rates[rate_index]}", flush=True)
                else:
                    stopping, stop = "validation_plateau", True
            row = {
                "epoch": epoch, "train_loss": float(accumulated.item() / count),
                "validation": metrics, "best_validation_loss": best, "best_epoch": best_epoch,
                "learning_rate": used_rate, "next_learning_rate": optimizer.param_groups[0]["lr"],
                "stale_epochs": stale, "epoch_seconds": monotonic() - epoch_started,
                "elapsed_seconds": monotonic() - started,
            }
            history.append(row)
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "config": effective, "run_id": logger.run_id, "history": history,
                "best_validation_loss": best, "best_epoch": best_epoch,
                "plateau_threshold": threshold, "stale_epochs": stale, "rate_index": rate_index,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            if improved:
                save_checkpoint(output / "best.pt", state)
                logger.link_output(output / "best.pt")
            save_checkpoint(output / "last.pt", state)
            logger.link_output(output / "last.pt")
            write(output / "history.json", history)
            write(output / "progress.json", dict(row, run_id=logger.run_id, status="running"))
            logger.log({"train/loss": row["train_loss"], "validation": metrics,
                        "learning_rate": used_rate, "epoch_seconds": row["epoch_seconds"]}, step=epoch)
            print("EPOCH " + json.dumps(row), flush=True)
            if stop:
                break
        if file_sha256(config["tracker"]["checkpoint"]) != config["tracker"]["sha256"]:
            raise RuntimeError("Frozen tracker checkpoint changed during GNN training")
        completion = dict(row, run_id=logger.run_id, status="completed", stopping=stopping)
        write(output / "completed.json", completion)
        write(output / "progress.json", completion)
        logger.link_output(output / "completed.json")
        logger.link_output(output / "history.json")
        print("COMPLETED " + json.dumps(completion), flush=True)
