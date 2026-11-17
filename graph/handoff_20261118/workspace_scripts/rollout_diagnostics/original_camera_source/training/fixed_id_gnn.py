"""Train only the Object21 GNN on frozen fixed-ID temporal predictions."""

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.build_graph import EdgeType, GraphBatch, NodeType
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.gnn.loss import object21_loss
from utils.experiment_logger import ExperimentLogger, write


def load_temporal(config, device):
    """Load and freeze the exact recorded temporal checkpoint."""
    settings = config["temporal"]
    path = Path(settings["checkpoint"]).resolve()
    if file_sha256(path) != settings["sha256"]:
        raise ValueError("Temporal checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["dataset_sha256"] != config["data"]["dataset_sha256"]
            or checkpoint["config"]["model"]["architecture"] != settings["architecture"]
            or checkpoint["config"]["seed"] != config["seed"]):
        raise ValueError("Temporal checkpoint dataset, architecture or seed mismatch")
    repo = Path(__file__).resolve().parents[1]
    source_hashes = checkpoint["config"]["implementation"]["files_sha256"]
    for name in ("graph/temporal/fixed_id_corner.py", "graph/temporal/causal_transformer.py",
                 "data_collection/temporal/fixed_id_corner_sequences.py"):
        expected = source_hashes[name]
        if file_sha256(repo / name) != expected:
            raise ValueError(f"Temporal checkpoint source hash mismatch: {name}")
    model = FixedIdCornerPredictor(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def predict_frozen(model, sequences, frame_count, device, batch_size):
    """Evaluate full causal histories once and scatter outputs to frame/ID slots."""
    result = torch.zeros((frame_count, 24, 3), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            length = max(len(item["inputs"]) for item in batch)
            values = np.zeros((len(batch), length, 13), dtype=np.float32)
            mask = np.zeros((len(batch), length), dtype=bool)
            for index, sequence in enumerate(batch):
                size = len(sequence["inputs"])
                values[index, :size] = sequence["inputs"]
                mask[index, :size] = True
            inputs = torch.as_tensor(values, device=device)
            context = torch.as_tensor(mask, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                delta = model(inputs, context).float()
            if not torch.isfinite(delta[context]).all():
                raise FloatingPointError("Frozen temporal predictor returned nonfinite coordinates")
            for index, sequence in enumerate(batch):
                size = len(sequence["inputs"])
                frame = sequence["frame_offset"]
                result[frame:frame + size, sequence["vertex_id"]] = delta[index, :size]
            if start == 0 or start + batch_size >= len(sequences) or (start // batch_size) % 20 == 0:
                print(f"Frozen predictions: {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)
    return result


def make_graph(xyz, seen, observed, tcp):
    """Batch complete directed frame graphs, preserving detector provenance."""
    batch, count, _ = xyz.shape
    if count != 24:
        raise ValueError("Expected 24 fixed-ID slots")
    kinds = torch.tensor([0] * 8 + [1] * 8 + [2] * 8, device=xyz.device).expand(batch, -1)
    kinds = torch.where(seen, kinds, NodeType.PAD)
    coordinates = torch.where(seen.unsqueeze(-1), xyz, 0)
    candidate = seen[:, :, None] & seen[:, None, :]
    candidate &= ~torch.eye(count, dtype=torch.bool, device=xyz.device)[None]
    sample, source, target = candidate.nonzero(as_tuple=True)
    edges = torch.stack((sample * count + source, sample * count + target))
    source_kind, target_kind = kinds[sample, source], kinds[sample, target]
    edge_types = torch.where(source_kind == target_kind, source_kind, EdgeType.CROSS)
    relative = coordinates[sample, source] - coordinates[sample, target]
    graph = GraphBatch(
        xyz_world=coordinates, object_type=kinds, node_valid=seen,
        semantic_id=torch.full_like(kinds, -1), semantic_valid=torch.zeros_like(seen),
        confidence=coordinates.new_zeros((batch, count)), confidence_valid=torch.zeros_like(seen),
        tcp_xyz_world=tcp, edge_index=edges, edge_type=edge_types,
        relative_xyz=relative, distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    return graph, observed.to(xyz.dtype).unsqueeze(-1)


def evaluate_batch(output, target):
    """Sum physical-unit errors; quaternions use sign-invariant angles."""
    predicted = output["object21"].float()
    values = {}
    for name, segment in (("peg_position_mm", slice(0, 3)), ("endpoint_mm", slice(7, 10)),
                          ("hole_position_mm", slice(10, 13))):
        error = torch.linalg.vector_norm(predicted[:, segment] - target[:, segment], dim=-1) * 1000
        values[name] = error.sum()
    for name, segment in (("peg_rotation_deg", slice(3, 7)), ("hole_rotation_deg", slice(13, 17))):
        a = torch.nn.functional.normalize(predicted[:, segment], dim=-1)
        b = torch.nn.functional.normalize(target[:, segment], dim=-1)
        dot = (a * b).sum(-1).abs().clamp(max=1)
        values[name] = (2 * torch.acos(dot) * (180 / np.pi)).sum()
    values["size_mae_mm"] = (predicted[:, 17:21] - target[:, 17:21]).abs().mean(-1).sum() * 1000
    return values


def run_epoch(gnn, arrays, indices, config, optimizer, epoch, logger, output):
    """Train or validate on the same frame order and 21D objective."""
    training = optimizer is not None
    gnn.train(training)
    batch_size = config["training"]["batch_size"]
    totals = {}
    count = 0
    started = time.monotonic()
    phase = "train" if training else "validation"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for number, left in enumerate(range(0, len(indices), batch_size), start=1):
            index = indices[left:left + batch_size]
            graph, source = make_graph(arrays["xyz"][index], arrays["seen"][index],
                                       arrays["observed"][index], arrays["tcp"][index])
            if training:
                optimizer.zero_grad(set_to_none=True)
            output_values = gnn(graph, source)
            losses = object21_loss(output_values, arrays["target"][index], **config["loss"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite {phase} loss at epoch {epoch}, batch {number}")
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), config["training"]["gradient_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
            size = len(index)
            count += size
            metrics = {"loss": losses["total"].detach() * size}
            metrics.update(evaluate_batch(output_values, arrays["target"][index]))
            for key, value in metrics.items():
                totals[key] = totals.get(key, torch.zeros((), device=index.device)) + value.detach()
            batches = (len(indices) + batch_size - 1) // batch_size
            if number == 1 or number % config["training"]["log_every_batches"] == 0 or number == batches:
                current_loss = float((totals["loss"] / count).item())
                print(f"epoch={epoch} {phase} batch={number}/{batches} loss={current_loss:.6f} "
                      f"frames={count} elapsed={time.monotonic() - started:.1f}s", flush=True)
                write(output / "progress.json", dict(epoch=epoch, phase=phase, batch=number,
                      batches=batches, loss=current_loss, frames=count))
    metrics = {key: float((value / count).item()) for key, value in totals.items()}
    metrics["frames"] = count
    logger.log(metrics, step=epoch, prefix=phase)
    return metrics


def train(config):
    """Train one GNN with frozen temporal model and matched seed/settings."""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(config["training"]["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This recipe requires CUDA for official Mamba and matched execution")
    cache_path = Path(config["data"]["cache"]).resolve()
    cache_manifest = json.loads((cache_path.parent / "manifest.json").read_text())
    if file_sha256(cache_path) != cache_manifest["cache_sha256"]:
        raise ValueError("Prepared frame cache SHA256 mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["dataset_sha256"] != config["data"]["dataset_sha256"]:
        raise ValueError("Prepared frame cache dataset mismatch")
    temporal, checkpoint = load_temporal(config, device)
    # Reset RNG after temporal construction so the same seed initializes the
    # GNN identically for the Mamba and Transformer branches.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gnn = VertexEdgeGatedMPNN(**config["gnn"]).to(device)
    if gnn.extra_node_dim != 1:
        raise ValueError("GNN must accept exactly one detection provenance feature")
    optimizer = torch.optim.AdamW(gnn.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    with ExperimentLogger(team="graph", model="fixed_id_gnn", config=config, seed=seed) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["temporal"]["checkpoint"], checkpoint=True)
        logger.record(dataset_sha256=cache["dataset_sha256"], cache_sha256=cache_manifest["cache_sha256"],
                      temporal_sha256=config["temporal"]["sha256"], temporal_run_id=checkpoint["run_id"],
                      temporal_frozen=True, artificial_occlusion=False)
        initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                               for value in gnn.state_dict().values())).hexdigest()
        logger.record(gnn_initial_sha256=initial_hash)
        raw = cache["arrays"]
        arrays = {key: torch.as_tensor(value, device=device) for key, value in raw.items()}
        delta = predict_frozen(temporal, cache["sequences"], len(raw["split"]), device,
                               config["training"]["temporal_batch_size"])
        arrays["xyz"] = torch.where(arrays["observed"].unsqueeze(-1), arrays["measurement"],
                                     arrays["last"] + delta)
        del delta, temporal, cache
        train_indices = torch.nonzero(~arrays["split"], as_tuple=True)[0]
        validation_indices = torch.nonzero(arrays["split"], as_tuple=True)[0]
        if not len(train_indices) or not len(validation_indices):
            raise ValueError("Train and validation must both contain frames")
        shuffle = np.random.default_rng(seed)
        history = []
        best = float("inf")
        best_epoch = 0
        stale = 0
        stop_reason = "max_epochs"
        for epoch in range(1, config["training"]["epochs"] + 1):
            order = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
            train_metrics = run_epoch(gnn, arrays, train_indices[order], config, optimizer, epoch, logger, output)
            validation = run_epoch(gnn, arrays, validation_indices, config, None, epoch, logger, output)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation))
            if validation["loss"] < best:
                best = validation["loss"]
                best_epoch = epoch
                stale = 0
                torch.save(dict(model=gnn.state_dict(), epoch=epoch, config=config,
                                dataset_sha256=cache_manifest["dataset_sha256"],
                                cache_sha256=cache_manifest["cache_sha256"],
                                gnn_initial_sha256=initial_hash,
                                temporal_sha256=config["temporal"]["sha256"],
                                run_id=logger.run_id), output / "best.pt")
            else:
                stale += 1
            write(output / "history.json", history)
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={validation['loss']:.6f} "
                  f"best_epoch={best_epoch} early_stop={stale}/{config['training']['patience']}", flush=True)
            if stale >= config["training"]["patience"]:
                stop_reason = "early_stopping"
                break
        summary = dict(epochs=epoch, best_epoch=best_epoch, best_validation_loss=best,
                       stop_reason=stop_reason, validation=history[best_epoch - 1]["validation"])
        write(output / "completed.json", summary)
        for name in ("best.pt", "history.json", "completed.json", "progress.json"):
            logger.link_output(output / name)
        logger.record(**summary)
        print(f"Completed: {output}; {summary}", flush=True)
        return str(output)
