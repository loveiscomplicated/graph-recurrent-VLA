"""Train Object21 GNNs on frozen detector-only temporal rollouts."""

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import load_source_episode, read_manifest, read_source_manifest
from graph.build_graph import EdgeType, GraphBatch, NodeType
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.fixed_id_gnn import evaluate_batch, load_temporal
from training.gnn.loss import object21_loss
from utils.experiment_logger import ExperimentLogger, write


def graph_batch(xyz, kinds, valid, observed, tcp):
    """Construct complete per-frame graphs from detector-only track outputs."""
    batch, count, _ = xyz.shape
    if kinds.shape != (batch, count) or valid.shape != kinds.shape or observed.shape != kinds.shape:
        raise ValueError("Graph node masks and detector kinds must have shape [B,N]")
    types = torch.where(valid, kinds, NodeType.PAD)
    xyz = torch.where(valid.unsqueeze(-1), xyz, 0)
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~torch.eye(count, dtype=torch.bool, device=xyz.device)[None]
    sample, source, target = allowed.nonzero(as_tuple=True)
    edges = torch.stack((sample * count + source, sample * count + target))
    edge_types = torch.where(types[sample, source] == types[sample, target],
                             types[sample, source], EdgeType.CROSS)
    relative = xyz[sample, source] - xyz[sample, target]
    graph = GraphBatch(
        xyz_world=xyz, object_type=types, node_valid=valid,
        semantic_id=torch.full_like(types, -1), semantic_valid=torch.zeros_like(valid),
        confidence=xyz.new_zeros((batch, count)), confidence_valid=torch.zeros_like(valid),
        tcp_xyz_world=tcp, edge_index=edges, edge_type=edge_types,
        relative_xyz=relative, distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    return graph, observed.to(xyz.dtype).unsqueeze(-1)


def rollout_episode(model, source, threshold_m, device):
    """Produce all detector-only tracked points without GT in the forward path."""
    tracker = DetectedCornerTracker(model, threshold_m, device)
    frames = []
    counts = dict(matches=0, new_tracks=0)
    for frame in range(len(source["robot"])):
        valid = source["valid"][frame]
        result = tracker.step(source["xyz"][frame, valid], source["kind"][frame, valid], source["robot"][frame])
        frames.append(dict(xyz=result["positions"], observed=result["observed"],
                           kinds=np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)))
        counts["matches"] += result["matches"]
        counts["new_tracks"] += result["new_tracks"]
    counts["final_tracks"] = len(tracker.tracks)
    return frames, counts


def build_rollouts(config, temporal, device, output):
    """Align detector-only episode rollouts with the existing Object21 labels."""
    fixed_root = Path(config["data"]["fixed_id_dataset"]).resolve()
    source_root = Path(config["data"]["source_dataset"]).resolve()
    fixed = read_manifest(fixed_root)
    source = read_source_manifest(source_root)
    if (fixed["dataset_sha256"] != config["data"]["dataset_sha256"]
            or source["dataset_sha256"] != config["data"]["source_sha256"]
            or fixed["source_dataset_sha256"] != source["dataset_sha256"]):
        raise ValueError("Corner dataset lineage or hash mismatch")
    frame_path = Path(config["data"]["frame_cache"]).resolve()
    frame_manifest = json.loads((frame_path.parent / "manifest.json").read_text())
    if (file_sha256(frame_path) != frame_manifest["cache_sha256"]
            or frame_manifest["dataset_sha256"] != fixed["dataset_sha256"]):
        raise ValueError("Synchronized Object21 frame cache hash mismatch")
    frame_cache = torch.load(frame_path, map_location="cpu", weights_only=False)["arrays"]
    selection_path = Path(config["selection_file"])
    if file_sha256(selection_path) != config["selection_sha256"]:
        raise ValueError("Calibrated threshold selection hash mismatch")
    selection = json.loads(selection_path.read_text())
    if selection["split"] != "calibration":
        raise ValueError("Distance threshold must come from calibration data")
    threshold = selection["threshold_mm"]
    validation_path = Path(config["source_validation"]["path"])
    if file_sha256(validation_path) != config["source_validation"]["sha256"]:
        raise ValueError("Detector-only validation result hash mismatch")
    validation = json.loads(validation_path.read_text())
    if (validation["config"]["temporal"]["sha256"] != config["temporal"]["sha256"]
            or validation["selection"]["threshold_mm"] != threshold):
        raise ValueError("Recorded validation uses a different temporal checkpoint or gate")
    expected = {row["episode"]: row["counts"] for row in validation["result"]["per_episode"]}
    source_rows = {row["episode"]: row for row in source["episodes"]}
    rows = [row for row in fixed["episodes"] if row["split"] in ("train", "validation")]
    if {row["episode"] for row in rows if row["split"] == "validation"} != set(expected):
        raise ValueError("Validation episode membership differs")
    data = []
    sample_ids = []
    offset = 0
    started = time.monotonic()
    for index, row in enumerate(rows):
        path = (fixed_root / row["file"]).resolve()
        if not path.is_relative_to(fixed_root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Prepared corner episode checksum mismatch: {row['episode']}")
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["sample_ids"].copy()
            robot = archive["robot_state"].copy()
        original = load_source_episode(source_root, source_rows[row["episode"]])
        if not np.array_equal(ids, original["sample_ids"]) or not np.array_equal(robot, original["robot"]):
            raise ValueError(f"Source samples or robot state differ: {row['episode']}")
        length = len(ids)
        segment = slice(offset, offset + length)
        expected_split = row["split"] == "validation"
        if (not np.all(frame_cache["split"][segment] == expected_split)
                or not np.allclose(frame_cache["tcp"][segment], robot[:, :3], atol=1e-6, rtol=0)):
            raise ValueError(f"Object21/TCP frame cache alignment mismatch: {row['episode']}")
        frames, counts = rollout_episode(temporal, original, threshold / 1000, device)
        if expected_split and any(counts[key] != expected[row["episode"]][key]
                                  for key in ("matches", "new_tracks", "final_tracks")):
            raise ValueError(f"Validation tracking replay differs: {row['episode']}")
        data.extend(frames)
        sample_ids.extend(ids.tolist())
        offset += length
        if index == 0 or (index + 1) % config["training"]["rollout_log_every_episodes"] == 0 or index + 1 == len(rows):
            print(f"rollout episodes={index+1}/{len(rows)} frames={offset} "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            write(output / "progress.json", dict(phase="rollout", episode=index + 1, episodes=len(rows),
                                                   frames=offset, elapsed_seconds=time.monotonic() - started))
    if offset != len(frame_cache["target"]):
        raise ValueError("Object21 cache frame count differs from rollout")
    max_nodes = max(1, max(len(frame["xyz"]) for frame in data))
    positions = np.zeros((offset, max_nodes, 3), dtype=np.float32)
    kinds = np.full((offset, max_nodes), -1, dtype=np.int64)
    valid = np.zeros((offset, max_nodes), dtype=bool)
    observed = np.zeros_like(valid)
    for index, frame in enumerate(data):
        count = len(frame["xyz"])
        positions[index, :count] = frame["xyz"]
        kinds[index, :count] = frame["kinds"]
        valid[index, :count] = True
        observed[index, :count] = frame["observed"]
    arrays = dict(xyz=positions, kinds=kinds, valid=valid, observed=observed,
                  tcp=frame_cache["tcp"], target=frame_cache["target"], split=frame_cache["split"],
                  sample_ids=np.asarray(sample_ids))
    return arrays, dict(source_dataset_sha256=source["dataset_sha256"],
                        dataset_sha256=fixed["dataset_sha256"], frame_cache_sha256=frame_manifest["cache_sha256"],
                        selection_sha256=config["selection_sha256"], threshold_mm=threshold,
                        train_frames=int((~arrays["split"]).sum()), validation_frames=int(arrays["split"].sum()),
                        max_nodes=max_nodes)


def run_epoch(gnn, arrays, indices, config, optimizer, epoch, logger, output):
    """Train or validate the GNN while the rollout inputs remain frozen."""
    training = optimizer is not None
    gnn.train(training)
    batch_size = config["training"]["batch_size"]
    phase = "train" if training else "validation"
    totals = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for number, left in enumerate(range(0, len(indices), batch_size), start=1):
            index = indices[left:left + batch_size]
            graph, features = graph_batch(arrays["xyz"][index], arrays["kinds"][index],
                                           arrays["valid"][index], arrays["observed"][index], arrays["tcp"][index])
            if training:
                optimizer.zero_grad(set_to_none=True)
            result = gnn(graph, features)
            loss = object21_loss(result, arrays["target"][index], **config["loss"])["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {phase} loss at epoch {epoch}, batch {number}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            size = len(index)
            count += size
            metrics = {"loss": loss.detach() * size}
            metrics.update(evaluate_batch(result, arrays["target"][index]))
            for key, value in metrics.items():
                totals[key] = totals.get(key, torch.zeros((), device=index.device)) + value.detach()
            batches = (len(indices) + batch_size - 1) // batch_size
            if number == 1 or number % config["training"]["log_every_batches"] == 0 or number == batches:
                current_loss = float((totals["loss"] / count).item())
                write(output / "progress.json", dict(phase=phase, epoch=epoch, batch=number,
                      batches=batches, loss=current_loss, frames=count))
    metrics = {key: float((value / count).item()) for key, value in totals.items()}
    metrics["frames"] = count
    logger.log(metrics, step=epoch, prefix=phase)
    return metrics


def train(config):
    """Train just the GNN for the configured total epoch budget."""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This matched workflow requires CUDA and official mamba_ssm")
    temporal, checkpoint = load_temporal(config, device)
    if any(parameter.requires_grad for parameter in temporal.parameters()):
        raise ValueError("Temporal predictor must be frozen")
    with ExperimentLogger(team="graph", model="detected_corner_gnn", config=config, seed=seed) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["temporal"]["checkpoint"], checkpoint=True)
        arrays, identity = build_rollouts(config, temporal, device, output)
        if any(parameter.grad is not None for parameter in temporal.parameters()):
            raise RuntimeError("Frozen temporal model unexpectedly received gradients")
        del temporal
        cache_path = output / "rollout_inputs.pt"
        torch.save(dict(arrays=arrays, identity=identity), cache_path)
        cache_sha = file_sha256(cache_path)
        logger.link_output(cache_path)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gnn = VertexEdgeGatedMPNN(**config["gnn"]).to(device)
        if gnn.extra_node_dim != 1:
            raise ValueError("GNN must use one detection-source feature")
        initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                               for value in gnn.state_dict().values())).hexdigest()
        logger.record(**identity, temporal_frozen=True, temporal_run_id=checkpoint["run_id"],
                      temporal_sha256=config["temporal"]["sha256"], gnn_initial_sha256=initial_hash,
                      rollout_cache_sha256=cache_sha, gt_ids_used_in_tracker=False)
        tensors = {name: torch.as_tensor(value, device=device) for name, value in arrays.items() if name != "sample_ids"}
        train_indices = torch.nonzero(~tensors["split"], as_tuple=True)[0]
        validation_indices = torch.nonzero(tensors["split"], as_tuple=True)[0]
        optimizer = torch.optim.AdamW(gnn.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        shuffle = np.random.default_rng(seed)
        best, best_epoch, history = float("inf"), 0, []
        for epoch in range(1, config["training"]["epochs"] + 1):
            order = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
            train_metrics = run_epoch(gnn, tensors, train_indices[order], config, optimizer, epoch, logger, output)
            validation = run_epoch(gnn, tensors, validation_indices, config, None, epoch, logger, output)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation))
            state = dict(model=gnn.state_dict(), epoch=epoch, config=config, identity=identity,
                         gnn_initial_sha256=initial_hash, rollout_cache_sha256=cache_sha,
                         temporal_sha256=config["temporal"]["sha256"], run_id=logger.run_id)
            torch.save(state, output / "last.pt")
            if validation["loss"] < best:
                best, best_epoch = validation["loss"], epoch
                torch.save(state, output / "best.pt")
            write(output / "history.json", history)
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={validation['loss']:.6f} "
                  f"best_epoch={best_epoch}", flush=True)
        summary = dict(epochs=len(history), best_epoch=best_epoch, best_validation_loss=best,
                       validation=history[best_epoch - 1]["validation"])
        write(output / "completed.json", summary)
        for name in ("best.pt", "last.pt", "history.json", "completed.json", "progress.json"):
            logger.link_output(output / name)
        logger.record(**summary)
        print(f"Completed: {output}; {summary}", flush=True)
        return str(output)
