"""Train the complete-graph Object21 GNN on frozen B2-3 tracker rollouts."""

import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.build_graph import EdgeType, GraphBatch, NodeType
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.state_estimation.graph import batch_graphs
from graph.temporal.spatial_temporal_corner_tracker import (
    OnlineSpatialTemporalTracker,
    load_tracker_checkpoint,
)
from training.fixed_id_gnn import evaluate_batch
from training.gnn.loss import object21_loss
from training.gnn.spatial_temporal_corner_data import (
    FiveModeRolloutDataset,
    rollout_dataset_identity,
)
from utils.experiment_logger import ExperimentLogger, write


def complete_track_graph(xyz, kinds, valid, observed, uncertainty, age, tcp, settings):
    """Build the downstream complete directed graph and reliability features."""
    batch, count, dimensions = xyz.shape
    expected = (batch, count)
    if dimensions != 3 or any(value.shape != expected for value in (kinds, valid, observed, uncertainty, age)):
        raise ValueError("B2-3 rollout arrays have inconsistent node shapes")
    types = torch.where(valid, kinds, NodeType.PAD)
    coordinates = torch.where(valid[..., None], xyz, 0)
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~torch.eye(count, dtype=torch.bool, device=xyz.device)[None]
    sample, source, target = allowed.nonzero(as_tuple=True)
    edges = torch.stack((sample * count + source, sample * count + target))
    source_type = types[sample, source]
    target_type = types[sample, target]
    edge_types = torch.where(source_type == target_type, source_type, EdgeType.CROSS)
    relative = coordinates[sample, source] - coordinates[sample, target]
    graph = GraphBatch(
        xyz_world=coordinates,
        object_type=types,
        node_valid=valid,
        semantic_id=torch.full_like(types, -1),
        semantic_valid=torch.zeros_like(valid),
        confidence=coordinates.new_zeros(expected),
        confidence_valid=torch.zeros_like(valid),
        tcp_xyz_world=tcp,
        edge_index=edges,
        edge_type=edge_types,
        relative_xyz=relative,
        distance=torch.linalg.vector_norm(relative, dim=-1, keepdim=True),
    )
    uncertainty_scale = settings.get("uncertainty_scale_m", 0.01)
    age_scale = settings.get("age_scale_frames", 10.0)
    if uncertainty_scale <= 0 or age_scale <= 0:
        raise ValueError("Downstream uncertainty and age scales must be positive")
    extra = torch.stack(
        (
            observed.to(xyz.dtype),
            torch.log1p(age.to(xyz.dtype)) / np.log1p(age_scale),
            uncertainty.to(xyz.dtype) / uncertainty_scale,
        ),
        dim=-1,
    )
    return graph, torch.where(valid[..., None], extra, 0)


@torch.inference_mode()
def rollout_episode(model, episode, association, device):
    """Run one complete episode without GT identities in the forward path."""
    tracker = OnlineSpatialTemporalTracker(
        model,
        distance_limit_m=association["distance_limit_m"],
        embedding_weight_m=association["embedding_weight_m"],
    )
    frames = []
    counts = {"matches": 0, "new_tracks": 0}
    for frame_index, graph in enumerate(episode.graphs):
        result = tracker.step(batch_graphs([graph]).to(device))
        frames.append(
            {
                "xyz": result["positions"].cpu().numpy(),
                "kinds": result["kinds"].cpu().numpy(),
                "observed": result["observed"].cpu().numpy(),
                "uncertainty": result["uncertainty"].cpu().numpy(),
                "age": result["age"].cpu().numpy(),
                "tcp": episode.tcp[frame_index],
                "target": episode.object21[frame_index],
                "sample_id": f"{episode.episode_id}:{frame_index}",
            }
        )
        counts["matches"] += result["matches"]
        counts["new_tracks"] += result["new_tracks"]
    counts["final_tracks"] = len(tracker.tracks)
    return frames, counts


def collate_rollout_frames(arrays, selected, device):
    """Right-pad selected frames from the ragged frozen rollout cache."""
    selected = np.asarray(selected, dtype=np.int64)
    offsets = arrays["offsets"]
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected + 1 >= len(offsets)):
        raise ValueError("Rollout frame indices are outside the frozen cache")
    counts = offsets[selected + 1] - offsets[selected]
    width = max(1, int(counts.max(initial=0)))
    batch = len(selected)
    xyz = np.zeros((batch, width, 3), dtype=np.float32)
    kinds = np.full((batch, width), -1, dtype=np.int64)
    valid = np.zeros((batch, width), dtype=bool)
    observed = np.zeros((batch, width), dtype=bool)
    uncertainty = np.zeros((batch, width), dtype=np.float32)
    age = np.zeros((batch, width), dtype=np.int64)
    for destination, frame_index in enumerate(selected):
        first, last = offsets[frame_index : frame_index + 2]
        count = last - first
        if count:
            xyz[destination, :count] = arrays["xyz"][first:last]
            kinds[destination, :count] = arrays["kinds"][first:last]
            valid[destination, :count] = True
            observed[destination, :count] = arrays["observed"][first:last]
            uncertainty[destination, :count] = arrays["uncertainty"][first:last]
            age[destination, :count] = arrays["age"][first:last]
    host = {
        "xyz": xyz,
        "kinds": kinds,
        "valid": valid,
        "observed": observed,
        "uncertainty": uncertainty,
        "age": age,
        "tcp": arrays["tcp"][selected],
        "target": arrays["target"][selected],
    }
    return {name: torch.as_tensor(value, device=device) for name, value in host.items()}


def build_rollout_cache(config, model, checkpoint, device, output):
    """Freeze full unfiltered train/validation tracker outputs into one cache."""
    datasets = {
        split: FiveModeRolloutDataset(
            config["data"]["manifest"],
            split,
            max_edge_m=config["data"].get("max_edge_m", 0.08),
        )
        for split in ("train", "validation")
    }
    expected = config["data"].get("expected_episodes", {})
    for split, count in expected.items():
        if split in datasets and len(datasets[split]) != count:
            raise ValueError(f"Expected {count} {split} episodes, found {len(datasets[split])}")
    node_chunks = {
        "xyz": [],
        "kinds": [],
        "observed": [],
        "uncertainty": [],
        "age": [],
    }
    offsets = [0]
    tcp_rows = []
    target_rows = []
    validation_rows = []
    sample_ids = []
    per_episode = []
    frame_count = 0
    max_nodes = 0
    maximum_allowed_tracks = int(config["training"].get("max_tracks_per_frame", 128))
    if maximum_allowed_tracks < 1:
        raise ValueError("training.max_tracks_per_frame must be positive")
    started = time.monotonic()
    try:
        for split, dataset in datasets.items():
            for number, episode in enumerate(dataset, 1):
                episode_frames, counts = rollout_episode(
                    model, episode, config["association"], device
                )
                for frame in episode_frames:
                    count = len(frame["xyz"])
                    if count > maximum_allowed_tracks:
                        raise ValueError(
                            f"Tracker produced {count} tracks at {frame['sample_id']}; "
                            f"maximum is {maximum_allowed_tracks}. Calibrate association "
                            "before building a quadratic complete graph."
                        )
                    max_nodes = max(max_nodes, count)
                    offsets.append(offsets[-1] + count)
                    for name in node_chunks:
                        node_chunks[name].append(frame[name])
                    tcp_rows.append(frame["tcp"])
                    target_rows.append(frame["target"])
                    validation_rows.append(split == "validation")
                    sample_ids.append(frame["sample_id"])
                    frame_count += 1
                per_episode.append({"split": split, "episode": episode.episode_id, **counts})
                interval = config["training"].get("rollout_log_every_episodes", 25)
                if number == 1 or number % interval == 0 or number == len(dataset):
                    print(
                        f"rollout {split} episodes={number}/{len(dataset)} frames={frame_count} "
                        f"elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    write(
                        output / "progress.json",
                        {
                            "phase": "rollout",
                            "split": split,
                            "episode": number,
                            "episodes": len(dataset),
                            "frames": frame_count,
                        },
                    )
    finally:
        for dataset in datasets.values():
            dataset.close()

    if not frame_count:
        raise ValueError("No frozen tracker rollout frames were produced")
    arrays = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "xyz": np.concatenate(node_chunks["xyz"], axis=0).astype(np.float32, copy=False),
        "kinds": np.concatenate(node_chunks["kinds"], axis=0).astype(np.int64, copy=False),
        "observed": np.concatenate(node_chunks["observed"], axis=0).astype(bool, copy=False),
        "uncertainty": np.concatenate(node_chunks["uncertainty"], axis=0).astype(
            np.float32, copy=False
        ),
        "age": np.concatenate(node_chunks["age"], axis=0).astype(np.int64, copy=False),
        "tcp": np.asarray(tcp_rows, dtype=np.float32),
        "target": np.asarray(target_rows, dtype=np.float32),
        "validation": np.asarray(validation_rows, dtype=bool),
        "sample_ids": np.asarray(sample_ids),
    }
    identity = rollout_dataset_identity(datasets["train"])
    identity.update(
        tracker_checkpoint_sha256=file_sha256(Path(config["tracker"]["checkpoint"])),
        tracker_training_dataset_sha256=checkpoint["identity"]["dataset_sha256"],
        train_episodes=len(datasets["train"]),
        validation_episodes=len(datasets["validation"]),
        train_frames=int((~arrays["validation"]).sum()),
        validation_frames=int(arrays["validation"].sum()),
        max_nodes=max_nodes,
        total_track_rows=int(arrays["offsets"][-1]),
        rollout_storage="ragged_frame_offsets",
        association=config["association"],
        per_episode_counts=per_episode,
    )
    cache_path = output / "frozen_tracker_rollouts.pt"
    torch.save({"format": "spatial_temporal_rollouts_v2", "arrays": arrays, "identity": identity}, cache_path)
    return arrays, identity, cache_path


def run_gnn_epoch(gnn, arrays, indices, config, device, optimizer=None):
    """Train or evaluate the downstream complete-graph GNN on frozen frames."""
    training = optimizer is not None
    gnn.train(training)
    settings = config["training"]
    totals = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for left in range(0, len(indices), settings["frame_batch_size"]):
            selected = indices[left : left + settings["frame_batch_size"]]
            values = collate_rollout_frames(arrays, selected, device)
            graph, extra = complete_track_graph(
                values["xyz"],
                values["kinds"],
                values["valid"],
                values["observed"],
                values["uncertainty"],
                values["age"],
                values["tcp"],
                config["downstream_features"],
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = gnn(graph, extra)
            losses = object21_loss(output, values["target"], **config["loss"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite B2-3 Object21 loss")
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    gnn.parameters(), settings.get("gradient_clip", 1.0), error_if_nonfinite=True
                )
                optimizer.step()
            size = len(selected)
            count += size
            metrics = {"loss": losses["total"].detach() * size}
            metrics.update(evaluate_batch(output, values["target"]))
            for name, value in metrics.items():
                totals[name] = totals.get(name, value.new_zeros(())) + value.detach()
    if not count:
        raise ValueError("No downstream Object21 frames were evaluated")
    return {name: float((value / count).cpu()) for name, value in totals.items()} | {"frames": count}


def train(config):
    """Freeze stage A rollouts, then train stage B complete-graph Object21 GNN."""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    torch.set_num_threads(config["training"].get("cpu_threads", 4))
    tracker, tracker_checkpoint = load_tracker_checkpoint(
        config["tracker"]["checkpoint"],
        config["tracker"].get("sha256"),
        device,
    )
    if any(parameter.requires_grad for parameter in tracker.parameters()):
        raise ValueError("Stage A tracker must remain frozen during stage B training")

    with ExperimentLogger(
        team="graph",
        model="spatial_temporal_object21",
        seed=seed,
        config=config,
        job_type="training",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        arrays, identity, cache_path = build_rollout_cache(
            config, tracker, tracker_checkpoint, device, output
        )
        cache_sha256 = file_sha256(cache_path)
        logger.link_output(config["tracker"]["checkpoint"], checkpoint=True)
        logger.link_output(cache_path)
        del tracker

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gnn = VertexEdgeGatedMPNN(**config["gnn"]).to(device)
        if gnn.extra_node_dim != 3 or gnn.use_semantic_ids:
            raise ValueError("B2-3 downstream GNN needs three reliability features and no semantic IDs")
        initial_hash = hashlib.sha256(
            b"".join(value.detach().cpu().numpy().tobytes() for value in gnn.state_dict().values())
        ).hexdigest()
        optimizer = torch.optim.AdamW(gnn.parameters(), **config["optimizer"])
        train_indices = np.flatnonzero(~arrays["validation"])
        validation_indices = np.flatnonzero(arrays["validation"])
        shuffle = np.random.default_rng(seed)
        history = []
        best = float("inf")
        best_epoch = 0
        stale = 0
        patience = config["training"].get("patience", config["training"]["epochs"])
        for epoch in range(1, config["training"]["epochs"] + 1):
            order = train_indices[shuffle.permutation(len(train_indices))]
            train_metrics = run_gnn_epoch(gnn, arrays, order, config, device, optimizer)
            validation = run_gnn_epoch(gnn, arrays, validation_indices, config, device)
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
            logger.log({"train": train_metrics, "validation": validation}, step=epoch)
            state = {
                "format": "spatial_temporal_object21_v1",
                "epoch": epoch,
                "config": config,
                "identity": identity,
                "rollout_cache_sha256": cache_sha256,
                "gnn_initial_sha256": initial_hash,
                "model": gnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "run_id": logger.run_id,
            }
            if validation["loss"] < best:
                best = validation["loss"]
                best_epoch = epoch
                stale = 0
                torch.save(state, output / "best.pt")
            else:
                stale += 1
            torch.save(state, output / "last.pt")
            write(output / "history.json", history)
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
                f"validation_loss={validation['loss']:.6f} best_epoch={best_epoch} "
                f"stale={stale}/{patience}",
                flush=True,
            )
            if stale >= patience:
                break
        summary = {
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best,
            "validation": history[best_epoch - 1]["validation"],
            "tracker_frozen": True,
        }
        write(output / "completed.json", summary)
        for name in (
            "effective_config.yaml",
            "best.pt",
            "last.pt",
            "history.json",
            "completed.json",
            "progress.json",
        ):
            path = output / name
            if path.exists():
                logger.link_output(path, kind="checkpoint" if name.endswith(".pt") else None)
        logger.record(
            **{key: value for key, value in identity.items() if key != "per_episode_counts"},
            rollout_cache_sha256=cache_sha256,
            gnn_initial_sha256=initial_hash,
            **summary,
        )
        return str(output)
