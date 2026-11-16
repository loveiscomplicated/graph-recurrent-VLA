"""Prepare portable fixed-Hole graph data once for the A–C2 curriculum."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.state21 import Episode, Sample, audit_splits
from data_collection.state21_batches import PreparedWindows
from data_collection.state21_shared_tapnext import (
    _audit_full_split,
    _load_episode,
    _open_source,
    _reference_split,
    split_reset_seeds,
)
from graph.state_estimation.graph import Frame, Graph, build_graph
from utils.experiment_logger import write

NODE_ARRAYS = (
    "features",
    "position",
    "object_id",
    "vertex_id",
    "visible",
    "geometry",
    "quality",
)
STAGES = (
    "expert",
    "post_grasp_perturbation",
    "extended_perturbation",
    "expert_recovery",
)
MODES = ("expert", "random", "expert_random", "expert_random_expert")


def _write_episode(path, episode, graphs):
    """Store local edge indices and untouched graph and label tensors."""
    arrays = {
        name: torch.cat([getattr(graph, name) for graph in graphs]).numpy()
        for name in NODE_ARRAYS
    }
    arrays.update(
        node_offsets=np.r_[0, np.cumsum([len(graph.features) for graph in graphs])],
        edge_offsets=np.r_[
            0, np.cumsum([graph.edge_index.shape[1] for graph in graphs])
        ],
        edge_index=torch.cat([graph.edge_index for graph in graphs], dim=1).numpy(),
        edge_attr=torch.cat([graph.edge_attr for graph in graphs]).numpy(),
        methods=np.asarray([graph.methods for graph in graphs]),
        rejected_edges=np.asarray([graph.rejected_edges for graph in graphs]),
        tcp=np.asarray([sample.observation.tcp for sample in episode.samples]),
        targets=torch.stack([sample.target for sample in episode.samples]).numpy(),
        frame_sha256=np.asarray([sample.frame_sha256 for sample in episode.samples]),
    )
    pending = path.with_suffix(".partial")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def _prepare_episode(task, output, max_edge_m):
    source, index, entry, split = task
    episode = _load_episode(
        source.arrays, index, entry, split, source.raw_root, "base_camera", None
    )
    graphs = [
        build_graph(sample.observation, max_edge_m=max_edge_m)
        for sample in episode.samples
    ]
    path = output / f"{episode.episode_id}.npz"
    _write_episode(path, episode, graphs)
    switch = entry.get("switch_step")
    post_grasp = False
    if entry["mode"] == "expert_random" and entry["switched"]:
        # The raw frame used here was already SHA-256 checked by the observation adapter.
        with np.load(
            source.raw_root / episode.episode_id / f"frame_{switch:06d}.npz",
            allow_pickle=False,
        ) as raw:
            post_grasp = bool(raw["collection/grasp"])
    return {
        "episode_id": episode.episode_id,
        "parent_id": episode.parent_id,
        "reset_fingerprint": episode.reset_fingerprint,
        "split": episode.split,
        "policy_mode": entry["mode"],
        "switch_step": switch,
        "recovery_step": entry.get("recovery_step"),
        "grasped_at_switch": post_grasp,
        "frames": len(episode.samples),
        "file": path.name,
        "sha256": file_sha256(path),
    }


def prepare_cache(config):
    """Materialize train and validation graphs while auditing the complete split."""
    settings = config["data"]
    output = Path(settings["curriculum_cache"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Curriculum cache already exists: {manifest}")
    sources = {name: _open_source(item) for name, item in settings["sources"].items()}
    entries = [row for source in sources.values() for row in source.entries]
    if (
        len(entries) != 400
        or len({row["path"] for row in entries}) != 400
        or {row["mode"] for row in entries} != set(MODES)
    ):
        raise ValueError("Expected the 400 fixed-Hole pilot and recovery episodes")
    if (
        len(
            {
                source.metadata["tapnext_checkpoint_sha256"]
                for source in sources.values()
            }
        )
        != 1
    ):
        raise ValueError("The caches must use the same TAPNext++ checkpoint")
    groups = split_reset_seeds(
        [row["seed"] for row in entries],
        seed=settings["split_seed"],
        validation_count=settings["validation_seed_count"],
        test_count=settings["test_seed_count"],
    )
    split_ref = _reference_split(settings, sources, groups)
    full_split = _audit_full_split(entries, groups)
    owners = {
        seed: split for split in ("train", "validation") for seed in groups[split]
    }
    tasks = [
        (source, index, row, owners[row["seed"]])
        for source in sources.values()
        for index, row in enumerate(source.entries)
        if row["seed"] in owners
    ]
    # Expert data is available first if preparation is interrupted and inspected.
    tasks.sort(key=lambda task: MODES.index(task[2]["mode"]))
    records = []
    workers = config["curriculum"].get("prepare_workers", 1)
    torch.set_num_threads(1)
    prepare_episode = partial(
        _prepare_episode, output=output, max_edge_m=settings["max_edge_m"]
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = 0
        for index, record in enumerate(pool.map(prepare_episode, tasks), start=1):
            records.append(record)
            frames += record["frames"]
            if index % 10 == 0 or index == len(tasks):
                print(
                    f"Prepared {index}/{len(tasks)} episodes, {frames} frames",
                    flush=True,
                )
    bundle = {
        "format": "state21_fixed_hole_graphs_v1",
        "max_edge_m": settings["max_edge_m"],
        "full_split": full_split,
        "split_reference_sha256": split_ref,
        "source_caches": {
            name: source.archive_sha256 for name, source in sources.items()
        },
        "episodes": records,
    }
    write(manifest, bundle)
    print(f"Verified curriculum graph cache: {manifest}", flush=True)


def read_graph_episode(root, row):
    """Restore tensor views from the checksum-verified transport archive."""
    path = root / row["file"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"Graph cache checksum differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        tensors = {
            name: torch.from_numpy(archive[name])
            for name in (*NODE_ARRAYS, "edge_index", "edge_attr", "targets")
        }
        node_offsets, edge_offsets = archive["node_offsets"], archive["edge_offsets"]
        methods, rejected = archive["methods"], archive["rejected_edges"]
        tcp, frame_sha256 = archive["tcp"], archive["frame_sha256"]
    name = row["episode_id"]
    graphs, samples = [], []
    for index in range(row["frames"]):
        nodes = slice(*node_offsets[index : index + 2])
        edges = slice(*edge_offsets[index : index + 2])
        graphs.append(
            Graph(
                **{name: tensors[name][nodes] for name in NODE_ARRAYS},
                edge_index=tensors["edge_index"][:, edges],
                edge_attr=tensors["edge_attr"][edges],
                methods=tuple(methods[index]),
                rejected_edges=int(rejected[index]),
                episode_id=name,
                frame_index=index,
            )
        )
        frame = Frame(name, index, tuple(tcp[index].tolist()), ())
        samples.append(
            Sample(frame, index, tensors["targets"][index], str(frame_sha256[index]))
        )
    episode = Episode(
        name, row["parent_id"], row["split"], tuple(samples), row["reset_fingerprint"]
    )
    return episode, graphs


def frame_limit(row, stage, short_random_frames, auxiliary_random_frames):
    """Select a causal episode prefix for the configured fixed-Hole stage."""
    total = row["frames"]
    mode = row["policy_mode"]
    if mode == "expert":
        return total
    if stage == "expert":
        return 0
    if stage == "post_grasp_perturbation":
        if mode != "expert_random" or not row["grasped_at_switch"]:
            return 0
        # The first Random action is applied after the observation at switch_step.
        return min(total, row["switch_step"] + short_random_frames + 1)
    if mode == "expert_random":
        return total
    if mode == "random":
        return min(total, auxiliary_random_frames)
    return total if stage == "expert_recovery" else 0


def prepare_stage(bundle, cached_episodes, stage, config):
    """Index the same graph inputs by stage without rebuilding geometry."""
    options = config["curriculum"]
    data = config["data"]
    prepared = {
        split: PreparedWindows([], data["history"], data["max_edge_m"])
        for split in ("train", "validation")
    }
    counts = {split: {} for split in prepared}
    visible_vertices = dict.fromkeys(prepared, 0)
    selected = []
    windows = []
    for row, (episode, graphs) in zip(bundle["episodes"], cached_episodes, strict=True):
        count = frame_limit(
            row,
            stage,
            options["short_random_frames"],
            options["auxiliary_random_frames"],
        )
        if count == 0:
            continue
        cropped = replace(episode, samples=episode.samples[:count])
        stage_graphs = graphs[:count]
        prepared[episode.split].add_episode(cropped, stage_graphs)
        visible_vertices[episode.split] += sum(
            int((graph.visible & (graph.vertex_id >= 0)).sum())
            for graph in stage_graphs
        )
        selected.append(cropped)
        windows.append((episode.episode_id, count))
        policy = counts[episode.split].setdefault(
            row["policy_mode"], {"episodes": 0, "frames": 0}
        )
        policy["episodes"] += 1
        policy["frames"] += count
    report = audit_splits(selected)
    report.update(
        manifest_sha256=canonical_hash(
            {"bundle": bundle, "stage": stage, "prefix_lengths": windows}
        ),
        full_split=bundle["full_split"],
        policy_counts=counts,
        verified_visible_vertex_rows=visible_vertices,
    )
    return prepared["train"], prepared["validation"], report
