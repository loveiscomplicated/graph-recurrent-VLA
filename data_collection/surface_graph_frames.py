"""Replay a bounded subset of saved RGB-D trajectories into surface graphs."""

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from data_collection.episode_dataset import file_sha256, read_frame
from graph.build_graph import NodeType
from graph.build_graph.surface_graph import SurfaceGraph, delaunay_edges
from graph.utils.object21_targets import object21_from_gt


@dataclass(frozen=True)
class EpisodeSlice:
    """One selected trajectory with its recorded source identity."""

    name: str
    seed: int
    split: str
    directory: Path
    frame_hashes: list[str]


def select_episodes(config):
    """Select explicit episodes with disjoint train and validation reset seeds."""
    manifest_path = Path(config["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "COMPLETE":
        raise ValueError(f"Expected a COMPLETE trajectory manifest: {manifest_path}")
    by_name = {episode["path"]: episode for episode in manifest["episodes"]}
    episodes = []
    for split in ("train", "validation"):
        names = config["episodes"][split]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"{split} needs distinct episode names")
        for name in names:
            if name not in by_name:
                raise ValueError(f"Episode {name} is missing from {manifest_path}")
            row = by_name[name]
            directory = (manifest_path.parent / name).resolve()
            if not directory.is_relative_to(manifest_path.parent):
                raise ValueError(f"Episode directory is outside the dataset: {directory}")
            if len(row["frame_sha256"]) != row["observations"]:
                raise ValueError(f"Invalid frame checksums for {name}")
            episodes.append(EpisodeSlice(name, row["seed"], split, directory, row["frame_sha256"]))

    train_seeds = {episode.seed for episode in episodes if episode.split == "train"}
    val_seeds = {episode.seed for episode in episodes if episode.split == "validation"}
    if train_seeds & val_seeds:
        raise ValueError("Train and validation reset seeds must be disjoint")
    return manifest_path, episodes


def read_observation(path, digest, camera, selected):
    """Load one raw frame and read its observed TCP only for graph samples."""
    if file_sha256(path) != digest:
        raise ValueError(f"Frame checksum mismatch: {path}")
    sensor, gt = read_frame(path, camera)
    inputs = (
        sensor["rgb"], sensor["depth"].squeeze(-1) if sensor["depth"].ndim == 3 else sensor["depth"],
        sensor["intrinsic_cv"], sensor["extrinsic_cv"],
    )
    if not selected:
        return inputs, None, None
    with np.load(path, allow_pickle=False) as archive:
        tcp = archive["observation/extra/tcp_pose"][0, :3]
    if not np.allclose(tcp, gt["ee_pose"][0, :3], rtol=0, atol=1e-6):
        raise ValueError(f"Observed TCP and Object21 target origin disagree: {path}")
    return inputs, tcp, object21_from_gt(gt)[0].numpy()


class SurfaceReplay:
    """Run the main corner detector and causal TAPNext++ on Peg and Box masks."""

    def __init__(self, detector, tracker, config):
        from evaluation.tapnextpp_surface_tracking import resolve_detector_input_mode

        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.input_mode = resolve_detector_input_mode("auto", detector.device)

    def frames(self, episode, camera, stride):
        """Replay every source frame and build graphs at the configured stride."""
        from evaluation.tapnextpp_surface_tracking import detector_inputs

        states = {kind: None for kind in (NodeType.PEG, NodeType.BOX)}
        for index, digest in enumerate(episode.frame_hashes):
            path = episode.directory / f"frame_{index:06d}.npz"
            selected = index % stride == 0
            inputs, tcp, target = read_observation(path, digest, camera, selected)
            detection = self.detector(*detector_inputs(inputs, self.detector.device, self.input_mode))
            bgr = cv2.cvtColor(inputs[0], cv2.COLOR_RGB2BGR)
            masks = detection.masks.cpu().numpy().astype(bool)
            tracks = [self.advance(bgr, inputs, masks[kind], kind, states)
                      for kind in (NodeType.PEG, NodeType.BOX)]
            if selected:
                graph, triangulation_ms = build_frame(detection, tracks, tcp, target)
                yield index, graph, triangulation_ms

    def advance(self, bgr, inputs, mask, kind, states):
        """Initialize at the first usable mask and lift visible, depth-backed tracks."""
        from evaluation.tapnextpp_surface_tracking import choose_points, lift, track_surface_frame

        _, depth, intrinsic, extrinsic = inputs
        query = None
        if states[kind] is None:
            # The existing sampler requires this many valid interior pixels.
            try:
                query = choose_points(mask, depth, self.config["surface_queries_per_object"])
            except ValueError:
                return np.empty((0, 2)), np.empty((0, 3)), np.empty(0, dtype=np.int64)

        xy, visible, states[kind] = track_surface_frame(
            self.tracker, self.config, bgr, query=query, state=states[kind], observation_mask=mask,
        )
        ids, camera_xyz = lift(xy, depth, mask, intrinsic, visible)
        rotation = extrinsic[:, :3]
        world_xyz = (camera_xyz - extrinsic[:, 3]) @ rotation
        return xy[ids], world_xyz, ids


def build_frame(detection, tracks, tcp, target):
    """Merge the observed nodes and time construction of the undirected edges."""
    corner_uv = detection.uv.cpu().numpy()
    corner_xyz = detection.xyz.cpu().numpy()
    corner_kinds = detection.object_id.cpu().numpy()
    uv = np.concatenate([corner_uv, *(track[0] for track in tracks)]).astype(np.float32)
    xyz = np.concatenate([corner_xyz, *(track[1] for track in tracks)]).astype(np.float32)
    kinds = np.concatenate([
        corner_kinds, *(np.full(len(track[0]), kind) for kind, track in enumerate(tracks)),
    ]).astype(np.int64)
    track_id = np.concatenate([np.full(len(corner_uv), -1), *(track[2] for track in tracks)])
    if not (np.isfinite(uv).all() and np.isfinite(xyz).all()):
        raise ValueError("Graph nodes must have finite pixel and world coordinates")

    started = perf_counter()
    edges = delaunay_edges(uv)
    triangulation_ms = (perf_counter() - started) * 1000
    return SurfaceGraph(uv, xyz, kinds, track_id == -1, track_id, edges, tcp, target), triangulation_ms


def save_graphs(path, samples):
    """Store unpadded nodes and frame-local undirected edges as ragged arrays."""
    indices, graphs, timings = zip(*samples)
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(
            stream,
            frame_indices=np.asarray(indices),
            node_offsets=np.r_[0, np.cumsum([len(graph.xyz) for graph in graphs])],
            edge_offsets=np.r_[0, np.cumsum([len(graph.edges) for graph in graphs])],
            uv=np.concatenate([graph.uv for graph in graphs]),
            xyz_world=np.concatenate([graph.xyz for graph in graphs]),
            object_type=np.concatenate([graph.kinds for graph in graphs]),
            is_corner=np.concatenate([graph.is_corner for graph in graphs]),
            track_id=np.concatenate([graph.track_id for graph in graphs]),
            edges=np.concatenate([graph.edges for graph in graphs]),
            tcp_xyz_world=np.stack([graph.tcp for graph in graphs]),
            object21=np.stack([graph.target for graph in graphs]),
            triangulation_ms=np.asarray(timings),
        )
    path.with_suffix(".partial").replace(path)


def load_graphs(manifest):
    """Load verified graph archives without running vision inference again."""
    groups = {"train": [], "validation": []}
    seeds = {split: set() for split in groups}
    for receipt in json.loads(manifest.read_text())["episodes"]:
        split = receipt["split"]
        seeds[split].add(receipt["seed"])
        path = manifest.parent / receipt["file"]
        if file_sha256(path) != receipt["sha256"]:
            raise ValueError(f"Graph archive checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            frames = {key: archive[key] for key in archive.files}
        for index in range(len(frames["frame_indices"])):
            nodes = slice(*frames["node_offsets"][index:index + 2])
            edges = slice(*frames["edge_offsets"][index:index + 2])
            groups[split].append(SurfaceGraph(
                frames["uv"][nodes], frames["xyz_world"][nodes], frames["object_type"][nodes],
                frames["is_corner"][nodes], frames["track_id"][nodes], frames["edges"][edges],
                frames["tcp_xyz_world"][index], frames["object21"][index],
            ))
    if not all(groups.values()) or seeds["train"] & seeds["validation"]:
        raise ValueError("Graph manifest requires nonempty, seed-disjoint train and validation splits")
    return groups
