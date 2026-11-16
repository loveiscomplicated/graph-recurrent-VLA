"""Extend a corner cache with paired Recurrent states from the same observations."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.object21_bc import predict_graphs, write_npz
from data_collection.storage import write_json
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from recurrent.features import FEATURE_LAYOUT, build_features, canon_quat
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger


GT_FIELDS = {
    "tcp_pose": "ee_pose",
    "left_tip": "left_fingertip_pos",
    "right_tip": "right_fingertip_pos",
    "gripper": "gripper_qpos",
    "peg_pose": "peg_pose",
    "hole_pose": "hole_pose",
    "peg_half": "peg_half_sizes",
    "hole_radius": "hole_radius",
}


def load_models(config, device):
    """Load the frozen Recurrent baseline Transformer and its paired GNN."""
    for name in ("gnn", "temporal"):
        if file_sha256(config[name + "_checkpoint"]) != config[name + "_sha256"]:
            raise ValueError(f"{name} checkpoint SHA-256 mismatch")
    checkpoint = torch.load(config["gnn_checkpoint"], map_location="cpu", weights_only=False)
    settings = deepcopy(checkpoint["config"])
    if (settings["seed"] != config["model_seed"]
            or settings["temporal"]["architecture"] != "transformer"
            or checkpoint["identity"]["threshold_mm"] != 30
            or settings["temporal"]["sha256"] != config["temporal_sha256"]):
        raise ValueError("Expected the configured Transformer pair and its 30 mm association gate")
    settings["temporal"]["checkpoint"] = config["temporal_checkpoint"]
    temporal, _ = load_temporal(settings, device)
    gnn = VertexEdgeGatedMPNN(**settings["gnn"]).to(device).eval().requires_grad_(False)
    gnn.load_state_dict(checkpoint["model"], strict=True)
    return temporal, gnn


def read_frame(path, digest):
    """Read observed Robot9 for Graph and independent raw GT for State36."""
    if file_sha256(path) != digest:
        raise ValueError(f"Raw frame checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as frame:
        robot = np.concatenate((frame["observation/extra/tcp_pose"][0],
                                frame["observation/agent/qpos"][0, -2:])).astype(np.float32)
        gt = {name: frame["ground_truth/" + key][0] for name, key in GT_FIELDS.items()}
        gt["hole_radius"] = np.asarray(gt["hole_radius"]).reshape(1)
        action = frame["action"]
        valid = bool(frame["action_valid"])
    return robot, gt, action, valid


@torch.no_grad()
def augment_episode(episode, source, corner_path, parent_hash, temporal, gnn, device):
    """Add both State36 rows for each original frame, including the terminal frame."""
    if file_sha256(corner_path) != parent_hash:
        raise ValueError(f"Corner cache SHA-256 mismatch: {corner_path}")
    with np.load(corner_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    count = episode["observations"]
    if not np.array_equal(arrays["frame_indices"], np.arange(count)):
        raise ValueError(f"Corner cache frame order differs: {corner_path}")
    offsets = arrays["corner_offsets"]
    points = arrays["corner_world_xyz"]
    kinds = arrays["corner_object_id"]
    if (offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(points)
            or len(points) != len(kinds) or np.any(np.diff(offsets) < 0)):
        raise ValueError(f"Invalid per-frame corner offsets: {corner_path}")

    tracker = DetectedCornerTracker(temporal, 0.030, device)
    frames, states, robot_positions, counts, actions, valid_actions = [], [], [], [], [], []
    for index, digest in enumerate(episode["frame_sha256"]):
        path = source / episode["path"] / f"frame_{index:06d}.npz"
        robot, gt, action, valid = read_frame(path, digest)
        state = build_features(gt, np.float32(1), np.float32(0))[:36]
        np.testing.assert_allclose(
            state[np.r_[0:7, 13:15]],
            np.concatenate((robot[:3], canon_quat(robot[3:7]), robot[-2:])),
            rtol=0, atol=1e-6, err_msg=f"Observed robot and GT are misaligned at {path}",
        )

        start, stop = offsets[index:index + 2]
        tracked = tracker.step(points[start:stop], kinds[start:stop], robot)
        labels = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
        observed = tracked["observed"]
        frames.append((tracked["positions"], labels, observed))
        counts.append(np.bincount(labels[observed], minlength=2))
        states.append(state)
        robot_positions.append(robot[:3])
        actions.append(action)
        valid_actions.append(valid)

    gt = np.stack(states)
    # The Recurrent baseline evaluates the GNN on one frame at a time.
    estimate = predict_graphs(gnn, frames, np.stack(robot_positions), 1, device)
    graph = np.concatenate((gt[:, :15], estimate), axis=1)
    if (gt.shape != (count, 36) or graph.shape != gt.shape or graph.dtype != np.float32
            or not np.isfinite(gt).all() or not np.isfinite(graph).all()):
        raise ValueError(f"Expected finite paired float32 State36 in {corner_path}")
    if valid_actions != [True] * (count - 1) + [False]:
        raise ValueError(f"Expected one terminal observation without action in {corner_path}")
    arrays.update(state36_gt=gt, state36_transformer=graph,
                  counts=np.asarray(counts, dtype=np.int16),
                  graph_has_nodes=np.asarray([len(frame[0]) != 0 for frame in frames]),
                  action=np.stack(actions), action_valid=np.asarray(valid_actions))
    return arrays


def build(config):
    """Build a new archive per episode and record the source and model identities."""
    torch.set_num_threads(config.get("threads", 2))
    device = torch.device(config["device"])
    source_path = Path(config["source_manifest"])
    parent_path = Path(config["corner_manifest"])
    source_hash = file_sha256(source_path)
    parent_hash = file_sha256(parent_path)
    source = json.loads(source_path.read_text())
    parent = json.loads(parent_path.read_text())
    if parent["source_manifest_sha256"] != source_hash:
        raise ValueError("Source and corner manifests must describe the same recordings")
    episodes = {row["path"]: row for row in source["episodes"]}
    rows = parent["episodes"]
    selected = config.get("episodes")
    if selected is None:
        selected = list(episodes)
    if set(selected) - {row["source"] for row in rows} or set(selected) - episodes.keys():
        raise ValueError("The corner cache is missing selected raw episodes")
    temporal, gnn = load_models(config, device)

    with ExperimentLogger(team="integration", model="tapnextpp_state36_cache", config=config,
                          job_type="data_preparation") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        provenance = dict(source_manifest_sha256=source_hash, corner_manifest_sha256=parent_hash,
                          gnn_sha256=config["gnn_sha256"], temporal_sha256=config["temporal_sha256"])
        logger.record(**provenance, torch_version=str(torch.__version__))
        print(f"Output directory: {output}", flush=True)
        records = []
        for row in rows:
            if row["source"] not in selected:
                continue
            episode = episodes[row["source"]]
            if (row["seed"], row["mode"], row["frames"]) != (
                episode["seed"], episode["mode"], episode["observations"]
            ):
                raise ValueError(f"Corner receipt metadata differs for {row['source']}")
            arrays = augment_episode(episode, source_path.parent, parent_path.parent / row["file"],
                                     row["sha256"], temporal, gnn, device)
            path = output / row["file"]
            write_npz(path, **arrays)
            records.append({**row, "source_corner_sha256": row["sha256"], "sha256": file_sha256(path),
                            "action_rows": int(arrays["action_valid"].sum()),
                            "graph_without_nodes": int((~arrays["graph_has_nodes"]).sum())})
            print(f"{len(records)}/{len(selected)} {row['source']} paired_frames={row['frames']}", flush=True)

        if len(records) != len(selected):
            raise ValueError("Output episode count must equal selected episode count")
        manifest = output / "manifest.json"
        write_json(manifest, dict(**provenance, state36_layout=FEATURE_LAYOUT[:-2], episodes=records,
                                  paired_frames=sum(row["frames"] for row in records),
                                  action_rows=sum(row["action_rows"] for row in records)))
        logger.link_output(manifest)
