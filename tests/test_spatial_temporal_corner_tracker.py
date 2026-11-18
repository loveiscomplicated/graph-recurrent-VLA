"""Contracts for the B2-3 spatial-context tracker and complete-graph handoff."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile

import numpy as np
import pytest
import torch

from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.state_estimation.graph import Frame, Node, Source, batch_graphs, build_graph
from graph.temporal.spatial_temporal_corner_tracker import (
    ContextTrack,
    OnlineSpatialTemporalTracker,
    SpatialTemporalCornerTracker,
    load_tracker_checkpoint,
)
from data_collection.spatial_temporal_corner_tracking import (
    _pack_episode_arrays,
    _read_raw_pose_sequences,
)
from data_collection.spatial_corner_tracking import VERTICES
from data_collection.state21_vertex_targets import corners_from_pose
from training.gnn.spatial_temporal_corner_data import (
    CornerTrackingDataset,
    _tracking_episode,
    collate_corner_tracking,
)
from training.spatial_temporal_corner_tracker import association_objective, tracker_objective
from data_collection.spatial_temporal_corner_audit import audit_episode
from data_collection.spatial_temporal_corner_transfer import reconstruct_episode
from evaluation.spatial_temporal_corner_tracking import evaluate_online
from training.spatial_temporal_object21 import (
    build_rollout_cache,
    complete_track_graph,
    run_gnn_epoch,
)


def prepared_episode_arrays():
    """Make three frames with one hidden corner and non-corner surface context."""
    frame_nodes = (
        [
            ((0.00, 0.00, 0.50), 0, 0, True),
            ((0.02, 0.00, 0.50), 0, 1, True),
            ((0.01, 0.01, 0.50), 0, -1, False),
            ((0.20, 0.00, 0.50), 1, 8, True),
        ],
        [
            ((0.001, 0.00, 0.50), 0, 0, True),
            ((0.011, 0.01, 0.50), 0, -1, False),
            ((0.201, 0.00, 0.50), 1, 8, True),
        ],
        [
            ((0.002, 0.00, 0.50), 0, 0, True),
            ((0.022, 0.00, 0.50), 0, 1, True),
            ((0.012, 0.01, 0.50), 0, -1, False),
            ((0.202, 0.00, 0.50), 1, 8, True),
        ],
    )
    offsets = [0]
    xyz, kinds, vertex_ids, is_corner = [], [], [], []
    for nodes in frame_nodes:
        for point, kind, vertex, corner in nodes:
            xyz.append(point)
            kinds.append(kind)
            vertex_ids.append(vertex)
            is_corner.append(corner)
        offsets.append(len(xyz))
    targets = np.zeros((3, 24, 3), dtype=np.float32)
    targets[:, :, 2] = 0.5
    targets[:, 0, 0] = [0.0, 0.001, 0.002]
    targets[:, 1, 0] = [0.02, 0.021, 0.022]
    targets[:, 8, 0] = [0.20, 0.201, 0.202]
    arrays = {
        "offsets": np.asarray(offsets),
        "xyz": np.asarray(xyz, dtype=np.float32),
        "kinds": np.asarray(kinds, dtype=np.int64),
        "vertex_ids": np.asarray(vertex_ids, dtype=np.int64),
        "is_corner": np.asarray(is_corner, dtype=bool),
        "tcp": np.zeros((3, 3), dtype=np.float32),
        "targets": targets,
    }
    return arrays


def prepared_episode():
    """Convert the fixture observations to chronological graphs and labels."""
    return _tracking_episode(prepared_episode_arrays(), "fixture", 0.08)


def test_graph_cache_reuses_graphs_but_checks_source_bytes(tmp_path, monkeypatch):
    episode_path = tmp_path / "episode.npz"
    np.savez_compressed(episode_path, **prepared_episode_arrays())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{
        "episode": "fixture", "file": "episode.npz", "frames": 3,
        "split": "train", "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
    }]}))
    dataset = CornerTrackingDataset(manifest, "train", graph_cache_root=tmp_path / "graphs")
    first = dataset[0]
    assert len(list((tmp_path / "graphs").glob("*.pt"))) == 1
    def fail_rebuild(*args):
        raise AssertionError("Cache hit must not rebuild episode graphs")
    monkeypatch.setattr("training.gnn.spatial_temporal_corner_data._tracking_episode", fail_rebuild)
    second = dataset[0]
    np.testing.assert_array_equal(first.targets, second.targets)
    for left, right in zip(first.graphs, second.graphs):
        torch.testing.assert_close(left.features, right.features, rtol=0, atol=0)
        torch.testing.assert_close(left.edge_index, right.edge_index, rtol=0, atol=0)
    episode_path.write_bytes(b"corrupt source")
    with pytest.raises(ValueError, match="checksum/path mismatch"):
        dataset[0]


def test_full_release_preparation_keeps_detector_rows_and_labels_legacy_surfaces():
    frames = 2
    pose = np.asarray(
        [
            0.00, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.20, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.20, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.02, 0.01, 0.01,
            0.015,
        ],
        dtype=np.float32,
    )
    references = corners_from_pose(pose)
    observed_vertices = VERTICES[:2]
    corner_xyz = np.asarray(
        [references[vertex] for _ in range(frames) for vertex in observed_vertices],
        dtype=np.float32,
    )
    frame_indices = np.arange(frames)
    tables = {
        "corners": {
            "frame_indices": frame_indices,
            "corner_offsets": np.asarray([0, 2, 4]),
            "corner_world_xyz": corner_xyz,
            "corner_object_id": np.zeros(4, dtype=np.int8),
        },
        "surface": {
            "frame_indices": frame_indices,
            "surface_offsets": np.asarray([0, 1, 2]),
            "surface_world_xyz": np.asarray(
                [[0.01, 0.00, 0.50], [0.011, 0.00, 0.50]], dtype=np.float32
            ),
        },
        "targets": {
            "frame_indices": frame_indices,
            "tcp": np.zeros((frames, 3), dtype=np.float32),
        },
    }
    record = {
        "seed": 1,
        "policy_mode": "Fixed_Hole_expert",
        "frames": frames,
    }
    arrays, surface_mode = _pack_episode_arrays(
        record,
        tables,
        np.repeat(pose[None], frames, axis=0),
        gate_m=0.005,
        margin_m=0.002,
    )
    assert surface_mode == "historical_peg_only"
    np.testing.assert_array_equal(arrays["offsets"], [0, 3, 6])
    np.testing.assert_array_equal(arrays["vertex_ids"], [0, 1, -1, 0, 1, -1])
    assert arrays["is_corner"].tolist() == [True, True, False] * frames
    assert arrays["targets"].shape == (frames, 24, 3)
    restored = reconstruct_episode(tables, arrays)
    for name, value in arrays.items():
        np.testing.assert_array_equal(restored[name], value)
    original_buffer, restored_buffer = io.BytesIO(), io.BytesIO()
    np.savez_compressed(original_buffer, **arrays)
    np.savez_compressed(restored_buffer, **restored)
    assert original_buffer.getvalue() == restored_buffer.getvalue()
    result = audit_episode(arrays, record, tables, {"gate_m": 0.005, "margin_m": 0.002})
    assert result == {"frames": 2, "detector_rows": 4, "surface_rows": 2, "certified_rows": 4}
    arrays["xyz"][0, 0] += 0.001
    with pytest.raises(AssertionError):
        audit_episode(arrays, record, tables, {"gate_m": 0.005, "margin_m": 0.002})


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is required by raw archives")
@pytest.mark.parametrize("index_first", [True, False])
def test_raw_pose_reader_streams_episode_frames_in_archive_order(tmp_path, index_first):
    pose = np.asarray(
        [
            0.00, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.20, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.20, 0.00, 0.50, 1.0, 0.0, 0.0, 0.0,
            0.02, 0.01, 0.01,
            0.015,
        ],
        dtype=np.float32,
    )
    source_tar = tmp_path / "raw.tar"
    episode_path = "recovery/pilot_1_expert"
    index = [
        {
            "seed": 1,
            "policy_mode": "Fixed_Hole_expert",
            "split": "train",
            "observations": 2,
            "path": episode_path,
        }
    ]
    with tarfile.open(source_tar, "w") as archive:
        content = json.dumps(index).encode()
        info = tarfile.TarInfo("./episode_index.json")
        info.size = len(content)
        index_content = content
        index_info = info
        if index_first:
            archive.addfile(index_info, io.BytesIO(index_content))
        for frame_index in (1, 0):
            buffer = io.BytesIO()
            start = 0
            fields = {}
            for name, width in (
                ("peg_pose", 7),
                ("box_pose", 7),
                ("hole_pose", 7),
                ("peg_half_sizes", 3),
                ("hole_radius", 1),
            ):
                fields[f"ground_truth/{name}"] = pose[start : start + width][None]
                start += width
            np.savez(buffer, **fields)
            content = buffer.getvalue()
            info = tarfile.TarInfo(f"{episode_path}/frame_{frame_index:06d}.npz")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if not index_first:
            archive.addfile(index_info, io.BytesIO(index_content))
    compressed = tmp_path / "episodes_00000_00002.tar.zst"
    subprocess.run(
        ["zstd", "-q", "-f", str(source_tar), "-o", str(compressed)], check=True
    )
    digest = hashlib.sha256(compressed.read_bytes()).hexdigest()
    record = {
        "seed": 1,
        "policy_mode": "Fixed_Hole_expert",
        "split": "train",
        "frames": 2,
    }
    result = _read_raw_pose_sequences(compressed, digest, [record])
    np.testing.assert_array_equal(result[(1, "Fixed_Hole_expert")], [pose, pose])


def test_spatial_transformer_shapes_loss_and_surface_gradient():
    batch = collate_corner_tracking([prepared_episode()])
    model = SpatialTemporalCornerTracker(
        hidden=16,
        spatial_layers=1,
        temporal_layers=1,
        heads=4,
        association_dim=8,
    )
    output = model(
        batch.graph,
        batch.rows,
        batch.frame_mask,
        batch.last_observed,
        batch.age,
    )
    assert output["position"].shape == (1, 3, 24, 3)
    assert output["track_embedding"].shape == (1, 3, 24, 8)
    assert output["detection_embedding"].shape[:3] == (1, 3, batch.graph.features.shape[1])
    assert torch.isfinite(output["position"]).all()
    losses = tracker_objective(
        model,
        output,
        batch,
        {
            "association_temperature": 0.1,
            "association_weight": 0.25,
            "rigidity_weight": 0.25,
            "uncertainty_weight": 0.05,
        },
    )
    losses["total"].backward()
    assert model.spatial.node_encoder[0].weight.grad.abs().sum() > 0
    assert model.temporal_blocks[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert batch.seen_before[0, 1, 1] and not batch.visible[0, 1, 1]
    assert batch.reappearance[0, 2, 1]


def test_batched_association_matches_objectwise_reference_loss_and_gradients():
    batch = collate_corner_tracking([prepared_episode(), prepared_episode()])
    generator = torch.Generator().manual_seed(7)
    predicted = torch.randn((2, 3, 24, 8), generator=generator, dtype=torch.float64, requires_grad=True)
    detected = torch.randn((2, 3, 4, 8), generator=generator, dtype=torch.float64, requires_grad=True)
    output = {"track_embedding": predicted, "detection_embedding": detected}
    actual, actual_correct, actual_count = association_objective(output, batch, 0.1)
    expected = predicted.new_zeros(())
    count, correct = 0, 0
    kinds = torch.tensor([0] * 8 + [1] * 16)
    for sample in range(2):
        for frame in range(3):
            for kind in (0, 1):
                valid = batch.visible[sample, frame] & batch.seen_before[sample, frame]
                slots = torch.nonzero(valid & (kinds == kind), as_tuple=True)[0]
                if not len(slots):
                    continue
                rows = batch.rows[sample, frame, slots]
                logits = predicted[sample, frame, slots] @ detected[sample, frame, rows].T / 0.1
                labels = torch.arange(len(slots))
                expected = expected + torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                count += len(slots)
                correct += int((logits.argmax(-1) == labels).sum())
    expected = expected / count
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert (actual_correct, actual_count) == (correct, count)
    actual_gradient = torch.autograd.grad(actual, (predicted, detected), retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, (predicted, detected))
    for actual_value, expected_value in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-12, atol=1e-12)
    batch.seen_before.zero_()
    empty, correct, count = association_objective(output, batch, 0.1)
    assert torch.isfinite(empty) and empty == 0 and correct == count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_episode_batch_moves_frame_graph_and_labels_to_cuda():
    batch = collate_corner_tracking([prepared_episode()]).to("cuda:0")
    assert batch.graph.features.device == batch.rows.device == torch.device("cuda:0")
    assert batch.graph.edge_index.device == batch.targets.device
    model = SpatialTemporalCornerTracker(
        hidden=16, spatial_layers=1, temporal_layers=1, heads=4, association_dim=8
    ).to("cuda:0")
    output = model(batch.graph, batch.rows, batch.frame_mask, batch.last_observed, batch.age)
    tracker_objective(model, output, batch, {})["total"].backward()
    assert model.displacement.weight.grad is not None


def observed_graph(offset=0.0):
    nodes = (
        Node((0.00 + offset, 0.00, 0.50), 0, True, None, True, None, Source.OBSERVED),
        Node((0.02 + offset, 0.00, 0.50), 0, True, None, True, None, Source.OBSERVED),
        Node((0.01 + offset, 0.01, 0.50), 0, False, None, False, None, Source.TRACKER),
        Node((0.20, 0.00, 0.50), 1, False, None, False, None, Source.TRACKER),
    )
    return batch_graphs([build_graph(Frame("online", int(offset > 0), (0, 0, 0), nodes))])


def test_online_tracker_keeps_ids_with_contextual_assignment():
    model = SpatialTemporalCornerTracker(
        hidden=16,
        spatial_layers=1,
        temporal_layers=1,
        heads=4,
        association_dim=8,
    ).eval()
    tracker = OnlineSpatialTemporalTracker(
        model, distance_limit_m=0.05, embedding_weight_m=0.0
    )
    first = tracker.step(observed_graph())
    second = tracker.step(observed_graph(0.001))
    assert first["new_tracks"] == 2
    assert second["matches"] == 2
    assert second["new_tracks"] == 0
    torch.testing.assert_close(second["detection_tracks"], torch.tensor([0, 1]))
    assert second["observed"].all()


def test_online_metrics_distinguish_forecasts_from_observed_passthrough():
    model = SpatialTemporalCornerTracker(
        hidden=16, spatial_layers=1, temporal_layers=1, heads=4, association_dim=8
    ).eval()
    metrics = evaluate_online(model, [prepared_episode()], {
        "distance_limit_m": 0.03, "embedding_weight_m": 0.0, "log_every_episodes": 0,
    }, torch.device("cpu"))
    assert metrics["association_accuracy"] == 1
    assert metrics["reappearance_same_id_rate"] == 1
    assert metrics["missing_rmse_mm"] == pytest.approx(1.0, abs=0.001)
    assert metrics["reappearance_rmse_mm"] == pytest.approx(2.0, abs=0.001)
    assert metrics["max_final_tracks"] == 3


def test_incremental_attention_matches_full_history_for_late_tracks():
    torch.manual_seed(11)
    model = SpatialTemporalCornerTracker(
        hidden=16, spatial_layers=1, temporal_layers=2, heads=4, association_dim=8
    ).eval()
    tracker = OnlineSpatialTemporalTracker(model)
    tracks = []
    with torch.inference_mode():
        for frame in range(5):
            for track in tracks:
                track.tokens.append(torch.randn(16))
            if frame in (0, 2):
                tracks.append(ContextTrack(
                    track_id=len(tracks), kind=0, last_observed=torch.zeros(3),
                    previous_hidden=torch.zeros(16), tokens=[torch.randn(16)],
                ))
            actual = tracker._advance_temporal(tracks)
            for index, track in enumerate(tracks):
                tokens = torch.stack(track.tokens)[None]
                valid = torch.ones(tokens.shape[:2], dtype=torch.bool)
                expected = model.temporal_forward(tokens, valid)[0, -1]
                torch.testing.assert_close(actual[index], expected, rtol=1e-5, atol=1e-6)
                assert all(len(history) == len(track.tokens) for history in track.layer_history)


@pytest.mark.parametrize("coordinate_frame", ["world", "tcp_relative"])
def test_world_context_retains_joint_peg_and_tcp_motion(coordinate_frame):
    points = [(0.125, 0.125, 0.5), (0.15625, 0.125, 0.5), (0.140625, 0.140625, 0.5)]
    graphs = []
    for shift in ((0, 0, 0), (0.125, 0.0625, 0)):
        nodes = tuple(
            Node(
                tuple(value + delta for value, delta in zip(point, shift)),
                0, index < 2, None, index < 2, None,
                Source.OBSERVED if index < 2 else Source.TRACKER,
            )
            for index, point in enumerate(points)
        )
        tcp = tuple(value + delta for value, delta in zip((0, 0, 0.5), shift))
        graphs.append(build_graph(Frame("translation", 0, tcp, nodes)))
    torch.testing.assert_close(graphs[0].features, graphs[1].features, rtol=0, atol=0)
    torch.manual_seed(0)
    model = SpatialTemporalCornerTracker(
        hidden=16, spatial_layers=1, temporal_layers=1, heads=4,
        association_dim=8, coordinate_frame=coordinate_frame,
    ).eval()
    with torch.no_grad():
        _, first, _ = model.encode_spatial(batch_graphs([graphs[0]]))
        _, moved, _ = model.encode_spatial(batch_graphs([graphs[1]]))
    if coordinate_frame == "world":
        assert (first[:, 0] - moved[:, 0]).abs().max() > 1e-4
    else:
        torch.testing.assert_close(first, moved, rtol=0, atol=0)


@pytest.mark.parametrize("stored_frame", [None, "world"])
def test_checkpoint_load_preserves_coordinate_interpretation(tmp_path, stored_frame):
    settings = {"hidden": 16, "spatial_layers": 1, "temporal_layers": 1,
                "heads": 4, "association_dim": 8}
    if stored_frame is not None:
        settings["coordinate_frame"] = stored_frame
    model = SpatialTemporalCornerTracker(**settings)
    path = tmp_path / "tracker.pt"
    torch.save({"format": "spatial_temporal_corner_tracker_v1",
                "config": {"model": settings}, "model": model.state_dict()}, path)
    loaded, _ = load_tracker_checkpoint(path, None, torch.device("cpu"))
    assert loaded.coordinate_frame == (stored_frame or "tcp_relative")
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_complete_graph_handoff_has_all_cross_object_pairs_and_reliability():
    xyz = torch.tensor([[[0.0, 0.0, 0.5], [0.02, 0.0, 0.5], [0.2, 0.0, 0.5]]])
    kinds = torch.tensor([[0, 0, 1]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    observed = torch.tensor([[True, False, True]])
    uncertainty = torch.tensor([[0.0, 0.01, 0.0]])
    age = torch.tensor([[0, 5, 0]])
    graph, extra = complete_track_graph(
        xyz,
        kinds,
        valid,
        observed,
        uncertainty,
        age,
        torch.zeros((1, 3)),
        {"uncertainty_scale_m": 0.01, "age_scale_frames": 10},
    )
    assert graph.edge_index.shape[1] == 3 * 2
    assert extra.shape == (1, 3, 3)
    assert extra[0, 0, 0] == 1 and extra[0, 1, 0] == 0
    source, target = graph.edge_index
    assert ((graph.object_type.flatten()[source] != graph.object_type.flatten()[target])).any()


def write_five_mode_release(tmp_path):
    """Create one train and one validation episode with both surface objects."""
    archive_name = "current_five_mode.tar"
    records = []
    with tarfile.open(tmp_path / archive_name, "w") as archive:
        for seed, split in ((1, "train"), (2, "validation")):
            frames = 2
            target = np.zeros((frames, 21), dtype=np.float32)
            target[:, 3] = 1
            target[:, 13] = 1
            target[:, 17:21] = np.asarray([0.04, 0.01, 0.01, 0.02])
            tables = {
                "corners": {
                    "corner_offsets": np.asarray([0, 2, 4]),
                    "corner_world_xyz": np.asarray(
                        [
                            [0.00, 0.00, 0.50],
                            [0.20, 0.00, 0.50],
                            [0.001, 0.00, 0.50],
                            [0.201, 0.00, 0.50],
                        ],
                        dtype=np.float32,
                    ),
                    "corner_object_id": np.asarray([0, 1, 0, 1], dtype=np.int8),
                },
                "surface": {
                    "surface_offsets": np.asarray([0, 2, 4]),
                    "surface_world_xyz": np.asarray(
                        [
                            [0.01, 0.01, 0.50],
                            [0.21, 0.01, 0.50],
                            [0.011, 0.01, 0.50],
                            [0.211, 0.01, 0.50],
                        ],
                        dtype=np.float32,
                    ),
                },
                "targets": {
                    "tcp": np.zeros((frames, 3), dtype=np.float32),
                    "object21": target,
                },
            }
            # Seed 1 emulates the historical Peg-only release (no redundant
            # object IDs); seed 2 exercises a current two-object surface cache.
            if seed == 2:
                tables["surface"]["surface_object_id"] = np.asarray(
                    [0, 1, 0, 1], dtype=np.int8
                )
            files = {}
            for kind, values in tables.items():
                member = f"episode_{seed}/{kind}/{kind}.npz"
                buffer = io.BytesIO()
                np.savez(buffer, frame_indices=np.arange(frames), **values)
                content = buffer.getvalue()
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
                files[kind] = {
                    "archive": archive_name,
                    "member": member,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            records.append(
                {
                    "seed": seed,
                    "policy_mode": "Fixed_Hole_expert",
                    "split": split,
                    "frames": frames,
                    "files": files,
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "five_mode_supervised_v1",
                "episodes": 2,
                "archives": [{"file": archive_name}],
                "records": records,
            }
        )
    )
    return manifest


def test_frozen_rollout_and_downstream_gnn_smoke(tmp_path):
    manifest = write_five_mode_release(tmp_path)
    tracker = SpatialTemporalCornerTracker(
        hidden=16,
        spatial_layers=1,
        temporal_layers=1,
        heads=4,
        association_dim=8,
    ).eval().requires_grad_(False)
    config = {
        "data": {"manifest": str(manifest), "max_edge_m": 0.08},
        "tracker": {"checkpoint": str(tmp_path / "tracker.pt")},
        "association": {"distance_limit_m": 0.05, "embedding_weight_m": 0.0},
        "training": {"rollout_log_every_episodes": 1, "frame_batch_size": 2},
        "downstream_features": {"uncertainty_scale_m": 0.01, "age_scale_frames": 10},
        "loss": {
            "pos_unit_m": 0.01,
            "consistency_weight": 0.05,
            "size_units_m": [0.04, 0.01, 0.01, 0.02],
        },
    }
    torch.save({"fixture": True}, config["tracker"]["checkpoint"])
    arrays, identity, cache_path = build_rollout_cache(
        config,
        tracker,
        {"identity": {"dataset_sha256": "fixture"}},
        torch.device("cpu"),
        tmp_path,
    )
    assert cache_path.exists()
    assert identity["train_frames"] == 2 and identity["validation_frames"] == 2
    assert arrays["offsets"][-1] >= 8
    assert identity["rollout_storage"] == "ragged_frame_offsets"
    gnn = VertexEdgeGatedMPNN(
        num_semantic_ids=24,
        hidden=16,
        layers=1,
        use_semantic_ids=False,
        extra_node_dim=3,
    )
    metrics = run_gnn_epoch(
        gnn,
        arrays,
        np.flatnonzero(arrays["validation"]),
        config,
        torch.device("cpu"),
    )
    assert metrics["frames"] == 2 and np.isfinite(metrics["loss"])
