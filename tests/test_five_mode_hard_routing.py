"""Tests for detector-based C1/C2 hard routing and graph construction."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from data_collection.detected_corner_visibility import classify_episode, classify_frame
from data_collection.episode_dataset import file_sha256
from graph.build_graph import NodeType
from graph.five_mode_hard_routing import (
    FiveModeHardRoutingPipeline,
    build_hard_routing_pipeline,
    load_surface_model,
)
from training.gnn.packed_five_mode_data import PackedFrameBatches, pack_episode


class RecordingModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, graph):
        self.calls.append(graph)
        batch_size = graph.xyz_world.shape[0]
        return {"object21": graph.xyz_world.new_full((batch_size, 21), self.value)}


def detection(peg_xyz, rim_count, exterior_count=0):
    peg_xyz = np.asarray(peg_xyz, dtype=np.float32).reshape(-1, 3)
    box_count = rim_count + exterior_count
    box_xyz = np.zeros((box_count, 3), dtype=np.float32)
    xyz = np.concatenate((peg_xyz, box_xyz))
    kinds = np.asarray(
        [NodeType.PEG] * len(peg_xyz) + [NodeType.BOX] * box_count,
        dtype=np.int64,
    )
    metadata = (
        [{"contour_hole": False}] * len(peg_xyz)
        + [{"contour_hole": True}] * rim_count
        + [{"contour_hole": False}] * exterior_count
    )
    return SimpleNamespace(
        xyz=torch.from_numpy(xyz),
        object_id=torch.from_numpy(kinds),
        metadata=metadata,
    )


def pipeline(detector=None):
    models = {
        "condition_1": RecordingModel(1),
        "condition_2": RecordingModel(2),
    }
    return FiveModeHardRoutingPipeline(
        models,
        detector=detector,
        device="cpu",
        coplanarity_tolerance_m=0.001,
    ), models


def test_routes_noncoplanar_frame_to_condition_1_and_appends_surface_points():
    routed, models = pipeline()
    corners = detection(
        [[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.02]],
        rim_count=3,
        exterior_count=1,
    )
    surface = np.asarray([[0.01, 0.01, 0.01], [0.02, 0.01, 0.01]], dtype=np.float32)

    result = routed.predict_detection(corners, surface, np.zeros(3, dtype=np.float32))

    assert result.condition == "condition_1"
    assert result.selected_model == "condition_1"
    assert result.peg_corner_count == 4
    assert result.hole_rim_count == 3
    assert result.peg_surface_count == 2
    assert result.prediction_status == "available"
    assert result.condition_evidence.peg_noncoplanar is True
    assert result.condition_evidence.peg_plane_rms_m > 0.001
    torch.testing.assert_close(result.object21, torch.ones(21))
    assert len(models["condition_1"].calls) == 1
    assert not models["condition_2"].calls
    graph = models["condition_1"].calls[0]
    node_count = len(corners.xyz) + len(surface)
    assert graph.node_valid.sum().item() == node_count
    assert graph.edge_index.shape[1] == node_count * (node_count - 1)
    assert graph.object_type[0, -2:].tolist() == [NodeType.PEG, NodeType.PEG]


def test_routes_coplanar_condition_1_candidate_to_condition_2():
    routed, models = pipeline()
    corners = detection(
        [[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0]],
        rim_count=3,
    )

    result = routed.predict_detection(corners, np.empty((0, 3)), np.zeros(3))

    assert result.condition == "condition_2"
    assert result.selected_model == "condition_2"
    assert result.condition_evidence.peg_noncoplanar is False
    torch.testing.assert_close(result.object21, torch.full((21,), 2.0))
    assert not models["condition_1"].calls
    assert len(models["condition_2"].calls) == 1


def test_condition_3_returns_without_running_an_expert():
    routed, models = pipeline()
    corners = detection([[0, 0, 0]], rim_count=4)

    result = routed.predict_detection(corners, np.empty((0, 3)), np.zeros(3))

    assert result.condition == "condition_3"
    assert result.selected_model is None
    assert result.object21 is None
    assert result.prediction_status == "unavailable"
    assert result.condition_evidence.peg_plane_rms_m is None
    assert not models["condition_1"].calls
    assert not models["condition_2"].calls


@pytest.mark.parametrize("peg,rim_count,condition", [
    ([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.02]], 3, "condition_1"),
    ([[0, 0, 0], [0.03, 0, 0]], 1, "condition_2"),
    ([[0, 0, 0]], 1, "condition_3"),
])
def test_rgbd_path_runs_detector_once_before_routing(peg, rim_count, condition):
    corners = detection(peg, rim_count=rim_count)

    class Detector:
        def __init__(self):
            self.calls = 0

        def __call__(self, rgb, depth, intrinsic, extrinsic, masks=None):
            self.calls += 1
            assert masks == "mask"
            return corners

    detector = Detector()
    routed, models = pipeline(detector)
    result = routed.predict_rgbd(
        "rgb", "depth", "intrinsic", "extrinsic",
        np.empty((0, 3)), np.zeros(3), masks="mask",
    )

    assert detector.calls == 1
    assert result.condition == condition
    for name, model in models.items():
        assert len(model.calls) == int(condition == name)


@pytest.mark.parametrize("peg_count,rim_count,condition", [
    (0, 0, "condition_3"),
    (0, 4, "condition_3"),
    (1, 4, "condition_3"),
    (2, 0, "condition_3"),
    (4, 0, "condition_3"),
    (2, 1, "condition_2"),
    (3, 3, "condition_2"),
])
def test_counts_exclude_box_exterior_and_surface_support(peg_count, rim_count, condition):
    routed, models = pipeline()
    corners = detection(np.zeros((peg_count, 3)), rim_count, exterior_count=8)
    # These noncoplanar Peg tracks must never turn missing detector corners into C1/C2.
    surface = np.asarray([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.02]])

    result = routed.predict_detection(corners, surface, np.zeros(3))

    assert result.condition == condition
    assert result.peg_corner_count == peg_count
    assert result.hole_rim_count == rim_count
    assert not models["condition_1"].calls
    assert len(models["condition_2"].calls) == int(condition == "condition_2")


@pytest.mark.parametrize("peg,rim_count,condition", [
    ([[0, 0, 0], [0.03, 0, 0]], 1, "condition_2"),
    ([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.02]], 3, "condition_1"),
])
def test_graph_matches_packed_training_graph_field_for_field(peg, rim_count, condition):
    routed, models = pipeline()
    corners = detection(peg, rim_count, exterior_count=2)
    surface = np.asarray([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]], dtype=np.float32)
    tcp = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    episode = dict(seed=7, policy_mode="Fixed_Hole_expert", frames=1)
    tables = {
        "corners": dict(
            frame_indices=np.asarray([0]), corner_offsets=np.asarray([0, len(corners.xyz)]),
            corner_world_xyz=corners.xyz.numpy(), corner_object_id=corners.object_id.numpy(),
        ),
        "surface": dict(
            frame_indices=np.asarray([0]), surface_offsets=np.asarray([0, 3]),
            surface_world_xyz=np.concatenate((surface, [[99, 99, 99]])).astype(np.float32),
            surface_object_id=np.asarray([0, 0, 1]),
        ),
        "targets": dict(frame_indices=np.asarray([0]), tcp=tcp[None],
                        object21=np.zeros((1, 21), dtype=np.float32)),
    }
    packed = pack_episode(episode, tables, include_surface=True)
    reference = next(iter(PackedFrameBatches([(episode, packed, [0])], "cpu", 1, None))).graph

    routed.predict_detection(corners, surface, tcp)

    actual = models[condition].calls[0]
    for field in fields(reference):
        expected_value = getattr(reference, field.name)
        actual_value = getattr(actual, field.name)
        if torch.is_tensor(expected_value):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("name,value", [
    ("surface", np.zeros(3)),
    ("surface", np.empty((0,))),
    ("surface", np.zeros((1, 2))),
    ("surface", [[np.nan, 0, 0]]),
    ("surface", [[np.inf, 0, 0]]),
    ("surface", [[1e100, 0, 0]]),
    ("surface", [["x", "y", "z"]]),
    ("tcp", np.zeros((1, 3))),
    ("tcp", [0, np.inf, 0]),
    ("tcp", [0, 1e100, 0]),
])
@pytest.mark.parametrize("rim_count", [0, 1])
def test_invalid_surface_or_tcp_fails_even_on_unavailable_frames(name, value, rim_count):
    routed, models = pipeline()
    corners = detection([[0, 0, 0], [0.03, 0, 0]], rim_count)
    surface = value if name == "surface" else np.empty((0, 3))
    tcp = value if name == "tcp" else np.zeros(3)
    expected_name = "peg_surface_xyz" if name == "surface" else "tcp_xyz"

    with pytest.raises(ValueError, match=expected_name):
        routed.predict_detection(corners, surface, tcp)
    assert not any(model.calls for model in models.values())


@pytest.mark.parametrize("attribute,value,message", [
    ("xyz", np.zeros((3, 2)), "detector corners"),
    ("xyz", [[np.nan, 0, 0]], "detection.xyz"),
    ("object_id", np.zeros((3, 1), dtype=np.int64), "object_id"),
    ("object_id", np.asarray([0.0, 0.0, 1.0]), "detector corners"),
    ("object_id", np.asarray([0, 0, 3]), "detector corners"),
    ("metadata", [], "one row"),
    ("metadata", [None, {}, {}], "mapping"),
    ("metadata", [{}, {}, {"contour_hole": "false"}], "boolean"),
])
def test_invalid_detector_output_has_clear_error(attribute, value, message):
    routed, _ = pipeline()
    corners = detection([[0, 0, 0], [0.03, 0, 0]], rim_count=1)
    setattr(corners, attribute, value)
    with pytest.raises(ValueError, match=message):
        routed.predict_detection(corners, np.empty((0, 3)), np.zeros(3))


@pytest.mark.parametrize("output", [
    {}, {"object21": None}, {"object21": [0] * 21},
    {"object21": torch.zeros(21)}, {"object21": torch.zeros((1, 21), dtype=torch.long)},
    {"object21": torch.full((1, 21), float("nan"))},
])
def test_invalid_model_output_has_clear_error(output):
    class InvalidModel(nn.Module):
        def forward(self, graph):
            return output

    routed, _ = pipeline()
    routed.models["condition_2"] = InvalidModel()
    with pytest.raises(ValueError, match="condition_2 model"):
        routed.predict_detection(
            detection([[0, 0, 0], [0.03, 0, 0]], 1), np.empty((0, 3)), np.zeros(3),
        )


@pytest.mark.parametrize("detector_device,model_device", [
    ("cpu", "cpu"), ("mps", "cpu"), ("cpu", "mps"),
    ("cuda", "cpu"), ("cpu", "cuda"),
])
def test_detector_and_model_devices_can_differ(detector_device, model_device):
    if "mps" in (detector_device, model_device) and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if "cuda" in (detector_device, model_device) and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    corners = detection([[0, 0, 0], [0.03, 0, 0]], 1)
    corners.xyz = corners.xyz.to(detector_device).requires_grad_()
    corners.object_id = corners.object_id.to(detector_device)
    surface = torch.zeros((1, 3), device=detector_device, requires_grad=True)
    models = {
        "condition_1": RecordingModel(1),
        "condition_2": RecordingModel(2),
    }
    routed = FiveModeHardRoutingPipeline(models, device=model_device)

    result = routed.predict_detection(corners, surface, torch.zeros(3, device=detector_device))

    assert result.object21.device.type == model_device
    assert not result.object21.requires_grad
    assert len(models["condition_2"].calls) == 1


def checkpoint_spec(tmp_path, *, condition="condition_1", include_surface=True):
    """Save a tiny state with the production checkpoint's config/model envelope."""
    model = nn.Linear(3, 2)
    checkpoint = {
        "config": {
            "model": {"name": "vertex_edge_gated_mpnn"},
            "data": {
                "factory": "training.gnn.packed_five_mode_data:build_data",
                "condition": condition,
                "include_surface": include_surface,
            },
        },
        "model": model.state_dict(),
    }
    path = tmp_path / "best.pt"
    torch.save(checkpoint, path)
    return dict(path=str(path), sha256=file_sha256(path)), checkpoint


def test_checkpoint_hash_mismatch_prevents_deserialization(tmp_path, monkeypatch):
    settings, _ = checkpoint_spec(tmp_path)
    settings["sha256"] = "0" * 64

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Checkpoint deserialization must follow hash validation")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="checksum"):
        load_surface_model(settings, torch.device("cpu"), "condition_1")


@pytest.mark.parametrize("change,message", [
    ("surface", "Peg-surface"), ("condition", "routed condition_1"),
    ("factory", "packed Five-mode"), ("model", "VertexEdgeGatedMPNN"),
    ("missing_model", "model contract"),
])
def test_checkpoint_contract_is_checked_before_building_model(tmp_path, monkeypatch, change, message):
    settings, checkpoint = checkpoint_spec(tmp_path)
    if change == "surface":
        checkpoint["config"]["data"]["include_surface"] = False
    elif change == "condition":
        checkpoint["config"]["data"]["condition"] = "condition_2"
    elif change == "factory":
        checkpoint["config"]["data"]["factory"] = "other:loader"
    elif change == "model":
        checkpoint["config"]["model"]["name"] = "other"
    else:
        del checkpoint["model"]
    torch.save(checkpoint, settings["path"])
    settings["sha256"] = file_sha256(settings["path"])

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Model construction must follow contract validation")

    monkeypatch.setattr("graph.five_mode_hard_routing.build_model", unexpected_build)
    with pytest.raises(ValueError, match=message):
        load_surface_model(settings, torch.device("cpu"), "condition_1")


def test_checkpoint_loader_restores_weights_and_freezes_eval_model(tmp_path, monkeypatch):
    settings, checkpoint = checkpoint_spec(tmp_path)
    monkeypatch.setattr("graph.five_mode_hard_routing.build_model", lambda config, device: nn.Linear(3, 2))

    model = load_surface_model(settings, torch.device("cpu"), "condition_1")

    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)


def test_builder_preserves_injected_detector_and_routes_both_pinned_models(tmp_path, monkeypatch):
    settings, _ = checkpoint_spec(tmp_path)
    loaded = []

    def load(settings, device, condition):
        loaded.append(condition)
        return RecordingModel(1)

    monkeypatch.setattr("graph.five_mode_hard_routing.load_surface_model", load)
    def detector(*args, **kwargs):
        return detection([[0, 0, 0], [0.03, 0, 0]], 1)

    config = {
        "device": "cpu",
        "checkpoints": {"condition_1": settings, "condition_2": settings},
    }
    routed = build_hard_routing_pipeline(config, detector=detector)
    assert loaded == ["condition_1", "condition_2"]
    assert routed.detector is detector
    assert routed.coplanarity_tolerance_m == 0.001


def test_rgbd_requires_detector():
    routed, _ = pipeline()
    with pytest.raises(RuntimeError, match="requires a detector"):
        routed.predict_rgbd(None, None, None, None, np.empty((0, 3)), np.zeros(3))


@pytest.mark.parametrize("tolerance_m", [0.0001, 0.001, 0.01])
def test_public_frame_classifier_matches_episode_helper(tolerance_m):
    generator = np.random.default_rng(7)
    frames = []
    for peg_count in range(9):
        for rim_count in range(5):
            peg = generator.normal(0, 0.02, size=(peg_count, 3))
            # Include both exactly planar and nonplanar detections.
            if rim_count % 2 == 0:
                peg[:, 2] = 0
            frames.append(detection(peg, rim_count, exterior_count=3))
    counts = [len(frame.xyz) for frame in frames]
    arrays = dict(
        frame_indices=np.arange(len(frames)), corner_offsets=np.r_[0, np.cumsum(counts)],
        corner_world_xyz=np.concatenate([frame.xyz.numpy() for frame in frames]),
        corner_object_id=np.concatenate([frame.object_id.numpy() for frame in frames]),
        corner_contour_hole=np.asarray([
            row["contour_hole"] for frame in frames for row in frame.metadata
        ]),
    )
    expected = {condition: [] for condition in ("condition_1", "condition_2", "condition_3")}
    for index, frame in enumerate(frames):
        condition = classify_frame(
            frame.xyz.numpy(), frame.object_id.numpy(),
            np.asarray([row["contour_hole"] for row in frame.metadata]), tolerance_m=tolerance_m,
        )
        expected[condition].append(index)
    assert classify_episode(arrays, len(frames), tolerance_m=tolerance_m) == expected
