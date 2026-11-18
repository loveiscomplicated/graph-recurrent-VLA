"""Verify additive surface generation, RGB-D gates, source identity and split isolation."""

from copy import deepcopy
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import cv2
import numpy as np
import pytest
import torch

from data_collection.episode_dataset import file_sha256
from data_collection import five_mode_box_surface as builder
from data_collection import five_mode_surface_source as source
from data_collection.five_mode_surface_export import merge_surfaces
from utils.experiment_logger import ExperimentLogger


class FixedQueryTracker:
    """A deterministic tracker for pipeline tests, not a learned tracking replacement."""

    def __init__(self):
        self.calls = []

    def track_frame(self, image, query_points_xy=None, state=None, **kwargs):
        self.calls.append((state is None, int(image[16, 16, 2])))
        queries = query_points_xy if state is None else state
        return queries.copy(), np.ones(len(queries), dtype=bool), queries.copy()


def encoded_npz(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def add_member(archive, name, contents):
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    archive.addfile(info, io.BytesIO(contents))


def rgbd_arrays(index, *, empty=False):
    hsv = np.full((32, 32, 3), (20, 120, 120 + 40 * index), dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if empty:
        rgb[:] = 0
    intrinsic = np.asarray([[100, 0, 16], [0, 100, 16], [0, 0, 1]], dtype=np.float32)
    extrinsic = np.column_stack((np.eye(3), [1, 0, 0])).astype(np.float32)
    return {
        source.FRAME_KEYS[0]: rgb[None],
        source.FRAME_KEYS[1]: np.full((1, 32, 32, 1), 1000, dtype=np.int16),
        source.FRAME_KEYS[2]: intrinsic[None], source.FRAME_KEYS[3]: extrinsic[None],
        # allow_pickle=False would fail if inference tried to decode this GT field.
        "ground_truth/must_not_decode": np.asarray([{"not_an_input": True}], dtype=object),
    }


def make_fixture(tmp_path, *, bad_raw_frame=False):
    release = tmp_path / "release"
    release.mkdir()
    raw_root = release / "original_rgbd"
    raw_root.mkdir()
    records = []
    with tarfile.open(release / "learning.tar", "w") as learning:
        for seed, split in enumerate(("train", "validation", "test"), start=1):
            batch = f"{split}_batch"
            key = f"{seed}_Fixed_Hole_expert"
            tables = {
                "corners": dict(
                    frame_indices=np.arange(2), corner_offsets=np.asarray([0, 1, 2]),
                    corner_uv=np.asarray([[16, 16], [16, 16]], dtype=np.float32),
                    corner_world_xyz=np.zeros((2, 3), dtype=np.float32),
                    corner_object_id=np.ones(2, dtype=np.int8),
                ),
                "surface": dict(frame_indices=np.arange(2), surface_offsets=np.arange(3),
                                surface_uv=np.asarray([[1, 1], [2, 2]], dtype=np.float32),
                                surface_track_id=np.zeros(2, dtype=np.int32),
                                surface_query_xy=np.asarray([[1, 1]], dtype=np.float32),
                                surface_anchor_frame=np.asarray(0),
                                surface_world_xyz=np.asarray([[0.1, 0, 1], [0.2, 0, 1]], dtype=np.float32)),
                "targets": dict(frame_indices=np.arange(2), tcp=np.zeros((2, 3), dtype=np.float32),
                                object21=np.ones((2, 21), dtype=np.float32)),
            }
            files = {}
            for kind, table in tables.items():
                data = encoded_npz(table)
                member = f"{batch}/{kind}/{key}.npz"
                files[kind] = dict(archive="learning.tar", member=member,
                                   sha256=hashlib.sha256(data).hexdigest())
                if split != "test":
                    add_member(learning, member, data)
            raw_digest = "0" * 64
            if split != "test":
                directory = raw_root / batch
                directory.mkdir()
                raw_tar = directory / "episodes.tar"
                frames = [encoded_npz(rgbd_arrays(index)) for index in range(2)]
                hashes = [hashlib.sha256(frame).hexdigest() for frame in frames]
                if bad_raw_frame and split == "train":
                    frames[1] = frames[1] + b"corrupt"
                case = dict(seed=seed, policy_mode="Fixed_Hole_expert", split=split,
                            observations=2, path=f"pilot/episode_{seed}")
                episode = dict(seed=seed, path=f"episode_{seed}", observations=2, frame_sha256=hashes)
                with tarfile.open(raw_tar, "w") as archive:
                    add_member(archive, "./episode_index.json", json.dumps([case]).encode())
                    add_member(archive, f"{case['path']}/episode.json", json.dumps(episode).encode())
                    # Collection stores frames in reverse order; replay must not follow TAR order.
                    for frame in (1, 0):
                        add_member(archive, f"{case['path']}/frame_{frame:06d}.npz", frames[frame])
                compressed = directory / "episodes.tar.zst"
                subprocess.run(["zstd", "-q", str(raw_tar), "-o", str(compressed)], check=True)
                raw_tar.unlink()
                raw_digest = file_sha256(compressed)
            records.append(dict(seed=seed, policy_mode="Fixed_Hole_expert", split=split,
                                frames=2, batch=batch, files=files, raw_sha256=raw_digest))
    manifest = release / "manifest.json"
    manifest.write_text(json.dumps(dict(
        format="five_mode_supervised_v1", episodes=3,
        records=records, archives=[dict(file="learning.tar")],
    )))
    index = release / "condition_index.json"
    index.write_text(json.dumps({"identity": "unchanged original condition labels"}))
    checkpoint = tmp_path / "tracker.pt"
    checkpoint.write_bytes(b"only a pinned checkpoint fixture; the tracker is injected")
    config = dict(
        splits=["train", "validation"], resume=True, mode="build",
        export_candidate=True, episodes_per_shard=2,
        manifest=dict(path=str(manifest), sha256=file_sha256(manifest)),
        condition_index=dict(path=str(index), sha256=file_sha256(index)),
        raw_root=str(raw_root), output_root=str(tmp_path / "cache"), scratch_root=str(tmp_path / "scratch"),
        minimum_free_gib=0, expected_counts={split: dict(episodes=1, frames=2) for split in source.ALLOWED_SPLITS},
        checkpoint_sha256=file_sha256(checkpoint),
        tracking=dict(checkpoint=str(checkpoint), device="cpu", surface_queries=2,
                      corner_exclusion_px=2, tap_input_resolution=256, tap_adaptive_crop=False),
    )
    return config, records


@pytest.mark.parametrize("optimized_flags", [False, True])
def test_build_preserves_originals_and_never_opens_test_members(tmp_path, monkeypatch, optimized_flags):
    config, original_records = make_fixture(tmp_path)
    release = Path(config["manifest"]["path"]).parent
    before = {path: file_sha256(path) for path in release.rglob("*") if path.is_file()}
    tracker = FixedQueryTracker()
    if optimized_flags:
        config["tracking"].update(optimized_tracking_settings())
        # Only the routing/context is exercised here; the injected tracker has
        # no CUDA operations and the result is explicitly marked SMOKE_ONLY.
        tracker.device = torch.device("cuda:0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(builder, "ExperimentLogger", partial(ExperimentLogger, root=tmp_path / "experiments"))
    original_extract = tarfile.TarFile.extractfile

    def guarded_extract(archive, member):
        name = member.name if isinstance(member, tarfile.TarInfo) else member
        assert "test_batch" not in name and "3_Fixed_Hole_expert" not in name
        return original_extract(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", guarded_extract)
    output = builder.build(config, tracker_factory=lambda settings: tracker)
    manifest = json.loads((output / "manifest.json").read_text())

    assert manifest["status"] == "SMOKE_ONLY"
    assert manifest["test_samples_read"] is False
    assert [row["split"] for row in manifest["records"]] == ["train", "validation"]
    for original, row in zip(original_records, manifest["records"]):
        assert {key: value for key, value in row.items() if key != "box_surface"} == original
        with np.load(output / row["box_surface"]["file"], allow_pickle=False) as arrays:
            assert arrays["frame_indices"].tolist() == [0, 1]
            assert arrays["surface_object_id"].tolist() == [1] * 4
            assert arrays["surface_offsets"].tolist() == [0, 2, 4]
            assert np.all(builder.separated_from_corners(arrays["surface_uv"], [[16, 16]], 2))
            # World-to-camera translates x by +1 m; verify known back-projection.
            expected_x = (arrays["surface_uv"][:, 0] - 16) / 100 - 1
            np.testing.assert_allclose(arrays["surface_world_xyz"][:, 0], expected_x, atol=1e-7)
    assert [initial for initial, _ in tracker.calls] == [True, False, True, False]
    assert tracker.calls[0][1] < tracker.calls[1][1]
    assert before == {path: file_sha256(path) for path in before}
    candidate_path = Path(manifest["candidate_manifest"]["path"])
    candidate = json.loads(candidate_path.read_text())
    assert candidate["status"] == "SMOKE_ONLY"
    assert candidate["format"] != "five_mode_supervised_v1"
    with tarfile.open(candidate_path.parent / candidate["archives"][0]["file"]) as archive:
        for original, row in zip(original_records, candidate["records"]):
            for kind in ("corners", "targets"):
                assert row["files"][kind]["sha256"] == original["files"][kind]["sha256"]
            with archive.extractfile(row["files"]["surface"]["member"]) as stream:
                with np.load(io.BytesIO(stream.read()), allow_pickle=False) as arrays:
                    peg = arrays["surface_object_id"] == 0
                    np.testing.assert_array_equal(arrays["surface_world_xyz"][peg],
                                                  np.asarray([[0.1, 0, 1], [0.2, 0, 1]], dtype=np.float32))
    # Exercise the production packing contract in memory; the actual fake-tracker
    # cache stays SMOKE_ONLY and remains rejected as a production release.
    from training.gnn.five_mode_experiment_data import verify_same_rows
    from training.gnn.packed_five_mode_data import read_splits

    original_manifest = json.loads(Path(config["manifest"]["path"]).read_text())
    compatible_envelope = dict(candidate, format="five_mode_supervised_v1")
    splits = ("train", "validation")
    reference = read_splits(Path(config["manifest"]["path"]), original_manifest, splits, "all", True)
    projected = read_splits(candidate_path, compatible_envelope, splits, "all", True)
    augmented = read_splits(candidate_path, compatible_envelope, splits, "all", True, include_box_surface=True)
    for split in splits:
        verify_same_rows(reference[split], projected[split])
        kinds = augmented[split][0][1][1]
        assert (kinds == 0).sum(axis=1).tolist() == [1, 1]
        assert (kinds == 1).sum(axis=1).tolist() == [3, 3]

    def unexpected_tracker(settings):
        raise AssertionError("A fully committed resume must not load a tracker")

    tracked = output / manifest["records"][0]["box_surface"]["file"]
    timestamp = tracked.stat().st_mtime_ns
    builder.build(config, tracker_factory=unexpected_tracker)
    assert tracked.stat().st_mtime_ns == timestamp
    tracked.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="checksum differs"):
        builder.build(config, tracker_factory=unexpected_tracker)


@pytest.mark.parametrize("split", ["test", "all"])
def test_test_split_or_all_is_rejected_before_any_input_read(split):
    with pytest.raises(ValueError, match="test is reserved"):
        builder.prepare(dict(splits=[split]))


def test_explicit_test_selection_and_mixed_test_batch_are_rejected(tmp_path):
    config, records = make_fixture(tmp_path)
    config["episodes"] = [dict(seed=3, policy_mode="Fixed_Hole_expert")]
    with pytest.raises(ValueError, match="test is reserved"):
        builder.prepare(config)
    manifest = dict(format="five_mode_supervised_v1", episodes=3, records=deepcopy(records))
    manifest["records"][2]["batch"] = manifest["records"][0]["batch"]
    with pytest.raises(ValueError, match="no test episodes"):
        source.selected_records(manifest)


def test_forbidden_legacy_locations_and_npz_inputs_are_rejected_without_io(tmp_path):
    for path in (tmp_path / "shared_tracking_cache/tapnext_state36.npz", tmp_path / "copy.npz"):
        with pytest.raises(ValueError, match="forbidden"):
            source.checked_path(path, ".json")


def test_bad_raw_frame_hash_stops_before_tracking(tmp_path, monkeypatch):
    config, _ = make_fixture(tmp_path, bad_raw_frame=True)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(builder, "ExperimentLogger", partial(ExperimentLogger, root=tmp_path / "experiments"))
    tracker = FixedQueryTracker()
    with pytest.raises(ValueError, match="Raw RGB-D frame checksum differs"):
        builder.build(config, tracker_factory=lambda settings: tracker)
    assert not tracker.calls


@pytest.mark.parametrize("invalid_frame", ["mask", "depth", "corners"])
def test_current_frame_gates_remove_unsupported_points(tmp_path, invalid_frame):
    for frame in range(2):
        arrays = rgbd_arrays(frame, empty=frame == 1 and invalid_frame == "mask")
        if frame == 1 and invalid_frame == "depth":
            arrays[source.FRAME_KEYS[1]][:] = -32768
        (tmp_path / f"frame_{frame:06d}.npz").write_bytes(encoded_npz(arrays))
    corners = dict(frame_indices=np.arange(2), corner_offsets=np.asarray([0, 0, 0]),
                   corner_uv=np.empty((0, 2), dtype=np.float32))
    tracker = FixedQueryTracker()
    settings = dict(surface_queries=2, corner_exclusion_px=2)
    if invalid_frame == "corners":
        first = builder.track_box_episode(tmp_path, corners, tracker, settings)
        corners["corner_offsets"] = np.asarray([0, 0, 2])
        corners["corner_uv"] = first["surface_query_xy"]
    result = builder.track_box_episode(tmp_path, corners, tracker, settings)
    assert result["surface_offsets"].tolist() == [0, 2, 2]
    assert result["frame_indices"].tolist() == [0, 1]


def test_depth_pixel_and_subpixel_locations_both_obey_corner_exclusion():
    xy = np.asarray([[4.4, 4.4], [4.6, 4.6]])
    actual = builder.separated_from_corners(xy, np.asarray([[4, 4]]), 0.5)
    assert actual.tolist() == [False, True]


def test_no_initial_box_keeps_empty_frames_and_late_initialization(tmp_path):
    for frame in range(2):
        arrays = rgbd_arrays(frame, empty=frame == 0)
        (tmp_path / f"frame_{frame:06d}.npz").write_bytes(encoded_npz(arrays))
    corners = dict(frame_indices=np.arange(2), corner_offsets=np.asarray([0, 0, 0]),
                   corner_uv=np.empty((0, 2), dtype=np.float32))
    result = builder.track_box_episode(tmp_path, corners, FixedQueryTracker(),
                                       dict(surface_queries=2, corner_exclusion_px=2))
    assert int(result["surface_anchor_frame"]) == 1
    assert result["surface_offsets"].tolist() == [0, 0, 2]


def test_merge_preserves_a_peg_only_episode_when_box_is_never_available():
    peg = dict(
        frame_indices=np.arange(2), surface_offsets=np.arange(3),
        surface_uv=np.asarray([[1, 1], [2, 2]], dtype=np.float32),
        surface_world_xyz=np.asarray([[1, 0, 1], [2, 0, 1]], dtype=np.float32),
        surface_track_id=np.zeros(2, dtype=np.int32), surface_anchor_frame=np.asarray(0),
        surface_query_xy=np.asarray([[1, 1]], dtype=np.float32),
    )
    box = dict(
        frame_indices=np.arange(2), surface_offsets=np.zeros(3, dtype=np.int64),
        surface_uv=np.empty((0, 2), dtype=np.float32),
        surface_world_xyz=np.empty((0, 3), dtype=np.float32),
        surface_track_id=np.empty(0, dtype=np.int32), surface_anchor_frame=np.asarray(-1),
        surface_query_xy=np.empty((0, 2), dtype=np.float32),
    )
    result = merge_surfaces(peg, box)
    for name in peg:
        np.testing.assert_array_equal(result[name], peg[name])
    assert result["surface_object_id"].tolist() == [0, 0]
    box["frame_indices"] = np.asarray([0, 2])
    with pytest.raises(ValueError, match="frame IDs"):
        merge_surfaces(peg, box)


def optimized_tracking_settings():
    """Use the same four execution switches as the optimized CUDA cache config."""
    return dict(
        device="cuda:0", surface_queries=2, corner_exclusion_px=2,
        tap_input_resolution=256, tap_adaptive_crop=True, tap_cuda_autocast=True,
        tap_cache_cuda_autocast=True, tap_cuda_graph=True,
        tap_mps_fp16_vit=False, tap_mps_fp16_recurrent_mlp=False,
        tap_mps_skip_past_query_embeddings=False, tap_mps_compile_recurrent=False,
    )


@pytest.mark.parametrize("overrides,message", [
    ({"device": "cpu"}, "CUDA tracking device"),
    ({"device": "mlx"}, "CUDA tracking device"),
    ({"tap_adaptive_crop": False}, "adaptive CUDA tracking"),
    ({"tap_cuda_autocast": False}, "CUDA FP16 autocast"),
    ({"tap_cuda_autocast": "true"}, "must be boolean"),
    ({"tap_cache_cuda_autocast": 1}, "must be boolean"),
    ({"tap_cuda_graph": "false"}, "must be boolean"),
])
def test_invalid_cuda_settings_fail_before_source_reads(overrides, message):
    settings = dict(optimized_tracking_settings(), **overrides)
    with pytest.raises(ValueError, match=message):
        builder.prepare(dict(splits=["train", "validation"], tracking=settings))


def test_cuda_cache_cannot_silently_run_with_cpu_tracker_or_inference_mode(tmp_path):
    from types import SimpleNamespace

    settings = optimized_tracking_settings()
    cpu_tracker = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA tracker instance"):
        builder.track_box_episode(tmp_path, {}, cpu_tracker, settings)
    cuda_tracker = SimpleNamespace(device=torch.device("cuda:0"))
    with torch.inference_mode(), pytest.raises(ValueError, match="enclosing inference_mode"):
        builder.track_box_episode(tmp_path, {}, cuda_tracker, settings)


def test_optimized_settings_select_adaptive_cuda_tracker_and_enable_graph(monkeypatch):
    from types import SimpleNamespace
    from evaluation.tapnextpp_surface_tracking import load_tracker
    from visual.tapnextpp_cuda import AdaptiveCudaTracker

    settings = dict(optimized_tracking_settings(), checkpoint="not_loaded_in_this_test.pt")
    tracker = SimpleNamespace(cuda_graph_enabled=False)
    calls = []

    def fake_checkpoint_loader(cls, checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return tracker

    monkeypatch.setattr(AdaptiveCudaTracker, "from_checkpoint", classmethod(fake_checkpoint_loader))
    assert load_tracker(settings) is tracker
    assert tracker.cuda_graph_enabled is True
    assert calls == [(settings["checkpoint"], dict(device="cuda:0", input_resolution=256))]


def test_weight_cache_reuses_casts_across_frames_without_autocasting_preprocessing(tmp_path, monkeypatch):
    """Use CPU BF16 casts to exercise PyTorch's cache lifetime without a CUDA device."""
    for frame in range(3):
        (tmp_path / f"frame_{frame:06d}.npz").write_bytes(encoded_npz(rgbd_arrays(frame)))
    corners = dict(frame_indices=np.arange(3), corner_offsets=np.zeros(4, dtype=np.int64),
                   corner_uv=np.empty((0, 2), dtype=np.float32))

    class CastProbeTracker(FixedQueryTracker):
        def __init__(self):
            super().__init__()
            # The declared device checks the CUDA routing contract only. All tensor
            # operations below are CPU BF16; this is not CUDA inference evidence.
            self.device = torch.device("cuda:0")
            self.layer = torch.nn.Linear(8, 8).eval()
            self.input = torch.ones(2, 8)
            self.flags = []

        def track_frame(self, image, query_points_xy=None, state=None, autocast=False, **kwargs):
            self.flags.append((autocast, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.layer(self.input)
            return super().track_frame(image, query_points_xy, state, **kwargs)

    real_box_mask = builder.box_mask

    def checked_box_mask(rgb):
        assert not torch.is_autocast_enabled("cuda")
        assert not torch.is_autocast_enabled("cpu")
        return real_box_mask(rgb)

    monkeypatch.setattr(builder, "box_mask", checked_box_mask)
    casts, predictions = {}, {}
    for cached in (False, True):
        settings = dict(optimized_tracking_settings(), tap_cache_cuda_autocast=cached)
        tracker = CastProbeTracker()
        # The implementation must explicitly enable caching, even if the caller
        # disabled it, and restore that setting when the episode scope exits.
        with torch.autocast("cpu", enabled=False, cache_enabled=False):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                predictions[cached] = builder.track_box_episode(tmp_path, corners, tracker, settings)
            assert torch.is_autocast_cache_enabled() is False
        casts[cached] = sum(event.count for event in profile.key_averages() if event.key == "aten::_to_copy")
        assert tracker.flags == [(True, False, False)] * 3
    assert casts[True] < casts[False]
    for name, values in predictions[False].items():
        np.testing.assert_array_equal(predictions[True][name], values)
