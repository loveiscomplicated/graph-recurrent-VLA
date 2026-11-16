"""Grouped splits and target isolation for the combined TAPNext++ archive."""

import json

import numpy as np
import pytest
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.state21 import load_dataset
from data_collection.state21_batches import PreparedWindows
from graph.state_estimation.models import MODELS, build_model
from recurrent.features import FEATURE_LAYOUT


@pytest.fixture
def source(tmp_path):
    seeds = [5, 5, 10, 20]
    episodes = []
    for episode_index, seed in enumerate(seeds):
        name = f"sample_{episode_index}"
        directory = tmp_path / name
        directory.mkdir()
        hashes = []
        for index in range(2):
            file = directory / f"frame_{index:06d}.npz"
            depth = np.full((1, 8, 8, 1), 500, dtype=np.int16)
            depth[0, 3, 3, 0] = 0
            np.savez(
                file,
                **{
                    "observation/sensor_data/base_camera/depth": depth,
                    "observation/sensor_param/base_camera/intrinsic_cv": np.array(
                        [[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 1.0]]]
                    ),
                    "observation/sensor_param/base_camera/extrinsic_cv": np.array(
                        [[[1.0, 0, 0, 0.1], [0, 1.0, 0, 0], [0, 0, 1.0, 0.2]]]
                    ),
                    "observation/extra/tcp_pose": np.array(
                        [[0.1 * seed, 0.2, 0.4, 1, 0, 0, 0]]
                    ),
                },
            )
            hashes.append(file_sha256(file))
        episodes.append(
            {
                "path": name,
                "seed": seed,
                "observations": 2,
                "steps": 1,
                "frame_sha256": hashes,
                "initial_fingerprint": f"seed-{seed}",
            }
        )
    raw = tmp_path / "manifest.json"
    raw.write_text(json.dumps({"status": "COMPLETE", "episodes": episodes}))
    labels = np.zeros((8, 36), dtype=np.float32)
    labels[:, [18, 28]] = 1
    labels[:, 32:] = 0.02
    arrays = {
        "episode_ids": np.array([episode["path"] for episode in episodes]),
        "episode_offsets": np.arange(0, 9, 2),
        "reset_seeds": np.array(seeds),
        "frame_indices": np.tile([0, 1], 4),
        "state36_gt": labels,
        "surface_xy": np.tile([[[1.0, 1.0], [3.0, 3.0]]], (8, 1, 1)).astype(np.float32),
        "surface_visible": np.ones((8, 2), dtype=bool),
        "corner_query_offsets": np.arange(0, 9, 2),
        "corner_frame_offsets": np.arange(0, 17, 2),
        "corner_query_object_id": np.tile([0, 1], 4),
        "corner_xy": np.tile([[2.0, 1.0], [3.0, 2.0]], (8, 1)).astype(np.float32),
        "corner_visible": np.tile([True, True, False, True], 4),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "format": "shared_tapnext_state36",
                    "format_version": 1,
                    "state36_layout": FEATURE_LAYOUT[:-2],
                    "source_manifest_sha256": file_sha256(raw),
                    "tapnext_checkpoint_sha256": "a" * 64,
                }
            )
        ),
    }
    cache = tmp_path / "tapnext_state36.npz"
    np.savez(cache, **arrays)
    settings = {
        "kind": "tapnext_state36",
        "cache": str(cache),
        "source_manifest": str(raw),
        "split_seed": 0,
        "validation_seed_count": 1,
        "test_seed_count": 1,
    }
    return settings, arrays


def test_grouped_split_and_depth_backed_queries(source):
    settings, _ = source
    episodes, report = load_dataset(settings)
    assert episodes[0].split == episodes[1].split
    assert len({episode.split for episode in episodes}) == 3
    assert report["full_split"]["reset_seed_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    first, second = episodes[0].samples
    assert first.observation.tcp == (0.5, 0.2, 0.4)
    nodes = first.observation.nodes
    assert [node.vertex_id for node in nodes] == [0, 1, None]
    np.testing.assert_allclose(nodes[0].position, [0.15, 0.125, 0.3])
    assert not second.observation.nodes[0].visible
    assert second.observation.nodes[0].position is None
    assert first.target.shape == (21,)


def test_changing_all_state36_columns_only_changes_targets(source):
    settings, arrays = source
    original, _ = load_dataset(settings)
    arrays["state36_gt"] += np.arange(36, dtype=np.float32) + 1
    np.savez(settings["cache"], **arrays)
    modified, _ = load_dataset(settings)
    for before, after in zip(original, modified, strict=True):
        for left, right in zip(before.samples, after.samples, strict=True):
            assert left.observation == right.observation
            assert not torch.equal(left.target, right.target)

    before_data = PreparedWindows(original[:1], history=2, max_edge_m=0.08)
    after_data = PreparedWindows(modified[:1], history=2, max_edge_m=0.08)
    before = list(before_data.batches(1, torch.device("cpu")))[-1][0]
    after = list(after_data.batches(1, torch.device("cpu")))[-1][0]
    for name in MODELS:
        model = build_model(name, hidden=8, layers=1, history=2).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(before)["object21"], model(after)["object21"], rtol=0, atol=0
            )


def test_smoke_audits_holdout_and_refuses_bad_frame_alignment(source):
    settings, arrays = source
    settings["smoke"] = {
        "train_seed_count": 1,
        "validation_seed_count": 1,
        "frames_per_episode": 1,
    }
    episodes, report = load_dataset(settings)
    assert {episode.split for episode in episodes} == {"train", "validation"}
    assert all(len(episode.samples) == 1 for episode in episodes)
    assert report["full_split"]["reset_seed_counts"]["test"] == 1
    arrays["frame_indices"][-1] = 9
    np.savez(settings["cache"], **arrays)
    with pytest.raises(ValueError, match="frame order"):
        load_dataset(settings)


def test_mixed_cache_keeps_split_and_reads_the_return_window(source, tmp_path):
    from pathlib import Path
    from shutil import copyfile

    pilot, arrays = source
    _, pilot_report = load_dataset(pilot)
    groups = pilot_report["full_split"]["reset_seeds"]
    reference = tmp_path / "split.json"
    reference.write_text(json.dumps({"full_split": {"reset_seeds": groups}}))

    raw = json.loads(Path(pilot["source_manifest"]).read_text())
    recovery_rows = []
    for row in raw["episodes"]:
        name = f"recovery_{row['path']}"
        directory = tmp_path / name
        directory.mkdir()
        for index in range(4):
            copyfile(
                tmp_path / row["path"] / f"frame_{index % 2:06d}.npz",
                directory / f"frame_{index:06d}.npz",
            )
        recovery_rows.append(
            {
                **row,
                "path": name,
                "observations": 4,
                "steps": 3,
                "recovery_step": 3,
                "frame_sha256": row["frame_sha256"] * 2,
            }
        )
    raw["episodes"] = recovery_rows
    recovery_manifest = tmp_path / "recovery_manifest.json"
    recovery_manifest.write_text(json.dumps(raw))
    recovery = dict(arrays)
    for key in (
        "state36_gt",
        "surface_xy",
        "surface_visible",
        "corner_xy",
        "corner_visible",
    ):
        groups_of_rows = np.split(arrays[key], 4)
        recovery[key] = np.concatenate(
            [np.concatenate((rows, rows)) for rows in groups_of_rows]
        )
    recovery["state36_gt"][:, 15] = np.arange(16, dtype=np.float32)
    recovery["episode_ids"] = np.array([row["path"] for row in recovery_rows])
    recovery["episode_offsets"] = np.arange(0, 17, 4)
    recovery["frame_indices"] = np.tile(np.arange(4), 4)
    recovery["corner_frame_offsets"] = np.arange(0, 33, 2)
    meta = json.loads(arrays["metadata_json"].item())
    recovery["metadata_json"] = np.asarray(
        json.dumps(
            {
                **meta,
                "source_manifest_sha256": file_sha256(recovery_manifest),
                "reset_seeds_by_split": groups,
            }
        )
    )
    recovery_cache = tmp_path / "recovery.npz"
    np.savez(recovery_cache, **recovery)
    config = {
        **pilot,
        "sources": {
            "pilot": pilot,
            "recovery": {
                "cache": str(recovery_cache),
                "source_manifest": str(recovery_manifest),
                "smoke_center": "recovery_step",
            },
        },
        "split_reference": str(reference),
        "smoke": {
            "train_seed_count": 1,
            "validation_seed_count": 1,
            "frames_per_episode": 2,
        },
    }
    episodes, report = load_dataset(config)
    assert report["full_split"]["reset_seeds"] == groups
    assert (
        report["sources"]["pilot"]["selected_frame_counts"]
        == report["sources"]["recovery"]["selected_frame_counts"]
    )
    for window in report["sources"]["recovery"]["selected_windows"]:
        assert (window["first_frame"], window["recovery_step"]) == (2, 3)
        episode = next(
            episode
            for episode in episodes
            if episode.episode_id == window["episode_id"]
        )
        assert [sample.observation.index for sample in episode.samples] == [2, 3]
        index = recovery["episode_ids"].tolist().index(episode.episode_id)
        assert episode.samples[0].target[0].item() == index * 4 + 2

    reference.write_text(json.dumps({"full_split": {"reset_seeds": {}}}))
    with pytest.raises(ValueError, match="pinned split"):
        load_dataset(config)
