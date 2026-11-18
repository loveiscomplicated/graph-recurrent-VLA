"""Contracts for unfiltered chronological Five-mode graph sequences."""

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import torch

from training.gnn.five_mode_sequence_data import build_train_loader


def write_release(tmp_path):
    """Create two variable-length train episodes and one reserved validation episode."""
    mode = "Fixed_Hole_expert"
    shard = "shard_000.tar"
    records = []
    specifications = ((11, "train", 3), (22, "train", 2), (33, "validation", 1))
    with tarfile.open(tmp_path / shard, "w") as archive:
        for seed, split, frames in specifications:
            corner_counts = np.arange(1, frames + 1)
            corner_offsets = np.r_[0, corner_counts.cumsum()]
            corner_values = np.concatenate([
                np.stack([
                    np.asarray([0.02 * node, 0.01 * frame, 0], dtype=np.float32)
                    for node in range(count)
                ])
                for frame, count in enumerate(corner_counts)
            ])
            corner_objects = np.concatenate([
                np.asarray(
                    [0] * max(1, int(count) - 1) + [1] * int(count > 1),
                    dtype=np.int8,
                )
                for count in corner_counts
            ])
            tables = {
                "corners": {
                    "corner_offsets": corner_offsets,
                    "corner_world_xyz": corner_values,
                    "corner_object_id": corner_objects,
                },
                "surface": {
                    "surface_offsets": np.arange(frames + 1),
                    "surface_world_xyz": np.stack([
                        np.asarray([0.03, 0.01 * frame, 0], dtype=np.float32)
                        for frame in range(frames)
                    ]),
                    "surface_object_id": np.zeros(frames, dtype=np.int8),
                },
                "targets": {
                    "tcp": np.stack(
                        [np.asarray([frame / 100, seed / 1000, 0], dtype=np.float32)
                         for frame in range(frames)]
                    ),
                    "object21": np.stack(
                        [np.full(21, seed * 10 + frame, dtype=np.float32)
                         for frame in range(frames)]
                    ),
                },
            }
            files = {}
            for kind, arrays in tables.items():
                member = f"batch_{seed}/{kind}/{seed}_{mode}.npz"
                buffer = io.BytesIO()
                np.savez(buffer, frame_indices=np.arange(frames), **arrays)
                content = buffer.getvalue()
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
                files[kind] = {
                    "archive": shard,
                    "member": member,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            records.append(
                {
                    "seed": seed,
                    "policy_mode": mode,
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
                "episodes": len(records),
                "episodes_by_split": {"train": 2, "validation": 1, "test": 0},
                "archives": [{"file": shard}],
                "records": records,
            }
        )
    )
    return manifest


@pytest.mark.parametrize("include_surface", (False, True))
def test_reads_every_train_episode_in_chronological_order(tmp_path, include_surface):
    manifest = write_release(tmp_path)
    config = {
        "data": {
            "manifest": manifest,
            "include_surface": include_surface,
            "shuffle_episodes": False,
            "expected_train_episodes": 2,
            "expected_train_frames": 5,
        },
        "training": {"episode_batch_size": 2},
    }
    loader, identity = build_train_loader(config)
    try:
        batch = next(iter(loader))
    finally:
        loader.dataset.close()

    assert identity["episodes"] == 2
    assert identity["frames"] == 5
    assert batch.episode_ids == [
        "11:Fixed_Hole_expert",
        "22:Fixed_Hole_expert",
    ]
    torch.testing.assert_close(batch.lengths, torch.tensor([3, 2]))
    torch.testing.assert_close(
        batch.frame_indices,
        torch.tensor([[0, 1, 2], [0, 1, -1]]),
    )
    torch.testing.assert_close(
        batch.frame_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    torch.testing.assert_close(
        batch.targets["object21"][0, :, 0], torch.tensor([110.0, 111.0, 112.0])
    )
    torch.testing.assert_close(
        batch.targets["object21"][1, :, 0], torch.tensor([220.0, 221.0, 0.0])
    )
    assert not batch.graph.valid[5].any()
    frame_output = torch.arange(6, dtype=torch.float32)[:, None]
    assert batch.reshape_frames(frame_output).shape == (2, 3, 1)

    first_frame_nodes = int(batch.graph.valid[0].sum())
    assert first_frame_nodes == (2 if include_surface else 1)

    source, target = batch.graph.edge_index
    object_types = batch.graph.object_id.flatten()
    edges = set(zip(source.tolist(), target.tolist()))
    assert edges
    assert all((target_row, source_row) in edges for source_row, target_row in edges)
    assert torch.equal(object_types[source], object_types[target])
    assert torch.all(batch.graph.edge_attr[:, 3] <= 0.08)
    assert identity["graph_topology"] == "b2_2_object_delaunay_bidirectional"
    assert identity["max_edge_m"] == 0.08


def test_rejects_condition_filter_fields(tmp_path):
    manifest = write_release(tmp_path)
    config = {
        "data": {"manifest": manifest, "condition": "condition_1"},
        "training": {"episode_batch_size": 1},
    }
    with pytest.raises(ValueError, match="does not accept condition filters"):
        build_train_loader(config)
