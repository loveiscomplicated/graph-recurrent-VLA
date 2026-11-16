"""Read the native TAPNext++ cache contract with independent raw target checks."""

import json

import numpy as np
import pytest

from data_collection.episode_dataset import file_sha256
from data_collection.state21_tapnext import load_tapnext_surface_manifest
from graph.state_estimation.codec import object21_from_gt


def test_tapnext_cache_checks_source_order_labels_and_tracks(tmp_path):
    source_dir = tmp_path / "source"
    episode_dir = source_dir / "trajectory"
    episode_dir.mkdir(parents=True)
    raw_file = episode_dir / "frame_000000.npz"
    camera = "base_camera"
    gt = {
        "ee_pose": np.array([[0.0, 0.0, 0.4, 1, 0, 0, 0]], dtype=np.float32),
        "peg_pose": [[0.0, 0.0, 0.5, 1, 0, 0, 0]],
        "hole_pose": [[0.1, 0.0, 0.5, 1, 0, 0, 0]],
        "peg_half_sizes": [[0.04, 0.01, 0.01]],
        "hole_radius": [0.02],
    }
    np.savez(
        raw_file,
        **{
            f"observation/sensor_data/{camera}/rgb": np.zeros(
                (1, 512, 512, 3), dtype=np.uint8
            ),
            f"observation/sensor_data/{camera}/depth": np.zeros(
                (1, 512, 512, 1), dtype=np.int16
            ),
            f"observation/sensor_param/{camera}/extrinsic_cv": np.zeros((1, 3, 4)),
            f"observation/sensor_param/{camera}/intrinsic_cv": np.zeros((1, 3, 3)),
            "observation/extra/tcp_pose": gt["ee_pose"],
            **{f"ground_truth/{key}": values for key, values in gt.items()},
        },
    )
    raw_manifest = source_dir / "manifest.json"
    raw_manifest.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "episodes": [
                    {
                        "path": "trajectory",
                        "seed": 42,
                        "observations": 1,
                        "frame_sha256": [file_sha256(raw_file)],
                    }
                ],
            }
        )
    )
    cache = tmp_path / "episode.npz"
    manifest = tmp_path / "graph_manifest.json"
    arrays = {
        "frame_indices": np.array([0]),
        "node_offsets": np.array([0, 2]),
        "xyz_world": np.array([[0.01, 0.01, 0.5], [0.0, 0.01, 0.5]], dtype=np.float32),
        "object_type": np.array([0, 0]),
        "is_corner": np.array([False, True]),
        "track_id": np.array([4, -1]),
        "tcp_xyz_world": gt["ee_pose"][:, :3],
        "object21": object21_from_gt(gt).numpy(),
    }

    def save():
        np.savez(cache, **arrays)
        manifest.write_text(
            json.dumps(
                {
                    "source": str(raw_manifest),
                    "source_sha256": file_sha256(raw_manifest),
                    "tracker_sha256": "a" * 64,
                    "episodes": [
                        {
                            "episode": "trajectory",
                            "seed": 42,
                            "split": "train",
                            "graphs": 1,
                            "source_frames": 1,
                            "file": cache.name,
                            "sha256": file_sha256(cache),
                        }
                    ],
                }
            )
        )

    save()
    (episode,), report = load_tapnext_surface_manifest(manifest)
    assert episode.parent_id == "reset_seed:42"
    surface, corner = episode.samples[0].observation.nodes
    assert surface.visible and not surface.is_vertex
    assert corner.is_vertex and corner.vertex_id is None and corner.confidence is None
    assert report["provenance"]["tracker"] == "TAPNext++"

    for field, value, message in (
        ("frame_indices", np.array([1]), "outside source"),
        ("frame_indices", np.array([0, 1]), "frame_indices"),
        ("node_offsets", np.array([1, 2]), "ragged node offsets"),
        ("node_offsets", np.array([0, 3]), "offsets do not match"),
        ("xyz_world", np.zeros((2, 2)), "share the same rows"),
        ("tcp_xyz_world", np.zeros((2, 3)), "one TCP"),
        ("is_corner", np.array([0, 1]), "boolean corner mask"),
    ):
        original = arrays[field]
        arrays[field] = value
        save()
        with pytest.raises(ValueError, match=message):
            load_tapnext_surface_manifest(manifest)
        arrays[field] = original

    for field, value, message in (
        ("tracker_sha256", "absent", "tracker_sha256"),
        ("source_sha256", "f" * 64, "manifest SHA-256"),
        ("episode", "../outside", "not listed"),
        ("file", "../outside.npz", "relative path"),
        ("seed", 7, "raw source disagree"),
    ):
        save()
        content = json.loads(manifest.read_text())
        target = content if field in content else content["episodes"][0]
        target[field] = value
        manifest.write_text(json.dumps(content))
        with pytest.raises(ValueError, match=message):
            load_tapnext_surface_manifest(manifest)

    with raw_file.open("ab") as stream:
        stream.write(b"mismatched raw file")
    save()
    with pytest.raises(ValueError, match="Raw frame checksum"):
        load_tapnext_surface_manifest(manifest)
    with raw_file.open("r+b") as stream:
        stream.seek(-len(b"mismatched raw file"), 2)
        stream.truncate()

    arrays["object21"][0, 0] += 0.1
    save()
    with pytest.raises(ValueError, match="not aligned"):
        load_tapnext_surface_manifest(manifest)
    arrays["track_id"][0] = -1
    save()
    with pytest.raises(ValueError, match="query IDs"):
        load_tapnext_surface_manifest(manifest)
    cache.write_bytes(b"unexpected bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_tapnext_surface_manifest(manifest)
