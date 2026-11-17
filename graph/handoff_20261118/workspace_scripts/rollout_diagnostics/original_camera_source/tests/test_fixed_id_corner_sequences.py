"""First-sighting, measurement-preservation and causal-input dataset contracts."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    corner_catalog,
    iter_sequences,
    pack_episode,
    prepare,
    read_manifest,
    read_source_manifest,
    sequences_from_arrays,
    validate_packed_episode,
)
from main import run


def source_arrays(episode="episode", length=8):
    """Make noisy detections, a late first sighting, a duplicate and an unknown."""
    references = np.zeros((length, 24, 3), dtype=np.float32)
    references[:, :, 0] = np.arange(24, dtype=np.float32)[None] * 0.1
    references[:, :, 1] = np.arange(length, dtype=np.float32)[:, None] * 0.01
    xyz = np.zeros((length, 4, 3), dtype=np.float32)
    identities = np.full((length, 4), -1, dtype=np.int64)
    valid = np.zeros((length, 4), dtype=bool)
    kind = np.zeros((length, 4), dtype=np.int64)
    for frame, vertex_id in ((0, 0), (1, 0), (4, 0), (2, 9), (3, 9), (6, 9), (7, 17)):
        xyz[frame, 0] = references[frame, vertex_id] + np.float32(0.001)
        identities[frame, 0] = vertex_id
        kind[frame, 0] = int(vertex_id >= 8)
        valid[frame, 0] = True
    # Duplicate certificate: keep one unchanged detector point, never their mean.
    xyz[4, 1] = references[4, 0] + np.float32(0.002)
    identities[4, 1] = 0
    valid[4, 1] = True
    # This raw point must not create an arbitrary vertex identity.
    xyz[:, 2] = 0.2
    valid[:, 2] = True
    robot = np.arange(length * 9, dtype=np.float32).reshape(length, 9) * 0.01
    return dict(
        xyz=xyz, kind=kind, valid=valid, physical_ids=identities,
        reference_xyz=references, reference_kind=np.array([0] * 8 + [1] * 16, dtype=np.int64),
        robot=robot,
        sample_ids=np.array([f"{episode}/frame_{frame:06d}.npz" for frame in range(length)]),
    )


def write_source(root):
    """Preserve all development splits and include an intentionally absent test file."""
    root.mkdir()
    episodes = []
    for split in ("train", "calibration", "validation"):
        filename = f"{split}.npz"
        np.savez_compressed(root / filename, **source_arrays(split))
        episodes.append(dict(
            episode=split, parent=f"parent_{split}", split=split, frames=8,
            file=filename, sha256=file_sha256(root / filename),
        ))
    episodes.append(dict(
        episode="reserved", parent="parent_test", split="test", frames=8,
        file="reserved_test_must_not_be_read.npz", sha256="0" * 64,
    ))
    manifest = dict(format="observed_corner_tracks_v1", time_unit="saved_frame", episodes=episodes)
    manifest["dataset_sha256"] = canonical_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class FixedIdCornerSequenceTests(unittest.TestCase):
    def test_catalog_defines_distinct_local_corners_for_all_families(self):
        catalog = corner_catalog()
        self.assertEqual([row["vertex_id"] for row in catalog], list(range(24)))
        for offset, family in ((0, "peg"), (8, "box_outer"), (16, "hole_rim")):
            rows = catalog[offset:offset + 8]
            self.assertEqual({row["family"] for row in rows}, {family})
            self.assertEqual(len({tuple(row["local_signs"]) for row in rows}), 8)
            self.assertEqual(rows[0]["local_signs"], [-1, -1, -1])
            self.assertEqual(rows[-1]["local_signs"], [1, 1, 1])

    def test_first_sightings_targets_and_actual_coordinates(self):
        source = source_arrays()
        packed, stats = pack_episode(source)
        validate_packed_episode(packed)
        self.assertEqual(packed["vertex_ids"].tolist(), [0, 9])
        self.assertEqual(packed["start_frames"].tolist(), [0, 2])
        self.assertEqual(stats["never_seen_vertices"], 21)
        self.assertEqual(stats["terminal_only_vertices"], 1)
        self.assertEqual(stats["unknown_detections"], 8)
        self.assertEqual(stats["duplicate_certificates"], 1)
        sequences = list(sequences_from_arrays(packed, dict(episode="episode", parent="parent")))
        peg, box = sequences
        self.assertEqual(peg.observed.tolist(), [True, True, False, False, True, False, False, False])
        self.assertEqual(box.observed.tolist(), [True, True, False, False, True, False])
        self.assertEqual(box.prediction_valid.tolist(), [False, True, True, True, True, True])
        # Both visible and missing post-initialization rows receive supervision.
        np.testing.assert_array_equal(box.targets[1], source["xyz"][3, 0])
        np.testing.assert_array_equal(box.targets[2], source["reference_xyz"][4, 9])
        self.assertFalse(np.array_equal(box.targets[1], source["reference_xyz"][3, 9]))
        np.testing.assert_array_equal(peg.targets[4], source["xyz"][4, 0])
        self.assertEqual(box.inputs.shape, (6, 13))
        np.testing.assert_array_equal(box.inputs[1, :3], source["xyz"][2, 0])
        np.testing.assert_array_equal(box.inputs[:, 4:], source["robot"][2:])
        np.testing.assert_array_equal(box.last_observed_xyz_world[4], source["xyz"][3, 0])
        np.testing.assert_array_equal(box.last_observed_xyz_world[5], source["xyz"][6, 0])
        np.testing.assert_allclose(box.last_observed_xyz_world + box.target_delta_xyz, box.targets)

    def test_frame_node_order_does_not_change_identity_or_targets(self):
        source = source_arrays()
        expected, expected_stats = pack_episode(source)
        shuffled = {name: value.copy() for name, value in source.items()}
        for frame in range(8):
            order = np.roll(np.arange(4)[::-1], frame % 4)
            for name in ("xyz", "kind", "valid", "physical_ids"):
                shuffled[name][frame] = source[name][frame, order]
        actual, actual_stats = pack_episode(shuffled)
        self.assertEqual(actual_stats, expected_stats)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_current_measurements_and_hidden_gt_cannot_leak_into_inputs(self):
        source = source_arrays()
        packed, _ = pack_episode(source)
        row = dict(episode="episode", parent="parent")
        original = list(sequences_from_arrays(packed, row))[1]
        source["xyz"][3, 0] += 0.02
        changed, _ = pack_episode(source)
        visible_changed = list(sequences_from_arrays(changed, row))[1]
        np.testing.assert_array_equal(visible_changed.inputs[:2], original.inputs[:2])
        np.testing.assert_array_equal(visible_changed.last_observed_xyz_world[:2], original.last_observed_xyz_world[:2])
        self.assertFalse(np.array_equal(visible_changed.targets[1], original.targets[1]))
        self.assertFalse(np.array_equal(visible_changed.inputs[2], original.inputs[2]))
        source["reference_xyz"][5, 9] += 0.5
        changed, _ = pack_episode(source)
        hidden_changed = list(sequences_from_arrays(changed, row))[1]
        np.testing.assert_array_equal(hidden_changed.inputs, visible_changed.inputs)
        self.assertFalse(np.array_equal(hidden_changed.targets[3], visible_changed.targets[3]))

    def test_no_observed_or_only_terminal_corners_make_no_prediction_samples(self):
        source = source_arrays()
        source["physical_ids"][:] = -1
        packed, stats = pack_episode(source)
        self.assertIsNone(packed)
        self.assertEqual(stats["never_seen_vertices"], 24)
        source["physical_ids"][7, 0] = 17
        packed, stats = pack_episode(source)
        self.assertIsNone(packed)
        self.assertEqual(stats["never_seen_vertices"], 23)
        self.assertEqual(stats["terminal_only_vertices"], 1)

    def test_root_config_roundtrip_split_hash_and_reserved_test_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            write_source(source)
            hashes_before = {path.name: file_sha256(path) for path in source.iterdir()}
            config = dict(
                trainer="data_collection.temporal.fixed_id_corner_sequences:prepare",
                source=dict(corner_dataset=str(source)), output=dict(root=str(root / "output")),
            )
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            environment = dict(WANDB_MODE="disabled", EXPERIMENTS_DIR=str(root / "experiments"))
            with patch.dict(os.environ, environment):
                run(config_path)
                output, = (root / "output").iterdir()
                manifest = read_manifest(output)
                self.assertEqual(manifest["reserved_test_episodes_unread"], 1)
                for split in ("train", "calibration", "validation"):
                    sequences = list(iter_sequences(output, split))
                    self.assertEqual(len(sequences), 2)
                    self.assertEqual({sample.parent for sample in sequences}, {f"parent_{split}"})
                self.assertEqual(list(iter_sequences(output, "test")), [])
                hashes_after = {path.name: file_sha256(path) for path in source.iterdir()}
                self.assertEqual(hashes_after, hashes_before)
                # Failed source integrity must not produce a completed dataset.
                with (source / "train.npz").open("ab") as file:
                    file.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum/path mismatch"):
                    prepare(config)
                incomplete, = [path for path in (root / "output").iterdir() if path != output]
                self.assertFalse((incomplete / "manifest.json").exists())
                # Readers also reject modified prepared files.
                prepared_path = output / manifest["episodes"][0]["file"]
                with prepared_path.open("ab") as file:
                    file.write(b"corrupt")
                with self.assertRaisesRegex(ValueError, "checksum/path mismatch"):
                    list(iter_sequences(output, "train"))

    def test_source_parents_must_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            manifest = write_source(root)
            manifest["episodes"][1]["parent"] = manifest["episodes"][0]["parent"]
            del manifest["dataset_sha256"]
            manifest["dataset_sha256"] = canonical_hash(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Parent crosses"):
                read_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
