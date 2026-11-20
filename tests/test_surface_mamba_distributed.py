"""Contracts for distributed Surface-Mamba collection and release assembly."""

import json
from pathlib import Path
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.surface_mamba_dataset import (
    DERIVED_FORMAT,
    EXPECTED_BUCKETS,
    SHARD_PACKAGE_FORMAT,
    SHARDED_DERIVED_FORMAT,
    _partition_buckets,
    _read_published_shard_packages,
    materialize_state36_release,
    merge_shard_packages,
)


GLOBAL_BUCKETS = [
    {
        "name": name,
        "split": split,
        "dart": dart,
        "count": count,
        "start_seed": start_seed,
        "max_attempts": max_attempts,
    }
    for (name, (split, dart, count)), start_seed, max_attempts in zip(
        EXPECTED_BUCKETS.items(),
        (4_000_000, 4_020_000, 4_040_000, 4_050_000),
        (20_000, 20_000, 10_000, 10_000),
        strict=True,
    )
]


class DistributedBucketTests(unittest.TestCase):
    def test_eight_shards_cover_each_quota_and_attempt_range_once(self):
        partitions = [
            _partition_buckets(GLOBAL_BUCKETS, {"index": index, "count": 8})
            for index in range(8)
        ]

        self.assertEqual(
            [sum(bucket["count"] for bucket in shard) for shard in partitions],
            [114, 114, 112, 112, 112, 112, 112, 112],
        )
        for bucket_index, original in enumerate(GLOBAL_BUCKETS):
            assigned = [shard[bucket_index] for shard in partitions]
            self.assertEqual(sum(bucket["count"] for bucket in assigned), original["count"])
            self.assertEqual(
                sum(bucket["max_attempts"] for bucket in assigned),
                original["max_attempts"],
            )
            ranges = [
                range(bucket["start_seed"], bucket["start_seed"] + bucket["max_attempts"])
                for bucket in assigned
            ]
            self.assertEqual(ranges[0].start, original["start_seed"])
            self.assertEqual(
                ranges[-1].stop,
                original["start_seed"] + original["max_attempts"],
            )
            for left, right in zip(ranges[:-1], ranges[1:], strict=True):
                self.assertEqual(left.stop, right.start)


class ShardMergeTests(unittest.TestCase):
    def _write_package(self, root: Path, shard_index: int) -> Path:
        directory = root / f"shard_{shard_index:03d}"
        directory.mkdir()
        for name in ("raw.tar", "state36.tar"):
            (directory / name).write_bytes(f"{shard_index}:{name}".encode())

        buckets = _partition_buckets(
            GLOBAL_BUCKETS, {"index": shard_index, "count": 8}
        )
        episodes = []
        for bucket in buckets:
            for offset in range(bucket["count"]):
                episodes.append(
                    {
                        "seed": bucket["start_seed"] + offset,
                        "split": bucket["split"],
                        "dart": bucket["dart"],
                        "path": f"buckets/{bucket['name']}/episode_{offset:06d}",
                    }
                )
        archives = {}
        for kind, name in (("raw", "raw.tar"), ("state36", "state36.tar")):
            path = directory / name
            archives[kind] = {
                "file": name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        index = {
            "format": SHARD_PACKAGE_FORMAT,
            "run_id": f"run_{shard_index}",
            "shard": {"index": shard_index, "count": 8},
            "counts": {bucket["name"]: bucket["count"] for bucket in buckets},
            "global_counts": {
                name: expected[2] for name, expected in EXPECTED_BUCKETS.items()
            },
            "global_buckets": GLOBAL_BUCKETS,
            "bucket_assignments": buckets,
            "episodes": episodes,
            "archives": archives,
        }
        path = directory / "shard_package.json"
        path.write_text(json.dumps(index))
        return path

    def test_merge_accepts_exact_complete_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexes = [self._write_package(root, index) for index in range(8)]
            output = root / "merged"

            merge_shard_packages(
                {
                    "shard_package_indexes": [str(path) for path in indexes],
                    "output": str(output),
                }
            )

            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["episode_count"], 900)
            self.assertEqual(manifest["counts"]["train_clean"], 360)
            self.assertEqual(len({row["seed"] for row in manifest["episodes"]}), 900)

    def test_published_package_checks_archive_metadata_without_downloading_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            index = self._write_package(Path(temporary), 0)
            package = json.loads(index.read_text())
            run_id = "a" * 32
            package["run_id"] = run_id
            index.write_text(json.dumps(package))
            prefix = f"recurrent/{run_id}/artifacts"
            index_path = f"{prefix}/episode_metadata/{file_sha256(index)}/shard_package.json"
            files = [SimpleNamespace(path=index_path, blob_id="index")]
            for archive in package["archives"].values():
                sha256 = archive["sha256"]
                files.append(SimpleNamespace(
                    path=f"{prefix}/trajectory/{sha256}/{archive['file']}",
                    blob_id="archive",
                    size=archive["size"],
                    lfs=SimpleNamespace(sha256=sha256),
                ))
            source = {"repository": "org/dataset", "revision": "b" * 40, "run_ids": [run_id]}
            with patch("huggingface_hub.HfApi") as api, patch(
                "huggingface_hub.hf_hub_download", return_value=str(index)
            ) as download:
                api.return_value.list_repo_tree.return_value = files
                [(loaded, provenance)] = _read_published_shard_packages(source)
                self.assertEqual(loaded, package)
                self.assertEqual(provenance["huggingface"]["archive_paths"]["raw"], files[1].path)
                download.assert_called_once_with(
                    source["repository"], index_path, repo_type="dataset",
                    revision=source["revision"], token=False,
                )
                files[1].lfs.sha256 = "0" * 64
                with self.assertRaisesRegex(ValueError, "Published archive size or checksum differs"):
                    _read_published_shard_packages(source)

    def test_merge_rejects_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexes = [self._write_package(root, index) for index in range(8)]
            first = json.loads(indexes[0].read_text())
            second = json.loads(indexes[1].read_text())
            second["episodes"][0]["seed"] = first["episodes"][0]["seed"]
            indexes[1].write_text(json.dumps(second))

            with self.assertRaisesRegex(ValueError, "assigned range"):
                merge_shard_packages(
                    {
                        "shard_package_indexes": [str(path) for path in indexes],
                        "output": str(root / "merged"),
                    }
                )


class State36MaterializationTests(unittest.TestCase):
    def _write_archive(self, root: Path, shard_index: int) -> Path:
        assignment = {"index": shard_index, "count": 8}
        buckets = _partition_buckets(GLOBAL_BUCKETS, assignment)
        state_root = root / f"source_{shard_index:03d}" / "state36"
        state_root.mkdir(parents=True)
        episodes = []
        split_rows = {name: [] for name in ("train", "validation", "test")}
        for bucket in buckets:
            for offset in range(bucket["count"]):
                seed = bucket["start_seed"] + offset
                record = {
                    "seed": seed,
                    "split": bucket["split"],
                    "dart": bucket["dart"],
                    "steps": 1,
                    "file": f"{seed}.npz",
                    "sha256": "unused-in-materialized-split",
                }
                episodes.append(record)
                split_rows[bucket["split"]].append(record)

        splits = {}
        for split, rows in split_rows.items():
            path = state_root / f"{split}.npz"
            count = len(rows)
            state = np.zeros((count, 36), dtype=np.float32)
            state[:, 3] = 1
            state[:, 18] = 1
            state[:, 28] = 1
            np.savez(
                path,
                state36_gt=state,
                state36_graph=state.copy(),
                action_target=np.zeros((count, 7), dtype=np.float32),
                episode_offsets=np.arange(count + 1, dtype=np.int64),
                reset_seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
                episode_dart=np.asarray([row["dart"] for row in rows], dtype=bool),
            )
            splits[split] = {
                "file": path.name,
                "episodes": count,
                "rows": count,
                "sha256": file_sha256(path),
            }

        manifest = {
            "format": SHARDED_DERIVED_FORMAT,
            "run_id": f"derived_{shard_index}",
            "shard": assignment,
            "frontend": {"identity": "frozen-test-frontend"},
            "splits": splits,
            "episodes": episodes,
            "state36_layout": "Robot15 followed by Object21",
            "controller_input": "44D test contract",
        }
        (state_root / "dataset_manifest.json").write_text(json.dumps(manifest))
        archive = root / f"state36_shard_{shard_index:03d}.tar"
        with tarfile.open(archive, "w") as bundle:
            for path in state_root.iterdir():
                bundle.add(path, arcname=f"state36/{path.name}")
        return archive

    def test_materializes_exact_training_splits_from_eight_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = [self._write_archive(root, index) for index in range(8)]
            output = root / "release"

            materialize_state36_release(
                {
                    "state36_archives": [str(path) for path in archives],
                    "output": str(output),
                }
            )

            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["format"], DERIVED_FORMAT)
            self.assertEqual(
                {name: split["episodes"] for name, split in manifest["splits"].items()},
                {"train": 720, "validation": 90, "test": 90},
            )
            with np.load(output / "train.npz", allow_pickle=False) as train:
                self.assertEqual(len(train["reset_seed"]), 720)
                self.assertEqual(train["episode_offsets"][-1], 720)
                self.assertEqual(len(set(train["reset_seed"].tolist())), 720)


if __name__ == "__main__":
    unittest.main()
