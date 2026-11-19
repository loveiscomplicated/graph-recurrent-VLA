"""Contracts for distributed Surface-Mamba collection and release assembly."""

import json
from pathlib import Path
import tempfile
import unittest

from data_collection.episode_dataset import file_sha256
from data_collection.surface_mamba_dataset import (
    EXPECTED_BUCKETS,
    SHARD_PACKAGE_FORMAT,
    _partition_buckets,
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
    def test_seven_shards_cover_each_quota_and_attempt_range_once(self):
        partitions = [
            _partition_buckets(GLOBAL_BUCKETS, {"index": index, "count": 7})
            for index in range(7)
        ]

        self.assertEqual(
            [sum(bucket["count"] for bucket in shard) for shard in partitions],
            [130, 130, 130, 128, 128, 128, 126],
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
            GLOBAL_BUCKETS, {"index": shard_index, "count": 7}
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
            "shard": {"index": shard_index, "count": 7},
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
            indexes = [self._write_package(root, index) for index in range(7)]
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

    def test_merge_rejects_duplicate_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            indexes = [self._write_package(root, index) for index in range(7)]
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


if __name__ == "__main__":
    unittest.main()
