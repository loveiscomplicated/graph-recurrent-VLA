"""Regression checks for the paired collection's range and receipt contracts."""

import json
from pathlib import Path
import tempfile
import unittest

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_cache import load_collection
from data_collection.five_mode_collection import collection_ranges, resume_collection
from data_collection.storage import write_json


class FiveModeCollectionTest(unittest.TestCase):
    def test_resume_recollects_only_a_missing_or_damaged_range(self):
        config = {"batch_seed_count": 2, "record_range": [4, 8]}
        batches = collection_ranges(config, 10)
        protocol = {"records": [{} for _ in range(10)]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            manifest, pending = resume_collection(target, config, protocol, batches)
            self.assertEqual(pending, [(4, 6), (6, 8)])
            for begin, end in batches:
                path = target / f"episodes_{begin:05d}_{end:05d}.tar.zst"
                path.write_bytes(bytes([begin]))
                manifest["archives"].append({
                    "file": path.name, "episodes": 10, "sha256": file_sha256(path),
                    "record_range": [begin, end],
                })
            manifest.update(status="COMPLETE", episodes=20)
            write_json(target / "manifest.json", manifest)
            self.assertEqual(load_collection(target), (manifest, protocol))

            path.write_bytes(b"corrupt")
            recovered, pending = resume_collection(target, config, protocol, batches)
            self.assertEqual(recovered["episodes"], 10)
            self.assertEqual(pending, [(6, 8)])

    def test_cache_requires_contiguous_assigned_range(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            protocol_path = target / "protocol.json"
            protocol_path.write_text(json.dumps({"records": [{}, {}, {}]}))
            manifest = {
                "status": "COMPLETE", "protocol_sha256": file_sha256(protocol_path),
                "effective_config": {"record_range": [0, 3]}, "episodes": 15,
                "archives": [
                    {"record_range": [1, 3], "episodes": 10,
                     "file": "episodes_00001_00003.tar.zst"},
                ],
            }
            write_json(target / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Unexpected collection archive"):
                load_collection(target)
