"""Contracts for causal inputs and the extracted graph2 Transformer."""

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data_collection.five_mode_corner_tracking import (
    read_raw_truth,
    sequence_arrays,
    training_fallback,
)
from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker


class FiveModeCornerTrackerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_extraction_matches_historical_model_exactly(self):
        """Copying the same weights must retain the existing forward computation."""
        path = (
            Path(__file__).resolve().parents[1]
            / "graph/handoff_20261118/workspace_scripts/spatiotemporal_corner_tracker_convergence/tracker.py"
        )
        spec = importlib.util.spec_from_file_location("reference_tracker", path)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference = reference_module.CornerTransformer(spatial=True).eval()
        with torch.no_grad():
            reference.output[1].weight.normal_(std=0.1)
        extracted = SpatialTemporalCornerTracker().eval()
        extracted.load_state_dict(reference.state_dict(), strict=True)
        features = torch.randn(2, 17, 20, 10)
        anchor, detected = torch.randn(2, 17, 20, 3), torch.randn(2, 17, 20, 3)
        observed = torch.rand(2, 17, 20) > 0.5
        valid = torch.ones(2, 17, dtype=torch.bool)
        valid[1, 12:] = False
        with torch.no_grad():
            expected = reference(features, anchor, detected, observed, valid)
            actual = extracted(features, anchor, detected, observed, valid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[observed], detected[observed], rtol=0, atol=0)

    def test_features_are_causal_and_do_not_read_targets(self):
        length = 14
        episode = {
            "gt": np.ones((length, 20, 3), dtype=np.float32),
            "detected": np.zeros((length, 20, 3), dtype=np.float32),
            "observed": np.zeros((length, 20), dtype=bool),
            "tcp": np.zeros((length, 3), dtype=np.float32),
            "depth_mm": np.zeros(length, dtype=np.float32),
            "mode": "test",
        }
        episode["observed"][3, 0] = True
        episode["detected"][3, 0] = [0.1, 0.2, 0.3]
        fallback = training_fallback([episode])
        original, _, _ = sequence_arrays([episode], fallback)
        changed = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in episode.items()
        }
        changed["gt"] += 100
        changed["tcp"][8:] += 10
        changed["detected"][8:] += 10
        altered, _, _ = sequence_arrays([changed], fallback)
        np.testing.assert_array_equal(
            original["features"][:, :8], altered["features"][:, :8]
        )
        np.testing.assert_array_equal(
            original["anchor"][:, :8], altered["anchor"][:, :8]
        )
        np.testing.assert_array_equal(
            original["anchor"][0, 4:8, 0],
            np.tile([0.1, 0.2, 0.3], (4, 1)).astype(np.float32),
        )
        np.testing.assert_array_equal(
            original["anchor"][0, :, 1], np.tile(fallback[1], (length, 1))
        )

    @unittest.skipUnless(shutil.which("zstd"), "zstd executable required")
    def test_raw_archive_allows_metadata_after_frames(self):
        """TAR ordering must not discard frames or bypass their checksum checks."""
        pose = np.array([[0, 0, 0.1, 1, 0, 0, 0]], dtype=np.float32)
        frame = io.BytesIO()
        np.savez(
            frame,
            **{
                "ground_truth/peg_pose": pose,
                "ground_truth/box_pose": pose,
                "ground_truth/hole_pose": pose,
                "ground_truth/peg_half_sizes": np.array([[0.1, 0.02, 0.02]]),
                "ground_truth/hole_radius": np.array([0.025]),
            },
        )
        contents = frame.getvalue()
        metadata = {"frame_sha256": [hashlib.sha256(contents).hexdigest()]}
        index = [
            {
                "seed": 9,
                "policy_mode": "Fixed_Hole_expert",
                "path": "episode",
                "observations": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "raw.tar"
            with tarfile.open(archive, "w") as target:
                members = [
                    ("episode/frame_000000.npz", contents),
                    ("episode/episode.json", json.dumps(metadata).encode()),
                    ("episode_index.json", json.dumps(index).encode()),
                ]
                for name, payload in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target.addfile(info, io.BytesIO(payload))
            compressed = archive.with_suffix(".tar.zst")
            subprocess.run(
                [shutil.which("zstd"), "-q", str(archive), "-o", str(compressed)],
                check=True,
            )
            digest = hashlib.sha256(compressed.read_bytes()).hexdigest()
            result = read_raw_truth(compressed, digest, shutil.which("zstd"))
            self.assertEqual(result[9, "Fixed_Hole_expert"][0].shape, (1, 20, 3))


if __name__ == "__main__":
    unittest.main()
