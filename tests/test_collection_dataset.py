"""Dataset verification must reject resized or misaligned source frames."""

import tempfile
import json
import unittest
from pathlib import Path

import numpy as np

from data_collection.collect_dataset import array_headers, load_config, shard_command


class DatasetWorkflowTests(unittest.TestCase):
    def test_npz_header_preserves_raw_dimensions_and_signed_depth(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.npz"
            np.savez_compressed(path, **{
                "camera/rgb": np.zeros((1, 512, 512, 3), dtype=np.uint8),
                "camera/depth": np.full((1, 512, 512, 1), -32768, dtype=np.int16),
            })
            headers = array_headers(path)
            self.assertEqual(headers["camera/rgb"], ((1, 512, 512, 3), np.dtype("uint8")))
            self.assertEqual(headers["camera/depth"], ((1, 512, 512, 1), np.dtype("int16")))
            with np.load(path) as frame:
                self.assertEqual(frame["camera/depth"].min(), -32768)

    def test_object_payload_is_rejected_before_array_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.npz"
            np.savez(path, unsafe=np.array([{}], dtype=object))
            with self.assertRaisesRegex(ValueError, "object dtype"):
                array_headers(path)

    def test_config_and_command_use_bounded_independent_seed_ranges(self):
        config_path = Path(__file__).resolve().parents[1] / "data_collection/configs/visible_peg.yaml"
        config = load_config(config_path)
        command = shard_command(config, Path("new_shard"), 30, 101000)
        self.assertEqual(command[command.index("--start-seed") + 1], "101000")
        self.assertEqual(command[command.index("--max-attempts") + 1], "1000")
        self.assertIn("--require-success", command)
        self.assertIn("--save-step-gt", command)

    def test_moving_hole_settings_are_forwarded_to_workers(self):
        config_path = Path(__file__).resolve().parents[1] / "data_collection/configs/moving_hole_collection.yaml"
        config = load_config(config_path)
        command = shard_command(config, Path("new_shard"), 1, 0)
        settings = json.loads(command[command.index("--hole-motion") + 1])
        self.assertEqual(settings, config["hole_motion"])
        self.assertFalse(config["publish"])


if __name__ == "__main__":
    unittest.main()
