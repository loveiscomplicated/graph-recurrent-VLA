"""Saved-mask integration: source rows, dense artifacts and local run identity."""

from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np

from visual.runtime.geometry import GeometricFeatures, process_geometry
from visual.runtime.rgbd import BOX, PEG, iter_saved_rgbd
from visual.runtime.test.run_geometry import evaluate


class GeometryEvaluationTests(unittest.TestCase):
    def test_sparse_sources_and_empty_mask_preserve_dense_features_without_rgb(self):
        count, height, width = 3, 9, 11
        depth = np.ones((count, height, width), np.float32)
        depth[2] *= 2
        depth[2, 4, 4] = np.nan
        source = np.array([0, 2], np.int32)
        peg = np.zeros((2, height, width), bool)
        peg[:, 1:8, 1:6] = True
        box = np.zeros_like(peg)
        box[1, 1:8, 7:10] = True
        # Deliberately no simulator GT in the input.
        episode = {"camera/rgb": np.full((count, height, width, 3), 110, np.uint8),
                   "camera/depth": depth,
                   "camera/intrinsic": np.tile([[100, 0, 5.5], [0, 100, 4.5], [0, 0, 1]], (count, 1, 1)),
                   "camera/cam2world": np.tile(np.eye(4), (count, 1, 1)),
                   "tcp_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (count, 1)),
                   "frame_idx": np.array([10, 11, 12]), "timestamp": np.array([0, 0.05, 0.1])}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path, masks_path = root / "episode.npz", root / "masks.npz"
            np.savez(episode_path, **episode)
            np.savez(masks_path, source_frame=source, peg=peg, box=box)
            with patch.dict(os.environ, {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")}):
                output = evaluate(episode_path, masks_path, camera="camera", output_root=root / "out",
                                  selected_sources=[0, 2], repeats=1)
            data = json.loads((output / "geometry.json").read_text())
            self.assertEqual([r["source_frame"] for r in data["frames"]], [0, 2])
            self.assertEqual([r["frame_idx"] for r in data["frames"]], [10, 12])
            self.assertEqual(data["summary"]["box"]["no_points_frames"], 1)
            self.assertEqual(data["frames"][0]["objects"]["box"]["observation_state"], "empty_mask")
            self.assertEqual(data["timing_ms"]["total"]["count"], 2)
            self.assertEqual(data["config"]["gt_segmentation_ids"], {})
            self.assertEqual(len(data["config"]["input_sha256"]), 64)
            for frame in iter_saved_rgbd(episode_path, masks_path, camera="camera"):
                expected = process_geometry(frame)
                with np.load(output / f"f{frame.source_frame:04d}.npz", allow_pickle=False) as saved:
                    self.assertNotIn("rgb", saved)
                    self.assertEqual(int(saved["source_frame"]), frame.source_frame)
                    self.assertEqual(int(saved["frame_idx"]), frame.frame_idx)
                    np.testing.assert_array_equal(saved["xyz"], expected.geometry.xyz)
                    for name, object_id in (("peg", PEG), ("box", BOX)):
                        for field in fields(GeometricFeatures):
                            np.testing.assert_array_equal(saved[f"{name}/{field.name}"],
                                                          getattr(expected.objects[object_id], field.name))
                image = mpimg.imread(output / f"f{frame.source_frame:04d}.png")
                self.assertGreater(image.shape[0], 1000)
                self.assertGreater(float(image.std()), 0.1)
            for name in ("geometry.html", "geometry.json", "f0000.npz", "f0000.png", "f0002.npz", "f0002.png"):
                sidecar = json.loads((output / f"{name}.tracking.json").read_text())
                self.assertEqual(sidecar["run_id"], output.name)
            metadata = json.loads((root / "experiments" / output.name / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "finished")
            self.assertEqual(metadata["evaluation_device"], "cpu")
            self.assertFalse(metadata["gpu_executed"])

    def test_invalid_diagnostic_parameters_fail_before_output(self):
        for arguments in (dict(repeats=0), dict(variation_degrees=np.nan), dict(variation_degrees=181)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                evaluate("missing_episode", "missing_masks", output_root="unused", **arguments)


if __name__ == "__main__":
    unittest.main()
