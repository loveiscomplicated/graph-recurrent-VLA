"""Analytical GT error cases and the two-artifact RGB-D evaluation workflow."""

import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

from visual.runtime.rgbd import BOX, PEG, TrackedRGBDFrame
from visual.runtime.test.run_rgbd import compare_object, depth_colors, evaluate, summarize
from utils.experiment_artifacts import plan_upload


def frame(predicted, depth=None):
    height, width = predicted.shape
    return TrackedRGBDFrame(
        source_frame=0, camera="test_camera",
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.ones((height, width), dtype=np.float32) if depth is None else depth,
        intrinsic=np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float32),
        cam2world=np.eye(4, dtype=np.float32),
        tcp_pose=np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        masks={PEG: predicted, BOX: np.zeros_like(predicted)}, frame_idx=0, timestamp=0.0,
    )


class RGBDEvaluationTest(unittest.TestCase):
    def test_perfect_mask_has_unit_iou_and_zero_surface_error(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 1:3] = True
        result = compare_object(frame(mask), PEG, mask)
        for key in ("iou", "precision", "recall"):
            self.assertEqual(result[key], 1)
        for key in ("centroid_error_mm", "chamfer_mm", "predicted_to_gt_mm", "gt_to_predicted_mm"):
            self.assertEqual(result[key], 0)

    def test_shifted_pixel_has_analytical_ten_mm_error(self):
        predicted = np.zeros((3, 4), dtype=bool)
        gt = np.zeros_like(predicted)
        predicted[1, 2] = gt[1, 1] = True
        observation = frame(predicted)
        observation.cam2world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        observation.cam2world[:3, 3] = [1, 2, 3]
        result = compare_object(observation, PEG, gt)
        self.assertEqual(result["iou"], 0)
        self.assertAlmostEqual(result["chamfer_mm"], 10, places=3)
        self.assertAlmostEqual(result["centroid_error_mm"], 10, places=3)

    def test_extra_pixel_checks_chamfer_normalization_and_direction(self):
        predicted = np.zeros((3, 4), dtype=bool)
        gt = np.zeros_like(predicted)
        predicted[1, 1:3] = True
        gt[1, 1] = True
        result = compare_object(frame(predicted), PEG, gt)
        self.assertEqual(result["iou"], 0.5)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["predicted_to_gt_mm"], 5, places=4)
        self.assertEqual(result["gt_to_predicted_mm"], 0)
        self.assertAlmostEqual(result["chamfer_mm"], 2.5, places=4)
        self.assertAlmostEqual(result["centroid_error_mm"], 5, places=4)

    def test_missing_predictions_are_not_hidden_as_zero_geometry_error(self):
        empty = np.zeros((3, 4), dtype=bool)
        gt = empty.copy()
        gt[1, 1] = True
        missed = compare_object(frame(empty), PEG, gt)
        absent = compare_object(frame(empty), BOX, empty)
        self.assertEqual(missed["iou"], 0)
        self.assertEqual(missed["recall"], 0)
        self.assertIsNone(missed["precision"])
        self.assertIsNone(missed["chamfer_mm"])
        self.assertIsNone(absent["iou"])
        hallucinated = compare_object(frame(gt), PEG, empty)
        summary = summarize([
            {"objects": {"peg": missed, "box": absent}},
            {"objects": {"peg": hallucinated, "box": absent}},
        ])
        self.assertEqual(summary["peg"]["visible_iou"]["mean"], 0)
        self.assertEqual(summary["peg"]["missed_visible_frames"], 1)
        self.assertEqual(summary["peg"]["false_positive_invisible_frames"], 1)
        self.assertEqual(summary["peg"]["geometry_unavailable_visible_frames"], 1)
        self.assertEqual(summary["peg"]["chamfer_mm"]["count"], 0)
        self.assertIsNone(summary["peg"]["chamfer_mm"]["mean"])

    def test_invalid_depth_is_excluded_and_video_uses_a_fixed_scale(self):
        depth = np.array([[0, -1, np.nan, np.inf], [0.5, 0.5, 1, 2]], dtype=np.float32)
        mask = np.ones(depth.shape, dtype=bool)
        result = compare_object(frame(mask, depth), PEG, mask)
        self.assertEqual(result["predicted_depth_pixels"], 4)
        self.assertEqual(result["predicted_valid_depth_fraction"], 0.5)
        self.assertEqual(result["chamfer_mm"], 0)
        colored = depth_colors(depth, 0.2, 1.5)
        self.assertFalse(colored[0].any())
        np.testing.assert_array_equal(colored[1, 0], colored[1, 1])
        np.testing.assert_array_equal(colored[1, 0], depth_colors(np.array([[0.5]]), 0.2, 1.5)[0, 0])
        np.testing.assert_array_equal(colored[1, 3], depth_colors(np.array([[1.5]]), 0.2, 1.5)[0, 0])
        self.assertFalse(np.array_equal(colored[1, 0], colored[1, 2]))

    def test_report_and_decodable_video_follow_sparse_source_frames(self):
        count, height, width = 3, 24, 32
        segmentation = np.zeros((count, height, width), dtype=np.int16)
        segmentation[:, 5:10, 3:8] = 18
        segmentation[:, 10:16, 19:25] = 19
        source = np.array([0, 2], dtype=np.int32)
        peg = segmentation[source] == 18
        peg[1] = False  # Only the second selected frame is missed.
        rgb = np.full((count, height, width, 3), 120, dtype=np.uint8)
        episode = {
            "test_camera/rgb": rgb,
            "test_camera/depth": np.ones((count, height, width), dtype=np.float32),
            "test_camera/intrinsic": np.repeat(np.array([[[100, 0, 16], [0, 100, 12], [0, 0, 1]]], dtype=np.float32), count, axis=0),
            "test_camera/cam2world": np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0),
            "test_camera/seg": segmentation,
            "tcp_pose": np.tile([0, 0, 0, 1, 0, 0, 0], (count, 1)).astype(np.float32),
            "frame_idx": np.array([10, 11, 12]), "timestamp": np.array([0, 0.05, 0.1]),
            "peg_id": 18, "box_id": 19, "seed": 42,
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            episode_path, masks_path = root / "episode.npz", root / "masks.npz"
            np.savez(episode_path, **episode)
            np.savez(masks_path, source_frame=source, peg=peg, box=segmentation[source] == 19)
            with patch.dict(os.environ, {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")}):
                output = evaluate(episode_path, masks_path, camera="test_camera", output_root=root / "results")
            self.assertEqual({path.name for path in output.iterdir() if not path.name.endswith(".tracking.json")},
                             {"gt_report.html", "depth.mp4"})
            report = (output / "gt_report.html").read_text()
            data = json.loads(re.search(r'id="evaluation-data">(.*?)</script>', report, re.S).group(1))
            self.assertEqual([row["source_frame"] for row in data["frames"]], [0, 2])
            self.assertEqual([row["frame_idx"] for row in data["frames"]], [10, 12])
            self.assertEqual(data["gt_segmentation_ids"], {"peg": 18, "box": 19})
            self.assertEqual(data["summary"]["peg"]["visible_iou"]["mean"], 0.5)
            self.assertEqual(data["summary"]["peg"]["chamfer_mm"]["count"], 1)
            self.assertEqual(data["summary"]["box"]["visible_iou"]["mean"], 1)
            self.assertEqual(data["video_fps"], 10)
            with imageio.get_reader(output / "depth.mp4") as video:
                self.assertEqual(video.count_frames(), 2)
                self.assertEqual(video.get_meta_data()["fps"], 10)
                rendered = video.get_data(0)
                self.assertEqual(rendered.shape[-1], 3)
                self.assertGreater(rendered.std(), 10)
            for artifact in ("gt_report.html", "depth.mp4"):
                link = json.loads((output / f"{artifact}.tracking.json").read_text())
                self.assertEqual(link["run_id"], output.name)
            metadata = json.loads((root / "experiments" / output.name / "metadata.json").read_text())
            self.assertEqual(metadata["team"], "visual")
            self.assertEqual(metadata["status"], "finished")
            self.assertEqual(metadata["evaluated_frames"], 2)
            plans = plan_upload(root / "experiments" / output.name, ["reports", "rollouts", "metrics"])
            self.assertEqual(set(plans), {"rollouts"})
            self.assertTrue(any(item.path.endswith("gt_report.html") for item in plans["rollouts"]["files"]))


if __name__ == "__main__":
    unittest.main()
