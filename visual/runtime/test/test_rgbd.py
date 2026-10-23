"""Frame alignment and object ID conversion in the Visual runtime output."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from visual.runtime.rgbd import BOX, PEG, attach_rgbd, iter_saved_rgbd
from visual.runtime.tracking import SAM2Tracker


def observations(camera="base_camera"):
    count, height, width = 5, 2, 3
    rows = np.arange(count)
    return {
        f"{camera}/rgb": np.broadcast_to(rows[:, None, None, None], (count, height, width, 3)).astype(np.uint8),
        f"{camera}/depth": np.broadcast_to(rows[:, None, None] / 10, (count, height, width)).astype(np.float32),
        f"{camera}/intrinsic": np.stack([np.eye(3) * (row + 1) for row in rows]).astype(np.float32),
        f"{camera}/cam2world": np.stack([np.eye(4) for _ in rows]).astype(np.float32),
        "tcp_pose": np.array([[row, 0, 0, 1, 0, 0, 0] for row in rows], dtype=np.float32),
        # Simulator steps intentionally differ from episode row numbers.
        "frame_idx": rows * 2 + 10,
        "timestamp": rows * 0.1 + 0.5,
    }


class TrackedRGBDTest(unittest.TestCase):
    def test_tracker_iterator_attaches_selected_camera_and_remaps_ids(self):
        class Predictor:
            def propagate_in_video(self, state, start_frame_idx):
                self.start_frame = start_frame_idx
                # The predictor can return IDs in any order.
                logits = torch.full((2, 1, 2, 3), -1.0)
                logits[0, 0, 1, 2] = 1.0  # box
                logits[1, 0, 0, 1] = 1.0  # peg
                yield start_frame_idx, [2, 1], logits

        episode = observations("hand_camera")
        episode["hand_camera/cam2world"][4, :3, 3] = [1, 2, 3]
        # Invalid depth is preserved, and metre depth must not be scaled again.
        episode["hand_camera/depth"][4, 0, 0] = np.nan
        predictor = Predictor()
        tracker = SAM2Tracker(predictor=predictor)
        frame, = attach_rgbd(tracker.track({}, start_frame=1), episode,
                             camera="hand_camera", source_frames=[1, 4])
        self.assertEqual(predictor.start_frame, 1)
        self.assertEqual((frame.source_frame, frame.frame_idx, frame.camera), (4, 18, "hand_camera"))
        self.assertEqual(frame.timestamp, episode["timestamp"][4])
        for name in ("rgb", "depth", "intrinsic", "cam2world"):
            np.testing.assert_array_equal(getattr(frame, name), episode[f"hand_camera/{name}"][4])
        np.testing.assert_array_equal(frame.tcp_pose, episode["tcp_pose"][4])
        self.assertEqual(set(frame.masks), {0, 1})
        self.assertTrue(frame.masks[PEG][0, 1])
        self.assertTrue(frame.masks[BOX][1, 2])
        self.assertFalse(frame.masks[PEG][1, 2])

    def test_saved_sparse_masks_follow_source_rows_including_empty_masks(self):
        episode = observations()
        # No GT segmentation, poses, sizes, or simulator IDs are supplied.
        source = np.array([4, 1, 3], dtype=np.int32)
        peg = np.zeros((3, 2, 3), dtype=bool)
        box = np.zeros_like(peg)
        peg[0, 0, 2] = True
        box[2, 1, 0] = True
        with tempfile.TemporaryDirectory() as root:
            episode_path = Path(root) / "episode.npz"
            masks_path = Path(root) / "masks.npz"
            np.savez(episode_path, **episode)
            np.savez(masks_path, source_frame=source, peg=peg, box=box)
            frames = list(iter_saved_rgbd(episode_path, masks_path))
        for row, frame in enumerate(frames):
            src = source[row]
            self.assertEqual(frame.source_frame, src)
            for name in ("rgb", "depth", "intrinsic", "cam2world"):
                np.testing.assert_array_equal(getattr(frame, name), episode[f"base_camera/{name}"][src])
            np.testing.assert_array_equal(frame.tcp_pose, episode["tcp_pose"][src])
            np.testing.assert_array_equal(frame.masks[PEG], peg[row])
            np.testing.assert_array_equal(frame.masks[BOX], box[row])
        self.assertFalse(frames[1].masks[PEG].any())
        self.assertFalse(frames[1].masks[BOX].any())

    def test_missing_object_stays_empty_and_timing_metadata_is_optional(self):
        episode = observations()
        del episode["frame_idx"], episode["timestamp"]
        peg = np.ones((2, 3), dtype=bool)
        frame, = attach_rgbd([(2, {1: peg})], episode)
        self.assertEqual(frame.source_frame, 2)
        self.assertIsNone(frame.frame_idx)
        self.assertIsNone(frame.timestamp)
        np.testing.assert_array_equal(frame.masks[PEG], peg)
        self.assertFalse(frame.masks[BOX].any())

    def test_misaligned_or_incomplete_observations_raise(self):
        for key, value in (("base_camera/depth", np.zeros((4, 2, 3), dtype=np.float32)),
                           ("base_camera/intrinsic", np.eye(3)),
                           ("tcp_pose", np.zeros((4, 7))),
                           ("timestamp", np.zeros(4))):
            with self.subTest(key=key):
                episode = observations()
                episode[key] = value
                with self.assertRaisesRegex(ValueError, "must have shape"):
                    list(attach_rgbd([(0, {})], episode))
        episode = observations()
        del episode["base_camera/cam2world"]
        with self.assertRaisesRegex(ValueError, "missing required observations"):
            list(attach_rgbd([(0, {})], episode))
        episode = observations()
        episode["base_camera/depth"] = np.ones((5, 2, 3), dtype=np.int16)
        with self.assertRaisesRegex(ValueError, "metres"):
            list(attach_rgbd([(0, {})], episode))

    def test_invalid_source_or_tracker_indices_raise(self):
        for source in ([-1], [5], [1.5], [[1]]):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "source_frame"):
                list(attach_rgbd([(0, {})], observations(), source_frames=source))
        for local_frame in (-1, 5, 0.5):
            with self.subTest(local_frame=local_frame), self.assertRaisesRegex(ValueError, "tracker frame"):
                list(attach_rgbd([(local_frame, {})], observations()))
        with self.assertRaisesRegex(ValueError, "tracker frame"):
            list(attach_rgbd([(1, {})], observations(), source_frames=[4]))

    def test_unknown_id_and_wrong_mask_shape_or_dtype_raise(self):
        for masks in ({3: np.ones((2, 3), dtype=bool)},
                      {1: np.ones((3, 2), dtype=bool)},
                      {2: np.ones((2, 3), dtype=np.float32)}):
            with self.subTest(masks=masks), self.assertRaisesRegex(ValueError, "tracker"):
                list(attach_rgbd([(0, masks)], observations()))

    def test_saved_masks_require_source_indices_and_matching_counts(self):
        with tempfile.TemporaryDirectory() as root:
            episode_path = Path(root) / "episode.npz"
            masks_path = Path(root) / "masks.npz"
            np.savez(episode_path, **observations())
            cases = [
                {"peg": np.zeros((1, 2, 3), bool), "box": np.zeros((1, 2, 3), bool)},
                {"source_frame": np.array([0, 2]), "peg": np.zeros((1, 2, 3), bool), "box": np.zeros((2, 2, 3), bool)},
            ]
            for saved in cases:
                with self.subTest(keys=list(saved)):
                    np.savez(masks_path, **saved)
                    with self.assertRaisesRegex(ValueError, "missing required arrays|len\\(source_frame\\)"):
                        list(iter_saved_rgbd(episode_path, masks_path))


if __name__ == "__main__":
    unittest.main()
