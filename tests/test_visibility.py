import unittest
from dataclasses import replace
import numpy as np
from data_collection.visibility import evaluate, CameraObservation


class VisibilityTests(unittest.TestCase):
    def setup_case(self, center=(0, 0, 1)):
        K = np.array([[100, 0, 64], [0, 100, 64], [0, 0, 1]], np.float32)
        E = np.c_[np.eye(3), np.zeros(3)].astype(np.float32)
        gl = np.diag([1, -1, -1, 1]).astype(np.float32)
        tcp = np.array([0, 0, 0.5, 1, 0, 0, 0], np.float32)
        c = CameraObservation(
            "synthetic",
            0,
            "base_camera",
            0.0,
            np.zeros((128, 128, 3), np.uint8),
            np.full((128, 128), 6000, np.int16),
            K,
            E,
            gl,
            tcp,
        )
        p = np.array([*center, 1, 0, 0, 0], np.float32)
        half = np.array([0.2, 0.04, 0.04])
        _, ev = evaluate(c, p, half, tcp)
        depth = np.full((128, 128), 6000, np.int16)
        m = ev["expected_mask"]
        depth[m] = np.floor(ev["expected_depth"][m] * 1000).astype(np.int16)
        return replace(c, depth_raw_mm=depth), p, half, tcp

    def test_complete_camera_facing_surface_passes(self):
        args = self.setup_case()
        r, e = evaluate(*args)
        self.assertTrue(r["requested_pass"])
        self.assertTrue(r["strict_100_pass"])
        self.assertTrue(all(x["pixels"] > 0 for x in r["regions"].values()))

    def test_cropped_end_fails(self):
        r, _ = evaluate(*self.setup_case((0.58, 0, 1)))
        self.assertFalse(r["full_in_frame"])
        self.assertFalse(r["requested_pass"])

    def test_grasp_occluder_fails_even_when_ends_visible(self):
        c, p, h, tcp = self.setup_case()
        _, ev = evaluate(c, p, h, tcp)
        depth = c.depth_raw_mm.copy()
        depth[ev["grasp"]] -= 30
        r, _ = evaluate(replace(c, depth_raw_mm=depth), p, h, tcp)
        self.assertFalse(r["requested_pass"])
        self.assertIn("grasp_occluded", r["reasons"])
        self.assertTrue(r["regions"]["red_end"]["passed"])
        self.assertTrue(r["regions"]["white_end"]["passed"])

    def test_optional_full_silhouette_gate_is_stricter(self):
        c, p, h, tcp = self.setup_case()
        _, ev = evaluate(c, p, h, tcp)
        other = ev["expected_mask"] & ~(ev["grasp"] | ev["red_end"] | ev["white_end"])
        self.assertTrue(other.any())
        depth = c.depth_raw_mm.copy()
        depth[other] -= 30
        r, _ = evaluate(replace(c, depth_raw_mm=depth), p, h, tcp)
        self.assertTrue(r["requested_pass"])
        self.assertFalse(r["strict_100_pass"])

    def test_invalid_observation_is_not_visibility(self):
        c, p, h, tcp = self.setup_case()
        _, ev = evaluate(c, p, h, tcp)
        depth = c.depth_raw_mm.copy()
        depth[ev["grasp"]] = 0
        r, _ = evaluate(replace(c, depth_raw_mm=depth), p, h, tcp)
        self.assertFalse(r["requested_pass"])
        self.assertIn("grasp_depth_unconfirmed", r["reasons"])

    def test_camera_behind_object_fails(self):
        r, _ = evaluate(*self.setup_case((0, 0, -1)))
        self.assertFalse(r["all_corners_in_front"])
        self.assertFalse(r["requested_pass"])


if __name__ == "__main__":
    unittest.main()
