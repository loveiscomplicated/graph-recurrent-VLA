import unittest
import numpy as np
import cv2
from visual.corner_detection import detect
from visual.corner_detection.geometry import (
    unproject,
    _visible_at_corner,
    _physical_edges,
    _unoccluded_footprint,
    _face_envelopes,
)
from visual.corner_detection.evaluation import score


class DetectorTests(unittest.TestCase):
    def test_box_recovery_respects_explicit_empty_mask(self):
        from visual.corner_detection.box_recovery import recover

        hsv = np.zeros((512, 512, 3), np.uint8)
        hsv[100:250, 100:250] = (18, 118, 180)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        points, diagnostics = recover(
            rgb,
            np.full((512, 512), 1000, np.int16),
            np.eye(3),
            np.c_[np.eye(3), np.zeros(3)],
            [],
            masks=np.zeros((2, 512, 512), bool),
        )
        self.assertEqual(points, [])
        self.assertEqual(diagnostics["features"], 0)

    def test_observed_cube_fit_uses_image_geometry(self):
        from itertools import product
        from scipy.spatial.transform import Rotation
        from visual.corner_detection.box_recovery import _observed_box_corners

        k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]])
        yy, xx = np.indices((512, 512))
        rays = np.stack(((xx - 256) / 500, (yy - 256) / 500, np.ones_like(xx)), -1)
        rotation = Rotation.from_euler("xy", [15, 20], degrees=True).as_matrix()
        center = np.array([0.0, 0.0, 1.0])
        half = 0.1
        origin = (-center) @ rotation
        directions = rays @ rotation
        lower = (-half - origin) / directions
        upper = (half - origin) / directions
        enter = np.minimum(lower, upper)
        exit = np.maximum(lower, upper)
        near = enter.max(-1)
        far = exit.min(-1)
        mask = (far >= near) & (near > 0)
        depth = np.where(mask, near, 2.0)
        xyz = unproject(depth, k)
        hit = enter.argmax(-1)
        planes = []
        for axis in sorted(
            range(3), key=lambda a: -np.count_nonzero(mask & (hit == a))
        ):
            sign = 1 if origin[axis] > 0 else -1
            normal = rotation[:, axis] * sign
            position = center + sign * half * rotation[:, axis]
            planes.append((normal, -normal @ position))
        result = _observed_box_corners(
            mask,
            xyz,
            planes,
            [0, 1, 2],
            k,
            depth,
            np.ones(mask.shape, bool),
            np.zeros(mask.shape, bool),
        )
        self.assertGreaterEqual(len(result), 1)
        expected = (
            np.array(list(product((-half, half), repeat=3))) @ rotation.T + center
        )
        for uv, point, evidence, size in result:
            self.assertLess(np.linalg.norm(expected - point, axis=1).min(), 0.003)
            self.assertLess(abs(size - 0.2), 0.003)

    def test_near_aperture_cannot_create_hidden_wall_corner(self):
        depth = np.ones((512, 512))
        valid = np.ones((512, 512), bool)
        depth[256, 256:258] = 0.8
        self.assertFalse(
            _unoccluded_footprint(
                np.array([0.0, 0.0, 1.0]), np.array([256.0, 256.0]), depth, valid
            )
        )

    def test_clipped_face_does_not_complete_a_rectangle(self):
        mask = np.zeros((512, 512), np.uint8)
        mask[100:200, :150] = 1
        planes = [(np.array([0.0, 0.0, 1.0]), -1.0), (np.array([1.0, 0.0, 0.0]), -0.2)]
        self.assertEqual(
            _face_envelopes(
                mask,
                0,
                planes,
                {0, 1},
                np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]]),
            ),
            [],
        )

    def test_nearer_neighbour_does_not_hide_observed_corner(self):
        k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]])
        depth = np.full((512, 512), 0.5)
        depth[256, 256] = 1.0
        support = np.zeros((512, 512), bool)
        support[256, 256] = True
        self.assertTrue(
            _visible_at_corner(
                np.array([0.0, 0.0, 1.0]),
                np.array([256.0, 256.0]),
                support,
                unproject(depth, k),
            )
        )

    def test_fully_hidden_corner_has_no_current_support(self):
        k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]])
        self.assertFalse(
            _visible_at_corner(
                np.array([0.0, 0.0, 1.0]),
                np.array([256.0, 256.0]),
                np.ones((512, 512), bool),
                unproject(np.full((512, 512), 0.5), k),
            )
        )

    def test_same_object_self_occlusion_is_not_a_crease(self):
        k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]])
        plane_mask = np.zeros((512, 512), np.uint8)
        plane_mask[256:, 256:] = 1
        depth = np.where(plane_mask, 1.0, 0.8)
        physical, _ = _physical_edges(
            np.array([256.0, 256.0]),
            [np.array([280.0, 256.0]), np.array([256.0, 280.0])],
            np.array([0.0, 0.0, 1.0]),
            -1.0,
            plane_mask,
            np.ones((512, 512), bool),
            depth,
            np.ones((512, 512), bool),
            unproject(depth, k),
            k,
            np.zeros((512, 512), bool),
        )
        self.assertFalse(physical)

    def test_image_crop_is_not_four_object_corners(self):
        result = detect(
            np.zeros((512, 512, 3), np.uint8),
            np.full((512, 512), 1000, np.int16),
            np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]]),
            np.c_[np.eye(3), np.zeros(3)],
            masks=np.stack((np.ones((512, 512), bool), np.zeros((512, 512), bool))),
        )
        self.assertEqual(result["corners"], [])

    def test_visible_white_tail_does_not_require_orange_end(self):
        hsv = np.zeros((512, 512, 3), np.uint8)
        hsv[100:200, 120:240] = (98, 12, 220)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        depth = np.full((512, 512), 2000, np.int16)
        depth[100:200, 120:240] = 1000
        found = detect(
            rgb,
            depth,
            np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]]),
            np.c_[np.eye(3), np.zeros(3)],
        )["corners"]
        self.assertEqual(len(found), 4)
        self.assertTrue(all(c["object_id"] == 0 for c in found))

    def test_gray_surfaces_are_not_peg_material(self):
        hsv = np.zeros((512, 512, 3), np.uint8)
        hsv[100:200, 120:240] = (100, 5, 168)
        hsv[250:350, 120:240] = (110, 26, 119)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        found = detect(
            rgb,
            np.full((512, 512), 1000, np.int16),
            np.eye(3),
            np.c_[np.eye(3), np.zeros(3)],
        )["corners"]
        self.assertEqual(found, [])

    def test_planar_rectangle_has_four_subpixel_corners(self):
        rgb = np.zeros((512, 512, 3), np.uint8)
        depth = np.full((512, 512), 2000, np.int16)
        depth[100:200, 120:240] = 1000
        masks = np.zeros((2, 512, 512), bool)
        masks[0, 100:200, 120:240] = True
        k = np.array([[500.0, 0, 256], [0, 500, 256], [0, 0, 1.0]])
        ext = np.c_[np.eye(3), np.array([0.1, 0.2, 0.3])]
        found = detect(rgb, depth, k, ext, masks=masks)["corners"]
        self.assertEqual(len(found), 4)
        for corner in found:
            u, v = corner["uv"]
            self.assertLess(min(abs(u - 119.5), abs(u - 239.5)), 0.05)
            self.assertLess(min(abs(v - 99.5), abs(v - 199.5)), 0.05)
            camera = np.array([(u - 256) / 500, (v - 256) / 500, 1.0])
            np.testing.assert_allclose(corner["xyz"], camera - ext[:, 3], atol=1e-6)

    def test_invalid_depth_yields_no_corners(self):
        found = detect(
            np.zeros((512, 512, 3), np.uint8),
            np.full((512, 512), -32768, np.int16),
            np.eye(3),
            np.c_[np.eye(3), np.zeros(3)],
            masks=np.ones((2, 512, 512), bool),
        )
        self.assertEqual(found["corners"], [])

    def test_duplicate_predictions_do_not_inflate_recall(self):
        p = [dict(object_id=0, xyz=[0.0, 0.0, 1.0])] * 2
        refs = [dict(object_id=0, xyz=[0.0, 0.0, 1.0], kind="peg")]
        metrics = score(p, refs, refs)[0]
        self.assertEqual(metrics["matched_3mm"], 1)
        self.assertEqual(metrics["precision_3mm"], 0.5)
        self.assertEqual(metrics["recall_3mm"], 1.0)

    def test_native_resolution_is_enforced(self):
        with self.assertRaises(ValueError):
            detect(
                np.zeros((128, 128, 3), np.uint8),
                np.zeros((128, 128), np.int16),
                np.eye(3),
                np.c_[np.eye(3), np.zeros(3)],
            )


if __name__ == "__main__":
    unittest.main()
