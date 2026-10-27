"""Legacy FPS depth/coordinate adaptation and empty observation contracts."""

import unittest

import numpy as np

from visualization.render_pointcloud import extract_frame, legacy_module


class LegacyPointcloudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy, _ = legacy_module()

    def test_signed_mm_depth_and_world_to_cv_transform(self):
        depth = np.full((4, 4), 1000, np.int16)
        depth[0, :2] = [-32768, 0]
        extrinsic = np.column_stack((np.eye(3), [1, 2, 3])).astype(np.float32)
        sensor = {'main/depth_raw_mm': depth, 'main/intrinsic': np.eye(3, dtype=np.float32),
                  'main/extrinsic_cv': extrinsic}
        masks = {'main': np.stack((np.ones((4, 4), bool), np.zeros((4, 4), bool)))}
        clouds, stats = extract_frame(sensor, masks, 'main', np.random.default_rng(0), self.legacy.extract)
        self.assertEqual(len(clouds[0]), 14)
        # Pixel (v=0,u=2), +0.5 pixel center, 1000mm=1m, translation inverted.
        np.testing.assert_allclose(clouds[0][0], [1.5, -1.5, -2.0], atol=1e-6)
        np.testing.assert_allclose(clouds[0][:, 2], -2.0, atol=1e-6)
        self.assertEqual(clouds[1].shape, (0, 3))
        self.assertEqual(stats[1]['points'], 0)

    def test_original_fps_cap_and_reproducible_episode_seed(self):
        sensor = {'main/depth_raw_mm': np.full((70, 70), 1000, np.int16),
                  'main/intrinsic': np.diag([100, 100, 1]).astype(np.float32),
                  'main/extrinsic_cv': np.column_stack((np.eye(3), np.zeros(3))).astype(np.float32)}
        masks = {'main': np.ones((2, 70, 70), bool)}
        first, _ = extract_frame(sensor, masks, 'main', np.random.default_rng(0), self.legacy.extract)
        second, _ = extract_frame(sensor, masks, 'main', np.random.default_rng(0), self.legacy.extract)
        for a, b in zip(first, second):
            self.assertEqual(a.shape, (1024, 3))
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, b)
