"""Exercise causal rim reuse with and without current Box depth support."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from visual.corner_detection.rim_memory import StaticRimMemory


class RimMemoryTest(unittest.TestCase):
    def test_reuses_earlier_corner_only_with_current_depth_support(self):
        memory = StaticRimMemory()
        mask = torch.zeros((2, 9, 9), dtype=torch.bool)
        mask[1, 4, 4] = True
        depth = np.zeros((9, 9), dtype=np.int16)
        intrinsic = np.array([[256, 0, 4], [0, 256, 4], [0, 0, 1]])
        extrinsic = np.eye(4)[:3]
        corner = {"xyz": [0, 0, 1], "object_id": 1, "method": "box_opening"}

        def observation(corners):
            return SimpleNamespace(
                masks=mask, as_dict=lambda: {"corners": list(corners)}
            )

        depth[4, 4] = 1000
        self.assertEqual(
            memory.update(observation([]), depth, intrinsic, extrinsic), []
        )
        memory.update(observation([corner]), depth, intrinsic, extrinsic)
        recovered = memory.update(observation([]), depth, intrinsic, extrinsic)
        self.assertEqual(recovered[0]["method"], "rim_memory")
        np.testing.assert_allclose(recovered[0]["xyz"], [0, 0, 1])
        depth[4, 4] = 950
        self.assertEqual(
            memory.update(observation([]), depth, intrinsic, extrinsic), []
        )
