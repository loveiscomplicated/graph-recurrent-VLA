"""Opt-in parity checks for the Objective-C++ RANSAC evaluation backend."""

import os
from tempfile import TemporaryDirectory
import unittest

import torch

from visual.corner_detection.backends import metal_ops
from visual.corner_detection.backends.native_ransac import NativeRansac
from visual.corner_detection.geometry import Parameters


@unittest.skipUnless(
    torch.backends.mps.is_available() and os.environ.get("TEST_NATIVE_MPS_RANSAC") == "1",
    "Set TEST_NATIVE_MPS_RANSAC=1 on a Mac with the Metal build tools",
)
class NativeMpsRansacTests(unittest.TestCase):
    def test_ordered_plane_bits_and_partial_compaction_on_an_offset_view(self):
        generator = torch.Generator().manual_seed(30)
        parameters = Parameters()
        with TemporaryDirectory() as directory, NativeRansac(directory) as native:
            for size in (33, 97, 257, 1000, 4077):
                with self.subTest(size=size):
                    points = torch.rand((size + 1, 3), generator=generator)
                    points[:size // 2, 2] = 0.8
                    points[size // 2:size * 4 // 5, 1] = 0.2
                    points = points.to("mps")[1:]
                    self.assertGreater(points.storage_offset(), 0)
                    expected = metal_ops.fit_planes(points, parameters, 42, 8, optimized=True)
                    with native.activate():
                        actual = metal_ops.fit_planes(points, parameters, 42, 8, optimized=True)
                    self.assertEqual(actual.device, points.device)
                    self.assertTrue(torch.equal(
                        actual.cpu().view(torch.int32), expected.cpu().view(torch.int32)
                    ))
            self.assertEqual(native.calls, 4)
            self.assertIsNone(metal_ops._native_ransac.get())
