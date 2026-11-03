"""Coordinate/source contract tests runnable on Torch CPU and real CUDA."""
import unittest
import numpy as np
import torch
try:
    from .masked_torch import project
except ImportError:
    from masked_torch import project


def reference(depth, masks, k):
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.indices(depth.shape)
    z = np.where(valid, depth, 0)
    xyz = np.stack(((u+.5-k[0, 2])*z/k[0, 0],
                    (v+.5-k[1, 2])*z/k[1, 1], z), axis=-1).astype(np.float32)
    rows, points, ids = [], [], []
    for oid in (0, 1):
        pixels = np.flatnonzero(masks[oid] & valid & ~(masks[0] & masks[1]))
        rows.append(pixels); points.append(xyz.reshape(-1, 3)[pixels])
        ids.append(np.full(len(pixels), oid, np.int64))
    return dict(points=np.concatenate(points), pixels=np.concatenate(rows), object_id=np.concatenate(ids))


class TorchCPU(unittest.TestCase):
    device = 'cpu'

    def test_coordinates_sources_empty_overlap_invalid_and_noncontiguous(self):
        rng = np.random.default_rng(23)
        for shape in ((1, 1), (3, 5), (29, 37)):
            for mode in ('random', 'empty', 'full', 'overlap'):
                depth = rng.uniform(.1, 12, shape).astype(np.float32)
                if depth.size >= 5:depth.flat[:5] = [0, -1, np.nan, np.inf, -np.inf]
                masks = rng.random((2, *shape)) < .3
                if mode == 'empty':masks[:] = False
                if mode == 'full':masks[0] = True; masks[1] = False
                if mode == 'overlap':masks[:] = True
                k = np.array([[171.7, 0, shape[1]/2+.13], [0, 205.3, shape[0]/2-.23], [0, 0, 1]], np.float64)
                expected = reference(depth, masks, k)
                td = torch.as_tensor(depth, device=self.device).T.contiguous().T
                tm = torch.as_tensor(masks, device=self.device).transpose(1, 2).contiguous().transpose(1, 2)
                actual = project(td, tm, torch.as_tensor(k, device=self.device))
                for name in ('pixels', 'object_id'):
                    np.testing.assert_array_equal(actual[name].cpu().numpy(), expected[name])
                np.testing.assert_allclose(actual['points'].cpu().numpy(), expected['points'], atol=1e-6, rtol=0)
                self.assertEqual(actual['points'].shape, (len(expected['pixels']), 3))

    def test_reject_invalid_shape_dtype_and_calibration(self):
        d = torch.ones((8, 9), device=self.device)
        m = torch.zeros((2, 8, 9), dtype=torch.bool, device=self.device)
        k = torch.tensor([[100., 0, 4.], [0, 100., 4.], [0, 0, 1.]], device=self.device)
        for args in ((d, m.float(), k), (d, m[:, :, :8], k), (d, m, torch.zeros_like(k))):
            with self.assertRaises(ValueError):project(*args)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA host required')
class TorchCUDA(TorchCPU):
    device = 'cuda'


if __name__ == '__main__':
    unittest.main()
