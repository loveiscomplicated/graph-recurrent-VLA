import unittest

import torch

from graph.prototype.geometry import transform, inverse_transform, rotation
from graph.prototype.surface_sampling import peg_surface, hole_surface, task_box_bounds


class GeometryTests(unittest.TestCase):
    def test_rigid_transform(self):
        pose = torch.tensor([[1., 2., 3., 2**-.5, 0., 0., 2**-.5]])
        points = torch.tensor([[[1., 0., 0.], [0., 2., 0.]]])
        world = transform(points, pose)
        torch.testing.assert_close(world, torch.tensor([[[1., 3., 3.], [-1., 2., 3.]]]))
        torch.testing.assert_close(inverse_transform(world, pose), points, atol=1e-6, rtol=0)
        torch.testing.assert_close(rotation(pose[:, 3:]) @ rotation(pose[:, 3:]).transpose(-1, -2), torch.eye(3)[None], atol=1e-6, rtol=0)

    def test_surfaces_randomization_order_and_normal(self):
        half = torch.tensor([[.10, .02, .02], [.12, .024, .024]])
        points, normals = peg_surface(half)
        self.assertEqual(points.shape, (2, 64, 3))
        torch.testing.assert_close(normals.norm(dim=-1), torch.ones(2, 64))
        ratios = points / half[:, None]
        self.assertTrue(((ratios.abs()-1).abs().min(-1).values < 1e-6).all())
        torch.testing.assert_close(ratios[0], ratios[1])
        torch.testing.assert_close(points, peg_surface(half)[0])

    def test_hole_points_are_exposed_union_boundary(self):
        half = torch.tensor([[.1]*3, [.125]*3])
        r = torch.tensor([[.023], [.028]])
        offset = torch.tensor([[0., .025, -.035], [0., -.031, .041]])
        p, n = hole_surface(half, r, offset)
        bounds = task_box_bounds(half, r, offset)
        def inside(q):
            return ((q[:, :, None] >= bounds[:, None, :, 0]) &
                    (q[:, :, None] <= bounds[:, None, :, 1])).all(-1).any(-1)
        self.assertEqual(p.shape, (2, 64, 3))
        self.assertTrue(inside(p-1e-5*n).all())
        self.assertFalse(inside(p+1e-5*n).any())
        rel = p-offset[:, None]
        empty = (rel[..., 1].abs() < r-1e-6) & (rel[..., 2].abs() < r-1e-6)
        self.assertFalse(empty.any())
        torch.testing.assert_close(n.norm(dim=-1), torch.ones(2, 64))
        torch.testing.assert_close(p, hole_surface(half, r, offset)[0])
        self.assertFalse(torch.equal(p[0], p[1]))

    def test_coverage_guard(self):
        with self.assertRaises(ValueError):
            peg_surface(torch.ones(1, 3), 5)
        with self.assertRaises(ValueError):
            hole_surface(torch.ones(1, 3), torch.ones(1, 1), torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
