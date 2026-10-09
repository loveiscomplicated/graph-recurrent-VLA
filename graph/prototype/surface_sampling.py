"""Deterministic exposed surfaces of the exact task collision box union."""
import torch


def _rectangles(rects, count, seed):
    """Round-robin patch coverage; stratified UV in each patch, stable ordering."""
    if count < len(rects):
        raise ValueError(f"need at least {len(rects)} points to cover all patches")
    generator = torch.Generator().manual_seed(seed)
    uv = torch.rand(count, 2, generator=generator)
    points, normals = [], []
    for i in range(count):
        axis, plane, lo, hi, sign = rects[i % len(rects)]
        free = [j for j in range(3) if j != axis]
        # Stratify the first coordinate within this rectangle.
        ordinal = i // len(rects)
        n = (count - 1 - i % len(rects)) // len(rects) + 1
        u = (ordinal + uv[i, 0]) / n
        v = uv[i, 1]
        p = torch.zeros(plane.shape[0], 3, device=plane.device, dtype=plane.dtype)
        normal = torch.zeros_like(p)
        p[:, axis] = plane
        p[:, free[0]] = lo[:, 0] + u.to(plane.device) * (hi[:, 0]-lo[:, 0])
        p[:, free[1]] = lo[:, 1] + v.to(plane.device) * (hi[:, 1]-lo[:, 1])
        normal[:, axis] = sign
        points.append(p)
        normals.append(normal)
    return torch.stack(points, 1), torch.stack(normals, 1)


def peg_surface(half, count=64, seed=17):
    rects = []
    for axis in range(3):
        free = [j for j in range(3) if j != axis]
        for sign in (-1, 1):
            rects.append((axis, half[:, axis]*sign, -half[:, free], half[:, free], sign))
    return _rectangles(rects, count, seed)


def hole_surface(half, radius, offset, count=64, seed=17):
    """16 non-overlapping surface patches of a rectangular tube.

    Four outer walls, four INNER walls, and four rim rectangles on EACH
    x end. No internal overlaps between task primitives are sampled.
    """
    r = radius.reshape(-1)
    cy, cz = offset[:, 1], offset[:, 2]
    x, y, z = half.unbind(-1)
    yl, yh, zl, zh = cy-r, cy+r, cz-r, cz+r
    if not torch.all((yl > -y) & (yh < y) & (zl > -z) & (zh < z)):
        raise ValueError("hole must be strictly inside box")
    stack = lambda a, b: torch.stack((a, b), -1)
    rects = []
    for sign in (-1, 1):
        rects += [(1, sign*y, stack(-x, -z), stack(x, z), sign),
                  (2, sign*z, stack(-x, -y), stack(x, y), sign)]
    rects += [(1, yl, stack(-x, zl), stack(x, zh), 1),
              (1, yh, stack(-x, zl), stack(x, zh), -1),
              (2, zl, stack(-x, yl), stack(x, yh), 1),
              (2, zh, stack(-x, yl), stack(x, yh), -1)]
    for sign in (-1, 1):
        rects += [(0, sign*x, stack(-y, -z), stack(yl, z), sign),
                  (0, sign*x, stack(yh, -z), stack(y, z), sign),
                  (0, sign*x, stack(yl, -z), stack(yh, zl), sign),
                  (0, sign*x, stack(yl, zh), stack(yh, z), sign)]
    return _rectangles(rects, count, seed)


def task_box_bounds(half, radius, offset):
    """Exact four primitive bounds, in installed _build_box_with_hole order."""
    x, y, z = half.unbind(-1)
    r = radius.reshape(-1)
    cy, cz = offset[:, 1], offset[:, 2]
    s = lambda a, b, c: torch.stack((a, b, c), -1)
    return torch.stack((
        torch.stack((s(-x, cy+r, -z), s(x, y, z)), 1),
        torch.stack((s(-x, -y, -z), s(x, cy-r, z)), 1),
        torch.stack((s(-x, -y, cz+r), s(x, y, z)), 1),
        torch.stack((s(-x, -y, -z), s(x, y, cz-r)), 1)), 1)
