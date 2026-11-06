"""Collision-defined region labels and an optional sampling-only ablation."""
from dataclasses import replace
import torch
from ..geometry import rotation, transform
from ..surface_sampling import hole_surface, task_box_bounds

REGIONS = ('robot', 'peg_head', 'peg_body', 'peg_grasp', 'hole_entry',
           'hole_inner', 'hole_outer', 'hole_exit')


def regions(g):
    """Entry is negative box X; grasp is tail-half side/cap, not a contact label.

    Entry labels cover the whole end face, not only a thin aperture rim.
    Peg head includes the leading quarter of its length and the end cap.
    All definitions are current local collision geometry, independent of actions.
    """
    p, h = g.peg_local, g.hole_local
    peg = torch.full(p.shape[:2], 2, device=p.device, dtype=torch.long)
    peg[p[..., 0] < 0] = 3
    peg[p[..., 0] >= g.peg_half_size[:, None, 0]*.5] = 1
    hole = torch.full(h.shape[:2], 6, device=h.device, dtype=torch.long)
    q = h-g.hole_offset[:, None]
    r = g.hole_radius
    inner = (((q[..., 1].abs()-r).abs() < 1e-6) & (q[..., 2].abs() <= r+1e-6)) | \
            (((q[..., 2].abs()-r).abs() < 1e-6) & (q[..., 1].abs() <= r+1e-6))
    hole[inner] = 5
    hole[torch.isclose(h[..., 0], -g.box_half_size[:, None, 0], atol=1e-6, rtol=0)] = 4
    hole[torch.isclose(h[..., 0], g.box_half_size[:, None, 0], atol=1e-6, rtol=0)] = 7
    return torch.cat((torch.zeros(len(p), 3, device=p.device, dtype=torch.long), peg, hole), 1)


def boundary_check(g):
    bounds = task_box_bounds(g.box_half_size, g.hole_radius, g.hole_offset)
    # Restore local normals, because stored normals use world axes.
    normals = g.hole_normals @ rotation(g.box_pose[:, 3:])
    def inside(q):
        return ((q[:, :, None] >= bounds[:, None, :, 0]) &
                (q[:, :, None] <= bounds[:, None, :, 1])).all(-1).any(-1)
    return dict(hole_inward_inside=bool(inside(g.hole_local-1e-5*normals).all()),
                hole_outward_outside=bool(~inside(g.hole_local+1e-5*normals).any()),
                peg_world_transform_max_error_m=float((transform(g.peg_local, g.peg_pose)-g.peg_points).abs().max()),
                hole_world_transform_max_error_m=float((transform(g.hole_local, g.box_pose)-g.hole_points).abs().max()))


def resample(g, count=64, weights=(1, 3, 3, 1), seed=17, entry_band=None):
    """C ablation: outer/inner/entry/exit patch quotas, all 16 patches retained.

    Balanced weights (1,1,1,1) reproduce the original sample exactly at 64.
    Inner and entry quotas improve coverage; no new geometry information.
    This is not a uniform-area sampler or an optimized sampling policy.
    """
    if count < 16 or len(weights) != 4 or min(weights) <= 0 or (entry_band is not None and entry_band <= 0):
        raise ValueError('positive four region weights and >=16 hole points required')
    # Allocate whole rounds over four patches; distribute remainder by index.
    budget = count-16
    w = torch.tensor(weights, dtype=torch.float64)
    raw = budget*w/w.sum()
    quotas = raw.floor().long()+4
    for i in torch.argsort(raw-raw.floor(), descending=True, stable=True)[:count-int(quotas.sum())]:
        quotas[i] += 1
    dense_count = int(quotas.max().item()+3)//4*16
    points, normals = hole_surface(g.box_half_size, g.hole_radius, g.hole_offset, dense_count, seed)
    chosen = []
    for region, quota in enumerate(quotas):
        candidates = [i for i in range(dense_count) if (i % 16)//4 == region]
        chosen.extend(candidates[:int(quota)])
    chosen.sort()
    p, n = points[:, chosen], normals[:, chosen]
    if entry_band is not None:
        # Retain one broad point on each entry patch, place remaining entry
        # points on the exposed end face within a band around the aperture.
        generator = torch.Generator().manual_seed(seed+1)
        uv = torch.rand(count, 2, generator=generator).to(p.device)
        seen = set()
        y, z = g.box_half_size[:, 1], g.box_half_size[:, 2]
        r = g.hole_radius[:, 0]; cy, cz = g.hole_offset[:, 1], g.hole_offset[:, 2]
        yl, yh, zl, zh = cy-r, cy+r, cz-r, cz+r
        for index, dense_index in enumerate(chosen):
            patch = dense_index % 16
            if patch not in (8, 9, 10, 11): continue
            if patch not in seen:
                seen.add(patch); continue
            if patch == 8: low=(torch.maximum(-y, yl-entry_band),torch.maximum(-z,zl-entry_band)); high=(yl,torch.minimum(z,zh+entry_band))
            elif patch == 9: low=(yh,torch.maximum(-z,zl-entry_band)); high=(torch.minimum(y,yh+entry_band),torch.minimum(z,zh+entry_band))
            elif patch == 10: low=(yl,torch.maximum(-z,zl-entry_band)); high=(yh,zl)
            else: low=(yl,zh); high=(yh,torch.minimum(z,zh+entry_band))
            for axis in (1,2): p[:, index, axis] = low[axis-1]+uv[index,axis-1]*(high[axis-1]-low[axis-1])
    return replace(g, hole_local=p, hole_points=transform(p, g.box_pose),
                   hole_normals=n @ rotation(g.box_pose[:, 3:]).transpose(-1, -2))
