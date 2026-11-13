"""Device-resident geometry for the independent 512 corner detector.

No GT/simulator/model imports. OpenCV topology is handled by detector;
all dense depth, RANSAC, plane masks, line fits and depth validation live here.
No silent device fallback. MPS uses float32 (it cannot represent float64).
"""

from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F


def resolve_device(device="auto"):
    if str(device) == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type not in ("cuda", "mps", "cpu"):
        raise ValueError("device must be cpu, cuda[:index], mps or auto")
    # Canonicalize unspecified GPU indices: tensor.device reports mps:0 /
    # cuda:0 while torch.device('mps') / ('cuda') compares unequal to those.
    return torch.empty(0, device=device).device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@lru_cache(maxsize=8)
def image_grid(device):
    y, x = torch.meshgrid(
        torch.arange(512, device=device, dtype=torch.float32),
        torch.arange(512, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return x, y


def unproject(depth, k):
    if depth.device.type == "mps":
        from . import metal_ops

        return metal_ops.unproject(depth, k)
    x, y = image_grid(depth.device)
    # Original probe uses pixel centres at integer u/v, without the +.5 used
    # by another live projector. Sharing that API would change this contract.
    return torch.stack(
        ((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth), -1
    )


def rays(uv, k):
    return torch.cat(
        (
            (uv - k[:2, 2]) / torch.stack((k[0, 0], k[1, 1])),
            torch.ones_like(uv[..., :1]),
        ),
        -1,
    )


def project(points, k):
    return (
        points[..., :2] / points[..., 2:3] * torch.stack((k[0, 0], k[1, 1])) + k[:2, 2]
    )


def plane_fit(points, selected=None):
    if selected is None:
        center = points.mean(0)
    else:
        weight = selected.to(points.dtype)
        center = (points * weight[:, None]).sum(0) / weight.sum().clamp_min(1)
    # The 3x3 covariance avoids the large rectangular SVD and is supported by
    # CPU/CUDA/MPS. Center first to preserve millimetre precision in float32.
    centered = points - center
    covariance = (
        centered.T @ centered
        if selected is None
        else centered.T @ (centered * weight[:, None])
    )
    _, vectors = torch.linalg.eigh(covariance)
    normal = vectors[:, 0]
    return torch.cat((normal, -(normal @ center)[None]))


def fit_planes(points, parameters, seed, maximum):
    if points.device.type == "mps":
        from . import metal_ops

        return metal_ops.fit_planes(points, parameters, seed, maximum)
    remaining = torch.arange(len(points), device=points.device)
    result = []
    rng = np.random.default_rng(seed)
    for _ in range(maximum):
        if len(remaining) < parameters.minimum_plane_pixels:
            break
        # Small deterministic index schedules preserve the frozen CPU seed and
        # do not copy the point cloud to the host.
        sample_index = torch.as_tensor(
            rng.choice(len(remaining), min(len(remaining), 1300), replace=False),
            device=points.device,
        )
        sample = remaining[sample_index]
        triples = torch.as_tensor(
            rng.choice(len(sample), (parameters.ransac_trials, 3), replace=True),
            device=points.device,
        )
        a, b, c = points[sample[triples]].unbind(1)
        normal = torch.linalg.cross(b - a, c - a)
        lengths = torch.linalg.vector_norm(normal, dim=1)
        normal = normal / lengths.clamp_min(1e-7)[:, None]
        offsets = -(normal * a).sum(1)
        residual = (points[sample] @ normal.T + offsets).abs()
        counts = (
            (residual < parameters.plane_tolerance_m) & (lengths > 1e-7)[None]
        ).sum(0)
        winner = counts.argmax()
        current = points[remaining]
        selected = (
            current @ normal[winner] + offsets[winner]
        ).abs() < parameters.plane_tolerance_m
        sufficient = selected.sum() >= parameters.minimum_plane_pixels
        for _ in range(3):
            # Fixed-shape weighted reductions avoid a GPU -> CPU size sync for
            # every dynamic compaction in each refinement iteration.
            plane = plane_fit(current, selected)
            selected = (
                (current @ plane[:3] + plane[3]).abs() < parameters.plane_tolerance_m
            ) & sufficient
            sufficient = sufficient & (
                selected.sum() >= parameters.minimum_plane_pixels
            )
        if not bool(sufficient):
            break
        plane = plane_fit(current, selected)
        result.append(plane)
        remaining = remaining[~selected]
    return torch.stack(result) if result else points.new_empty((0, 4))


def dilate(mask, size=3):
    return F.max_pool2d(mask.float(), size, 1, size // 2) > 0.5


def erode(mask, size=3):
    return -F.max_pool2d(-mask.float(), size, 1, size // 2) > 0.5


def plane_masks(xyz, support, planes, tolerance):
    if not len(planes):
        return support.new_empty((0, 512, 512)), support.new_empty((0, 512, 512))
    if xyz.device.type == "mps":
        from . import metal_ops

        return metal_ops.plane_masks(xyz, support, planes, tolerance)
    pixels = support.flatten().nonzero()[:, 0]
    points = xyz.reshape(-1, 3)[pixels]
    distances = (points @ planes[:, :3].T + planes[:, 3]).abs()
    labels = distances.argmin(1)
    masks = torch.zeros((len(planes), 512 * 512), dtype=torch.bool, device=xyz.device)
    ids = torch.arange(len(planes), device=xyz.device)[:, None]
    masks[:, pixels] = (labels[None] == ids) & (distances.T < tolerance)
    masks = erode(dilate(masks.reshape(-1, 512, 512)))
    return masks, erode(masks)


def fit_lines(parts, device):
    """Fit all contour legs in one batch, including the original robust refit."""
    if not parts:
        return np.empty((0, 4))
    # Preserve this batched reduction's residual ranking. A sequential fused
    # line sum can reorder near-tied candidates in the later greedy merge.
    width = max(map(len, parts))
    packed = np.zeros((len(parts), width, 2), np.float32)
    valid = np.zeros((len(parts), width), bool)
    for i, points in enumerate(parts):
        packed[i, : len(points)] = points
        valid[i, : len(points)] = True
    points = torch.as_tensor(packed, device=device)
    valid = torch.as_tensor(valid, device=device)

    def fit(weight):
        count = weight.sum(1).clamp_min(1)
        center = (points * weight[..., None]).sum(1) / count[:, None]
        delta = points - center[:, None]
        cov = delta.transpose(1, 2) @ (delta * weight[..., None])
        angle = 0.5 * torch.atan2(2 * cov[:, 0, 1], cov[:, 0, 0] - cov[:, 1, 1])
        normal = torch.stack((-angle.sin(), angle.cos()), -1)
        residual = (delta * normal[:, None]).sum(-1).abs()
        return center, normal, residual

    center, normal, residual = fit(valid)
    keep = valid & (residual < 1.3)
    refit = (keep.sum(1) >= 5) & (keep.sum(1) != valid.sum(1))
    weight = torch.where(refit[:, None], keep, valid)
    center, normal, residual = fit(weight)
    rms = ((residual.square() * weight).sum(1) / weight.sum(1).clamp_min(1)).sqrt()
    return (
        torch.cat((normal, -(normal * center).sum(1)[:, None], rms[:, None]), 1)
        .cpu()
        .numpy()
    )


def sample_patch(image, uv, radius, floor=False):
    center = torch.floor(uv) if floor else torch.round(uv)
    t = torch.arange(-radius, radius + 1, device=uv.device)
    y, x = torch.meshgrid(t, t, indexing="ij")
    pix = center.to(torch.int64)[:, None, None] + torch.stack((x, y), -1)
    inside = ((pix >= 0) & (pix < 512)).all(-1)
    pix = pix.clamp(0, 511)
    return image[pix[..., 1], pix[..., 0]], inside, pix


def visible(points, uv, support, xyz):
    if points.device.type == "mps":
        from . import metal_ops

        return metal_ops.visible(points, uv, support, xyz)
    observed, inside, pix = sample_patch(support, uv, 2)
    cloud = xyz[pix[..., 1], pix[..., 0]]
    footprint = (pix.to(uv.dtype) - uv[:, None, None]).square().sum(-1) <= 1.5**2
    bounds = ((torch.round(uv) >= 2) & (torch.round(uv) < 510)).all(1)
    return bounds & (
        observed
        & inside
        & footprint
        & ((cloud - points[:, None, None]).square().sum(-1) < 0.004**2)
    ).any((1, 2))


def unoccluded(points, uv, depth, valid):
    if points.device.type == "mps":
        from . import metal_ops

        return metal_ops.unoccluded(points, uv, depth, valid)
    pix = torch.floor(uv).to(torch.int64)
    shifts = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=uv.device)
    pix = pix[:, None] + shifts
    bounds = ((pix >= 1) & (pix <= 510)).all((1, 2))
    pix = pix.clamp(0, 511)
    foreground = valid[pix[..., 1], pix[..., 0]] & (
        depth[pix[..., 1], pix[..., 0]] < points[:, 2:3] - 0.020
    )
    return bounds & (foreground.sum(1) < 2)


def snap(points, primary, planes, eligible):
    if len(planes) < 2 or not len(points):
        return points, torch.ones(len(points), dtype=torch.int64, device=points.device)
    if points.device.type == "mps":
        from . import metal_ops

        return metal_ops.snap(points, primary, planes, eligible)
    normal, offset = planes[:, :3], planes[:, 3]
    distance = (points @ normal.T + offset).abs()
    first = normal[primary]
    possible = (distance <= 0.004) & eligible[None] & ((first @ normal.T).abs() < 0.25)
    second_id = torch.where(possible, distance, torch.inf).argmin(1)
    second_ok = possible.any(1)
    second = normal[second_id]
    possible3 = possible & ((second @ normal.T).abs() < 0.25) & second_ok[:, None]
    third_id = torch.where(possible3, distance, torch.inf).argmin(1)
    third_ok = possible3.any(1)
    ids = torch.stack((primary, second_id, third_id), 1)
    active = torch.stack((torch.ones_like(second_ok), second_ok, third_ok), 1)
    normals = normal[ids] * active[..., None]
    residual = ((normal[ids] * points[:, None]).sum(-1) + offset[ids]) * active
    gram = normals @ normals.transpose(1, 2) + torch.diag_embed(
        (~active).to(points.dtype)
    )
    weights = torch.linalg.solve(gram, residual[..., None])[..., 0]
    refined = points - (weights[..., None] * normals).sum(1)
    ok = (refined - points).square().sum(-1) < 0.0045**2
    return torch.where(ok[:, None], refined, points), torch.where(ok, active.sum(1), 1)


def physical_edges(
    uv, adjacent, planes, plane_ids, masks, object_mask, depth, valid, xyz, k, neutral
):
    """Validate every candidate's two legs in a single tensor batch."""
    if uv.device.type == "mps":
        from . import metal_ops

        return metal_ops.physical_edges(
            uv,
            adjacent,
            planes,
            plane_ids,
            masks,
            object_mask,
            depth,
            valid,
            xyz,
            k,
            neutral,
        )
    vector = adjacent - uv[:, None]
    length = torch.linalg.vector_norm(vector, dim=-1)
    direction = vector / length.clamp_min(1e-12)[..., None]
    side = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    fraction = torch.linspace(0, 1, 6, device=uv.device)
    along = (
        1.5
        + (torch.minimum(length * 0.75, length.new_tensor(8.0)) - 1.5)[..., None]
        * fraction
    )
    center = uv[:, None, None] + along[..., None] * direction[:, :, None]
    signs = uv.new_tensor([-1.0, 1.0])
    raw = torch.round(
        center[:, :, None]
        + 2 * signs[None, None, :, None, None] * side[:, :, None, None]
    ).to(torch.int64)
    bounds = ((raw >= 1) & (raw <= 510)).all((1, 2, 3, 4)) & (length >= 3).all(1)
    pix = raw.clamp(0, 511)
    values = masks[plane_ids[:, None, None, None], pix[..., 1], pix[..., 0]].sum(-1)
    outside_id = (values[..., 0] >= values[..., 1]).to(torch.int64)
    select = outside_id[..., None, None, None].expand(-1, -1, 1, 6, 2)
    outside = pix.gather(2, select).squeeze(2)
    inside = pix.gather(2, 1 - select).squeeze(2)
    u, v = outside.unbind(-1)
    iu, iv = inside.unbind(-1)
    observed = xyz[v, u]
    normal, offset = planes[plane_ids, :3], planes[plane_ids, 3]
    denominator = (rays(outside.to(uv.dtype), k) * normal[:, None, None]).sum(-1)
    predicted = -offset[:, None, None] / torch.where(
        denominator.abs() > 0.02, denominator, torch.nan
    )
    delta = depth[v, u] - predicted
    gap = (observed - xyz[iv, iu]).square().sum(-1)
    away = (
        (observed * normal[:, None, None]).sum(-1) + offset[:, None, None]
    ).abs() > 0.0012
    crease = (object_mask[v, u] | ~neutral[v, u]) & away & (gap < 0.012**2)
    background = ~valid[v, u] | ((delta > 0.002) & away)
    occluding = valid[v, u] & (delta < -0.002) & ~crease
    inside_ok = valid[iv, iu] & masks[plane_ids[:, None, None], iv, iu]
    counts = torch.stack(
        (((crease | background) & inside_ok).sum(-1), occluding.sum(-1)), -1
    )
    return bounds & ((counts[..., 0] >= 3) & (counts[..., 1] <= 1)).all(1), counts
