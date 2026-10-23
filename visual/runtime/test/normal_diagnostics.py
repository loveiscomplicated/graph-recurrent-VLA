"""Task 02-A evaluation helpers. No runtime estimator or predicted classes.

Analytic geometry is evaluation truth only. Actual features always come from the
unchanged Task 01 extractor and recorded depth/tracker masks. Raster subsampling
is a diagnostic scale intervention without averaging or a replacement estimator.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation

from visual.runtime.geometry import process_geometry
from visual.runtime.rgbd import PEG, TrackedRGBDFrame

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
THRESHOLDS = (5, 10, 20, 30)


def erode(mask, radius):
    return binary_erosion(mask, CROSS, iterations=radius, border_value=0) if radius else mask.copy()


def dilate(mask, radius):
    return binary_dilation(mask, CROSS, iterations=radius) if radius else mask.copy()


def pairs():
    yield (slice(None), slice(None, -1)), (slice(None), slice(1, None))
    yield (slice(None, -1), slice(None)), (slice(1, None), slice(None))


def endpoints(mask, other):
    """Pixels in mask with an in-image four-neighbour in other."""
    result = np.zeros_like(mask)
    for a, b in pairs():
        result[a] |= mask[a] & other[b]
        result[b] |= mask[b] & other[a]
    return result


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, max=None)
    return dict(count=int(a.size), mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)), max=float(a.max()))


def angular_error(normals, truth):
    """Oriented angle in degrees, stable for identical rounded unit normals."""
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(normals, truth), axis=-1),
                                np.sum(normals * truth, axis=-1)))


def metrics(feature, region, truth=None):
    points = region & feature.valid_points
    nv = points & feature.normal_valid
    vv = points & feature.normal_variation_valid
    variation = np.rad2deg(feature.normal_variation_rad[vv])
    n, nn, vn = int(points.sum()), int(nv.sum()), int(vv.sum())
    row = dict(region_pixels=int(region.sum()), points=n, normals=nn, variation_valid=vn,
               observed_fraction_of_region=n/int(region.sum()) if region.any() else None,
               normal_fraction_of_region=nn/int(region.sum()) if region.any() else None,
               normal_valid_fraction=nn/n if n else None,
               variation_valid_fraction=vn/n if n else None, variation_deg=stats(variation),
               variation_hist_5deg=np.histogram(variation, bins=np.arange(0, 185, 5))[0].tolist(),
               depth_discontinuity=int((points & feature.depth_discontinuity).sum()),
               mask_boundary=int((points & feature.mask_boundary).sum()),
               invalid_depth_boundary=int((points & feature.depth_validity_boundary).sum()),
               incomplete_normal_neighbours=int((nv & (feature.normal_neighbor_count < 4)).sum()))
    response = points & (feature.depth_discontinuity |
                         (feature.normal_variation_valid & (np.rad2deg(feature.normal_variation_rad) > 20)))
    row['jump_or_high20_count'] = int(response.sum())
    row['jump_or_high20_fraction_of_region'] = int(response.sum())/int(region.sum()) if region.any() else None
    for t in THRESHOLDS:
        count = int(np.count_nonzero(variation > t))
        row[f'high_{t}_count'] = count
        row[f'high_{t}_fraction_of_valid'] = count/vn if vn else None
        row[f'high_{t}_fraction_of_points'] = count/n if n else None
    if truth is not None:
        row['angular_error_deg'] = stats(angular_error(feature.normals[nv], truth[nv]))
    return row


def rays(shape, intrinsic):
    v, u = np.indices(shape, dtype=float)
    return np.stack(((u+.5-intrinsic[0, 2])/intrinsic[0, 0],
                     (v+.5-intrinsic[1, 2])/intrinsic[1, 1], np.ones(shape)), axis=-1)


def synthetic_frame(depth, intrinsic, mask):
    return TrackedRGBDFrame(0, 'synthetic', np.zeros((*depth.shape, 3), np.uint8),
                            np.asarray(depth, np.float32), np.asarray(intrinsic), np.eye(4),
                            np.array([0, 0, 0, 1, 0, 0, 0.]), {PEG: mask.copy()}, 0, 0.)


def quantize(depth, mode='floor'):
    # Use float64 arithmetic on the same unquantized float32 input in A and B.
    scaled = np.asarray(depth, np.float64) * 1000
    return ((np.floor(scaled) if mode == 'floor' else np.rint(scaled)) / 1000).astype(np.float32)


def synthetic_case(kind, intrinsic, shape=(512, 512), z0=.25, phase_m=0.):
    ray = rays(shape, intrinsic)
    x, y = ray[..., 0], ray[..., 1]
    a = np.zeros(shape)
    b = np.zeros(shape)
    if kind == 'tilted':
        a.fill(.4); b.fill(-.25)
    elif kind == 'crease':
        a[:] = np.where(x < 0, -.6, .6); b.fill(.15)
    elif kind not in ('front', 'step'):
        raise ValueError('unknown synthetic geometry')
    depth = (z0 + phase_m) / (1 - a*x - b*y)
    if kind == 'step':
        depth = depth + np.where(x >= 0, .05, 0.)
    truth = np.stack((a, b, -np.ones(shape)), axis=-1)
    truth /= np.linalg.norm(truth, axis=-1, keepdims=True)
    h, w = shape
    mask = np.zeros(shape, bool)
    mask[h//4:3*h//4, w//4:3*w//4] = True
    inner = erode(mask, 3)  # Variation's complete dependency has Manhattan radius 2.
    edge = inner & (np.abs(np.indices(shape)[1] + .5 - intrinsic[0, 2]) <= 3)
    planar = inner & ~edge if kind in ('crease', 'step') else inner
    return synthetic_frame(depth, intrinsic, mask), truth, {'plane': planar, 'edge_band': edge}


def subframe(frame, stride):
    """Exact lattice [::s,::s], preserving pixel-centre CV projection, no filter."""
    k = frame.intrinsic.astype(float).copy()
    k[0, 0] /= stride; k[1, 1] /= stride
    k[0, 2] = (k[0, 2]-.5)/stride+.5
    k[1, 2] = (k[1, 2]-.5)/stride+.5
    return replace(frame, depth=frame.depth[::stride, ::stride], rgb=frame.rgb[::stride, ::stride],
                   masks={i: m[::stride, ::stride] for i, m in frame.masks.items()}, intrinsic=k)


def scale_probe(frame, baseline, object_id, plane_base, truth):
    """Compare each stride with stride 1 on IDENTICAL centres and safe flat support.

    Larger strides change both stencil span and normal-comparison separation.
    Only analytic planar ROIs are used, so the ideal variation remains zero.
    """
    common = erode(plane_base, 8)  # Covers radius 2*s for the largest stride, s=4.
    rows = []
    for stride in (1, 2, 4):
        sub = subframe(frame, stride)
        feature = baseline if stride == 1 else process_geometry(sub).objects[object_id]
        roi = common[::stride, ::stride]
        t = truth[::stride, ::stride]
        base_region = np.zeros_like(common)
        base_region[::stride, ::stride] = roi
        rows.append(dict(stride_px=stride, normal_span_px=2*stride,
                         sample_origin_uv=[0, 0],
                         baseline_same_centres=metrics(baseline, base_region, truth),
                         larger_spacing=metrics(feature, roi, t)))
    return rows


def pose_rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()


def ray_box(origin, direction, centre, half):
    """First positive ray/AABB hit, with CV-z parameter and signed face ID."""
    delta_lo, delta_hi = centre-half-origin, centre+half-origin
    moving = np.abs(direction) > 1e-14
    lo = np.divide(delta_lo, direction, out=np.full_like(direction, -np.inf), where=moving)
    hi = np.divide(delta_hi, direction, out=np.full_like(direction, np.inf), where=moving)
    near_axis, far_axis = np.minimum(lo, hi), np.maximum(lo, hi)
    parallel_outside = (~moving) & ((origin < centre-half) | (origin > centre+half))
    near, far = near_axis.max(-1), far_axis.min(-1)
    hit = (near > 0) & (near <= far) & ~parallel_outside.any(-1)
    axis = near_axis.argmax(-1)
    chosen_dir = np.take_along_axis(direction, axis[..., None], -1)[..., 0]
    face = axis*2 + (chosen_dir < 0).astype(int)  # 0=-x, 1=+x, etc.
    return np.where(hit, near, np.nan), np.where(hit, face, -1)


def oracle(frame, pose, boxes):
    """Analytic cuboid-union truth, never passed to actual feature extraction."""
    rotation = pose_rotation(pose)
    cv_to_local = frame.cam2world[:3, :3].T @ rotation
    origin = (frame.cam2world[:3, 3]-pose[:3]) @ rotation
    direction = rays(frame.depth.shape, frame.intrinsic) @ cv_to_local
    depth = np.full(frame.depth.shape, np.nan)
    face = np.full(frame.depth.shape, -1, np.int8)
    for centre, half in boxes:
        z, f = ray_box(origin, direction, np.asarray(centre), np.asarray(half))
        use = np.isfinite(z) & (~np.isfinite(depth) | (z < depth))
        depth[use], face[use] = z[use], f[use]
    local = origin + np.nan_to_num(depth)[..., None]*direction
    axes = np.eye(3)[np.maximum(face, 0)//2]
    local_n = axes * np.where(face % 2, 1., -1.)[..., None]
    normals = local_n @ cv_to_local.T
    normals[face < 0] = np.nan
    return dict(depth=depth, face=face, local=local, normals=normals, origin=origin)


def box_primitives(half_length, hole_radius, hole_centre):
    """Installed ManiSkill _build_box_with_hole: four boxes, local x hole axis."""
    t = (half_length-hole_radius)/2
    cy, cz = hole_centre
    offset = t + hole_radius
    return [(np.array([0, offset+cy/2, 0]), np.array([half_length, t-cy/2, half_length])),
            (np.array([0, -offset+cy/2, 0]), np.array([half_length, t+cy/2, half_length])),
            (np.array([0, 0, offset+cz/2]), np.array([half_length, half_length, t-cz/2])),
            (np.array([0, 0, -offset+cz/2]), np.array([half_length, half_length, t+cz/2]))]


def real_regions(frame, result, gt, obj_oracle, object_id, half=None):
    feature = result.objects[object_id]
    valid = result.geometry.valid_depth
    hit = np.isfinite(obj_oracle['depth'])
    consistent = hit & valid & (np.abs(frame.depth-obj_oracle['depth']) <= .0015)
    visible = gt & consistent
    face = obj_oracle['face']
    seam = np.zeros_like(gt)
    cut = np.zeros_like(gt)
    for a, b in pairs():
        both = visible[a] & visible[b]
        different = both & (face[a] != face[b])
        seam[a] |= different; seam[b] |= different
        same = both & (face[a] == face[b])
        cut[a] |= same & feature.object_mask[a] & ~feature.object_mask[b]
        cut[b] |= same & feature.object_mask[b] & ~feature.object_mask[a]
    silhouette = endpoints(hit, ~hit) & visible
    occluder = hit & ~gt & valid & (frame.depth < obj_oracle['depth']-.0015)
    occlusion = endpoints(visible, occluder)
    regions = dict(all=np.ones_like(gt), gt_visible=visible,
                   contamination=~gt, face_seam_2px=dilate(seam, 2) & visible,
                   silhouette_2px=dilate(silhouette, 2) & visible,
                   occlusion_2px=dilate(occlusion, 2) & visible,
                   same_plane_tracker_cut_2px=dilate(cut, 2) & visible,
                   invalid_depth_2px=dilate(feature.depth_validity_boundary, 2))
    plane_base = np.zeros_like(gt)
    if half is not None:
        # Metric margin from all OTHER cuboid faces; separate signed-face ROIs.
        margin = half - np.abs(obj_oracle['local'])
        for f in range(6):
            axis = f//2
            other = [i for i in range(3) if i != axis]
            candidate = visible & (face == f) & np.all(margin[..., other] > .004, axis=-1)
            plane_base |= candidate
            regions[f'plane_face_{f}'] = erode(candidate, 3)
        regions['plane_interior'] = np.logical_or.reduce([regions[f'plane_face_{f}'] for f in range(6)])
    return regions, plane_base, dict(seam=seam, silhouette=silhouette, occlusion=occlusion, cut=cut)


def rim_region(frame, obj_oracle, visible, pose, half_length, radius, centre):
    """Visible entrance rim: analytic 3D distance <=4mm and raster band <=2px.

    Geometry truth identifies a physical line. A GT segmentation contour alone
    is never treated as a rim. Hidden rim projections fail the 3D-distance test.
    """
    local = obj_oracle['local']
    x = np.sign(obj_oracle['origin'][0])*half_length
    cy, cz = centre
    distance2 = np.full(frame.depth.shape, np.inf)
    for axis, other, c, oc in ((1, 2, cy, cz), (2, 1, cz, cy)):
        for sign in (-1, 1):
            distance2 = np.minimum(distance2, (local[..., 0]-x)**2
                                   +(local[..., axis]-(c+sign*radius))**2
                                   +np.maximum(np.abs(local[..., other]-oc)-radius, 0)**2)
    raster = np.zeros(frame.depth.shape, bool)
    rotation = pose_rotation(pose)
    world_to_cv = np.linalg.inv(frame.cam2world)
    for axis, other, c, oc in ((1, 2, cy, cz), (2, 1, cz, cy)):
        for sign in (-1, 1):
            points = np.zeros((2001, 3)); points[:, 0] = x
            points[:, axis] = c+sign*radius
            points[:, other] = np.linspace(oc-radius, oc+radius, len(points))
            cv = (points @ rotation.T + pose[:3]) @ world_to_cv[:3, :3].T + world_to_cv[:3, 3]
            uv = cv[:, :2]/cv[:, 2:3] * np.array([frame.intrinsic[0, 0], frame.intrinsic[1, 1]])
            uv += frame.intrinsic[:2, 2] - .5
            uv = np.rint(uv[cv[:, 2] > 0]).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < raster.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < raster.shape[0])
            uv = uv[ok]; raster[uv[:, 1], uv[:, 0]] = True
    projected_band = dilate(raster, 2)
    return visible & (distance2 <= .004**2) & projected_band, projected_band


def normal_blockers(result, object_id):
    """Nonexclusive reasons for unavailable normals on observed points."""
    g, f = result.geometry, result.objects[object_id]
    support = f.valid_points
    masks = dict(mask=np.zeros_like(support), invalid_depth=np.zeros_like(support),
                 overlap=np.zeros_like(support))
    for a, b in pairs():
        for p, q in ((a, b), (b, a)):
            masks['mask'][p] |= support[p] & ~f.object_mask[q]
            masks['invalid_depth'][p] |= support[p] & ~g.valid_depth[q]
            masks['overlap'][p] |= support[p] & g.overlap_mask[q]
    masks['depth_jump'] = support & f.depth_discontinuity
    masks['image'] = support & f.image_boundary
    invalid = support & ~f.normal_valid
    union = np.logical_or.reduce(list(masks.values()))
    return dict(unavailable=int(invalid.sum()),
                reasons_nonexclusive={k: int((invalid & v).sum()) for k, v in masks.items()},
                unexplained=int((invalid & ~union).sum()))
