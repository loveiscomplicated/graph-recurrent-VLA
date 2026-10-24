"""Independent PCA5 implementation optimizations; Task 01/02-B stay unchanged.

Same observed 5x5 patch-local connected component, minimum six points, camera
orientation and rank gates. Complete patches use reusable box moments; partial
patches retain explicit connectivity. Float64 is retained throughout PCA.
"""
from __future__ import annotations

from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

from visual.runtime.geometry import (
    GeometricFeatures, GeometryFrame, GeometryParameters, OrganizedGeometry,
    compute_geometric_features,
)
from visual.runtime.rgbd import BOX, PEG
from visual.runtime.normal_estimators import _assign, _empty, _jump


def build_geometry(frame):
    """Dense broadcasting of the original backprojection, same operation order."""
    depth = np.asarray(frame.depth)
    k = np.asarray(frame.intrinsic, dtype=np.float64)
    if depth.ndim != 2 or not min(depth.shape) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('depth must be a nonempty floating point (H,W) array')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1])):
        raise ValueError('invalid zero-skew CV intrinsic')
    if set(frame.masks)-{PEG, BOX}:
        raise ValueError('unknown object ID')
    masks = {}
    for oid in (PEG, BOX):
        mask = np.asarray(frame.masks.get(oid, np.zeros(depth.shape, bool)))
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError('object masks must be boolean (H,W)')
        masks[oid] = mask.copy()
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0)
    xyz = np.empty((*depth.shape, 3), np.float32)
    xyz[..., 0] = (np.arange(depth.shape[1])[None, :]+.5-k[0, 2])*z/k[0, 0]
    xyz[..., 1] = (np.arange(depth.shape[0])[:, None]+.5-k[1, 2])*z/k[1, 1]
    xyz[..., 2] = z
    if not np.isfinite(xyz).all():
        raise ValueError('depth/intrinsic must yield representable float32 XYZ')
    xyz[~valid] = np.nan
    return OrganizedGeometry(xyz, valid, masks, masks[PEG] & masks[BOX])


def _bounds(mask, halo):
    v, u = np.nonzero(mask)
    if not len(v):
        return None
    return (slice(max(0, v.min()-halo), min(mask.shape[0], v.max()+halo+1)),
            slice(max(0, u.min()-halo), min(mask.shape[1], u.max()+halo+1)))


def covariance_patches(xyz, support, centres, parameters, *, batch_size=4096):
    """Original partial-patch covariance, using globally precomputed unit links."""
    n = len(centres)
    covariance = np.zeros((n, 3, 3), float)
    enough = np.zeros(n, bool)
    if not n:
        return covariance, enough
    h, w = support.shape
    z = np.where(support, xyz[..., 2], 0)
    horizontal = support[:, :-1] & support[:, 1:] & ~_jump(z[:, :-1], z[:, 1:], parameters)
    vertical = support[:-1] & support[1:] & ~_jump(z[:-1], z[1:], parameters)
    horizontal = np.pad(horizontal, ((0, 0), (0, 1)))
    vertical = np.pad(vertical, ((0, 1), (0, 0)))
    dv, du = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing='ij')
    for start in range(0, n, batch_size):
        vu = centres[start:start+batch_size]
        v, u = vu[:, 0, None, None]+dv, vu[:, 1, None, None]+du
        inside = (v >= 0) & (v < h) & (u >= 0) & (u < w)
        v, u = np.clip(v, 0, h-1), np.clip(u, 0, w-1)
        available = inside & support[v, u]
        hl = horizontal[v[:, :, :-1], u[:, :, :-1]] & available[:, :, :-1] & available[:, :, 1:]
        vl = vertical[v[:, :-1], u[:, :-1]] & available[:, :-1] & available[:, 1:]
        # 25 patch vertices fit in one uint32. Links encode their source bit;
        # no horizontal wrap because a row's last source bit is always zero.
        bit = np.left_shift(np.uint32(1), np.arange(25, dtype=np.uint32)).reshape(5, 5)
        right = np.sum(hl.astype(np.uint32)*bit[:, :-1], axis=(1, 2), dtype=np.uint32)
        down = np.sum(vl.astype(np.uint32)*bit[:-1], axis=(1, 2), dtype=np.uint32)
        reached_bits = np.full(len(vu), np.uint32(1 << 12))
        for _ in range(24):
            nxt = (reached_bits | ((reached_bits & right) << np.uint32(1))
                   | ((reached_bits >> np.uint32(1)) & right)
                   | ((reached_bits & down) << np.uint32(5))
                   | ((reached_bits >> np.uint32(5)) & down))
            if np.array_equal(nxt, reached_bits): break
            reached_bits = nxt
        reached = ((reached_bits[:, None, None] & bit) != 0)
        count = reached.sum((1, 2))
        good = count >= 6
        enough[start:start+len(vu)] = good
        chosen = np.flatnonzero(good)
        if not len(chosen): continue
        delta = np.where(reached[good, ..., None],
                         xyz[v[good], u[good]]-xyz[vu[good, 0], vu[good, 1]][:, None, None], 0)
        mean = delta.sum((1, 2))/count[good, None]
        c = np.einsum('nijk,nijl->nkl', delta, delta)/count[good, None, None]
        c -= mean[:, :, None]*mean[:, None, :]
        covariance[start+chosen] = c
    return covariance, enough


def smallest_eigenvector(covariance):
    """Closed-form symmetric 3x3 eigensystem with residual/degeneracy fallback.

    Solves the same eigenproblem, not z=f(x,y) regression. Near a repeated
    eigenvalue or an inaccurate residual, use the original LAPACK eigensolver.
    """
    n = len(covariance)
    if not n: return np.empty((0, 3)), np.empty((0, 3)), 0
    scale = np.max(np.abs(covariance), axis=(1, 2))
    a = covariance/np.where(scale > 0, scale, 1)[:, None, None]
    q = np.trace(a, axis1=1, axis2=2)/3
    b = a-q[:, None, None]*np.eye(3)
    p = np.sqrt(np.sum(b*b, axis=(1, 2))/6)
    c = b/np.where(p > 0, p, 1)[:, None, None]
    determinant = (c[:, 0, 0]*(c[:, 1, 1]*c[:, 2, 2]-c[:, 1, 2]**2)
                   -c[:, 0, 1]*(c[:, 0, 1]*c[:, 2, 2]-c[:, 1, 2]*c[:, 0, 2])
                   +c[:, 0, 2]*(c[:, 0, 1]*c[:, 1, 2]-c[:, 1, 1]*c[:, 0, 2]))
    phi = np.arccos(np.clip(determinant/2, -1, 1))/3
    largest = q+2*p*np.cos(phi)
    smallest = q+2*p*np.cos(phi+2*np.pi/3)
    middle = 3*q-smallest-largest
    eigenvalues = np.column_stack((smallest, middle, largest))*scale[:, None]
    rows = a-smallest[:, None, None]*np.eye(3)
    crosses = np.stack((np.cross(rows[:, 0], rows[:, 1]), np.cross(rows[:, 0], rows[:, 2]),
                        np.cross(rows[:, 1], rows[:, 2])), axis=1)
    norms = np.sum(crosses*crosses, axis=2)
    vectors = crosses[np.arange(n), norms.argmax(1)]
    lengths = np.linalg.norm(vectors, axis=1)
    vectors /= np.where(lengths > 0, lengths, 1)[:, None]
    residual = np.linalg.norm(np.einsum('nij,nj->ni', a, vectors)-smallest[:, None]*vectors, axis=1)
    threshold = np.maximum(1e-16, eigenvalues[:, 2]*1e-8)
    # Preserve delicate original rank decisions with LAPACK, too.
    delicate = ((eigenvalues[:, 1] <= threshold*100)
                | (eigenvalues[:, 1]-eigenvalues[:, 0] <= threshold*100))
    fallback = (lengths < 1e-10) | (residual > 1e-10) | delicate | ~np.isfinite(residual)
    if fallback.any():
        values, basis = np.linalg.eigh(covariance[fallback])
        eigenvalues[fallback], vectors[fallback] = values, basis[:, :, 0]
    return eigenvalues, vectors, int(fallback.sum())


def local_plane_optimized(geometry, object_id, parameters=GeometryParameters(), *, solver='eigh', profile=None):
    if solver not in ('eigh', 'analytic'):
        raise ValueError('solver must be eigh or analytic')
    t0 = perf_counter()
    normals, valid = _empty(geometry)
    support_full = geometry.observed_mask(object_id)
    bounds = _bounds(support_full, 2)
    if bounds is None:
        if profile is not None: profile.update(points=0, fast_points=0, partial_points=0)
        return normals, valid
    support = support_full[bounds]
    xyz = geometry.xyz[bounds].astype(float)
    v, u = np.nonzero(support); vu = np.column_stack((v, u))
    z = np.where(support, xyz[..., 2], 0)
    hl = np.zeros_like(support); vl = np.zeros_like(support)
    hl[:, :-1] = support[:, :-1] & support[:, 1:] & ~_jump(z[:, :-1], z[:, 1:], parameters)
    vl[:-1] = support[:-1] & support[1:] & ~_jump(z[:-1], z[1:], parameters)
    fast_map = (minimum_filter(support, size=5, mode='constant', cval=0)
                & minimum_filter(hl, size=(5, 4), mode='constant', cval=0)
                & minimum_filter(vl, size=(4, 5), mode='constant', cval=0))
    fast = fast_map[v, u]
    t1 = perf_counter()
    covariance = np.zeros((len(v), 3, 3), float)
    enough = fast.copy()
    if fast.any():
        # A constant translation limits cancellation in second-moment sums.
        shifted = np.where(support[..., None], xyz-xyz[v[0], u[0]], 0)
        means = [uniform_filter(shifted[..., axis], size=5, mode='constant')[v[fast], u[fast]] for axis in range(3)]
        for i in range(3):
            for j in range(i, 3):
                value = uniform_filter(shifted[..., i]*shifted[..., j], size=5, mode='constant')[v[fast], u[fast]]-means[i]*means[j]
                covariance[fast, i, j] = value; covariance[fast, j, i] = value
    t2 = perf_counter()
    slow = ~fast
    if slow.any():
        c, ok = covariance_patches(xyz, support, vu[slow], parameters)
        covariance[slow], enough[slow] = c, ok
    t3 = perf_counter()
    idx = np.flatnonzero(enough)
    fallback = 0
    if solver == 'eigh':
        values, vectors = np.linalg.eigh(covariance[idx]); vectors = vectors[:, :, 0]
    else:
        values, vectors, fallback = smallest_eigenvector(covariance[idx])
    rank2 = values[:, 1] > np.maximum(1e-16, values[:, 2]*1e-8)
    unique = values[:, 1]-values[:, 0] > np.maximum(1e-16, values[:, 2]*1e-8)
    good = rank2 & unique & np.isfinite(values).all(1)
    global_vu = vu[idx[good]]+np.array([bounds[0].start, bounds[1].start])
    _assign(geometry, global_vu, vectors[good], normals, valid)
    t4 = perf_counter()
    if profile is not None:
        profile.update(points=len(v), fast_points=int(fast.sum()), partial_points=int(slow.sum()), eigen_fallback=fallback,
                       prepare_ms=(t1-t0)*1000, box_moments_ms=(t2-t1)*1000,
                       partial_connectivity_covariance_ms=(t3-t2)*1000, eigensystem_scatter_ms=(t4-t3)*1000)
    return normals, valid


def cropped_features(geometry, oid, normals, normal_valid, parameters=GeometryParameters()):
    """Same feature operator on all raw mask pixels + one neighbour halo.

    This is computational cropping, never an evaluation ROI or point sampling.
    Outputs remain full-resolution and all true image boundaries are retained.
    """
    mask = geometry.object_masks[oid]
    shape = mask.shape
    bounds = _bounds(mask, 1)
    arrays = dict(object_mask=mask, valid_points=geometry.observed_mask(oid), normals=normals,
                  normal_valid=normal_valid)
    floating = {'depth_jump_m', 'normal_variation_rad'}
    for key in GeometricFeatures.__dataclass_fields__:
        if key in arrays: continue
        arrays[key] = (np.full(shape, np.nan, np.float32) if key in floating
                       else np.zeros(shape, np.uint8 if key=='normal_neighbor_count' else bool))
    if bounds is not None:
        cropped = OrganizedGeometry(geometry.xyz[bounds], geometry.valid_depth[bounds],
                                    {i: m[bounds] for i, m in geometry.object_masks.items()}, geometry.overlap_mask[bounds])
        result = compute_geometric_features(cropped, oid, normals[bounds], normal_valid[bounds], parameters)
        for key in arrays:
            if key not in ('object_mask', 'valid_points', 'normals', 'normal_valid'):
                arrays[key][bounds] = getattr(result, key)
    return GeometricFeatures(**arrays)


@lru_cache(maxsize=1)
def _two_object_executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix='pca5-object')


def process_optimized(frame, parameters=GeometryParameters(), *, solver='eigh', profile=None, parallel=False):
    start = perf_counter(); geometry = build_geometry(frame); after_geometry = perf_counter()
    profiles = {i: {} for i in geometry.object_masks}
    if parallel:
        executor=_two_object_executor()
        futures={i:executor.submit(local_plane_optimized,geometry,i,parameters,solver=solver,profile=profiles[i]) for i in geometry.object_masks}
        normals={i:f.result() for i,f in futures.items()}
    else:
        normals = {i: local_plane_optimized(geometry, i, parameters, solver=solver, profile=profiles[i]) for i in geometry.object_masks}
    after_normals = perf_counter()
    if parallel:
        futures={i:executor.submit(cropped_features,geometry,i,*normals[i],parameters) for i in geometry.object_masks}
        objects={i:f.result() for i,f in futures.items()}
    else:
        objects = {i: cropped_features(geometry, i, *normals[i], parameters) for i in geometry.object_masks}
    end = perf_counter()
    if profile is not None: profile.update(profiles)
    return GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx, frame.timestamp, geometry, objects,
                         dict(geometry=(after_geometry-start)*1000, normals=(after_normals-after_geometry)*1000,
                              discontinuity_features=(end-after_normals)*1000, total=(end-start)*1000))
