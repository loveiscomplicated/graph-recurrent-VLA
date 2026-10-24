"""Observation-only normal estimators promoted unchanged from Task 02-B.

Inputs are Task 01 organized observed XYZ and tracker support only. Unit normals
face the CV camera. The original estimator and variation implementation remain
in geometry.py. No GT, RGB, resampling, smoothing or imputation is used here.
"""
from __future__ import annotations

import numpy as np

from visual.runtime.geometry import GeometryParameters, OrganizedGeometry

METHODS = ('baseline', 'fd2', 'fd4', 'pca5', 'pca7')


def _jump(a, b, parameters):
    return np.abs(a-b) > parameters.depth_jump_abs_m + parameters.depth_jump_rel*np.minimum(a, b)


def _empty(geometry):
    return np.full_like(geometry.xyz, np.nan), np.zeros(geometry.valid_depth.shape, bool)


def _assign(geometry, vu, vectors, normals, valid):
    lengths = np.linalg.norm(vectors, axis=1)
    good = np.isfinite(lengths) & (lengths > 0)
    vu, vectors = vu[good], vectors[good]/lengths[good, None]
    if not len(vu):
        return
    xyz = geometry.xyz[vu[:, 0], vu[:, 1]]
    vectors[np.einsum('ij,ij->i', vectors, xyz) > 0] *= -1
    normals[vu[:, 0], vu[:, 1]] = vectors.astype(np.float32)
    valid[vu[:, 0], vu[:, 1]] = True


def finite_difference(geometry: OrganizedGeometry, object_id: int, spacing: int,
                      parameters: GeometryParameters = GeometryParameters()):
    """Full-resolution central secants with complete, gated axis paths.

    Every intermediate pixel from centre to each of four endpoints must be
    observed for this object; EVERY adjacent pair must pass the original jump
    gate. Thus even an endpoint beyond a narrow gap cannot bridge it. The gate
    is on unit pixel steps, not multiplied by spacing or applied to longer
    secants. No one-sided fallback: incomplete stencils are invalid.
    """
    if not isinstance(spacing, int) or isinstance(spacing, bool) or spacing < 1:
        raise ValueError('spacing must be a positive integer')
    normals, valid = _empty(geometry)
    h, w = valid.shape
    if min(h, w) <= 2*spacing:
        return normals, valid
    support = geometry.observed_mask(object_id)
    v, u = np.nonzero(support)
    inside = (v >= spacing) & (v < h-spacing) & (u >= spacing) & (u < w-spacing)
    v, u = v[inside], u[inside]
    good = np.ones(len(v), bool)
    z = np.where(geometry.valid_depth, geometry.xyz[..., 2], 0)
    for dv, du in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        for step in range(1, spacing+1):
            a = (v+dv*(step-1), u+du*(step-1))
            b = (v+dv*step, u+du*step)
            good &= support[b] & ~_jump(z[a], z[b], parameters)
    v, u = v[good], u[good]
    dx = geometry.xyz[v, u+spacing].astype(float)-geometry.xyz[v, u-spacing]
    dy = geometry.xyz[v+spacing, u].astype(float)-geometry.xyz[v-spacing, u]
    _assign(geometry, np.column_stack((v, u)), np.cross(dx, dy), normals, valid)
    return normals, valid


def local_plane(geometry: OrganizedGeometry, object_id: int, window: int,
                parameters: GeometryParameters = GeometryParameters(), *,
                min_points: int = 6, batch_size: int = 4096):
    """PCA of the centre-connected component inside an odd square patch.

    Four-connected links require same-object finite positive depth and the
    original adjacent z-jump gate. Flood fill is confined to this patch, so
    disconnected islands/holes cannot be bridged via remote connections. All
    accepted points have equal weight, with no fitted residual trimming. True
    continuous creases below the gate CAN mix and blur: measured by this task.
    At least six points (including centre), rank >=2 and a unique smallest
    covariance eigenvector are required. Partial square patches are permitted.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 3 or window % 2 != 1:
        raise ValueError('window must be an odd integer >=3')
    if not isinstance(min_points, int) or min_points < 3 or batch_size < 1:
        raise ValueError('min_points must be >=3 and batch_size positive')
    normals, valid = _empty(geometry)
    support = geometry.observed_mask(object_id)
    centres = np.column_stack(np.nonzero(support))
    h, w = support.shape
    radius = window//2
    dv, du = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), indexing='ij')
    for start in range(0, len(centres), batch_size):
        vu = centres[start:start+batch_size]
        v, u = vu[:, 0, None, None]+dv, vu[:, 1, None, None]+du
        inside = (v >= 0) & (v < h) & (u >= 0) & (u < w)
        v, u = np.clip(v, 0, h-1), np.clip(u, 0, w-1)
        available = inside & support[v, u]
        xyz = geometry.xyz[v, u].astype(float)
        z = np.where(available, xyz[..., 2], 0)
        horizontal = available[:, :, :-1] & available[:, :, 1:] & ~_jump(z[:, :, :-1], z[:, :, 1:], parameters)
        vertical = available[:, :-1, :] & available[:, 1:, :] & ~_jump(z[:, :-1, :], z[:, 1:, :], parameters)
        reached = np.zeros(available.shape, bool)
        reached[:, radius, radius] = True
        for _ in range(window*window-1):
            nxt = reached.copy()
            nxt[:, :, 1:] |= reached[:, :, :-1] & horizontal
            nxt[:, :, :-1] |= reached[:, :, 1:] & horizontal
            nxt[:, 1:, :] |= reached[:, :-1, :] & vertical
            nxt[:, :-1, :] |= reached[:, 1:, :] & vertical
            if np.array_equal(nxt, reached):
                break
            reached = nxt
        count = reached.sum((1, 2))
        enough = count >= min_points
        if not enough.any():
            continue
        vu, count, reached, xyz = vu[enough], count[enough], reached[enough], xyz[enough]
        # Centre-relative coordinates and float64 covariance avoid cancellation
        # from metre-scale absolute positions and millimetre-scale support.
        delta = np.where(reached[..., None], xyz-geometry.xyz[vu[:, 0], vu[:, 1]][:, None, None, :], 0)
        mean = delta.sum((1, 2))/count[:, None]
        covariance = np.einsum('nijk,nijl->nkl', delta, delta)/count[:, None, None]
        covariance -= mean[:, :, None]*mean[:, None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        rank2 = eigenvalues[:, 1] > np.maximum(1e-16, eigenvalues[:, 2]*1e-8)
        unique = eigenvalues[:, 1]-eigenvalues[:, 0] > np.maximum(1e-16, eigenvalues[:, 2]*1e-8)
        good = rank2 & unique & np.isfinite(eigenvalues).all(1)
        _assign(geometry, vu[good], eigenvectors[good, :, 0], normals, valid)
    return normals, valid
