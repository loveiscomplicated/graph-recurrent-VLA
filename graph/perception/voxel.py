"""Validated source ordering, nominal scale and deterministic voxel representatives."""
from __future__ import annotations
import numpy as np

def sources(flat, xyz):
    flat, xyz = np.asarray(flat), np.asarray(xyz)
    if flat.ndim != 1 or flat.dtype.kind not in 'iu' or np.any(flat < 0):
        raise ValueError('nonnegative integer sources required')
    if len(flat)>1 and not np.all(flat[1:]>flat[:-1]): raise ValueError('strict source order required')
    if xyz.shape != (len(flat),3) or not np.isfinite(xyz).all(): raise ValueError('finite source XYZ required')
    return flat, xyz

def nominal_scale(xyz, intrinsic):
    xyz, k = np.asarray(xyz), np.asarray(intrinsic, float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or (xyz[:,2]<=0).any():
        raise ValueError('original positive-depth X required')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('valid intrinsic required')
    if not len(xyz): return dict(status='N/A', z_ref_m=None, epsilon_m=None, a_m=None)
    z = float(np.median(xyz[:,2].astype(float)))
    epsilon = z*np.hypot(2/k[0,0], 2/k[1,1])
    a = epsilon/np.sqrt(3)
    if not np.isfinite(a) or a<=0: raise ValueError('nonpositive nominal scale')
    return dict(status='defined', z_ref_m=z, fx_px=float(k[0,0]), fy_px=float(k[1,1]),
        epsilon_m=float(epsilon), a_m=float(a), multiplier=1.,
        pair_footprint_m=float(z*max(np.hypot(3/k[0,0],2/k[1,1]),np.hypot(2/k[0,0],3/k[1,1]))))

def voxel_keys(xyz, a):
    xyz=np.asarray(xyz, float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or not np.isfinite(a) or a<=0:
        raise ValueError('finite XYZ and positive size required')
    with np.errstate(over='ignore',invalid='ignore'):
        q=np.floor(xyz/a)
    # float64(int64.max) rounds to 2**63, so the upper comparison is strict.
    if not np.isfinite(q).all() or (q < -(2.**63)).any() or (q >= 2.**63).any():
        raise ValueError('voxel key int64 overflow')
    return q.astype(np.int64)

def representatives(flat, xyz, a):
    flat,xyz=sources(flat,xyz)
    key=voxel_keys(xyz,a)
    cells,membership,capacity=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    rep_rows=np.empty(len(cells),np.int64)
    for cell in range(len(cells)):
        rows=np.flatnonzero(membership==cell)  # already source ascending
        points=xyz[rows].astype(float)
        centroid=points.mean(axis=0)
        distance=np.sum((points-centroid)**2,axis=1)
        rep_rows[cell]=rows[int(distance.argmin())]
    same_rows=rep_rows[membership]
    same_distance=np.linalg.norm(xyz.astype(float)-xyz[same_rows].astype(float),axis=1)
    if len(flat) and np.max(same_distance)>np.sqrt(3)*a+64*np.finfo(float).eps*max(1.,np.max(np.abs(xyz))):
        raise AssertionError('same-cell bound failed')
    return dict(keys=key,cells=cells,membership=membership,capacity=capacity,
        cell_rep_rows=rep_rows, representative_rows=np.sort(rep_rows),
        same_representative_source=flat[same_rows], same_cell_distance_m=same_distance)
