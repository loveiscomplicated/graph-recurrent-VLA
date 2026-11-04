"""Adopted point-cloud numerical helpers; historical plotting/CLI stay in visual.scripts."""
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ERODE_PX = 2

MIN_PTS = 50

SOR_K = 16

SOR_STD = 2.0

MAX_PTS = 1024

FPS_PRE = 2048

CLUSTER_R = 0.015  # m

CLUSTER_KEEP = 0.2

CLUSTER_GAP = 0.05  # m

def backproject(depth, K, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float32)

def to_world(pts, c2w):
    return (pts @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)

def sor_filter(pts, k=SOR_K, std_ratio=SOR_STD):
    if len(pts) <= k:
        return pts
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    md = d[:, 1:].mean(1)
    return pts[md <= md.mean() + std_ratio * md.std()]

def drop_small_clusters(pts, r=CLUSTER_R, keep=CLUSTER_KEEP, gap=CLUSTER_GAP):
    if len(pts) < 2:
        return pts
    pairs = cKDTree(pts).query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2)
    _, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    main = cnt.argmax()
    drop = np.zeros(len(cnt), bool)
    small = np.nonzero(cnt < keep * cnt[main])[0]
    if len(small):
        dist, _ = cKDTree(pts[lab == main]).query(pts)
        for c in small:
            drop[c] = dist[lab == c].min() > gap
    return pts[~drop[lab]]

def fps(pts, n, rng):
    if len(pts) <= n:
        return pts
    x, y, z = (np.ascontiguousarray(pts[:, i]) for i in range(3))
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(len(pts))
    dist = np.full(len(pts), np.inf, np.float32)
    for i in range(1, n):
        j = idx[i - 1]
        np.minimum(dist, (x - x[j]) ** 2 + (y - y[j]) ** 2 + (z - z[j]) ** 2, out=dist)
        idx[i] = dist.argmax()
    return pts[idx]

def extract(mask, depth, K, c2w, rng, erode_px=ERODE_PX, max_pts=MAX_PTS, cluster=False):
    """mask (H,W) bool → (points_cam, points_world, info). 비어 있으면 (0,3)."""
    info = dict(n_mask_px=int(mask.sum()))
    m = mask
    if erode_px > 0 and info["n_mask_px"]:
        er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px) > 0
        if (er & (depth > 0)).sum() >= MIN_PTS:
            m = er
    pts = backproject(depth, K, m)
    info["n_raw_pts"] = len(pts)
    if len(pts) > FPS_PRE:
        pts = pts[rng.choice(len(pts), FPS_PRE, replace=False)]
    pts = sor_filter(pts)
    info["n_sor_pts"] = len(pts)
    if cluster:
        n = len(pts)
        pts = drop_small_clusters(pts)
        info["cluster_removed_frac"] = round(1 - len(pts) / n, 4) if n else 0.0
    pts = fps(pts, max_pts, rng)
    return pts, to_world(pts, c2w), info
