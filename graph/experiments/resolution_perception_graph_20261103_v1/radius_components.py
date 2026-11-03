"""Task-local radius connectivity prototype; no point sampling or edge list.

This is not installed into PlanA or the paused experiment. Cells have diameter
less than the radius, hence each occupied cell is already a connected clique.
Two cell components are joined only if an exact KD-tree radius count is nonzero.
Only connectivity is retained; the potentially quadratic list of edges is not.
"""
from itertools import product

import numpy as np
from scipy.spatial import cKDTree


def _source_order_labels(labels):
    """Match scipy connected_components' first-source ordering, including ties."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    remap = np.empty(len(first), dtype=np.int64)
    remap[np.argsort(first)] = np.arange(len(first))
    return remap[inverse]


def components(xyz, radius):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape != (len(xyz), 3) or not np.isfinite(xyz).all():
        raise ValueError('finite (N,3) coordinates required')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('positive finite radius required')
    if not len(xyz):
        return np.empty(0, dtype=np.int64)

    # sqrt(3)*radius/2 < radius: every within-cell pair is an edge.
    # Relative coordinates avoid large cell IDs from the world-frame offset.
    cell = np.floor((xyz-xyz.min(axis=0))/(radius/2)).astype(np.int64)
    keys, inverse = np.unique(cell, axis=0, return_inverse=True)
    count = len(keys)
    if count == 1:
        return np.zeros(len(xyz), dtype=np.int64)
    order = np.argsort(inverse, kind='stable')
    groups = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    points = [xyz[g] for g in groups]
    trees = [cKDTree(p) for p in points]
    lower = np.array([p.min(axis=0) for p in points])
    upper = np.array([p.max(axis=0) for p in points])
    lookup = {tuple(k): i for i, k in enumerate(keys)}
    parent = np.arange(count)
    size = np.ones(count, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    # +/-2 covers possible radius edges; an extra cell protects boundary cases.
    offsets = [d for d in product(range(-3, 4), repeat=3) if d > (0, 0, 0)]
    offsets.sort(key=lambda d: (sum(x*x for x in d), d))
    for i, key in enumerate(keys):
        x, y, z = map(int, key)
        for dx, dy, dz in offsets:
            j = lookup.get((x+dx, y+dy, z+dz))
            if j is None:
                continue
            a, b = find(i), find(j)
            if a == b:
                continue
            gap = np.maximum(0, np.maximum(lower[i]-upper[j], lower[j]-upper[i]))
            if np.dot(gap, gap) > radius*radius:
                continue
            # Scalar count: never allocate query_pairs' [E,2] edge array.
            if trees[i].count_neighbors(trees[j], radius) == 0:
                continue
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
    roots = np.array([find(i) for i in range(count)])
    return _source_order_labels(roots[inverse])
