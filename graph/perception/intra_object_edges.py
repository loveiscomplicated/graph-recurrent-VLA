"""GT-free observed-component constrained symmetric Euclidean k-NN (k=8).

Components use the FULL observed raster support and existing depth-jump pairs.
Neighbors use final point rows and original camera XYZ promoted to float64.
This is geometric proximity, not a physical edge/crease/contact detector.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

EDGE_POLICY = dict(name='observed_component_symmetric_knn', k=8,
                   distance_frame='original_CV_camera_metres_float64',
                   tie_break='squared_distance_then_final_point_index',
                   component_gate='existing_same_object_supported_pair_and_not_depth_jump')


def observed_components(evidence):
    """Return full source-index component maps; unobserved pixels have ID -1."""
    if evidence.estimator != 'pca5':
        raise ValueError('existing pca5 observation evidence required')
    geometry = evidence.frame.geometry
    maps, stats = {}, {}
    for oid in (0, 1):
        support = geometry.observed_mask(oid)
        source = np.flatnonzero(support)
        pairs = evidence.pairs[oid]
        pixels = np.asarray(pairs.pixels)
        if (tuple(pairs.shape) != support.shape or pixels.ndim != 2 or pixels.shape[1] != 2
                or pixels.dtype.kind not in 'iu' or (pixels < 0).any() or (pixels >= support.size).any()):
            raise ValueError('invalid original raster pair indices')
        p = len(pixels)
        if (pairs.supported.dtype != bool or pairs.supported.shape != (p, 2)
                or pairs.depth_jump.dtype != bool or pairs.depth_jump.shape != (p,)
                or not np.array_equal(pairs.supported, support.ravel()[pixels])):
            raise ValueError('pair/support identity mismatch')
        width = support.shape[1]
        step = pixels[:, 1] - pixels[:, 0]
        adjacent = ((step == width) | ((step == 1) & (pixels[:,0]//width == pixels[:,1]//width)))
        if not adjacent.all():
            raise ValueError('expected original right/down four-neighbour pairs')
        good = pairs.supported.all(1) & ~pairs.depth_jump
        lookup = np.full(support.size, -1, np.int64)
        lookup[source] = np.arange(len(source), dtype=np.int64)
        links = lookup[pixels[good]]
        graph = coo_array((np.ones(len(links), bool), (links[:,0], links[:,1])),
                          shape=(len(source),len(source))).tocsr()
        if len(source):
            count, labels = connected_components(graph,directed=False,return_labels=True)
        else:
            count, labels = 0, np.empty(0,np.int64)
        dense = np.full(support.size, -1, np.int64)
        dense[source] = labels
        maps[oid] = dense.reshape(support.shape)
        stats[oid] = dict(observed_points=len(source),supported_links=int(good.sum()),
                          components=int(count),component_support_counts=np.bincount(labels,minlength=count))
    return maps, stats


def directed_knn(points, object_id, component_id):
    """Deterministic directed k=8 neighbors, with only O(max component N) scratch.

    Public for numerical ordering audits: arbitrary rigidly transformed point
    arrays may be compared, but the production topology always uses camera XYZ.
    """
    xyz = np.asarray(points,dtype=np.float64)
    oid = np.asarray(object_id)
    component = np.asarray(component_id)
    n = len(xyz)
    if (xyz.shape != (n,3) or not np.isfinite(xyz).all()
            or oid.shape != (n,) or oid.dtype != np.int64 or not np.isin(oid,[0,1]).all()
            or component.shape != (n,) or component.dtype != np.int64 or (component < 0).any()):
        raise ValueError('finite points and valid object/component rows required')
    edges = []
    for obj in (0, 1):
        for c in np.unique(component[oid == obj]):
            rows = np.flatnonzero((oid == obj) & (component == c))
            for i in rows:
                candidates = rows[rows != i]
                delta = xyz[candidates]-xyz[i]
                distance = np.einsum('ij,ij->i',delta,delta)
                order = np.lexsort((candidates,distance))[:8]
                edges.extend((int(i),int(j)) for j in candidates[order])
    return np.asarray(edges,np.int64).reshape(-1,2).T.copy()


def build_intra_object_edges(camera_points, object_id, source_index, component_maps):
    """Return point-local symmetric unique edges and row-aligned component IDs."""
    xyz, oid, source = map(np.asarray,(camera_points,object_id,source_index))
    n = len(xyz)
    if (xyz.dtype != np.float32 or xyz.shape != (n,3) or not np.isfinite(xyz).all()
            or oid.dtype != np.int64 or oid.shape != (n,) or not np.isin(oid,[0,1]).all()
            or source.dtype != np.int64 or source.shape != (n,) or (source < 0).any()
            or len(set(zip(oid.tolist(),source.tolist()))) != n):
        raise ValueError('invalid final camera point/object/source rows')
    component = np.empty(n,np.int64)
    for obj in (0,1):
        rows = np.flatnonzero(oid == obj)
        mapping = np.asarray(component_maps[obj])
        if mapping.ndim != 2 or mapping.dtype != np.int64 or (source[rows] >= mapping.size).any():
            raise ValueError('invalid component map/source')
        component[rows] = mapping.ravel()[source[rows]]
    if (component < 0).any():raise ValueError('point not on observed support')
    directed = directed_knn(xyz,oid,component)
    edges = np.unique(np.concatenate((directed.T,directed[::-1].T)),axis=0).T.copy()
    return edges,component


def edge_diagnostics(packet):
    """All component counts/degrees plus longest source/UV/XYZ pairs for review."""
    n = len(packet['points']);edge = packet['intra_object_edge_index']
    oid = packet['object_id']; comp = packet['observed_component_id']
    source = packet['provenance']['source_index']; width = int(packet['image_shape'][1])
    xyz = packet['camera_points'].astype(float)
    length = np.linalg.norm(xyz[edge[0]]-xyz[edge[1]],axis=1)
    degree = np.bincount(edge[0],minlength=n)
    result = {}
    for obj in (0,1):
        objects = oid == obj; keep = objects[edge[0]]
        components = []
        for c in np.unique(comp[objects]):
            rows = np.flatnonzero(objects & (comp == c))
            selected = keep & (comp[edge[0]] == c)
            lengths = length[selected]
            components.append(dict(component_id=int(c),point_count=len(rows),edge_count=int(selected.sum()),
                                   isolated_nodes=int((degree[rows] == 0).sum()),degree=degree[rows],
                                   length_mean_m=float(lengths.mean()) if len(lengths) else None,
                                   length_p95_m=float(np.percentile(lengths,95)) if len(lengths) else None,
                                   length_max_m=float(lengths.max()) if len(lengths) else None))
        unique = np.flatnonzero(keep & (edge[0] < edge[1]))
        top = unique[np.argsort(-length[unique],kind='stable')[:10]]
        longest = []
        for e in top:
            rows = edge[:,e];s = source[rows]
            longest.append(dict(point_rows=rows,source_index=s,
                                uv=np.column_stack((s%width,s//width)),camera_xyz=xyz[rows],
                                ee_xyz=packet['points'][rows],length_m=float(length[e])))
        lengths = length[keep]
        result[obj] = dict(point_count=int(objects.sum()),edge_count=int(keep.sum()),
                           degree=degree[objects],isolated_nodes=int((degree[objects]==0).sum()),
                           length_mean_m=float(lengths.mean()) if len(lengths) else None,
                           length_p95_m=float(np.percentile(lengths,95)) if len(lengths) else None,
                           length_max_m=float(lengths.max()) if len(lengths) else None,
                           components=components,longest_edges=longest)
    return dict(objects=result,cross_object_edges=int((oid[edge[0]]!=oid[edge[1]]).sum()),
                cross_component_edges=int((comp[edge[0]]!=comp[edge[1]]).sum()))
