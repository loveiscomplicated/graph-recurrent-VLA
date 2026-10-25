"""Evaluation-only DE/EDE adapter over Task05's unchanged segment validator.

Only stored final rows and observed component IDs enter candidate selection.
Normals, GT, feature extraction and coordinate transforms are not consulted.
"""
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay, QhullError

from visual.runtime.triangle_intra_object_edges import (
    QHULL_OPTIONS, supercover, prepare_validation, validate_cover, triangles_to_edges,
)


def _policy(extended):
    return dict(name='extended_delaunay_edgewise' if extended else 'delaunay_edgewise',
                version=1, evaluation_only=True,
                candidate_mode='extended_delaunay' if extended else 'delaunay',
                component_gate='existing_same_object_supported_pair_and_not_depth_jump',
                qhull_options=QHULL_OPTIONS,
                raster='closed_convex_supercover_exact_integer_SAT',
                pairs='right_down_both_cells_in_edge_cover',
                depth_jump_abs_m=0.01, depth_jump_rel=0.02, normal_veto=False,
                rejection='independent_edge_no_triangle_interior')


DE_POLICY = _policy(False)
EDE_POLICY = _policy(True)
ORIGINS = ('delaunay_edge', 'two_point_fallback', 'collinear_chain')


def _segment_supercover_single_sat(vertices_uv):
    """Opt-in identical integer SAT, omitting only the opposite segment normal.

    Let M bound absolute endpoint coordinates. Each normal coordinate is at
    most 2M, doubled vertices/centres at most 2M, so either dot product is
    bounded by 8M² and the cell radius by 4M. Python-int checking
    8M²+4M <= INT64_MAX also bounds differences, negation, doubling, partial
    products/sums, c +/- radius, and bbox hi+1. Both normal orientations are
    thus safe. Outside this conservative domain preserve the original path.

    Negating the normal negates c and swaps/negates projection min/max while
    keeping radius fixed: the same two inclusive inequalities merely exchange.
    Bbox, centres, closed cells, corner touches and row-major filtering stay
    identical. Triangles/unsupported input retain the original implementation.
    """
    p = np.asarray(vertices_uv)
    if p.shape != (2, 2) or p.dtype.kind not in 'iu':
        return supercover(vertices_uv)
    lo, hi = p.min(0), p.max(0)
    m = max(abs(int(lo[0])), abs(int(lo[1])), abs(int(hi[0])), abs(int(hi[1])))
    if 8*m*m+4*m > np.iinfo(np.int64).max:
        return supercover(vertices_uv)
    p = p.astype(np.int64)
    lo, hi = lo.astype(np.int64), hi.astype(np.int64)
    v, u = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    centres = np.column_stack((u.ravel(), v.ravel()))
    keep = np.ones(len(centres), bool)
    vertices = 2*p
    d = p[1]-p[0]
    axis = np.array([-d[1], d[0]], np.int64)
    if axis.any():
        projection = vertices @ axis
        c = 2*(centres @ axis)
        radius = np.abs(axis).sum()
        keep &= (c+radius >= projection.min()) & (c-radius <= projection.max())
    return centres[keep]


def collinear_order(uv, source, rows):
    """A2 exact Python-integer cross/projection; return chain order or None.

    Temporary indices only; final input rows are never reordered. n=0/1/2 are
    handled separately by the caller. Unique (object, source) is a prerequisite.
    """
    points = [tuple(map(int, p)) for p in uv]
    lex = sorted(range(len(rows)), key=lambda i: (*points[i], int(source[i]), int(rows[i])))
    if len(lex) < 2:
        return np.asarray(rows, np.int64).copy()
    a, b = points[lex[0]], points[lex[-1]]
    du, dv = b[0]-a[0], b[1]-a[1]
    if any(du*(v-a[1])-dv*(u-a[0]) != 0 for u, v in points):
        return None
    order = sorted(lex, key=lambda i: ((points[i][0]-a[0])*du+(points[i][1]-a[1])*dv,
                                       int(source[i]), int(rows[i])))
    return np.asarray(rows, np.int64)[order].copy()


def directed_edges(undirected):
    pairs = np.asarray(undirected, np.int64).reshape(-1, 2)
    return np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0).T.copy()


def candidate_edges(object_id, source_index, component_id, image_shape, *, extended,
                    stored_triangles=None):
    """Generate candidates, or read DE's frozen T0 reference without validation.

    For standalone cost measurement omit stored_triangles: each call executes
    its own Qhull call. Qhull errors and omitted vertices never trigger fallback.
    """
    start = perf_counter()
    oid, source, component = map(np.asarray, (object_id, source_index, component_id))
    n = len(source)
    shape = np.asarray(image_shape)
    if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or (shape <= 0).any()
            or oid.shape != (n,) or oid.dtype != np.int64 or not np.isin(oid, (0, 1)).all()
            or source.shape != (n,) or source.dtype != np.int64 or (source < 0).any()
            or (source >= int(shape[0])*int(shape[1])).any()
            or component.shape != (n,) or component.dtype != np.int64 or (component < 0).any()
            or len(set(zip(oid.tolist(), source.tolist()))) != n):
        raise ValueError('invalid final object/source/component rows')
    if stored_triangles is not None and extended:
        raise ValueError('stored T0 is the DE reference only; EDE must generate independently')
    frozen = None
    if stored_triangles is not None:
        frozen = np.asarray(stored_triangles)
        if (frozen.dtype != np.int64 or frozen.ndim != 2 or frozen.shape[1] != 3
                or (frozen < 0).any() or (frozen >= n).any()
                or (oid[frozen] != oid[frozen[:, :1]]).any()
                or (component[frozen] != component[frozen[:, :1]]).any()):
            raise ValueError('invalid frozen triangle membership')
    w = int(shape[1]); uv = np.column_stack((source % w, source//w))
    candidates, origins, statuses, triangles = [], [], [], []
    timing = dict(delaunay_ms=0., collinearity_chain_ms=0., two_point_ms=0.)
    object_ms = {}
    for obj in (0, 1):
        object_start = perf_counter()
        ids = np.unique(component[oid == obj])
        if not len(ids):
            statuses.append(dict(object_id=obj, component_id=None, N=0, status='empty',
                                 rows=[], source=[], chain_rows=[], omitted_rows=[],
                                 omitted_source=[], coplanar_count=0))
        for cid in ids:
            rows = np.flatnonzero((oid == obj) & (component == cid))
            p = uv[rows]
            status = dict(object_id=obj, component_id=int(cid), N=len(rows), status='ok',
                          rows=rows.tolist(), source=source[rows].tolist(), chain_rows=[],
                          omitted_rows=[], omitted_source=[], coplanar_count=0)
            pairs = np.empty((0, 2), np.int64); origin = 'delaunay_edge'
            if len(rows) == 1:
                status['status'] = 'singleton'
            elif len(rows) == 2:
                tick = perf_counter(); status['status'] = 'two_point'
                if extended:
                    pairs = rows.reshape(1, 2); origin = 'two_point_fallback'
                timing['two_point_ms'] += (perf_counter()-tick)*1000
            else:
                tick = perf_counter()
                order = collinear_order(p, source[rows], rows)
                if order is not None:
                    status.update(status='collinear', chain_rows=order.tolist(),
                                  chain_source=source[order].tolist())
                    if extended:
                        pairs = np.column_stack((order[:-1], order[1:])); origin = 'collinear_chain'
                timing['collinearity_chain_ms'] += (perf_counter()-tick)*1000
                if order is None:
                    tick = perf_counter()
                    if frozen is not None:
                        t = frozen[(oid[frozen[:, 0]] == obj) & (component[frozen[:, 0]] == cid)]
                        omitted = sorted(set(rows.tolist())-set(t.ravel().tolist()))
                        if omitted:
                            status.update(status='omitted_vertices', omitted_rows=omitted,
                                          omitted_source=source[omitted].tolist())
                    else:
                        try:
                            tri = Delaunay(p.astype(np.float64), qhull_options=QHULL_OPTIONS)
                            local = np.asarray(tri.simplices)
                            if (local.ndim != 2 or local.shape[1] != 3 or (local < 0).any()
                                    or (local >= len(rows)).any()):
                                raise ValueError('Qhull returned invalid simplex indices')
                            t = np.sort(rows[local], axis=1)
                            omitted = sorted(set(rows.tolist())-set(t.ravel().tolist()))
                            status.update(omitted_rows=omitted, omitted_source=source[omitted].tolist(),
                                          coplanar_count=len(tri.coplanar),
                                          coplanar=np.asarray(tri.coplanar).tolist())
                            if omitted:status['status'] = 'omitted_vertices'
                        except QhullError as exc:
                            status.update(status='qhull_error', error_type=type(exc).__name__)
                            t = np.empty((0, 3), np.int64)
                    triangles.extend(t.tolist())
                    edges = triangles_to_edges(t)
                    pairs = edges[:, edges[0] < edges[1]].T
                    timing['delaunay_ms'] += (perf_counter()-tick)*1000
            pairs = np.sort(pairs, axis=1)
            candidates.extend(pairs.tolist()); origins.extend([origin]*len(pairs))
            status['candidate_count'] = len(pairs)
            statuses.append(status)
        object_ms[obj] = (perf_counter()-object_start)*1000
    pairs = np.asarray(candidates, np.int64).reshape(-1, 2)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, origin = pairs[order], np.asarray(origins, dtype='<U18')[order]
    if len(np.unique(pairs, axis=0)) != len(pairs):raise ValueError('duplicate candidate')
    timing['candidate_total_ms'] = (perf_counter()-start)*1000
    return dict(candidates=pairs, origins=origin, components=statuses,
                triangles=np.unique(np.asarray(triangles, np.int64).reshape(-1, 3), axis=0),
                timing_ms=timing, candidate_object_ms=object_ms)


def build_extended_delaunay_edges(camera_points, object_id, source_index, component_id,
                                  evidence, *, extended=True, contexts=None,
                                  stored_triangles=None, optimized=False):
    """Return directed edge_index and an audit over independently checked edges.

    A caller may prepare common validation lookup once; core measurements must
    disclose that shared cost separately. No candidate/validation cache is shared.
    optimized=True explicitly selects single-normal segment SAT; default is the
    unchanged baseline raster. Policy metadata is independent of this option.
    """
    begin = perf_counter()
    xyz, oid, source = map(np.asarray, (camera_points, object_id, source_index))
    g = evidence.frame.geometry
    n = len(source)
    if (xyz.shape != (n, 3) or xyz.dtype != np.float32 or not np.isfinite(xyz).all()
            or not np.array_equal(xyz, g.xyz.reshape(-1, 3)[source])):
        raise ValueError('source camera XYZ mismatch')
    audit = candidate_edges(oid, source, component_id, g.valid_depth.shape,
                            extended=extended, stored_triangles=stored_triangles)
    for obj in (0, 1):
        if not g.observed_mask(obj).ravel()[source[oid == obj]].all():
            raise ValueError('selected source not observed')
    timing = audit['timing_ms']
    tick = perf_counter()
    contexts = prepare_validation(evidence) if contexts is None else contexts
    timing['validation_lookup_ms'] = (perf_counter()-tick)*1000
    timing.update(supercover_ms=0., support_validation_ms=0., depth_validation_ms=0.)
    w = g.valid_depth.shape[1]; uv = np.column_stack((source % w, source//w))
    accepted, checks = [], []
    object_ms = {0: 0., 1: 0.}; cover_bytes = bbox_bytes = 0
    raster = _segment_supercover_single_sat if optimized else supercover
    for pair in audit['candidates']:
        tick = perf_counter(); cover = raster(uv[pair])
        timing['supercover_ms'] += (perf_counter()-tick)*1000
        check_start = perf_counter()
        result = validate_cover(cover, contexts[int(oid[pair[0]])], timing)
        object_ms[int(oid[pair[0]])] += (perf_counter()-tick)*1000
        accepted.append(not result['reasons']); checks.append(result)
        cover_bytes = max(cover_bytes, cover.nbytes)
        bbox_bytes = max(bbox_bytes, int(np.prod(uv[pair].max(0)-uv[pair].min(0)+1))*8*8)
    accepted = np.asarray(accepted, bool)
    tick = perf_counter(); edge = directed_edges(audit['candidates'][accepted])
    timing['edge_packaging_ms'] = (perf_counter()-tick)*1000
    timing['helper_wall_ms'] = (perf_counter()-begin)*1000
    audit.update(accepted=accepted, checks=checks, validation_object_ms=object_ms,
                 raster_work=dict(cover_peak_array_bytes=cover_bytes,
                                  bbox_peak_scratch_estimate_bytes=bbox_bytes))
    return edge, audit


def edge_metadata(audit, *, extended):
    """Packet algebra evidence; full rejected witnesses stay in the audit."""
    return dict(edge_policy=dict(EDE_POLICY if extended else DE_POLICY),
                accepted_edge_pairs=audit['candidates'][audit['accepted']].copy(),
                accepted_edge_origins=audit['origins'][audit['accepted']].tolist())
