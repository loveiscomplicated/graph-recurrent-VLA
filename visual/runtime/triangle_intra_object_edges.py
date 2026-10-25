"""Evaluation-only 2D triangles on unchanged observed point rows.

Closed pixel cells intersecting the convex primitive are found by exact integer
separating axes. Pairs are only right/down neighbours whose BOTH cells belong
to that cover, never all pairs in its bbox. A cell-touch is conservative by up
to half a pixel; this is observation validation, not physical reconstruction.
No normals, face labels, GT, sampling, feature estimation or transforms here.
"""
from time import perf_counter

import numpy as np
from scipy.spatial import Delaunay, QhullError

QHULL_OPTIONS = 'Qbb Qc Qz Q12'
TRIANGLE_POLICY = dict(
    name='observed_component_2d_delaunay_support_depth', version=1,
    component_gate='existing_same_object_supported_pair_and_not_depth_jump',
    uv='integer_pixel_centres_closed_half_pixel_cells', qhull_options=QHULL_OPTIONS,
    raster='closed_convex_supercover_exact_integer_SAT',
    pairs='right_down_both_cells_in_cover_boundary_and_interior',
    depth_jump_abs_m=0.01, depth_jump_rel=0.02, normal_veto=False,
    triangle_rejection='whole_triangle_no_edge_salvage_no_fallback')
SUPPORT_REASONS = frozenset(('mask_outside', 'invalid_depth', 'overlap', 'image_outside'))
DEPTH_REASONS = frozenset(('pair_missing', 'pair_unsupported', 'delta_invalid', 'depth_jump'))


def supercover(vertices_uv):
    """All (u,v) cells touching a segment or triangle, including corner-touch.

    Vertices are integer pixel centres. All projections are int64 at doubled
    coordinates; the cell has half width 1 there. No epsilon or margins.
    Only one primitive bbox is allocated at a time (not T x H x W).
    """
    p = np.asarray(vertices_uv)
    if p.shape not in ((2, 2), (3, 2)) or p.dtype.kind not in 'iu':
        raise ValueError('integer UV segment or triangle required')
    p = p.astype(np.int64)
    lo, hi = p.min(0), p.max(0)
    v, u = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
    centres = np.column_stack((u.ravel(), v.ravel()))
    keep = np.ones(len(centres), bool)
    vertices = 2*p
    for d in np.roll(p, -1, axis=0)-p:
        axis = np.array([-d[1], d[0]], np.int64)
        if not axis.any(): continue
        projection = vertices @ axis
        c = 2*(centres @ axis)
        radius = np.abs(axis).sum()
        keep &= (c+radius >= projection.min()) & (c-radius <= projection.max())
    # x/y separating axes are already enforced by the integer bbox.
    return centres[keep]


def adjacent_pairs(cover_uv, shape):
    """Original flat-index right/down pairs with both touched cells in image."""
    h, w = shape
    cells = {(int(u), int(v)) for u, v in cover_uv if 0 <= u < w and 0 <= v < h}
    pairs = [(v*w+u, y*w+x) for u, v in cells
             for x, y in ((u+1, v), (u, v+1)) if (x, y) in cells]
    return np.asarray(sorted(pairs), np.int64).reshape(-1, 2)


def prepare_validation(evidence):
    """Index existing PairEvidence; do not recompute gate, normals or geometry."""
    if evidence.estimator != 'pca5':raise ValueError('existing pca5 evidence required')
    g = evidence.frame.geometry
    h, w = g.valid_depth.shape
    result = {}
    for obj in (0, 1):
        pairs = evidence.pairs[obj]
        pixels = np.asarray(pairs.pixels)
        if (tuple(pairs.shape) != (h, w) or pixels.ndim != 2 or pixels.shape[1] != 2
                or pixels.dtype.kind not in 'iu' or (pixels < 0).any() or (pixels >= h*w).any()):
            raise ValueError('pair indices/shape mismatch')
        count = len(pixels)
        if (pairs.supported.shape != (count, 2) or pairs.supported.dtype != bool
                or pairs.depth_jump.shape != (count,) or pairs.depth_jump.dtype != bool
                or pairs.signed_depth_delta_m.shape != (count,)):
            raise ValueError('pair fields mismatch')
        support = g.observed_mask(obj)
        if not np.array_equal(pairs.supported, support.ravel()[pixels]):
            raise ValueError('pair/support contradiction')
        step = pixels[:, 1]-pixels[:, 0]
        right = (step == 1) & (pixels[:, 0]//w == pixels[:, 1]//w)
        down = step == w
        if not (right | down).all():raise ValueError('non right/down evidence pair')
        rmap, dmap = np.full((h, w), -1, np.int64), np.full((h, w), -1, np.int64)
        # Duplicate pairs would make lookup non-injective and are input errors.
        for sel, mapping in ((right, rmap), (down, dmap)):
            s = pixels[sel, 0]
            if len(np.unique(s)) != len(s):raise ValueError('duplicate PairEvidence')
            mapping.ravel()[s] = np.flatnonzero(sel)
        result[obj] = dict(shape=(h, w), mask=g.object_masks[obj], valid=g.valid_depth,
                           overlap=g.overlap_mask, support=support, pairs=pairs,
                           right=rmap, down=dmap)
    return result


def validate_cover(cover_uv, context, timings=None):
    """Separate support causes from absent evidence and actual local jumps.

    Enumerate pairs among touched cells only. Unsupported endpoints are reported
    as pair_unsupported; they do not masquerade as missing supported evidence.
    Store complete cause counts and up to four witnesses per cause.
    """
    start = perf_counter()
    uv = np.asarray(cover_uv, np.int64)
    h, w = context['shape']
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    reasons, witnesses, counts = [], {}, {}
    def note(name, values):
        if len(values):
            reasons.append(name); counts[name] = len(values)
            witnesses[name] = np.asarray(values)[:4].copy()
    note('image_outside', uv[~inside])
    s = uv[inside, 1]*w+uv[inside, 0]
    note('mask_outside', s[~context['mask'].ravel()[s]])
    note('invalid_depth', s[~context['valid'].ravel()[s]])
    note('overlap', s[context['overlap'].ravel()[s]])
    after_mask = perf_counter()
    pairs = adjacent_pairs(uv, (h, w))
    supported = context['support'].ravel()[pairs].all(1)
    note('pair_unsupported', pairs[~supported])
    checked = pairs[supported]
    right = ((checked[:, 1]-checked[:, 0]) == 1) & (checked[:, 1]//w == checked[:, 0]//w)
    idx = np.where(right, context['right'].ravel()[checked[:, 0]],
                   context['down'].ravel()[checked[:, 0]])
    note('pair_missing', checked[idx < 0])
    present = idx >= 0
    idx, checked = idx[present], checked[present]
    evidence = context['pairs']
    note('delta_invalid', checked[~np.isfinite(evidence.signed_depth_delta_m[idx])])
    note('depth_jump', checked[evidence.depth_jump[idx]])
    if timings is not None:
        timings['support_validation_ms'] += (after_mask-start)*1000
        timings['depth_validation_ms'] += (perf_counter()-after_mask)*1000
    return dict(reasons=reasons, counts=counts, witnesses=witnesses,
                cover_pixels=len(uv), enumerated_pairs=len(pairs), supported_pairs=int(supported.sum()))


def triangles_to_edges(triangles):
    """Canonical unique bidirectional union; no node is created or removed."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    sides = np.concatenate((t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]))
    undirected = np.unique(np.sort(sides, axis=1), axis=0)
    return np.unique(np.concatenate((undirected, undirected[:, ::-1])), axis=0).T.copy()


def build_triangle_intra_object_edges(camera_points, object_id, source_index,
                                     component_maps, evidence):
    """Return edge_index, component rows, and candidate/stage/rejection audit."""
    start = perf_counter()
    xyz, oid, source = map(np.asarray, (camera_points, object_id, source_index))
    n = len(xyz)
    shape = evidence.frame.geometry.valid_depth.shape
    h, w = shape
    if (xyz.shape != (n, 3) or xyz.dtype != np.float32 or not np.isfinite(xyz).all()
            or oid.shape != (n,) or oid.dtype != np.int64 or not np.isin(oid, [0, 1]).all()
            or source.shape != (n,) or source.dtype != np.int64 or (source < 0).any()
            or (source >= h*w).any() or len(set(zip(oid.tolist(), source.tolist()))) != n):
        raise ValueError('invalid/duplicate final point source rows')
    if not np.array_equal(xyz, evidence.frame.geometry.xyz.reshape(-1, 3)[source]):
        raise ValueError('source XYZ mismatch')
    component = np.empty(n, np.int64)
    for obj in (0, 1):
        mapping = np.asarray(component_maps[obj])
        if mapping.shape != shape or mapping.dtype != np.int64:raise ValueError('component map mismatch')
        rows = oid == obj
        component[rows] = mapping.ravel()[source[rows]]
        if not evidence.frame.geometry.observed_mask(obj).ravel()[source[rows]].all():
            raise ValueError('selected point not supported')
    if (component < 0).any():raise ValueError('unobserved selected point')
    uv = np.column_stack((source % w, source//w))
    contexts = prepare_validation(evidence)
    timing = dict(input_lookup_ms=(perf_counter()-start)*1000, delaunay_ms=0.,
                  supercover_ms=0., support_validation_ms=0., depth_validation_ms=0.,
                  edge_packaging_ms=0.)
    triangles, statuses = [], []
    tick = perf_counter()
    for obj in (0, 1):
        for cid in np.unique(component[oid == obj]):
            rows = np.flatnonzero((oid == obj) & (component == cid))
            p = uv[rows]
            status = dict(object_id=obj, component_id=int(cid), N=len(rows), status='ok',
                          omitted_rows=[], coplanar_count=0)
            if len(rows) < 3:
                status['status'] = 'insufficient'
            else:
                d = p-p[0]
                nonzero = np.flatnonzero(np.any(d != 0, axis=1))
                a = d[nonzero[0]]
                if not (a[0]*d[:, 1]-a[1]*d[:, 0]).any():
                    status['status'] = 'collinear'
                else:
                    try:
                        tri = Delaunay(p.astype(np.float64), qhull_options=QHULL_OPTIONS)
                        local = np.asarray(tri.simplices)
                        if local.ndim != 2 or local.shape[1] != 3 or (local < 0).any() or (local >= len(rows)).any():
                            raise ValueError('Qhull returned invalid simplex indices')
                        omitted = sorted(set(range(len(rows)))-set(local.ravel().tolist()))
                        status['omitted_rows'] = rows[omitted].tolist()
                        status['coplanar_count'] = len(tri.coplanar)
                        if omitted:status['status'] = 'omitted_vertices'
                        triangles.extend(np.sort(rows[local], axis=1).tolist())
                    except QhullError as exc:
                        status.update(status='qhull_error', error_type=type(exc).__name__)
            statuses.append(status)
    t = np.unique(np.asarray(triangles, np.int64).reshape(-1, 3), axis=0)
    timing['delaunay_ms'] = (perf_counter()-tick)*1000
    support_pass, depth_pass, checks = [], [], []
    bbox_cells, cover_cells, scratch_peak = 0, 0, 0
    for triangle in t:
        p = uv[triangle]
        area2 = (p[1, 0]-p[0, 0])*(p[2, 1]-p[0, 1])-(p[1, 1]-p[0, 1])*(p[2, 0]-p[0, 0])
        if area2 == 0:raise ValueError('Qhull returned zero-area triangle')
        tick = perf_counter()
        covers = [supercover(p)] + [supercover(p[[i, (i+1)%3]]) for i in range(3)]
        timing['supercover_ms'] += (perf_counter()-tick)*1000
        bbox = int(np.prod(p.max(0)-p.min(0)+1)); bbox_cells += bbox
        cover_cells += sum(map(len, covers))
        scratch_peak = max(scratch_peak, sum(c.nbytes for c in covers)+bbox*8*8)
        context = contexts[int(oid[triangle[0]])]
        results = [validate_cover(c, context, timing) for c in covers]
        all_reasons = set().union(*(r['reasons'] for r in results))
        support_pass.append(not bool(all_reasons & SUPPORT_REASONS))
        depth_pass.append(not bool(all_reasons & DEPTH_REASONS))
        checks.append(dict(interior=results[0], edges=results[1:], reasons=sorted(all_reasons)))
    support_pass, depth_pass = np.asarray(support_pass, bool), np.asarray(depth_pass, bool)
    accepted = support_pass & depth_pass
    tick = perf_counter()
    stages = dict(T0=triangles_to_edges(t), TM=triangles_to_edges(t[support_pass]),
                  TMD=triangles_to_edges(t[accepted]))
    timing['edge_packaging_ms'] = (perf_counter()-tick)*1000
    timing['helper_wall_ms'] = (perf_counter()-start)*1000
    audit = dict(triangles=t, support_pass=support_pass, depth_pass=depth_pass, accepted=accepted,
                 triangle_checks=checks, components=statuses, stage_edges=stages, timing_ms=timing,
                 raster_work=dict(bbox_cells=bbox_cells, cover_cells=cover_cells,
                    scratch_peak_array_bytes_estimate=scratch_peak,
                    lookup_array_bytes=sum(c['right'].nbytes+c['down'].nbytes+c['support'].nbytes
                                          for c in contexts.values())))
    return stages['TMD'], component, audit
