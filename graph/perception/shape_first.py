"""CPU Shape-First contour, voxel and FPS selection for the current graph path."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import json
from time import perf_counter
import cv2
import numpy as np
from graph.vision_input import receive_vision_packet
from graph.perception.sampling import fps_indices
import graph.perception.voxel as voxel
import graph.perception.vision_packet as prior

SURFACE = {0: 97, 1: 95}

def thin_indices(m, ratio, phase=0):
    if type(m) is not int or m < 0:
        raise ValueError('nonnegative integer M required')
    if not np.isfinite(float(ratio)) or not 0 < float(ratio) <= 1:
        raise ValueError('ratio must be finite in (0,1]')
    r = Fraction(str(ratio))
    if type(phase) is not int or phase not in (0, 1) or (phase == 1 and r != Fraction(1, 2)):
        raise ValueError('phase1 diagnostic is defined only at ratio .5')
    p, q = r.numerator, r.denominator
    count = (m*p+q-1)//q
    rows = np.array([(k*q)//p+phase for k in range(count)], dtype=np.int64)
    return rows[rows < m]

def canonical_walk(walk):
    walk = np.asarray(walk, np.int64)
    if len(walk) > 1 and walk[0] == walk[-1]:
        walk = walk[:-1]
    if not len(walk):
        return walk
    # Degenerate thin contours may revisit their minimum source.
    options = []
    for sequence in (walk, walk[::-1]):
        for start in np.flatnonzero(sequence == sequence.min()):
            options.append(tuple(np.roll(sequence, -int(start)).tolist()))
    return np.asarray(min(options), np.int64)

def stable_unique(values):
    return np.asarray(list(dict.fromkeys(map(int, values))), np.int64)

def observed_runs(good):
    """Walk positions; bridge closure only on a completely observed contour."""
    good = np.asarray(good, bool)
    if not len(good): return []
    if good.all(): return [(np.arange(len(good), dtype=np.int64), True)]
    # Start after a real exclusion, so a valid boundary-crossing run stays intact.
    order = np.roll(np.arange(len(good)), -int(np.flatnonzero(~good)[0])-1)
    runs, current = [], []
    for pos in order:
        if good[pos]: current.append(int(pos))
        elif current:
            runs.append((np.asarray(current, np.int64), False)); current = []
    if current: runs.append((np.asarray(current, np.int64), False))
    return runs

def outer_contours(raw_mask, observed, valid_depth, overlap):
    raw_mask, observed = np.asarray(raw_mask), np.asarray(observed)
    if raw_mask.ndim != 2 or raw_mask.dtype != bool:
        raise ValueError('raw bool [H,W] mask required')
    if any(np.asarray(a).shape != raw_mask.shape or np.asarray(a).dtype != bool
           for a in (observed, valid_depth, overlap)) or (observed & ~raw_mask).any():
        raise ValueError('invalid contour support masks')
    contours, _ = cv2.findContours(raw_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = raw_mask.shape
    walks = [canonical_walk(c[:, 0, 1].astype(np.int64)*w+c[:, 0, 0]) for c in contours]
    walks.sort(key=lambda x: tuple(x.tolist()))
    components = []
    ordered = []
    for component, walk in enumerate(walks):
        good = observed.ravel()[walk]
        v, u = np.divmod(walk, w)
        segments = np.full(len(walk), -1, np.int64)
        for sid, (positions, _) in enumerate(observed_runs(good)): segments[positions] = sid
        components.append(dict(component=component, walk=walk, observed=good, observed_segment_id=segments,
            invalid_depth=~valid_depth.ravel()[walk], overlap=overlap.ravel()[walk],
            image_boundary=(v == 0) | (u == 0) | (v == h-1) | (u == w-1),
            walk_position=np.arange(len(walk), dtype=np.int64),
            repeated_source_visits=len(walk)-len(np.unique(walk))))
        ordered.extend(walk[good].tolist())
    return dict(components=components, C=stable_unique(ordered),
        raw_unique=stable_unique([s for walk in walks for s in walk]),
        walk_count=sum(len(w) for w in walks), canonical_component_count=len(walks))

def source_rows(x, source):
    x, source = np.asarray(x), np.asarray(source)
    rows = np.searchsorted(x, source)
    if ((rows >= len(x)).any() or not np.array_equal(x[rows], source)):
        raise ValueError('candidate/selected source outside observed X')
    return rows

def prepare_object(x, xyz, pca, raw_mask, valid_depth, overlap, intrinsic):
    """Observation-only preprocessing, shared once by all five arms."""
    t0 = perf_counter()
    x, xyz = voxel.sources(x, xyz)
    pca = np.asarray(pca)
    voxel.sources(pca, xyz[source_rows(x, pca)])
    if xyz.dtype != np.float32:
        raise ValueError('original float32 geometry required')
    observed = np.zeros(raw_mask.shape, bool)
    observed.ravel()[x] = True
    contours = outer_contours(raw_mask, observed, valid_depth, overlap)
    t1 = perf_counter()
    scale = voxel.nominal_scale(xyz, intrinsic)
    t2 = perf_counter()
    candidates = {'pca': pca.copy(), 'sil': contours['C']}
    groups, reps = {}, {}
    for name, c in candidates.items():
        asc = np.sort(c)
        if len(x):
            group = voxel.representatives(asc, xyz[source_rows(x, asc)], scale['a_m'])
        else:
            group = dict(keys=np.empty((0, 3), np.int64), cells=np.empty((0, 3), np.int64),
                membership=np.empty(0, np.int64), capacity=np.empty(0, np.int64),
                cell_rep_rows=np.empty(0, np.int64), representative_rows=np.empty(0, np.int64),
                same_representative_source=np.empty(0, np.int64), same_cell_distance_m=np.empty(0))
        r = asc[group['representative_rows']]
        # Independent compression is ascending; restore contour order only afterwards.
        reps[name] = c[np.isin(c, r)]
        group['candidate_source_ascending'] = asc
        groups[name] = group
    t3 = perf_counter()
    return dict(X=x.copy(), xyz=xyz.copy(), C=candidates, R=reps, groups=groups,
        contours=contours, scale=scale, shape=raw_mask.shape,
        timings_ms=dict(contour=(t1-t0)*1000, nominal_scale=(t2-t1)*1000, voxel=(t3-t2)*1000))

def select_object(prepared, ratio, phase, requested_surface):
    """No GT, ROI, reference, cap, score, or random first-point input."""
    if type(requested_surface) is not int or requested_surface < 0:
        raise ValueError('nonnegative Surface request required')
    t0 = perf_counter()
    ts = {name: r[thin_indices(len(r), ratio, phase)] for name, r in prepared['R'].items()}
    t1 = perf_counter()
    p = stable_unique(np.r_[ts['pca'], ts['sil']])
    t2 = perf_counter()
    x, xyz = prepared['X'], prepared['xyz']
    count = min(requested_surface, len(x)-len(p))
    initial = source_rows(x, p)
    if not len(initial) and count:
        initial = np.array([0], np.int64)  # deterministic first Surface, inside request
    rows = fps_indices(xyz, len(p)+count, initial=initial)
    f = x[rows]
    np.testing.assert_array_equal(f[:len(p)], p)
    if len(np.unique(f)) != len(f):
        raise AssertionError('duplicate source after Surface')
    t3 = perf_counter()
    return dict(T=ts, P=p, S=f[len(p):], F=f, xyz=xyz[rows].copy(),
        shortfall=requested_surface-count, requested_surface=requested_surface,
        timings_ms=dict(thinning=(t1-t0)*1000, union=(t2-t1)*1000, surface_fps=(t3-t2)*1000))

def make_packet(source_frame, prepared, selected, feature_cases):
    """Exact stored-condition feature join; provenance stays separate from features."""
    t0 = perf_counter()
    fields = None
    object_rows = []
    compact_by_object = []
    for oid in (0, 1):
        prep, s, case = prepared[oid], selected[oid], feature_cases[oid]
        f = s['F']
        available, positions, features = prior.feature_rows(case, f)
        schema = [dict(name=name, shape=list(value.shape[1:]), dtype=value.dtype.str)
                  for name, value in sorted(case['saved_features'].items())]
        if fields is None:
            fields = schema
        elif fields != schema:
            raise ValueError('feature schema differs across objects')
        view = np.full((len(f), sum(int(np.prod(k['shape'])) if k['shape'] else 1 for k in fields)), np.nan)
        offset = 0
        for field in fields:
            values = features[field['name']]
            width = int(np.prod(field['shape'])) if field['shape'] else 1
            view[positions, offset:offset+width] = values.reshape(len(positions), width)
            offset += width
        evidence = np.column_stack([np.isin(f, prep['C'][n]) for n in ('pca', 'sil')])
        chosen = np.column_stack([np.isin(f, s['T'][n]) for n in ('pca', 'sil')])
        surface = np.arange(len(f)) >= len(s['P'])
        object_rows.append(dict(points=s['xyz'], object_id=np.full(len(f), oid, np.int64),
            geometric_features=view, provenance=dict(source_index=f.copy(), object_id=np.full(len(f), oid, np.int64),
                candidate_evidence=evidence, both_evidence=evidence.all(axis=1), selected_by_structural_source=chosen,
                selection_stage=surface.astype(np.uint8), surface_selected=surface, feature_available=available,
                feature_field_available=np.repeat(available[:, None], len(fields), axis=1))))
        compact_by_object.append((positions, features))
    packet = dict(episode='ep000', camera='base_camera', source_frame=int(source_frame),
        coordinate_frame='CV_camera_metres', feature_fields=fields, edge_status='not_implemented',
        intra_object_edge_index=np.empty((2, 0), np.int64))
    for name in ('points', 'object_id', 'geometric_features'):
        packet[name] = np.concatenate([o[name] for o in object_rows])
    packet['provenance'] = {name: np.concatenate([o['provenance'][name] for o in object_rows])
                            for name in object_rows[0]['provenance']}
    offset = len(selected[0]['F'])
    packet['compact_features'] = {name: dict(
        positions=np.r_[compact_by_object[0][0], compact_by_object[1][0]+offset],
        values=np.concatenate([v[name] for _, v in compact_by_object])) for name in compact_by_object[0][1]}
    received = receive_vision_packet(packet)
    if packet_digest(packet) != packet_digest(received.packet):
        raise AssertionError('receiver changed packet rows/identity')
    return packet, (perf_counter()-t0)*1000

from vla.data.packet_io import packet_digest
