"""Current-observation Structural priority / Surface FPS, internal NumPy metadata.

No GT, cache, final node features, graph edges or Vision--Graph packaging. XYZ
remains in the source CV-camera frame. Selection roles are not semantic types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from time import perf_counter

import numpy as np

from visual.runtime.geometry import (
    GeometryFrame, GeometryParameters, estimate_surface_normals,
)
from visual.runtime.pca import build_geometry, cropped_features, local_plane_optimized
from visual.runtime.normal_estimators import finite_difference
from visual.runtime.edge_evidence import PairEvidence, pair_evidence
from visual.runtime.rgbd import PEG, BOX, TrackedRGBDFrame

METHODS = ('S0', 'S1', 'S2', 'S3', 'S4')


def _integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


@dataclass(frozen=True)
class SamplingConfig:
    """Experimental defaults, not adopted architecture/hyperparameters.

    Reserve minimums first, then apportion remaining capacity by observed point
    counts (or their square roots). Largest remainder, capped by availability;
    exact ties use ascending object ID. Structural slots are a GLOBAL maximum
    floor(num_points*ratio), including independently capped S4 depth-only slots.
    """
    num_points: int = 256
    structural_ratio: float = .25
    min_points: tuple[int, int] = (16, 16)
    allocation: str = 'observed_points'
    seed: int = 0
    orientation_threshold_deg: float = 20.
    depth_priority_fraction: float = .25
    sampling: str = 'fps'

    def __post_init__(self):
        _integer(self.num_points, 'num_points'); _integer(self.seed, 'seed')
        if len(self.min_points) != 2:
            raise ValueError('min_points must contain Peg and Box minimums')
        for n in self.min_points: _integer(n, 'object minimum')
        if sum(self.min_points) > self.num_points:
            raise ValueError('sum of object minimums exceeds num_points')
        for name in ('structural_ratio', 'depth_priority_fraction'):
            x = getattr(self, name)
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f'{name} must be finite in [0,1]')
        if not np.isfinite(self.orientation_threshold_deg) or not 0 <= self.orientation_threshold_deg <= 180:
            raise ValueError('orientation threshold must be in [0,180] degrees')
        if self.allocation not in ('observed_points', 'sqrt_observed_points'):
            raise ValueError('unknown allocation rule')
        if self.sampling != 'fps': raise ValueError('only fps sampling is supported')


def allocate_budget(total, capacities, minimums=(0, 0), weights=None):
    """Capped proportional integer apportionment, deterministic and availability aware."""
    _integer(total, 'budget')
    cap = np.asarray(capacities)
    mins = np.asarray(minimums)
    if cap.shape != (2,) or mins.shape != (2,): raise ValueError('expected two object counts')
    for x in (*cap, *mins): _integer(x, 'capacity/minimum')
    if mins.sum() > total: raise ValueError('minimums exceed total budget')
    cap, mins = cap.astype(np.int64), mins.astype(np.int64)
    weight = cap.astype(float) if weights is None else np.asarray(weights, float)
    if weight.shape != (2,) or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('weights must be two finite nonnegative values')
    out = np.minimum(cap, mins)
    left = min(total, int(cap.sum()))-int(out.sum())
    while left:
        room = cap-out
        w = np.where(room > 0, weight, 0)
        if not w.any(): w = (room > 0).astype(float)
        quota = left*w/w.sum()
        take = np.minimum(room, np.floor(quota).astype(np.int64))
        out += take; left -= int(take.sum())
        # Largest fractional remainder; saturated objects are skipped. If a
        # quota exceeded capacity, the next pass redistributes the surplus.
        order = np.argsort(-(quota-np.floor(quota)), kind='stable')
        for i in order:
            if out[i] < cap[i] and left: out[i] += 1; left -= 1
    return out


def fps_indices(points, count, *, seed=0, initial=()):
    """Legacy pointcloud.fps distance update, returning unique source row indices.

    No random pre-reduction. Uniform seeded first row; argmax ties choose the
    first source row. An initial selection seeds distances for Surface coverage.
    All initial rows precede newly selected rows. Even coincident XYZ is never
    selected twice by row index. Computes squared distances in float64.
    """
    _integer(count, 'FPS count'); _integer(seed, 'FPS seed')
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('FPS requires finite [N,3] observed coordinates')
    initial = np.asarray(initial, dtype=np.int64)
    if initial.ndim != 1 or len(np.unique(initial)) != len(initial) or ((initial < 0) | (initial >= len(xyz))).any():
        raise ValueError('initial FPS rows must be distinct valid indices')
    count = min(count, len(xyz))
    if len(initial) > count: raise ValueError('initial rows exceed FPS count')
    if not count: return np.empty(0, np.int64)
    chosen = list(initial)
    distance = np.full(len(xyz), np.inf)
    x, y, z = (np.ascontiguousarray(xyz[:, i]) for i in range(3))
    def update(j):
        np.minimum(distance, (x-x[j])**2+(y-y[j])**2+(z-z[j])**2, out=distance)
        distance[j] = -np.inf
    for j in chosen: update(j)
    if not chosen:
        chosen.append(int(np.random.default_rng(seed).integers(len(xyz))))
        update(chosen[-1])
    while len(chosen) < count:
        j = int(distance.argmax()); chosen.append(j); update(j)
    return np.asarray(chosen, np.int64)


@dataclass(frozen=True)
class SamplingEvidence:
    """Detector output before thresholding. Contains observations only."""
    frame: GeometryFrame
    pairs: dict[int, PairEvidence]
    estimator: str
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class ObjectCandidates:
    """Only valid observed rows, in original raster order; all evidence stays aligned."""
    flat_index: np.ndarray
    points: np.ndarray
    metadata: dict[str, np.ndarray]


@dataclass(frozen=True)
class SampledPoints:
    """Internal selection, variable N; no padding, no final GNN feature encoding.

    priority_selected / depth_priority_selected record selection paths, whereas
    metadata['structural_candidate'] records detector eligibility. Surface FPS
    may select more detector candidates than the structural priority slot cap.
    """
    source_frame: int
    camera: str
    points: np.ndarray
    pixels_vu: np.ndarray
    flat_index: np.ndarray
    object_id: np.ndarray
    priority_selected: np.ndarray
    depth_priority_selected: np.ndarray
    metadata: dict[str, np.ndarray]
    object_budgets: dict[int, int]
    counts: dict[int, dict[str, int]]
    timings_ms: dict[str, float] = field(default_factory=dict)


def extract_sampling_evidence(frame: TrackedRGBDFrame, estimator='pca5',
                              parameters=GeometryParameters()):
    """Reuse Task02 estimators and pair operator; optimized PCA5 analytic solver."""
    if estimator not in ('baseline', 'fd2', 'pca5'): raise ValueError('unknown normal estimator')
    t0 = perf_counter(); geometry = build_geometry(frame); t1 = perf_counter()
    normals = {}
    for oid in (PEG, BOX):
        if estimator == 'baseline': normals[oid] = estimate_surface_normals(geometry, oid, parameters)
        elif estimator == 'fd2': normals[oid] = finite_difference(geometry, oid, 2, parameters)
        else: normals[oid] = local_plane_optimized(geometry, oid, parameters, solver='analytic')
    t2 = perf_counter()
    features = {i: cropped_features(geometry, i, *normals[i], parameters) for i in (PEG, BOX)}
    t3 = perf_counter()
    pairs = {i: pair_evidence(geometry, i, *normals[i], parameters) for i in (PEG, BOX)}
    t4 = perf_counter()
    timing = dict(geometry=(t1-t0)*1000, normals=(t2-t1)*1000,
                  features=(t3-t2)*1000, pair_evidence=(t4-t3)*1000, extraction_total=(t4-t0)*1000)
    result = GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx, frame.timestamp,
                           geometry, features, timing)
    return SamplingEvidence(result, pairs, estimator, timing)


def extract_candidates(evidence: SamplingEvidence, config: SamplingConfig, method='S3'):
    """Threshold at observed pixels; boundary flags alone never imply eligibility."""
    if method not in METHODS: raise ValueError('unknown sampling method')
    expected = dict(S1='baseline', S2='fd2', S3='pca5', S4='pca5')
    if method in expected and evidence.estimator != expected[method]:
        raise ValueError('method and normal estimator disagree')
    g = evidence.frame.geometry
    result = {}
    for oid in (PEG, BOX):
        idx = np.flatnonzero(g.observed_mask(oid))
        f = evidence.frame.objects[oid]
        maps = evidence.pairs[oid].candidates(config.orientation_threshold_deg)
        metadata = {name: getattr(f, name).reshape((-1, 3) if name == 'normals' else (-1,))[idx].copy()
                    for name in f.__dataclass_fields__}
        metadata.update({f'{name}_evidence': m.ravel()[idx] for name, m in maps.items() if name != 'union'})
        orientation = maps['orientation'].ravel()[idx]
        depth_only = maps['depth'].ravel()[idx] & ~orientation
        metadata.update(orientation_score_deg=np.rad2deg(f.normal_variation_rad.ravel()[idx]),
                        orientation_candidate=orientation, depth_only_candidate=depth_only,
                        structural_candidate=orientation | (depth_only if method == 'S4' and config.depth_priority_fraction > 0 else False))
        result[oid] = ObjectCandidates(idx, g.xyz.reshape(-1, 3)[idx].copy(), metadata)
    return result


def sample_points(evidence: SamplingEvidence, config=SamplingConfig(), *, method='S3', fps_fn=None):
    """Object budgets -> Structural FPS -> remaining-support FPS seeded by priority."""
    fps = fps_indices if fps_fn is None else fps_fn
    t0 = perf_counter(); objects = extract_candidates(evidence, config, method); t1 = perf_counter()
    available = np.array([len(objects[i].points) for i in (PEG, BOX)])
    weights = np.sqrt(available) if config.allocation == 'sqrt_observed_points' else available
    budgets = allocate_budget(config.num_points, available, config.min_points, weights)
    cap = int(np.floor(config.num_points*config.structural_ratio)) if method != 'S0' else 0
    capacities = np.array([min(budgets[i], int(objects[i].metadata['structural_candidate'].sum())) for i in (PEG, BOX)])
    slots = allocate_budget(cap, capacities, weights=budgets)
    depth_cap = int(np.floor(cap*config.depth_priority_fraction)) if method == 'S4' else 0
    depth_capacity = [min(slots[i], int(objects[i].metadata['depth_only_candidate'].sum())) for i in (PEG, BOX)]
    depth_slots = allocate_budget(depth_cap, depth_capacity, weights=budgets)
    t2 = perf_counter()
    selections, priority_sizes, depth_sizes = {}, {}, {}
    for i in (PEG, BOX):
        obj = objects[i]; m = obj.metadata
        # Each object always uses its own stream, invariant to missing other objects.
        seed = int(np.random.SeedSequence([config.seed, i]).generate_state(1)[0])
        oi = np.flatnonzero(m['orientation_candidate'])
        di = np.flatnonzero(m['depth_only_candidate'])
        # Unused depth reservation can be used by orientation, never vice versa.
        dn = min(int(depth_slots[i]), len(di))
        on = min(int(slots[i])-dn, len(oi))
        orientation = oi[fps(obj.points[oi], on, seed=seed)]
        depth = di[fps(obj.points[di], dn, seed=seed,)]
        selections[i] = np.concatenate((orientation, depth))
        priority_sizes[i], depth_sizes[i] = on+dn, dn
    t3 = perf_counter()
    for i in (PEG, BOX):
        seed = int(np.random.SeedSequence([config.seed, i]).generate_state(1)[0])
        selections[i] = fps(objects[i].points, int(budgets[i]), seed=seed, initial=selections[i])
    t4 = perf_counter()
    flat, xyz, ids, priority, depth_priority, metadata = [], [], [], [], [], {}
    counts = {}
    for i in (PEG, BOX):
        obj = objects[i]; rows = selections[i]; n = len(rows); pn = priority_sizes[i]; dn = depth_sizes[i]
        flat.append(obj.flat_index[rows]); xyz.append(obj.points[rows]); ids.append(np.full(n, i, np.int64))
        priority.append(np.arange(n) < pn)
        depth_priority.append((np.arange(n) >= pn-dn) & (np.arange(n) < pn))
        for name, values in obj.metadata.items(): metadata.setdefault(name, []).append(values[rows])
        counts[i] = dict(observed=len(obj.points), budget=int(budgets[i]), candidates=int(obj.metadata['structural_candidate'].sum()),
                         priority=pn, orientation_priority=pn-dn, depth_priority=dn, surface_fps=n-pn,
                         selected_candidates=int(obj.metadata['structural_candidate'][rows].sum()), selected=n)
    flat = np.concatenate(flat)
    timing = dict(candidate_selection=(t1-t0)*1000, budget_allocation=(t2-t1)*1000,
                  structural_sampling=(t3-t2)*1000, surface_fps=(t4-t3)*1000)
    result = SampledPoints(evidence.frame.source_frame, evidence.frame.camera, np.concatenate(xyz),
                          np.column_stack(np.unravel_index(flat, evidence.frame.geometry.valid_depth.shape)),
                          flat, np.concatenate(ids), np.concatenate(priority), np.concatenate(depth_priority),
                          {k: np.concatenate(v) for k, v in metadata.items()},
                          {i: int(budgets[i]) for i in (PEG, BOX)}, counts, timing)
    timing['sampling_total'] = (perf_counter()-t0)*1000
    timing['geometry_sampling_total'] = evidence.timings_ms['extraction_total']+timing['sampling_total']
    return result
