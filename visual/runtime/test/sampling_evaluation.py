"""Evaluation-only GT matching and surface/false-selection metrics for Task03."""
from dataclasses import replace
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.ndimage import label
from visual.runtime.test.normal_diagnostics import stats, dilate, endpoints, erode
from visual.runtime.test.run_normal_comparison import real_truth
from visual.runtime.test.run_structural_edges import raster_reference, rim_reference, negative_regions
from visual.runtime.rgbd import PEG, BOX

TOLERANCES = ((1., .002), (2., .003), (3., .005))


class CoverageMatcher:
    """Same-pair joint 2D/3D matching. No reference-dependent sampling inputs.

    Distances are unfiltered by a GT corridor. Missing selected points yield
    zero hits; distances then stay undefined and missing_reference_distances is
    explicit. Sampling retention denominator is only tracker-recoverable refs.
    """
    def __init__(self, geometry, oid, reference, tolerances=TOLERANCES):
        self.flat = np.flatnonzero(geometry.observed_mask(oid))
        self.uv = np.column_stack(np.unravel_index(self.flat, geometry.valid_depth.shape))[:, ::-1]
        self.xyz = geometry.xyz.reshape(-1, 3)[self.flat]
        self.ref_uv, self.ref_xyz = reference
        self.nref = len(self.ref_uv)
        self.tolerances = tolerances
        self.adjacencies = {}
        if len(self.flat) and self.nref:
            self.to_gt_px = cKDTree(self.ref_uv).query(self.uv)[0]
            self.to_gt_mm = cKDTree(self.ref_xyz).query(self.xyz)[0]*1000
        else:
            self.to_gt_px = np.full(len(self.flat), np.nan)
            self.to_gt_mm = self.to_gt_px.copy()
        for px, m in tolerances:
            if len(self.flat) and self.nref:
                matrix = cKDTree(self.uv).sparse_distance_matrix(cKDTree(self.ref_uv), px, output_type='coo_matrix')
                good = np.linalg.norm(self.xyz[matrix.row]-self.ref_xyz[matrix.col], axis=1) <= m
                adj = csr_matrix((np.ones(good.sum(), bool), (matrix.row[good], matrix.col[good])), shape=(len(self.flat), self.nref))
            else: adj = csr_matrix((len(self.flat), self.nref), dtype=bool)
            self.adjacencies[self.key(px,m)] = adj

    @staticmethod
    def key(px,m): return f'{px:g}px_{m*1000:g}mm'

    def reference_hits(self, selected_flat, px=2., m=.003):
        """Reference IDs and matched source nodes from the existing joint adjacency.

        IDs are row positions in the saved reference arrays, not rounded pixels.
        A node can cover multiple IDs; multiple nodes covering one ID count once.
        """
        selected_flat = np.asarray(selected_flat)
        if selected_flat.ndim != 1 or selected_flat.dtype.kind not in 'iu':
            raise ValueError('source indices must be a one-dimensional integer array')
        if len(np.unique(selected_flat)) != len(selected_flat):
            raise ValueError('duplicate source nodes')
        rows = np.searchsorted(self.flat, selected_flat)
        if len(rows) and ((rows >= len(self.flat)).any()
                          or not np.array_equal(self.flat[rows], selected_flat)):
            raise ValueError('evaluation points outside observed support')
        adjacency = self.adjacencies[self.key(px, m)][rows]
        return (np.flatnonzero(adjacency.getnnz(axis=0)),
                selected_flat[adjacency.getnnz(axis=1) > 0])

    def score(self, selected_flat):
        rows = np.searchsorted(self.flat, selected_flat)
        if len(rows) and ((rows >= len(self.flat)).any() or not np.array_equal(self.flat[rows], selected_flat)):
            raise ValueError('evaluation points outside observed support')
        out = dict(selected=len(rows), gt_visible=self.nref, tolerance={})
        arrays = dict(sample_to_gt_px=self.to_gt_px[rows], sample_to_gt_mm=self.to_gt_mm[rows])
        if len(rows) and self.nref:
            arrays['gt_to_sample_px'] = cKDTree(self.uv[rows]).query(self.ref_uv)[0]
            arrays['gt_to_sample_mm'] = cKDTree(self.xyz[rows]).query(self.ref_xyz)[0]*1000
        else:
            arrays['gt_to_sample_px'] = np.empty(0); arrays['gt_to_sample_mm'] = np.empty(0)
        out['missing_reference_distances'] = self.nref if not len(rows) else 0
        for key, adjacency in self.adjacencies.items():
            observable = adjacency.getnnz(axis=0) > 0
            covered = adjacency[rows].getnnz(axis=0) > 0
            matched = adjacency[rows].getnnz(axis=1) > 0
            row = dict(gt_visible=self.nref, tracker_observed=int(observable.sum()), sampled=int(covered.sum()),
                       selected=len(rows), selected_near_gt=int(matched.sum()))
            # Every visible reference remains in the denominator, including tracker losses.
            row.update(gt_coverage=row['sampled']/self.nref if self.nref else None,
                       tracker_coverage=row['tracker_observed']/self.nref if self.nref else None,
                       sampling_retention=row['sampled']/row['tracker_observed'] if row['tracker_observed'] else None)
            # Spatial distribution of preservation, fixed image-axis halves of visible references.
            if self.nref:
                mid = np.median(self.ref_uv, axis=0)
                quadrants = (self.ref_uv[:,0] >= mid[0]).astype(int)+2*(self.ref_uv[:,1] >= mid[1]).astype(int)
                row['reference_quadrants'] = [dict(gt_visible=int((quadrants==q).sum()),
                    tracker_observed=int((observable & (quadrants==q)).sum()), sampled=int((covered & (quadrants==q)).sum())) for q in range(4)]
            else: row['reference_quadrants'] = []
            out['tolerance'][key] = row
        out['distances'] = {k:stats(v) for k,v in arrays.items()}
        return out, arrays


def count_ratio(numerator, denominator):
    """Preserve raw counts, including undefined zero-denominator fractions."""
    return dict(numerator=int(numerator), denominator=int(denominator),
                value=numerator/denominator if denominator else None)


def staged_reference_coverage(matcher, candidates, priority, surface, final):
    """Saved-set diagnosis only; no candidate extraction or selection is run.

    C/P are None for the General FPS control, rather than empty candidate sets.
    Set differences overlap and must not be summed as causal loss fractions.
    """
    if (candidates is None) != (priority is None):
        raise ValueError('candidate and priority applicability must agree')
    nodes = dict(X=matcher.flat, C=candidates, P=priority, S=surface, F=final)
    hits, stages = {}, {}
    for name, flat in nodes.items():
        if flat is None:
            hits[name] = None
            stages[name] = None
            continue
        ids, matched = matcher.reference_hits(flat)
        hits[name] = set(ids.tolist())
        stages[name] = dict(node_count=len(flat), matched_node_count=len(matched),
                            unique_reference_count=len(ids))
    p = np.asarray(priority if priority is not None else [], dtype=np.int64)
    if np.intersect1d(p, surface).size or set(final) != set(p) | set(surface):
        raise ValueError('final nodes must be the disjoint priority/surface union')
    if candidates is not None and not set(p) <= set(candidates):
        raise ValueError('priority nodes outside candidate support')
    v, o = set(range(matcher.nref)), hits['X']
    c, p, s, f = hits['C'], hits['P'], hits['S'], hits['F']
    if not f <= o or (c is not None and not c <= o):
        raise AssertionError('covered references outside observable references')
    if f != (p or set()) | s:
        raise AssertionError('final reference union differs from priority/surface')
    differences = {'V_minus_O': v-o, 'O_minus_F': o-f}
    if c is not None:
        differences.update(O_minus_C=o-c, C_minus_P=c-p, C_minus_F=c-f,
                           P_intersect_S=p&s, S_minus_P=s-p, S_minus_C=s-c,
                           C_minus_P_intersect_S=(c-p)&s)
    else:
        differences.update({key: None for key in ('O_minus_C', 'C_minus_P', 'C_minus_F',
                            'P_intersect_S', 'S_minus_P', 'S_minus_C', 'C_minus_P_intersect_S')})
    coverage = dict(observation=count_ratio(len(o), len(v)))
    for name, key in [('C', 'candidate'), ('P', 'priority'), ('S', 'surface'), ('F', 'final_retention')]:
        coverage[key] = count_ratio(len(hits[name]), len(o)) if hits[name] is not None else None
    return dict(stages=stages, coverage=coverage,
                gt_final_coverage=count_ratio(len(f), len(v)),
                reference_ids={k: sorted(ids) if ids is not None else None
                               for k, ids in dict(V=v, O=o, **hits).items()},
                differences={k: dict(count=len(ids), reference_ids=sorted(ids)) if ids is not None else None
                             for k, ids in differences.items()})


def evaluation_context(frame, evidence, gt, camera, robot_ids):
    """Reuse unchanged Task02 visible seam/rim definition; GT read after extraction."""
    truths, regions, maps = real_truth(frame, evidence.frame, gt, camera)
    refs, matchers, negatives, surfaces = {}, {}, {}, {}
    for name, oid in [('peg',PEG), ('box',BOX)]:
        t = truths[name]; r = regions[name]
        robot = np.isin(gt[f'{camera}/seg'][frame.source_frame],robot_ids) & np.isfinite(t['depth']) & evidence.frame.geometry.valid_depth & (frame.depth < t['depth']-.0015)
        r['robot_occlusion_2px'] = dilate(endpoints(r['gt_visible'],robot),2) & r['gt_visible']
        if name == 'box':
            r['plane_interior'] = np.logical_or.reduce([erode(r['gt_visible'] & (t['face']==f),3) for f in range(6)]) & ~dilate(r['hole_rim'],4)
        reference = raster_reference(maps[name]['seam'],t,frame.intrinsic) if name=='peg' else rim_reference(frame,gt,t,r['hole_rim'])
        refs[name] = reference
        matchers[name] = CoverageMatcher(evidence.frame.geometry,oid,reference)
        # Include other physical box seams when excluding negatives, exactly as Task02-C.
        other = raster_reference(maps[name]['seam'], t, frame.intrinsic)
        all_ref = tuple(np.concatenate([reference[i], other[i]]) for i in (0,1))
        negatives[name] = negative_regions(r, maps[name], all_ref, frame.depth.shape)
        support = evidence.frame.geometry.observed_mask(oid)
        flat = np.flatnonzero(support)
        components, count = label(support)
        comp = components.ravel()[flat]
        sizes = np.bincount(comp, minlength=count+1)
        surfaces[name] = dict(flat=flat, xyz=evidence.frame.geometry.xyz.reshape(-1,3)[flat],
                             plane=r['plane_interior'].ravel()[flat], face=t['face'].ravel()[flat],
                             small_component=(sizes[comp] <= 64), component_count=count)
    return dict(truths=truths, regions=regions, maps=maps, references=refs, matchers=matchers,
                negatives=negatives, surfaces=surfaces)


def score_selection(result, context):
    row, arrays = dict(objects={}), {}
    for name, oid in [('peg',PEG),('box',BOX)]:
        choose = result.object_id==oid
        flat = result.flat_index[choose]
        structural, dist = context['matchers'][name].score(flat)
        arrays.update({name+'/'+k:v for k,v in dist.items()})
        surface = context['surfaces'][name]
        nearest = cKDTree(result.points[choose]).query(surface['xyz'])[0]*1000 if choose.any() else np.empty(0)
        surface_dist = {'all':nearest}
        if len(nearest):
            surface_dist.update(plane=nearest[surface['plane']],small_components=nearest[surface['small_component']])
            surface_dist.update({f'face_{f}':nearest[surface['face']==f] for f in range(6)})
        surface_metrics = {k:dict(distance_mm=stats(v), within_3mm=int((v<=3).sum()), points=len(v)) for k,v in surface_dist.items()}
        surface_metrics['component_count'] = surface['component_count']
        arrays.update({name+'/surface_'+k:v for k,v in surface_dist.items()})
        priority = result.priority_selected[choose]
        negative = {}
        for k, mask in context['negatives'][name].items():
            hit = mask.ravel()[flat]
            negative[k] = dict(observed=int(mask.ravel()[surface['flat']].sum()),
                               selected=int(hit.sum()), priority=int((hit&priority).sum()),
                               depth_priority=int((hit&result.depth_priority_selected[choose]).sum()))
        # Union is evaluated once; region counts deliberately overlap.
        false_union = np.logical_or.reduce(list(context['negatives'][name].values()))
        hit = false_union.ravel()[flat]
        negative['known_negative_union'] = dict(observed=int(false_union.ravel()[surface['flat']].sum()),
            selected=int(hit.sum()), priority=int((hit&priority).sum()), depth_priority=int((hit&result.depth_priority_selected[choose]).sum()))
        row['objects'][name] = dict(structural=structural,surface=surface_metrics,negative=negative,counts=result.counts[oid])
    row['timings_ms'] = result.timings_ms
    return row, arrays


def summarize(rows, arrays):
    """Micro counts plus per-frame coverage distributions; no independent-frame CI."""
    result = dict(frames=len(rows),objects={})
    for name in ('peg','box'):
        objects = [r['objects'][name] for r in rows]
        counts = {k:sum(o['counts'][k] for o in objects) for k in objects[0]['counts']}
        structural = {}
        for tol in objects[0]['structural']['tolerance']:
            original = [o['structural']['tolerance'][tol] for o in objects]
            summed = {k:sum(o[k] for o in original) for k in ('gt_visible','tracker_observed','sampled','selected','selected_near_gt')}
            n,obs,sel = summed['gt_visible'],summed['tracker_observed'],summed['sampled']
            summed.update(gt_coverage=sel/n if n else None,tracker_coverage=obs/n if n else None,
                          sampling_retention=sel/obs if obs else None,
                          frame_gt_coverage=stats([v['gt_coverage'] for v in original if v['gt_coverage'] is not None]),
                          frame_sampling_retention=stats([v['sampling_retention'] for v in original if v['sampling_retention'] is not None]))
            structural[tol] = summed
        negatives = {k:{c:sum(o['negative'][k][c] for o in objects) for c in objects[0]['negative'][k]} for k in objects[0]['negative']}
        for v in negatives.values():
            v['fraction_of_priority'] = v['priority']/counts['priority'] if counts['priority'] else None
            v['fraction_of_selected_budget'] = v['priority']/counts['selected'] if counts['selected'] else None
        distance = {k:stats(np.concatenate(v)) for k,v in arrays.items() if k.startswith(name+'/')}
        surfaces = {}
        for key in objects[0]['surface']:
            if key=='component_count': continue
            valid = [o['surface'][key] for o in objects if key in o['surface']]
            n=sum(v['points'] for v in valid); hit=sum(v['within_3mm'] for v in valid)
            surfaces[key] = dict(points=n,within_3mm=hit,coverage_3mm=hit/n if n else None,
                                 distance_mm=distance.get(name+'/surface_'+key))
        result['objects'][name] = dict(counts=counts,structural=structural,surface=surfaces,negative=negatives,
                                      distances=distance,missing_reference_distances=sum(o['structural']['missing_reference_distances'] for o in objects))
    result['timings_ms'] = {k:stats([r['timings_ms'][k] for r in rows]) for k in rows[0]['timings_ms']}
    return result
