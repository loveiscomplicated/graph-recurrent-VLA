"""Task07-G observation-only sampling diagnostics; production builders unchanged."""
from copy import deepcopy
import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from graph.integration_baseline.vision import shape_first
from visual.runtime.sampling import extract_sampling_evidence, extract_candidates, SamplingConfig
from visual.runtime.vision_packet import complete_vision_packet
from visual.runtime.intra_object_edges import observed_components
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede
from graph.vision_input import receive_vision_packet


def prepare(frame):
    evidence = extract_sampling_evidence(frame, estimator='pca5')
    candidates = extract_candidates(evidence, SamplingConfig(), method='S3')
    prepared, cases = {}, {}
    g = evidence.frame.geometry
    for oid in (0, 1):
        obj = candidates[oid]
        orientation = evidence.pairs[oid].candidates(20)['orientation']
        pca = obj.flat_index[orientation.ravel()[obj.flat_index]]
        prepared[oid] = shape_first.prepare_object(obj.flat_index, obj.points, pca,
            frame.masks[oid], g.valid_depth, g.overlap_mask, frame.intrinsic)
        cases[oid] = dict(saved_flat=obj.flat_index, saved_features=obj.metadata)
    maps, _ = observed_components(evidence)
    return evidence, prepared, cases, maps


def local_replace(prep, original, components, fraction):
    """Fixed evenly spaced Surface rows; nearest same-component observed source.

    Candidate: unused in original selection, within 8-neighbour pixel AND nominal
    metric scale, OR same camera-frame nominal voxel. No output/target access.
    Original structural prefix is fixed. No replacement is forced.
    """
    if fraction not in (.01, .05):
        raise ValueError('predeclared fractions only')
    s = deepcopy(original)
    x, xyz, a = prep['X'], prep['xyz'], prep['scale']['a_m']
    requested = int(math.ceil(len(s['S'])*fraction))
    chosen = np.linspace(0, len(s['S'])-1, requested, dtype=int) if requested else []
    unused = ~np.isin(x, original['F'])
    width = prep['shape'][1]
    cells = np.floor(xyz.astype(np.float64)/a).astype(np.int64) if len(x) else np.empty((0, 3), int)
    changes = []
    for j in chosen:
        old = int(original['S'][j]); row = int(shape_first.source_rows(x, [old])[0])
        distance = np.linalg.norm(xyz.astype(np.float64)-xyz[row].astype(np.float64), axis=1)
        pixel = np.hypot(x//width-old//width, x%width-old%width)
        eligible = unused & (components.ravel()[x] == components.ravel()[old])
        eligible &= ((pixel <= np.sqrt(2)) & (distance <= a)) | (cells == cells[row]).all(1)
        rows = np.flatnonzero(eligible)
        if not len(rows):
            continue
        pick = rows[np.lexsort((x[rows], pixel[rows], distance[rows]))[0]]
        unused[pick] = False
        s['S'][j] = x[pick]
        changes.append([old, int(x[pick])])
    s['F'] = np.r_[s['P'], s['S']]
    s['xyz'] = xyz[shape_first.source_rows(x, s['F'])].copy()
    np.testing.assert_array_equal(s['F'][:len(s['P'])], original['P'])
    assert len(s['F']) == len(original['F']) == len(np.unique(s['F']))
    return s, dict(label='CONTROLLED_LOCAL_REPLACEMENT', fraction=fraction,
        surface_count=len(original['S']), requested=requested, actual=len(changes),
        actual_surface_fraction=len(changes)/len(original['S']) if len(original['S']) else 0.,
        pairs=changes)


def build(frame, identity, evidence, prepared, cases, maps, condition):
    selected, replacements = {}, {}
    for oid in (0, 1):
        selected[oid] = shape_first.select_object(prepared[oid], .5,
            int(condition == 'phase1'), shape_first.SURFACE[oid])
        if condition.startswith('local'):
            selected[oid], replacements[oid] = local_replace(prepared[oid], selected[oid], maps[oid],
                .01 if condition == 'local01' else .05)
    selection, _ = shape_first.make_packet(frame.source_frame, prepared, selected, cases)
    selection.update(episode=identity, camera=frame.camera)
    packet = complete_vision_packet(selection, evidence, frame)
    source = packet['provenance']['source_index']
    component = np.empty(len(source), np.int64)
    for oid in (0, 1):
        rows = packet['object_id'] == oid
        component[rows] = maps[oid].ravel()[source[rows]]
        assert (component[rows] >= 0).all()
        assert np.isin(source[rows], prepared[oid]['X']).all()
    packet['observed_component_id'] = component
    edge, metadata, _ = build_raw_ede(packet)
    packet.update(metadata, edge_status='supplied', intra_object_edge_index=edge)
    receive_vision_packet(packet)
    return packet, replacements


def assert_packet_equal(a, b, path='packet'):
    """Exact recursive equality, including NaNs/dtypes and all provenance rows."""
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape, path
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys() ^ b.keys())
        for k in a:
            assert_packet_equal(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_packet_equal(x, y, path+'/'+str(i))
    else:
        assert a == b, (path, a, b)


def permute_graph(graph, rng):
    """Object node rows permuted; three identity-addressed robot slots stay fixed."""
    result = deepcopy(graph)
    order = np.r_[np.arange(3), rng.permutation(np.arange(3, len(graph.node_type)))]
    inverse = np.argsort(order)
    result.node_input = graph.node_input[order]
    result.node_type = graph.node_type[order]
    result.edge_index = torch.as_tensor(inverse)[graph.edge_index]
    # Edge column order/attributes/types describe the same directed relations.
    return result


def edge_keys(graph, kind, object_id=None, common_sources=None):
    p = graph.packet
    ids = [('robot', i) for i in range(3)] + list(zip(p['object_id'].tolist(), p['provenance']['source_index'].tolist()))
    edges = graph.edge_index.numpy()[:, graph.edge_type.numpy() == kind].T
    if object_id is not None:
        edges = np.asarray([e for e in edges if ids[e[0]][0] == object_id], dtype=int).reshape(-1, 2)
    keys = {(ids[i], ids[j]) for i, j in edges}
    if common_sources is not None:
        keys = {e for e in keys if all(i[0] == 'robot' or i in common_sources for i in e)}
    return keys


def jaccard(a, b):
    return len(a & b)/len(a | b) if a | b else 1.


def sampling_metrics(ga, gb, prepared):
    a, b = ga.packet, gb.packet
    result = []
    for oid in (0, 1):
        ia, ib = [np.flatnonzero(p['object_id'] == oid) for p in (a, b)]
        sa, sb = [p['provenance']['source_index'][i] for p, i in ((a, ia), (b, ib))]
        common, ca, cb = np.intersect1d(sa, sb, return_indices=True)
        xa, xb = a['points'][ia].astype(float), b['points'][ib].astype(float)
        nn = np.r_[cKDTree(xa).query(xb)[0], cKDTree(xb).query(xa)[0]] if len(xa) and len(xb) else np.empty(0)
        prep = prepared[oid]; scale = prep['scale']['a_m']
        divisor = scale if scale is not None else 1.
        vox = [set(map(tuple, np.floor(p['camera_points'][i].astype(float)/divisor).astype(int))) for p, i in ((a, ia), (b, ib))]
        fullvox = set(map(tuple, np.floor(prep['xyz'].astype(float)/divisor).astype(int)))
        coverage = []
        for p, i in ((a, ia), (b, ib)):
            d = cKDTree(p['camera_points'][i]).query(prep['xyz'])[0] if len(i) else np.empty(0)
            coverage.append(dict(support_nn_mean_m=float(d.mean()) if len(d) else None,
                support_nn_p95_m=float(np.quantile(d, .95)) if len(d) else None))
        np.testing.assert_array_equal(a['geometric_features'][ia[ca]], b['geometric_features'][ib[cb]])
        feature_nn = None
        if len(xa) and len(xb):
            nearest = cKDTree(xa).query(xb)[1]
            fa, fb = a['geometric_features'][ia[nearest]], b['geometric_features'][ib]
            feature_nn = []
            for j in range(5):
                good = np.isfinite(fa[:, j]) & np.isfinite(fb[:, j])
                feature_nn.append(float(np.mean(np.abs(fa[good, j]-fb[good, j]))) if good.any() else None)
        structural = [int((~p['provenance']['surface_selected'][i]).sum()) for p, i in ((a, ia), (b, ib))]
        valid = [p['feature_field_valid'][i].mean(0).tolist() if len(i) else [None]*3 for p, i in ((a, ia), (b, ib))]
        means = []
        for p, i in ((a, ia), (b, ib)):
            f = p['geometric_features'][i]
            means.append([float(f[np.isfinite(f[:, j]), j].mean()) if np.isfinite(f[:, j]).any() else None for j in range(5)])
        result.append(dict(object_id=oid, count_a=len(sa), count_b=len(sb), common=len(common),
            source_jaccard=jaccard(set(sa), set(sb)), presence_equal=bool(len(sa)>0)==bool(len(sb)>0),
            structural_a=structural[0], structural_b=structural[1], surface_a=len(sa)-structural[0], surface_b=len(sb)-structural[1],
            chamfer_mean_m=float(nn.mean()) if len(nn) else None,
            nn_p50_m=float(np.median(nn)) if len(nn) else None,
            nn_p95_m=float(np.quantile(nn, .95)) if len(nn) else None,
            nn_max_m=float(nn.max()) if len(nn) else None, voxel_scale_m=scale,
            voxel_jaccard=jaccard(*vox), voxel_coverage_a=len(vox[0])/len(fullvox) if fullvox else None,
            voxel_coverage_b=len(vox[1])/len(fullvox) if fullvox else None,
            coverage_a=coverage[0], coverage_b=coverage[1], f5_nearest_abs_difference=feature_nn,
            f5_mean_a=means[0], f5_mean_b=means[1], validity_a=valid[0], validity_b=valid[1],
            common_source_f5_exact=True, raw_ede_jaccard=jaccard(edge_keys(ga, 0, oid), edge_keys(gb, 0, oid))))
    shared = set(zip(a['object_id'].tolist(), a['provenance']['source_index'].tolist())) & set(zip(b['object_id'].tolist(), b['provenance']['source_index'].tolist()))
    return dict(objects=result, edge_counts_a=torch.bincount(ga.edge_type, minlength=3).tolist(),
        edge_counts_b=torch.bincount(gb.edge_type, minlength=3).tolist(),
        edge_jaccard=[jaccard(edge_keys(ga,k), edge_keys(gb,k)) for k in range(3)],
        common_induced_edge_jaccard=[jaccard(edge_keys(ga,k,common_sources=shared), edge_keys(gb,k,common_sources=shared)) for k in range(3)])
