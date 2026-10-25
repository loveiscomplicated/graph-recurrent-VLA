"""Task05-B fixed offline comparison. No sampler/estimator/simulator calls.

Numerics and report-only rendering are separate. All four methods execute
independently for benchmark cost; DE's correctness reference is frozen T0.
"""
import argparse
from collections import Counter
import copy
import difflib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np
import scipy
from scipy.spatial import _qhull

from graph.vision_input import receive_vision_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.extended_delaunay_edges import (
    DE_POLICY, EDE_POLICY, build_extended_delaunay_edges, edge_metadata, collinear_order,
)
from visual.runtime.intra_object_edges import EDGE_POLICY, build_intra_object_edges
from visual.runtime.triangle_intra_object_edges import (
    TRIANGLE_POLICY, triangles_to_edges, prepare_validation, supercover, validate_cover,
    build_triangle_intra_object_edges,
)
from visual.runtime.test.run_vision_feature_edge import read_packet, save_packet, array_bytes
from visual.runtime.test.run_adaptive_node_budget import load_evidence
from visual.runtime.test.run_variable_rate_shape_first import packet_digest, flatten_arrays
from visual.runtime.test.run_triangle_intra_object_edges import (
    REPO, BASE, SELECTION, FRAMES, RATIOS, serial, write, sha, stats, rss, graph_metrics,
)

OLD = REPO/'experiments/91ecd0413a87455cabc5e813cddaea8d'
REPORT = REPO/'docs/visual-graph-integration/05b_extended_delaunay_edgewise.md'
METHODS = ('K8', 'TMD', 'DE', 'EDE')
NEW_SOURCES = ('visual/runtime/extended_delaunay_edges.py',
               'visual/runtime/test/test_extended_delaunay_edges.py',
               'visual/runtime/test/run_extended_delaunay_edges.py',
               'graph/vision_input.py', 'docs/visual-graph-integration/00_interface.md')
TESTS = ('visual.runtime.test.test_extended_delaunay_edges',
         'visual.runtime.test.test_triangle_intra_object_edges',
         'visual.runtime.test.test_intra_object_edges', 'visual.runtime.test.test_vision_features',
         'visual.runtime.test.test_variable_rate_graph_contract')
MUTABLE = {'intra_object_edge_index', 'edge_policy', 'accepted_triangles',
           'accepted_edge_pairs', 'accepted_edge_origins'}


def require(condition, message):
    if not condition:raise AssertionError(message)


def undirected(edges):
    return np.asarray(edges)[:, edges[0] < edges[1]].T.copy()


def pair_set(edges, source=None):
    pairs = undirected(edges)
    if source is not None:pairs = np.sort(source[pairs], axis=1)
    return set(map(tuple, pairs.tolist()))


def preserved(original, candidate):
    for key in original:
        if key not in MUTABLE:
            require(packet_digest(original[key]) == packet_digest(candidate[key]), 'immutable field: '+key)


def lookup_maps(packet):
    """Selected-source lookup adapter, NOT a full-support component map."""
    maps = {obj:np.full(tuple(packet['image_shape']), -1, np.int64) for obj in (0, 1)}
    for obj in (0, 1):
        rows = packet['object_id'] == obj
        maps[obj].ravel()[packet['provenance']['source_index'][rows]] = packet['observed_component_id'][rows]
    return maps


def load_inputs(out):
    """All source/hash/index contracts pass BEFORE new candidate generation."""
    begin = perf_counter()
    old = json.loads((OLD/'triangle_intra_object_edges.json').read_text())
    old_hashes = json.loads((OLD/'source_hashes.json').read_text())
    input_hashes = dict(old_hashes['input_hashes'])
    for path, expected in input_hashes.items():
        require(sha(REPO/path) == expected, 'Task05 saved input hash mismatch: '+path)
    # Source snapshots establish which computation produced the baselines. The
    # documented Task05 report-only recovery is separate from numerical source.
    source_gate = {}
    for path, expected in old_hashes['evaluated_source_hashes'].items():
        snapshot = OLD/'source_snapshot'/path
        require(sha(snapshot) == expected, 'Task05 computation snapshot hash: '+path)
        if path not in ('graph/vision_input.py', 'docs/visual-graph-integration/00_interface.md',
                        'visual/runtime/test/run_triangle_intra_object_edges.py'):
            require(sha(REPO/path) == expected, 'baseline helper implementation changed: '+path)
        source_gate[path] = dict(computation_snapshot_sha256=expected, current_sha256=sha(REPO/path))
    cfg = json.loads((REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json').read_text())['config']
    freeze = json.loads((SELECTION/'selection_before_gt.json').read_text())
    require(sha(SELECTION/'selection_before_gt.npz') == freeze['sha256'], 'selection freeze hash')
    packets, evidences, baseline, gates = {}, {}, {}, {}
    totals = Counter(); degenerate_sources = {}
    with np.load(cfg['input'], allow_pickle=False) as raw, np.load(cfg['masks'], allow_pickle=False) as masks, \
            np.load(OLD/'triangle_intra_object_edges.npz', allow_pickle=False) as saved:
        # Named observation arrays only: never load GT mesh/pose/segmentation.
        depth = raw['base_camera/depth']; ms = masks['source_frame']
        for fid in FRAMES:
            ep = BASE/f'runtime_evidence/f{fid:04d}.npz'
            e = evidences[fid] = load_evidence(ep); g = e.frame.geometry
            require((e.frame.source_frame, e.frame.camera) == (fid, 'base_camera'), 'evidence identity')
            np.testing.assert_array_equal(g.valid_depth, np.isfinite(depth[fid]) & (depth[fid] > 0))
            np.testing.assert_array_equal(g.xyz[..., 2][g.valid_depth], depth[fid][g.valid_depth])
            require(np.isfinite(g.xyz[g.valid_depth]).all(), 'finite observed XYZ')
            mi = np.flatnonzero(ms == fid); require(len(mi) == 1, 'mask source_frame')
            for obj, name in ((0, 'peg'), (1, 'box')):
                np.testing.assert_array_equal(g.object_masks[obj], masks[name][mi[0]])
                np.testing.assert_array_equal(g.observed_mask(obj), e.frame.objects[obj].valid_points)
            np.testing.assert_array_equal(g.overlap_mask, g.object_masks[0] & g.object_masks[1])
            for pairs in e.pairs.values():
                z = g.xyz[..., 2].ravel()[pairs.pixels]
                valid = g.valid_depth.ravel()[pairs.pixels].all(1)
                delta = z[valid, 1]-z[valid, 0]
                np.testing.assert_array_equal(pairs.signed_depth_delta_m[valid], delta)
                jump = np.zeros(len(z), bool)
                jump[valid] = np.abs(delta) > (.01+.02*z[valid].min(1))
                np.testing.assert_array_equal(pairs.depth_jump, jump)
            prepare_validation(e)  # verifies saved pair/support/index identities
            for ratio in RATIOS:
                key = f'f{fid:04d}_r{ratio:g}'; p = read_packet(BASE/f'packets/{key}.npz')
                tmd = read_packet(OLD/f'packets/{key}.npz'); record = old['records'][key]
                require(p['edge_policy'] == EDGE_POLICY and tmd['edge_policy'] == TRIANGLE_POLICY, 'baseline policy')
                receive_vision_packet(p); receive_vision_packet(tmd); preserved(p, tmd)
                require((p['source_frame'], p['camera'], p['frame_idx'], p['timestamp']) ==
                        (fid, 'base_camera', int(raw['frame_idx'][fid]), float(raw['timestamp'][fid])), 'packet identity')
                for name, expected in (('camera_intrinsic', raw['base_camera/intrinsic'][fid]),
                                       ('camera_to_world', raw['base_camera/cam2world'][fid]),
                                       ('observation_ee_pose', raw['tcp_pose'][fid])):
                    np.testing.assert_array_equal(p[name], expected)
                with np.load(SELECTION/f'selection/{key}.npz', allow_pickle=False) as selection:
                    np.testing.assert_array_equal(p['camera_points'], selection['packet/points'])
                    np.testing.assert_array_equal(p['object_id'], selection['packet/object_id'])
                    for name, value in p['provenance'].items():
                        np.testing.assert_array_equal(value, selection['packet/provenance/'+name])
                for name, value in flatten_arrays(p, prefix=key+'/original').items():
                    require(value.dtype == saved[name].dtype, 'original dtype')
                    require(value.shape == saved[name].shape and value.tobytes() == saved[name].tobytes(),
                            'Task04/05 exact original bytes: '+name)
                src = p['provenance']['source_index']; comp = p['observed_component_id']; oid = p['object_id']
                np.testing.assert_array_equal(p['camera_points'], g.xyz.reshape(-1, 3)[src])
                triangles = saved[key+'/triangles']; accepted = saved[key+'/accepted']
                np.testing.assert_array_equal(tmd['accepted_triangles'], triangles[accepted])
                np.testing.assert_array_equal(tmd['intra_object_edge_index'], saved[key+'/TMD_edges'])
                np.testing.assert_array_equal(triangles_to_edges(triangles), saved[key+'/T0_edges'])
                k8, final = p['intra_object_edge_index'], tmd['intra_object_edge_index']
                for method, edge in (('K8', k8), ('TMD', final), ('T0', saved[key+'/T0_edges'])):
                    degree = np.bincount(edge[0], minlength=len(src))
                    np.testing.assert_array_equal(degree, saved[key+'/'+method+'_degree'])
                    np.testing.assert_array_equal(degree, record['stages'][method]['degree'])
                entries = []
                for status in record['component_status']:
                    obj, cid, count = status['object_id'], status['component_id'], status['N']
                    rows = np.flatnonzero((oid == obj) & (comp == cid))
                    require(len(rows) == count, 'saved observed component membership')
                    uv = np.column_stack((src[rows] % p['image_shape'][1], src[rows]//p['image_shape'][1]))
                    if count == 1:kind = 'singleton'
                    elif count == 2:kind = 'two_point'
                    elif collinear_order(uv, src[rows], rows) is not None:kind = 'collinear'
                    else:kind = 'noncollinear'
                    if kind == 'collinear':require(status['status'] == 'collinear', 'saved exact collinearity')
                    totals[kind] += count
                    if kind != 'noncollinear':entries.append(dict(object_id=obj, component_id=cid,
                                                                  kind=kind, rows=rows, source=src[rows]))
                degenerate_sources[key] = entries
                require(sum(s['N'] for s in record['component_status']) == len(src), 'component partition')
                totals['T0_isolated'] += int((saved[key+'/T0_degree'] == 0).sum())
                baseline[key] = dict(triangles=triangles.copy(), K8=k8.copy(), TMD=final.copy(),
                                     T0=saved[key+'/T0_edges'].copy(), accepted_triangles=tmd['accepted_triangles'].copy())
                packets[key] = p
                gates[key] = dict(status='PASS', N=len(src), packet_digest=packet_digest(p),
                                  source_hash_index_provenance_exact=True, original_bytes_exact=True)
    require([totals[k] for k in ('singleton', 'two_point', 'collinear', 'T0_isolated')] == [270, 72, 249, 591],
            'Task05 degenerate totals mismatch')
    result = dict(status='PASS', conditions=gates, input_hashes=input_hashes, source_gate=source_gate,
                  source_component_recount=dict(totals), degenerate_membership=degenerate_sources,
                  grouping='saved rows only; no support resegmentation',
                  wall_ms=(perf_counter()-begin)*1000)
    write(out/'input_gate.json', result)
    return packets, evidences, baseline, old, result


def execute(method, original, evidence, maps, contexts, *, frozen=None):
    """Actual core: copy, independent candidate/validation, package, receiver."""
    begin, cpu_begin = perf_counter(), process_time()
    tick = perf_counter(); p = copy.deepcopy(original)
    timing = dict(packet_copy_ms=(perf_counter()-tick)*1000)
    audit = None
    if method == 'K8':
        tick = perf_counter()
        edge, comp = build_intra_object_edges(p['camera_points'], p['object_id'], p['provenance']['source_index'], maps)
        timing['knn_candidate_dedup_ms'] = (perf_counter()-tick)*1000
    elif method == 'TMD':
        edge, comp, audit = build_triangle_intra_object_edges(p['camera_points'], p['object_id'],
            p['provenance']['source_index'], maps, evidence)
        timing.update(audit['timing_ms'])
        tick = perf_counter()
        p.update(edge_policy=dict(TRIANGLE_POLICY), accepted_triangles=audit['triangles'][audit['accepted']].copy())
        timing['policy_metadata_packaging_ms'] = (perf_counter()-tick)*1000
    else:
        comp = p['observed_component_id']
        edge, audit = build_extended_delaunay_edges(p['camera_points'], p['object_id'],
            p['provenance']['source_index'], comp, evidence, extended=method == 'EDE', contexts=contexts,
            stored_triangles=frozen if method == 'DE' else None)
        timing.update(audit['timing_ms'])
        for obj in (0, 1):
            timing[f'candidate_object{obj}_ms'] = audit['candidate_object_ms'][obj]
            timing[f'edge_validation_object{obj}_ms'] = audit['validation_object_ms'][obj]
        tick = perf_counter()
        p.update(**edge_metadata(audit, extended=method == 'EDE'))
        timing['policy_metadata_packaging_ms'] = (perf_counter()-tick)*1000
    tick = perf_counter()
    p.update(intra_object_edge_index=edge, observed_component_id=comp)
    timing['packet_packaging_ms'] = (perf_counter()-tick)*1000
    tick = perf_counter(); received = receive_vision_packet(p)
    timing['receiver_ms'] = (perf_counter()-tick)*1000
    timing.update(core_wall_ms=(perf_counter()-begin)*1000, core_process_cpu_ms=(process_time()-cpu_begin)*1000)
    return p, audit, timing


def check_execution(p, original, audit, expected):
    preserved(original, p)
    np.testing.assert_array_equal(p['intra_object_edge_index'], expected)
    if audit is not None:
        require(not [s for s in audit['components'] if s['status'] in ('qhull_error', 'omitted_vertices')],
                'incomplete triangulation; comparison must not pass')


def source_rows(packet, rows):
    rows = np.asarray(rows, np.int64)
    return dict(rows=rows, source=packet['provenance']['source_index'][rows])


def component_diagnostics(packet, audit):
    """Exact local incircle/near-line flags, outside measured core."""
    src = packet['provenance']['source_index']; w = int(packet['image_shape'][1])
    uv = np.column_stack((src % w, src//w)); output = []
    for status in audit['components']:
        item = dict(status); rows = status['rows']
        if len(rows) >= 3:
            p = [tuple(map(int, uv[r])) for r in rows]
            a, b = min(p), max(p); du, dv = b[0]-a[0], b[1]-a[1]
            cross = max(abs(du*(v-a[1])-dv*(u-a[0])) for u, v in p)
            ratio = cross/(du*du+dv*dv)
            item.update(normalized_cross=ratio, nearly_collinear_nonzero=bool(0 < ratio <= 1e-8))
        output.append(item)
    adjacent = {}
    for tri in audit['triangles']:
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            adjacent.setdefault(tuple(sorted(map(int, tri[[i, j]]))), []).append(int(tri[k]))
    ties = []
    for (a, b), opposite in adjacent.items():
        if len(opposite) != 2:continue
        c, d = opposite
        def cross(x, y, z):
            x, y, z = [tuple(map(int, uv[i])) for i in (x, y, z)]
            return (y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0])
        if cross(a, b, c)*cross(a, b, d) >= 0 or cross(c, d, a)*cross(c, d, b) >= 0:continue
        x, y = map(int, uv[d]); m = []
        for r in (a, b, c):
            u, v = map(int, uv[r]); u -= x; v -= y; m.append((u, v, u*u+v*v))
        det = sum(m[0][i]*(m[1][(i+1)%3]*m[2][(i+2)%3]-m[1][(i+2)%3]*m[2][(i+1)%3]) for i in range(3))
        if det == 0:ties.append(dict(shared_edge=[a, b], quad_rows=[a, b, c, d]))
    return dict(components=output, exact_cocircular_local_ties=ties)


def analyze_condition(key, original, evidence, baseline, old_record, de, ede):
    packets = dict(K8=original, TMD=copy.deepcopy(original), DE=de[0], EDE=ede[0])
    packets['TMD']['intra_object_edge_index'] = baseline['TMD']
    src = original['provenance']['source_index']; oid = original['object_id']; n = len(src)
    w = int(original['image_shape'][1]); uv = np.column_stack((src % w, src//w))
    row_sets = {m:pair_set(p['intra_object_edge_index']) for m, p in packets.items()}
    src_sets = {m:pair_set(p['intra_object_edge_index'], src) for m, p in packets.items()}
    violations = {}
    for label, sets in (('rows', row_sets), ('source', src_sets)):
        violations[label] = {'TMD_minus_DE': sorted(sets['TMD']-sets['DE']),
                             'DE_minus_EDE': sorted(sets['DE']-sets['EDE'])}
    require(not any(v for item in violations.values() for v in item.values()), 'subset contract violation: '+key)
    da, ea = de[1], ede[1]
    np.testing.assert_array_equal(da['candidates'], ea['candidates'][ea['origins'] == 'delaunay_edge'])
    cache = {tuple(item['rows']):copy.deepcopy(item) for item in old_record['edge_audit']}
    # Existing baseline path audits are reused. New edges use exactly the same helper.
    for pair, check in zip(ea['candidates'], ea['checks']):
        a, b = pair; normal = original['geometric_features'][pair, :3]
        angle = None
        if original['feature_field_valid'][pair, 0].all():
            angle = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(*normal)), np.dot(*normal))))
        cache[tuple(map(int, pair))] = dict(rows=pair, object_id=int(oid[a]), uv=uv[pair],
            source=src[pair], normal_angle_deg=angle, **check)
    stages = {}
    for method, packet in packets.items():
        edges = packet['intra_object_edge_index']; pairs = undirected(edges)
        metric = graph_metrics(original, edges); objects = {}
        for obj in (0, 1):
            checks = [cache[tuple(ab)] for ab in pairs if oid[ab[0]] == obj]
            reasons = Counter(r for c in checks for r in c['reasons'])
            counts = Counter()
            for c in checks:counts.update(c['counts'])
            measured = [c['normal_angle_deg'] for c in checks if c['normal_angle_deg'] is not None]
            op = pairs[oid[pairs[:, 0]] == obj]
            lengths = np.linalg.norm(original['camera_points'][op[:, 0]].astype(float)-original['camera_points'][op[:, 1]], axis=1)*1000
            uv_lengths = np.linalg.norm(uv[op[:, 0]]-uv[op[:, 1]], axis=1)
            endpoint_depth = original['camera_points'][op, 2]
            bins = [0., 1., 2., 5., 10., float('inf')]
            hist = [int(((endpoint_depth.max(1) >= lo) & (endpoint_depth.max(1) < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])]
            base_obj = metric['objects'][obj]
            base_obj.update(length_mm_raw=lengths, uv_length_raw=uv_lengths, uv_length=stats(uv_lengths),
                isolated_source=src[base_obj['isolated_rows']],
                isolated_fraction=base_obj['isolated_count']/base_obj['N'] if base_obj['N'] else None,
                observed_support_components=None, selected_support_components=int(len(np.unique(original['observed_component_id'][oid == obj]))),
                observation_invalid_edges=sum(bool(c['reasons']) for c in checks),
                reason_edge_counts=dict(reasons), reason_cell_pair_counts=dict(counts),
                normal_measured_edges=len(measured), normal_unmeasured_edges=len(checks)-len(measured),
                normal_over20_edges=sum(a > 20 for a in measured),
                endpoint_depth_m_raw=endpoint_depth, endpoint_max_depth_bins_m=['[0,1)', '[1,2)', '[2,5)', '[5,10)', '[10,inf)'],
                endpoint_max_depth_edge_counts=hist)
            objects[obj] = base_obj
        metric['objects'] = objects; stages[method] = metric
        if method in ('DE', 'EDE'):
            require(not sum(o['observation_invalid_edges'] for o in objects.values()), 'accepted observation violation')
    degrees = {m:stages[m]['degree'] for m in METHODS}
    fallback = ea['origins'] != 'delaunay_edge'
    candidate_nodes = np.unique(ea['candidates'][fallback])
    rejected = fallback & ~ea['accepted']; accepted = fallback & ea['accepted']
    rejected_nodes = np.unique(ea['candidates'][rejected])
    recovery = dict(T0_isolated=source_rows(original, np.flatnonzero(np.bincount(baseline['T0'][0], minlength=n) == 0)),
        TMD_isolated=source_rows(original, np.flatnonzero(degrees['TMD'] == 0)),
        DE_newly_connected=source_rows(original, np.flatnonzero((degrees['TMD'] == 0) & (degrees['DE'] > 0))),
        EDE_fallback_additionally_connected=source_rows(original, np.flatnonzero((degrees['DE'] == 0) & (degrees['EDE'] > 0))),
        fallback_candidate_nodes=source_rows(original, candidate_nodes),
        fallback_failed_incident_nodes=source_rows(original, rejected_nodes),
        fallback_all_candidates_failed_nodes=source_rows(original, candidate_nodes[degrees['EDE'][candidate_nodes] == 0]),
        EDE_still_isolated=source_rows(original, np.flatnonzero(degrees['EDE'] == 0)),
        fallback_candidates=int(fallback.sum()), fallback_accepted=int(accepted.sum()), fallback_rejected=int(rejected.sum()))
    changes = {}
    for before, after in (('TMD', 'DE'), ('DE', 'EDE'), ('TMD', 'EDE')):
        merges = []
        for label in np.unique(stages[after]['graph_labels']):
            rows = np.flatnonzero(stages[after]['graph_labels'] == label)
            labels, counts = np.unique(stages[before]['graph_labels'][rows], return_counts=True)
            if len(labels) > 1:merges.append(dict(rows=rows, before_labels=labels, before_sizes=counts))
        changes[before+'_to_'+after] = dict(added_edges=sorted(row_sets[after]-row_sets[before]),
            removed_edges=sorted(row_sets[before]-row_sets[after]), merges=merges,
            split_count=0 if row_sets[before] <= row_sets[after] else None)
    isolated = np.flatnonzero(degrees['EDE'] == 0)
    incident = np.bincount(ea['candidates'].ravel(), minlength=n)
    reasons = {int(row):('no_candidate' if incident[row] == 0 else 'all_incident_candidates_rejected') for row in isolated}
    return dict(frame=original['source_frame'], ratio=float(key.split('_r')[1]), N=n,
        stages=stages, recovery=recovery, changes=changes, subset_contract=dict(status='PASS', violations=violations),
        nondegenerate_candidates_exact=True,
        DE=dict(candidates=da['candidates'], origins=da['origins'], accepted=da['accepted'], checks=da['checks']),
        EDE=dict(candidates=ea['candidates'], origins=ea['origins'], accepted=ea['accepted'], checks=ea['checks']),
        component_diagnostics=component_diagnostics(original, ea),
        EDE_isolated_reason=reasons, edge_audit=list(cache.values()))


def track_remote(packet, record):
    src = packet['provenance']['source_index']; w = int(packet['image_shape'][1])
    rows = np.flatnonzero((packet['object_id'] == 1) & (packet['observed_component_id'] == 20))
    uv = np.column_stack((src % w, src//w))
    chain = collinear_order(uv[rows], src[rows], rows)
    require(len(rows) == 12 and chain is not None, 'Frame0 remote component identity')
    ed = record['EDE']; candidates = np.asarray(ed['candidates']); accepted = np.asarray(ed['accepted'])
    ids = np.flatnonzero(np.isin(candidates, rows).all(1)); edges = []
    for i in ids:
        pair = candidates[i]
        edges.append(dict(rows=pair, source=src[pair], uv=uv[pair], origin=ed['origins'][i],
            accepted=bool(accepted[i]), validation=ed['checks'][i],
            length_mm=float(np.linalg.norm(packet['camera_points'][pair[0]].astype(float)-packet['camera_points'][pair[1]])*1000),
            uv_length=float(np.linalg.norm(uv[pair[0]]-uv[pair[1]]))))
    pair = [int(np.flatnonzero((packet['object_id'] == 1) & (src == 206*w+u))[0]) for u in (251, 259)]
    final = record['stages']['EDE']; labels = final['graph_labels']
    direct = any(set(e['rows']) == set(pair) and e['accepted'] for e in edges)
    connected = bool(labels[pair[0]] == labels[pair[1]])
    return dict(object_id=1, observed_component_id=20, rows=rows, source=src[rows], uv=uv[rows],
        camera_xyz=packet['camera_points'][rows], ee_xyz=packet['points'][rows],
        chain_rows=chain, chain_source=src[chain], chain_uv=uv[chain], candidate_edges=edges,
        candidate_count=len(ids), accepted_count=sum(e['accepted'] for e in edges),
        original_pair_rows=pair, original_pair_direct=direct, original_pair_indirect=connected and not direct,
        original_pair_connected=connected, degree=final['degree'][rows],
        recovered_from_TMD=int(((record['stages']['TMD']['degree'][rows] == 0) & (final['degree'][rows] > 0)).sum()),
        remaining_isolated=int((final['degree'][rows] == 0).sum()),
        graph_component_sizes=np.unique(labels[rows], return_counts=True)[1],
        limitation='Continuous jointly wrong mask/depth plateau can pass. No depth/length/ROI veto applied.')


def memory_worker(key, method):
    before = rss(); fid = int(key[1:5])
    p = read_packet(BASE/f'packets/{key}.npz')
    e = load_evidence(BASE/f'runtime_evidence/f{fid:04d}.npz'); loaded = rss()
    maps = lookup_maps(p); contexts = prepare_validation(e); prepared = rss()
    result, audit, timing = execute(method, p, e, maps, contexts)
    peak = rss()
    print(json.dumps(serial(dict(import_baseline_peak_bytes=before, input_load_peak_bytes=loaded,
        common_prepared_peak_bytes=prepared, core_peak_bytes=peak,
        core_peak_increase_bytes=max(0, peak-prepared), common_peak_increase_bytes=max(0, prepared-loaded),
        packet_array_bytes=array_bytes(p), evidence_array_bytes=array_bytes(e),
        lookup_array_bytes=array_bytes(dict(maps=maps, contexts=contexts)),
        audit_array_bytes=array_bytes(audit), result_packet_array_bytes=array_bytes(result),
        raster_work=audit.get('raster_work') if audit else None,
        ru_maxrss_unit='bytes (Darwin native bytes; Linux KiB converted)',
        native_qhull_allocator_overhead='included in RSS; not separately identifiable',
        scope='fresh worker: imports, one condition load, common lookup, core incl receiver; no plotting/audit/IO'))), flush=True)


def numerics(out, packets, evidences, baselines, old):
    records, arrays, common, remote = {}, {}, {}, {}
    begin = perf_counter()
    for key, p in packets.items():
        e = evidences[p['source_frame']]; base = baselines[key]
        tick = perf_counter(); maps = lookup_maps(p); map_ms = (perf_counter()-tick)*1000
        tick = perf_counter(); contexts = prepare_validation(e); validation_ms = (perf_counter()-tick)*1000
        common[key] = dict(selected_lookup_map_ms=map_ms, validation_prepare_ms=validation_ms,
            lookup_array_bytes=array_bytes(dict(maps=maps, contexts=contexts)),
            component_segmentation='NOT_EXECUTED; stored IDs',
            observed_support=old['common'][str(p['source_frame'])]['observed_components'])
        # Correctness: DE reads frozen T0; EDE independently generates same base.
        de = execute('DE', p, e, maps, contexts, frozen=base['triangles'])
        ede = execute('EDE', p, e, maps, contexts)
        for method, result in (('DE', de), ('EDE', ede)):
            check_execution(result[0], p, result[1], result[0]['intra_object_edge_index'])
            save_packet(out/'packets'/f'{key}_{method}.npz', result[0])
        np.testing.assert_array_equal(de[1]['triangles'], base['triangles'])
        np.testing.assert_array_equal(ede[1]['triangles'], base['triangles'])
        tick = perf_counter()
        rec = analyze_condition(key, p, e, base, old['records'][key], de, ede)
        for method in METHODS:
            for obj in (0, 1):
                rec['stages'][method]['objects'][obj]['observed_support_components'] = common[key]['observed_support'][str(obj)]['components']
        rec['diagnostics_wall_ms'] = (perf_counter()-tick)*1000
        if p['source_frame'] == 0:remote[key] = track_remote(p, rec)
        records[key] = rec
        arrays.update(flatten_arrays(p, prefix=key+'/original'))
        arrays[key+'/uv'] = np.column_stack((p['provenance']['source_index'] % p['image_shape'][1],
                                            p['provenance']['source_index']//p['image_shape'][1]))
        for method in METHODS:
            edge = base[method] if method in ('K8', 'TMD') else (de if method == 'DE' else ede)[0]['intra_object_edge_index']
            arrays[key+'/'+method+'_edges'] = edge
            arrays[key+'/'+method+'_degree'] = rec['stages'][method]['degree']
            arrays[key+'/'+method+'_graph_labels'] = rec['stages'][method]['graph_labels']
        for method, a in (('DE', de[1]), ('EDE', ede[1])):
            for field in ('candidates', 'origins', 'accepted', 'triangles'):
                arrays[key+'/'+method+'_'+field] = a[field]
        for name, value in rec['recovery'].items():
            if isinstance(value, dict):
                arrays[key+'/recovery/'+name+'/rows'] = value['rows']
                arrays[key+'/recovery/'+name+'/source'] = value['source']
        # Method order rotates and reverses by repeat, with 3 unrecorded warmups.
        raw = {m:[] for m in METHODS}; prep_samples = []; orders = []
        expected = {m:arrays[key+'/'+m+'_edges'] for m in METHODS}
        for repeat in range(8):
            tick = perf_counter(); maps = lookup_maps(p); map_ms = (perf_counter()-tick)*1000
            tick = perf_counter(); contexts = prepare_validation(e); context_ms = (perf_counter()-tick)*1000
            rotation = repeat % 4; order = METHODS[rotation:]+METHODS[:rotation]
            if repeat % 2:order = order[::-1]
            orders.append(dict(repeat=repeat, warmup=repeat < 3, methods=order))
            for method in order:
                result, audit, timing = execute(method, p, e, maps, contexts)
                check_execution(result, p, audit, expected[method])
                if audit is not None and method == 'TMD':
                    np.testing.assert_array_equal(audit['triangles'], base['triangles'])
                    np.testing.assert_array_equal(audit['triangles'][audit['accepted']], base['accepted_triangles'])
                if repeat >= 3:raw[method].append(timing)
            if repeat >= 3:prep_samples.append(dict(selected_lookup_map_ms=map_ms, validation_prepare_ms=context_ms))
        rec.update(timings_raw=raw, method_order=orders,
            timing_ms={m:{field:stats([s[field] for s in samples]) for field in samples[0]} for m, samples in raw.items()},
            object_core_timing=dict(DE=dict(candidate_ms=de[1]['candidate_object_ms'], validation_ms=de[1]['validation_object_ms']),
                                    EDE=dict(candidate_ms=ede[1]['candidate_object_ms'], validation_ms=ede[1]['validation_object_ms'])),
            object_timing_scope='standalone correctness pass; receiver/copy/input shared, not apportioned')
        common[key]['raw_prepare_ms'] = prep_samples
        memory = {}
        for method in METHODS:
            worker = subprocess.run([sys.executable, '-m', 'visual.runtime.test.run_extended_delaunay_edges',
                '--memory-worker', key, method], cwd=REPO, env=os.environ.copy(), capture_output=True, text=True, check=True)
            memory[method] = json.loads(worker.stdout.strip().splitlines()[-1])
        rec['memory'] = memory
        write(out/'records'/f'{key}.json', rec)
        write(out/'progress.json', dict(completed_conditions=len(records), last_completed=key))
        print('complete', key, 'N', len(p['points']), 'U', {m:rec['stages'][m]['U'] for m in METHODS},
              'isolated', {m:int((rec['stages'][m]['degree'] == 0).sum()) for m in METHODS}, flush=True)
    np.savez_compressed(out/'extended_delaunay_edgewise.npz', **arrays)
    with np.load(out/'extended_delaunay_edgewise.npz', allow_pickle=False) as restored:
        require(set(restored.files) == set(arrays), 'aggregate NPZ keys')
        for name, value in arrays.items():
            require(value.dtype == restored[name].dtype and value.tobytes() == restored[name].tobytes(), 'NPZ readback: '+name)
    totals = {m:dict(U=sum(r['stages'][m]['U'] for r in records.values()),
                    isolated=sum(int((r['stages'][m]['degree'] == 0).sum()) for r in records.values()),
                    components=sum(sum(o['components'] for o in r['stages'][m]['objects'].values()) for r in records.values()),
                    observation_invalid=sum(sum(o['observation_invalid_edges'] for o in r['stages'][m]['objects'].values()) for r in records.values()),
                    core_wall_ms=stats([s['core_wall_ms'] for r in records.values() for s in r['timings_raw'][m]])) for m in METHODS}
    recovery = {}
    for field, value in next(iter(records.values()))['recovery'].items():
        recovery[field] = sum(len(r['recovery'][field]['rows']) if isinstance(value, dict) else r['recovery'][field] for r in records.values())
    return dict(records=records, common=common, remote=remote, totals=totals, recovery_totals=recovery,
                point_row_observations=sum(r['N'] for r in records.values()),
                numerical_pipeline_wall_ms=(perf_counter()-begin)*1000, analysis_process_peak_rss_bytes=rss())


def environment():
    try:
        from scipy.spatial import Delaunay
        Delaunay(np.zeros((3, 2)))
    except Exception as exc:
        qhull = next((s for s in str(exc).splitlines() if 'Qhull' in s and 'Options' in s), 'see extension hash')
    return dict(python=platform.python_version(), interpreter=sys.executable, platform=platform.platform(),
        cpu=platform.processor(), numpy=np.__version__, scipy=scipy.__version__,
        qhull_extension_sha256=sha(_qhull.__file__), qhull_version_diagnostic=qhull,
        thread_env={k:os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})


def snapshot(out, initial):
    for rel in NEW_SOURCES:
        target = out/'source_snapshot'/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/rel, target)
    for rel in ('visual/runtime/triangle_intra_object_edges.py', 'visual/runtime/intra_object_edges.py',
                'visual/runtime/vision_packet.py', 'visual/runtime/edge_evidence.py',
                'visual/runtime/test/run_triangle_intra_object_edges.py', 'visual/runtime/test/run_vision_feature_edge.py',
                'visual/runtime/test/run_adaptive_node_budget.py', 'utils/experiment_logger.py'):
        target = out/'source_snapshot'/rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO/rel, target)
    diff = []
    for rel in NEW_SOURCES:
        previous = OLD/'source_snapshot'/rel
        before = previous.read_text().splitlines(True) if previous.exists() else []
        diff.extend(difflib.unified_diff(before, (REPO/rel).read_text().splitlines(True), fromfile='task05/'+rel, tofile=rel))
    (out/'implementation.diff').write_text(''.join(diff))
    if Path('/tmp/task05b_initial.diff').exists():shutil.copyfile('/tmp/task05b_initial.diff', out/'preexisting_worktree.diff')
    if Path('/tmp/task05b_initial_status.txt').exists():shutil.copyfile('/tmp/task05b_initial_status.txt', out/'preexisting_status.txt')
    write(out/'source_hashes.json', dict(protected_before=initial,
        evaluated_sources={rel:sha(REPO/rel) for rel in NEW_SOURCES},
        computation_and_report_sources='same snapshot; report-only changes, if any, stored separately'))


def protected_check(initial):
    allowed = {'graph/vision_input.py', 'docs/visual-graph-integration/00_interface.md'}
    changed = [path for path, expected in initial.items() if path not in allowed and sha(REPO/path) != expected]
    require(not changed, 'protected source/artifact changed: '+str(changed))
    return dict(status='PASS', protected_file_count=len(initial)-len(allowed), unexpected_changes=changed,
                intentional_source_changes=sorted(allowed))


def figures(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    plt.rcParams.update({'font.size': 9})
    cfg = json.loads((REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json').read_text())['config']
    with np.load(cfg['input'], allow_pickle=False) as raw:rgb = {f:raw['base_camera/rgb'][f] for f in FRAMES}
    def save(fig, name):
        fig.savefig(out/'plots'/f'{name}.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    def lines(ax, uv, pairs, color='#00e1ca', width=.5):
        pairs = np.asarray(pairs, int).reshape(-1, 2)
        if len(pairs):ax.add_collection(LineCollection(uv[pairs], colors=color, linewidths=width, alpha=.8))
    examples = {}
    with np.load(out/'extended_delaunay_edgewise.npz', allow_pickle=False) as data:
        for key, rec in summary['records'].items():
            fid = rec['frame']; p = read_packet(BASE/f'packets/{key}.npz')
            uv = data[key+'/uv']; edges = {m:data[key+'/'+m+'_edges'] for m in METHODS}
            fig, axs = plt.subplots(1, 4, figsize=(19, 5), sharex=True, sharey=True)
            for ax, method in zip(axs, METHODS):
                ax.imshow(rgb[fid]); lines(ax, uv, undirected(edges[method]))
                ax.scatter(*uv.T, s=2, c='#e84578')
                iso = np.asarray(rec['stages'][method]['degree']) == 0
                ax.scatter(*uv[iso].T, s=8, c='#ffc400')
                ax.set(title=f'{method}: U={rec["stages"][method]["U"]}, isolated={iso.sum()}',
                       xlim=(0, 511), ylim=(511, 0), xlabel='u', ylabel='v')
            fig.suptitle(key+' | identical Point rows and axes | yellow = isolated')
            save(fig, key+'_overlay')
            candidates = np.asarray(rec['EDE']['candidates'], int).reshape(-1, 2)
            accepted = np.asarray(rec['EDE']['accepted'], bool); origins = np.asarray(rec['EDE']['origins'])
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
            for ax, origin, color in zip(axs, ('delaunay_edge', 'two_point_fallback', 'collinear_chain'),
                                        ('#00bfbd', '#fb7a1d', '#b046ee')):
                ax.imshow(rgb[fid]); ax.scatter(*uv.T, s=2, c='#c7c7c7')
                sel = origins == origin
                lines(ax, uv, candidates[sel], '#f52b42', .9)
                lines(ax, uv, candidates[sel & accepted], color, .7)
                ax.set(title=f'{origin}: candidate={sel.sum()}, accepted={(sel & accepted).sum()}',
                       xlim=(0, 511), ylim=(511, 0), xlabel='u', ylabel='v')
            fig.suptitle(key+' | red = rejected; other colors = accepted; all points retained')
            save(fig, key+'_candidates')
            fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
            for ax, field, title in zip(axs,
                    ('DE_newly_connected', 'EDE_fallback_additionally_connected'),
                    ('Triangle rejection removed: DE recovery', 'Fallback added: EDE recovery')):
                ax.imshow(rgb[fid]); lines(ax, uv, undirected(edges['EDE']), '#008c9e', .3)
                ax.scatter(*uv.T, s=3, c='#8e969c')
                recovered = np.asarray(rec['recovery'][field]['rows'], int)
                remaining = np.asarray(rec['recovery']['EDE_still_isolated']['rows'], int)
                ax.scatter(*uv[recovered].T, s=11, c='#25ef55', label=f'recovered {len(recovered)}')
                ax.scatter(*uv[remaining].T, s=11, c='#ffba00', label=f'still isolated {len(remaining)}')
                ax.set(title=title, xlim=(0, 511), ylim=(511, 0), xlabel='u', ylabel='v'); ax.legend(fontsize=7)
            fig.suptitle(key+' | observation connectivity; physical correctness NOT_ESTABLISHED')
            save(fig, key+'_recovery')
            for i, (pair, check) in enumerate(zip(candidates, rec['EDE']['checks'])):
                for cause in check['reasons']:
                    if cause not in examples or len(check['reasons']) < examples[cause]['cause_count']:
                        examples[cause] = dict(key=key, pair=pair.tolist(), candidate_index=i,
                                               cause_count=len(check['reasons']))
            # A local normal-angle example is diagnostic only; normals never select edges.
            accepted_set = pair_set(edges['EDE'])
            high = [a for a in rec['edge_audit'] if tuple(a['rows']) in accepted_set
                    and a['normal_angle_deg'] is not None and a['normal_angle_deg'] > 20]
            if high:
                a = max(high, key=lambda a:a['normal_angle_deg']); pair = np.array(a['rows'], int)
                e = load_evidence(BASE/f'runtime_evidence/f{fid:04d}.npz')
                cover = supercover(uv[pair]); z = e.frame.geometry.xyz[..., 2]
                fig, axs = plt.subplots(1, 3, figsize=(13, 4))
                axs[0].imshow(rgb[fid]); lines(axs[0], uv, [pair], '#15fc62', 2)
                mid = uv[pair].mean(0); radius = max(8, int(np.ptp(uv[pair], axis=0).max())+3)
                axs[0].set(xlim=(mid[0]-radius, mid[0]+radius), ylim=(mid[1]+radius, mid[1]-radius), title='Accepted edge / observed RGB')
                axs[1].scatter(cover[:, 0], cover[:, 1], c=z[cover[:, 1], cover[:, 0]], cmap='viridis', s=45, marker='s')
                axs[1].invert_yaxis(); axs[1].set(title='Edge-only cover / depth', xlabel='u', ylabel='v')
                normals = p['geometric_features'][pair, :3]
                ax = axs[2]; ax.remove(); ax = fig.add_subplot(1, 3, 3, projection='3d')
                ax.quiver([0, 0], [0, 0], [0, 0], *normals.T, color=['#097fc0', '#da3c50'])
                ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title=f'EE normals: {a["normal_angle_deg"]:.1f} degrees')
                fig.suptitle(key+' | normal change allowed; physical face/corner NOT_ESTABLISHED')
                save(fig, key+'_normal')
    # Synthetic branch illustration makes deterministic ordering visible at useful scale.
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, points, title in zip(axs, ([[1, 2], [7, 2]], [[7, 2], [1, 2], [4, 2], [2, 2]]),
                                 ('n=2: one candidate', 'Collinear: projection order; adjacent only')):
        # Illustration only: no adapter or validator is re-executed by rendering.
        points = np.array(points); order = np.argsort(points[:, 0])
        pairs = np.column_stack((order[:-1], order[1:]))
        lines(ax, points, pairs, '#138eac', 2); ax.scatter(*points.T, s=45)
        for i, point in enumerate(points):ax.annotate('row '+str(i), point+np.array([0, .25 if i % 2 else -.35]))
        ax.set(title=title+' (synthetic)', xlim=(0, 8), ylim=(0, 4), xlabel='u', ylabel='v')
    save(fig, 'fallback_rules')
    # Display every observed cause, without inventing causes absent in these inputs.
    for cause, example in examples.items():
        key, pair = example['key'], np.asarray(example['pair']); rec = summary['records'][key]
        p = read_packet(BASE/f'packets/{key}.npz'); src = p['provenance']['source_index']; w = int(p['image_shape'][1])
        uv = np.column_stack((src % w, src//w)); fid = p['source_frame']
        e = load_evidence(BASE/f'runtime_evidence/f{fid:04d}.npz'); g = e.frame.geometry
        cover = supercover(uv[pair]); check = rec['EDE']['checks'][example['candidate_index']]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].imshow(rgb[fid]); lines(axs[0], uv, [pair], '#ff2222', 2)
        lo, hi = uv[pair].min(0)-4, uv[pair].max(0)+4
        axs[0].set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title=f'{key}: rejected edge')
        obj = int(p['object_id'][pair[0]])
        axs[1].imshow(g.observed_mask(obj), cmap='gray'); lines(axs[1], uv, [pair], '#ed364b', 1.5)
        axs[1].scatter(*cover.T, marker='s', facecolors='none', edgecolors='#00dcf7', s=20)
        axs[1].set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title='Same-object support and edge-only cells')
        z = g.xyz[..., 2]; order = np.lexsort((cover[:, 1], cover[:, 0]))
        axs[2].plot(z[cover[order, 1], cover[order, 0]], '.-')
        axs[2].set(title='Covered pixel depth (display order)', xlabel='cell (u then v)', ylabel='depth (m)')
        fig.suptitle(cause+' | reasons: '+', '.join(check['reasons']))
        save(fig, 'validation_'+cause)
    summary['validation_examples'] = examples
    # A saved real triangle with exactly one failed side: show independent
    # salvage using frozen results only, without generating any new edges.
    independent = None
    with np.load(OLD/'triangle_intra_object_edges.npz', allow_pickle=False) as frozen:
        for key, rec in summary['records'].items():
            de_pairs = {tuple(pair) for pair, yes in zip(rec['DE']['candidates'], rec['DE']['accepted']) if yes}
            tmd_pairs = {tuple(a) for a in undirected(np.asarray(
                read_packet(OLD/f'packets/{key}.npz')['intra_object_edge_index']))}
            oldrec = json.loads((OLD/f'records/{key}.json').read_text())
            for index, triangle in enumerate(frozen[key+'/triangles']):
                if frozen[key+'/accepted'][index]:continue
                pairs = np.sort(np.array([triangle[[0, 1]], triangle[[1, 2]], triangle[[2, 0]]]), axis=1)
                good = np.array([tuple(pair) in de_pairs for pair in pairs])
                if good.sum() == 2 and any(tuple(pair) not in tmd_pairs for pair in pairs[good]):
                    independent = dict(key=key, triangle_id=index, rows=triangle.tolist(),
                        pairs=pairs.tolist(), edge_accepted=good.tolist(),
                        saved_triangle_checks=oldrec['triangle_checks'][index])
                    break
            if independent:break
    if independent:
        from matplotlib.collections import PolyCollection
        key = independent['key']; p = read_packet(BASE/f'packets/{key}.npz')
        src = p['provenance']['source_index']; w = int(p['image_shape'][1])
        uv = np.column_stack((src % w, src//w)); tri = np.asarray(independent['rows'])
        pairs = np.asarray(independent['pairs']); good = np.asarray(independent['edge_accepted'])
        obj = int(p['object_id'][tri[0]]); e = load_evidence(BASE/f'runtime_evidence/f{p["source_frame"]:04d}.npz')
        lo, hi = uv[tri].min(0)-3, uv[tri].max(0)+3
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
        for ax in axs:
            ax.imshow(rgb[p['source_frame']]); ax.scatter(*uv[tri].T, s=25, c='#ffbf00')
            ax.set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), xlabel='u', ylabel='v')
        axs[0].add_collection(PolyCollection([uv[tri]], facecolors='#ff3344', alpha=.25))
        lines(axs[0], uv, pairs, '#f52b42', 2); axs[0].set_title('TMD: this entire triangle rejected')
        lines(axs[1], uv, pairs[good], '#14ee59', 2.5); lines(axs[1], uv, pairs[~good], '#f52b42', 2)
        axs[1].set_title('DE: two valid edges retained independently')
        axs[2].imshow(e.frame.geometry.observed_mask(obj), cmap='gray', alpha=.9)
        failed = pairs[~good][0]; cover = supercover(uv[failed])
        axs[2].scatter(*cover.T, s=30, marker='s', edgecolors='#03d8eb', facecolors='none')
        lines(axs[2], uv, [failed], '#f52b42', 1.5); axs[2].set_title('Only the failed edge cover shown')
        fig.suptitle(key+f' | frozen triangle {independent["triangle_id"]} | green retained / red rejected')
        save(fig, 'edge_independence')
        summary['edge_independence_example'] = independent
    for key, rem in summary['remote'].items():
        p = read_packet(BASE/f'packets/{key}.npz'); src = p['provenance']['source_index']; w = int(p['image_shape'][1])
        uv = np.column_stack((src % w, src//w)); xyz = p['camera_points'].astype(float)
        pairs = np.array([e['rows'] for e in rem['candidate_edges']], int).reshape(-1, 2)
        good = np.array([e['accepted'] for e in rem['candidate_edges']], bool)
        rows = np.asarray(rem['rows']); chain = np.asarray(rem['chain_rows'])
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(2, 2, 1); ax.imshow(rgb[0]); lines(ax, uv, pairs[good], '#1bfd59', 2)
        ax.scatter(*uv[rows].T, s=15, c='#ffea00'); ax.set(title='Full observed image / remote component', xlim=(0, 511), ylim=(511, 0))
        ax = fig.add_subplot(2, 2, 2); ax.imshow(rgb[0]); lines(ax, uv, pairs[good], '#1bfd59', 2); lines(ax, uv, pairs[~good], '#ff4040', 2)
        ax.scatter(*uv[rows].T, s=25, c='#ffea00')
        for j, r in enumerate(chain):ax.annotate(f'{j}:r{r}', uv[r]+[0, (-1)**j*2], fontsize=7)
        lo, hi = uv[rows].min(0)-5, uv[rows].max(0)+5
        ax.set(title='Actual chain order / accepted green, rejected red', xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]))
        ax = fig.add_subplot(2, 2, 3, projection='3d'); ax.scatter(*xyz.T, s=1, c='#adb2bd')
        if len(pairs):ax.add_collection3d(Line3DCollection(xyz[pairs[good]], colors='#ed3c60', linewidths=2))
        ax.set(title='Full finite XYZ extent, no clipping', xlabel='camera x (m)', ylabel='camera y (m)', zlabel='camera z (m)')
        ax = fig.add_subplot(2, 2, 4); ax.bar(np.arange(len(pairs)), [e['length_mm'] for e in rem['candidate_edges']], color=['#168caa' if v else '#d33756' for v in good])
        ax.set(title='Every chain candidate / camera XYZ length', xlabel='chain edge (canonical row order in JSON)', ylabel='length (mm)')
        fig.suptitle(key+f' | depth ~8.431 m | accepted {rem["accepted_count"]}/{rem["candidate_count"]}; recovered {rem["recovered_from_TMD"]}/12 | contaminated plateau can pass')
        save(fig, key+'_remote')
    keys = list(summary['records']); x = np.arange(len(keys)); colors = ['#6b6f85', '#e5a22a', '#1b87a5', '#d44069']
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for method, color in zip(METHODS, colors):
        lengths, degree, sizes = [], [], []
        for r in summary['records'].values():
            for obj in r['stages'][method]['objects'].values():
                lengths.extend(obj['length_mm_raw']); degree.extend(obj['degree']); sizes.extend(obj['component_sizes'])
        for ax, values, title in zip(axs, (lengths, degree, sizes), ('Unique edge length (mm)', 'Node degree', 'Graph component size (singletons included)')):
            values = np.sort(np.asarray(values, float))
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=method, color=color)
            ax.set(xlabel=title, ylabel='Empirical CDF', xscale='symlog'); ax.grid(alpha=.2)
    axs[0].legend(); fig.suptitle('All 16 conditions / both objects; raw arrays and all slices retained')
    save(fig, 'length_degree_components')
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, (method, color) in enumerate(zip(METHODS, colors)):
        means = [summary['records'][k]['timing_ms'][method]['core_wall_ms']['mean'] for k in keys]
        p95 = [summary['records'][k]['timing_ms'][method]['core_wall_ms']['p95'] for k in keys]
        axs[0].plot(x, means, '.-', label=method, color=color)
        axs[0].scatter(x, p95, s=12, marker='x', color=color)
        peaks = [summary['records'][k]['memory'][method]['core_peak_bytes']/2**20 for k in keys]
        axs[1].plot(x, peaks, '.-', label=method, color=color)
    axs[0].set(title='3 warmups + 5 measurements; mean lines, sample p95 crosses', ylabel='Core wall (ms)'); axs[0].legend()
    axs[1].set(ylabel='Fresh-worker peak RSS (MiB)', title='Imports + input + common lookup + core; native allocator included')
    axs[1].set_xticks(x, keys, rotation=50, ha='right'); fig.tight_layout()
    save(fig, 'latency_memory')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fields = ('packet_copy_ms', 'delaunay_ms', 'collinearity_chain_ms', 'two_point_ms',
              'supercover_ms', 'support_validation_ms', 'depth_validation_ms', 'edge_packaging_ms',
              'policy_metadata_packaging_ms', 'packet_packaging_ms', 'receiver_ms')
    for ax, method in zip(axs, ('DE', 'EDE')):
        vals = [np.mean([s.get(f, 0) for r in summary['records'].values() for s in r['timings_raw'][method]]) for f in fields]
        ax.barh(fields, vals); ax.set(title=method+' | exclusive recorded stages (subtotals excluded)', xlabel='mean ms across fixed conditions')
    fig.tight_layout(); save(fig, 'cpu_stages')


def reports(out, summary):
    run_id = summary['run_id']; total = summary['totals']; recovery = summary['recovery_totals']
    def f(value):return 'N/A' if value is None else f'{value:.3f}'
    overview = ['| 방법 | U (undirected) | 고립 Point | Graph component | 관측 규칙 실패 Edge | Core 평균 ms |',
                '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        t = total[m]; overview.append(f'| {m} | {t["U"]} | {t["isolated"]} | {t["components"]} | {t["observation_invalid"]} | {f(t["core_wall_ms"]["mean"])} |')
    slices = ['| Frame / ratio | Object | N | U K8/TMD/DE/EDE | 고립 K8/TMD/DE/EDE | CC K8/TMD/DE/EDE |',
              '|---|---|---:|---|---|---|']
    recovery_table = ['| 조건 | TMD 고립 | DE 복구 | EDE 추가 복구 | fallback 후보/통과/거부 | 후보 있지만 모두 실패한 Point | EDE 잔존 고립 |',
                      '|---|---:|---:|---:|---|---:|---:|']
    reason_table = ['| 조건 / Object | EDE 후보 | 유지 | 거부 | 거부 Edge 사유 (중복 가능) |', '|---|---:|---:|---:|---|']
    remote_table = ['| 조건 | chain 후보 / 유지 | 기존 고립 복구 | 원래 263.469mm pair 직접 / 간접 | 최대 새 chain 길이 mm |',
                    '|---|---|---:|---|---:|']
    timing_table = ['| 조건 | K8 mean/p95 ms | TMD mean/p95 ms | DE mean/p95 ms | EDE mean/p95 ms |', '|---|---|---|---|---|']
    origin_totals = {origin:Counter() for origin in ('two_point_fallback', 'collinear_chain')}
    failed_fallback = []
    for key, rec in summary['records'].items():
        p = read_packet(BASE/f'packets/{key}.npz')
        for obj, name in ((0, 'Peg'), (1, 'Box')):
            data = {m:rec['stages'][m]['objects'].get(str(obj), rec['stages'][m]['objects'].get(obj)) for m in METHODS}
            counts = lambda field:'/'.join(str(data[m][field]) for m in METHODS)
            slices.append(f'| {key} | {name} | {data["K8"]["N"]} | {counts("U")} | {counts("isolated_count")} | {counts("components")} |')
            candidates = np.asarray(rec['EDE']['candidates'], int).reshape(-1, 2); accepted = np.asarray(rec['EDE']['accepted'], bool)
            sel = p['object_id'][candidates[:, 0]] == obj
            causes = Counter(cause for i, c in enumerate(rec['EDE']['checks']) if sel[i] and not accepted[i] for cause in c['reasons'])
            reason_table.append(f'| {key} / {name} | {sel.sum()} | {(sel & accepted).sum()} | {(sel & ~accepted).sum()} | {dict(causes)} |')
        r = rec['recovery']; size = lambda field:len(r[field]['rows'])
        recovery_table.append(f'| {key} | {size("TMD_isolated")} | {size("DE_newly_connected")} | {size("EDE_fallback_additionally_connected")} | {r["fallback_candidates"]}/{r["fallback_accepted"]}/{r["fallback_rejected"]} | {size("fallback_all_candidates_failed_nodes")} | {size("EDE_still_isolated")} |')
        cells = [f'{f(rec["timing_ms"][m]["core_wall_ms"]["mean"])}/{f(rec["timing_ms"][m]["core_wall_ms"]["p95"])}' for m in METHODS]
        timing_table.append('| '+key+' | '+' | '.join(cells)+' |')
        for pair, origin, accepted, check in zip(rec['EDE']['candidates'], rec['EDE']['origins'],
                                                rec['EDE']['accepted'], rec['EDE']['checks']):
            if origin in origin_totals:
                origin_totals[origin]['candidates'] += 1
                origin_totals[origin]['accepted'] += int(accepted)
                if not accepted:
                    src = p['provenance']['source_index'][pair]
                    failed_fallback.append(f'{key}: {origin}, rows {pair}, source {src.tolist()}, reasons {check["reasons"]}')
    for key, r in summary['remote'].items():
        remote_table.append(f'| {key} | {r["candidate_count"]}/{r["accepted_count"]} | {r["recovered_from_TMD"]}/12 | {r["original_pair_direct"]} / {r["original_pair_indirect"]} | {max(e["length_mm"] for e in r["candidate_edges"]):.3f} |')
    gallery = []
    for path in sorted((out/'plots').glob('*.png')):
        gallery.append(f'<figure><a href="plots/{escape(path.name)}"><img loading="lazy" src="plots/{escape(path.name)}" alt="{escape(path.stem)}"></a><figcaption>{escape(path.stem)}</figcaption></figure>')
    def html_table(lines):
        rows = [line.strip('|').split('|') for i, line in enumerate(lines) if i != 1]
        return '<table>'+''.join('<tr>'+''.join(f'<{"th" if i == 0 else "td"}>{escape(cell.strip())}</{"th" if i == 0 else "td"}>' for cell in row)+'</tr>' for i, row in enumerate(rows))+'</table>'
    added = total['DE']['U']-total['TMD']['U']; fallback_added = total['EDE']['U']-total['DE']['U']
    findings = (f'TMD→DE는 {added}개 undirected Edge를 추가 보존하고 고립 Point {recovery["DE_newly_connected"]}개를 연결했다. '
                f'DE→EDE fallback은 {fallback_added}개 Edge를 추가하고 {recovery["EDE_fallback_additionally_connected"]}개 Point를 추가 연결했다. '
                f'fallback 후보는 {recovery["fallback_candidates"]}개, 통과 {recovery["fallback_accepted"]}개, 거부 {recovery["fallback_rejected"]}개다. '
                f'후보가 있지만 모두 실패한 Point는 {recovery["fallback_all_candidates_failed_nodes"]}개이며 EDE 고립은 {total["EDE"]["isolated"]}개다.')
    limitations = ('DE/EDE 유지 Edge는 저장 관측 규칙을 모두 통과했다. 이 결과는 물리적 연결 정확성의 보장이 아니다. '
                  'Frame0 Box의 약 8.431m plateau에서도 연결이 다시 생성됐다. Mask와 Depth가 함께 잘못 관측된 연속 영역은 이번 규칙만으로 식별되지 않을 수 있다. '
                  '새 depth/length cutoff, known-negative ROI, 보완 k-NN, outlier 필터는 적용하지 않았다. 최종 Graph Edge 정책 채택은 보류한다.')
    scope = ('고정 8 frame × keep ratio 0.5/1.0 × phase 0, 16개 조건·32 object slice·11095 Point-row 관측이다. '
             '독립 episode 수나 episode 고유 Point 수가 아니다. Sampling/PCA5/Feature/EE 변환 재실행 0회. '
             'CPU NumPy/SciPy/Qhull offline 실행만 수행했으며 simulator/GPU/PCL/GNN/GRU/BC/추가 episode는 실행하지 않았다.')
    cost = ('동일 interpreter와 단일 thread 설정에서 방법별 3회 warmup·5회 측정, 순서 회전/역전. '
            'Core는 packet copy, 독립 후보 생성, 실제 validation·사유 기록, metadata/Edge packaging, Graph receiver를 포함한다. '
            '공통 selected-source component lookup 및 prepare_validation은 별도 기록한다. Component 재분할은 없다. '
            'TMD helper는 내부 lookup 준비·triangle 내부와 3변 진단 및 T0/TM/TMD 포장을 실제로 실행하고 그 비용을 포함한다. '
            'TMD supercover/validation 시간은 기존 helper의 triangle+3변 합계이며 이를 production 최적화 TMD 비용으로 해석하지 않는다. '
            'DE/EDE 간 후보·검증 cache를 공유하지 않았다. 5표본 p95는 기술통계다. 전체 edge audit/그림/파일 IO는 core 밖이다. '
            '기존 Task05 약 278ms와 직접 비교하지 않는다. Graph/GNN 20Hz 성능 검증이 아니다.')
    two, chain = origin_totals['two_point_fallback'], origin_totals['collinear_chain']
    fallback_detail = (f'2점 후보 {two["candidates"]}개 중 {two["accepted"]}개가 통과했고, '
                       f'collinear chain 후보 {chain["candidates"]}개는 {chain["accepted"]}개가 통과했다. '
                       '실패 fallback은 다음 두 개이며 source별 원본은 JSON/NPZ에 있다: '+'; '.join(failed_fallback)+'.')
    summary['fallback_by_origin'] = {origin:dict(counts) for origin, counts in origin_totals.items()}
    memory_lines = []
    for m in METHODS:
        peaks = [r['memory'][m]['core_peak_bytes']/2**20 for r in summary['records'].values()]
        memory_lines.append(f'{m} {min(peaks):.2f}–{max(peaks):.2f} MiB')
    memory_text = 'Fresh-worker core peak 범위(import/input/common 포함): '+', '.join(memory_lines)+'.'
    verify = summary.get('verification', {})
    md = f'''# Task05-B — Extended Delaunay + Edge-wise Validation

상태: 구현·고정 offline 비교 완료. **새 정책 후보는 EDE 하나이며 DE는 분석 대조군**이다. 최종 정책은 확정하지 않았다.

실행 UUID: `{run_id}`. [정적 HTML](../../experiments/{run_id}/extended_delaunay_edgewise.html) · [전체 JSON](../../experiments/{run_id}/extended_delaunay_edgewise.json) · [전체 NPZ](../../experiments/{run_id}/extended_delaunay_edgewise.npz) · [검증](../../experiments/{run_id}/verification.json) · [계획](05b_extended_delaunay_edgewise_plan.md)

## 1. 결과와 해석

{findings}

{chr(10).join(overview)}

{limitations}

{scope}

## 2. 입력·구현 계약

Task03-K `9185355a32b8433abc70f1738f983377`, Task04 `9f90293b227e4dbfaaa6290afad9cc3b`, Task05 `91ecd0413a87455cabc5e813cddaea8d`를 사용했다. [입력 gate](../../experiments/{run_id}/input_gate.json)는 모든 조건에서 PASS다. Source/hash/index와 원본 배열 byte·NaN 위치, 저장 T0/TMD·component membership를 먼저 검사했다. Singleton 270 + two-point 72 + collinear 249 = 기존 T0 고립 591 Point-row를 source별로 재확인했다. 321은 fallback 후보 제공 가능 Point 수이며 복구 보장이 아니다.

기존 Task05 `supercover`, `prepare_validation`, `validate_cover`, Qhull 옵션 `Qbb Qc Qz Q12`를 import했다. Collinear는 Python integer exact cross와 A2 projection/source/row 정렬을 사용한다. 2점은 한 후보, 전체 일직선은 이웃 chain, 비일직선은 기존 Delaunay 변이다. QJ/jitter/새 Point/normal·length·absolute depth veto는 없다. QhullError/omitted vertex는 별도 상태이며 자동 fallback하지 않는다.

DE는 저장 T0 변을 correctness 후보로 사용했다. EDE 비퇴화 후보는 이를 exact 재현했다. `TMD ⊆ DE ⊆ EDE`는 **undirected row pair와 source pair 모두 16/16 PASS**다. 별도 benchmark에서 K8/TMD/DE/EDE를 독립 재실행해 저장 correctness 결과와 exact 대조했다. Receiver는 DE/EDE 명시 정책·canonical accepted pair·출처와 whole-component 기하 계약·정확한 양방향 Edge 집합을 검사한다. Dense RGB-D 검사는 helper 책임이다.

공통 ExperimentLogger `team='visual', job_type='evaluate'`, `WANDB_MODE=disabled`를 사용했다. 성공한 Task03-K/04 평가 실행과 실제 Logger API를 확인했다. 원본 artifact sidecar를 변경하지 않았다. [소스 해시](../../experiments/{run_id}/source_hashes.json)와 [구현 diff](../../experiments/{run_id}/implementation.diff), source snapshot·config를 같은 UUID에 저장했다. 게시·sync·Git commit은 하지 않았다.

## 3. 모든 frame / ratio / object

다음 개수는 unique undirected Edge와 singleton을 포함한 Graph component 기준이다. 빈 Box도 누락하지 않았다. 세부 degree·length·UV·normal 측정 분모·depth 구간·full observed support component·source·EE/Camera XYZ는 JSON/NPZ와 condition별 records에 보존했다.

{chr(10).join(slices)}

## 4. 고립 회복과 거부 사유

{chr(10).join(recovery_table)}

{fallback_detail}

실패 incident Point와 모든 후보 실패 Point는 구분한다. 한 실패 Edge에 incident해도 다른 통과 Edge로 연결될 수 있으므로 실패 incident Point 수를 잔존 고립 수로 해석하지 않는다. 추가된 모든 DE/EDE Edge는 동일 관측 검증을 통과했지만, 아래 원거리 사례와 같이 오염된 관측의 연결도 함께 늘 수 있다. Normal >20°를 넘는 유지 Edge는 diagnostics에 별도 기록하며 물리 face/corner는 `NOT_ESTABLISHED`다.

{chr(10).join(reason_table)}

## 5. Frame0 Box 8.431m collinear component

{chr(10).join(remote_table)}

두 ratio 모두 실제 chain 순서·12개 row/source/UV/Camera·EE XYZ, 11개 후보 각각의 cover/pair 수·Mask/Depth 사유·통과, degree·component 크기를 JSON `remote`에 저장했다. 원래 `(251,206)↔(259,206)` pair의 직접 연결과 짧은 변을 통한 간접 연결은 구분했다. 아래 그림은 전체 이미지·전체 XYZ extent와 확대 chain을 함께 표시한다. 이는 정상 물리 연결로 확인된 회복이 아니라 오염된 plateau를 다시 연결하는 실패 가능성이다.

두 ratio의 실제 순서는 `(251,206), (252,206), …, (262,206)`이다. 11개 인접 후보가 모두 Mask/Depth 검사를 통과해 유지됐고, 각 길이는 약 32.9336mm, degree는 양 끝 1·내부 10점 2, Graph component는 크기 12 하나다. 기존 263.469mm pair는 직접 후보가 아니며 8개 chain Edge를 통한 간접 연결이다. 각 ratio에서 기존 고립 12개가 모두 연결됐고 잔존 고립은 0이다. 이 24 Point-row는 전체 fallback 복구 317개에 포함된다. 그 외 복구 Point의 물리 연결 정확성도 관측 계약만으로 확정하지 않는다.

![Frame0 primary](../../experiments/{run_id}/plots/f0000_r0.5_remote.png)

![Frame0 reference](../../experiments/{run_id}/plots/f0000_r1_remote.png)

## 6. CPU / 메모리

{cost}

{memory_text}

{chr(10).join(timing_table)}

![CPU and memory](../../experiments/{run_id}/plots/latency_memory.png)

![Stages](../../experiments/{run_id}/plots/cpu_stages.png)

Fresh worker를 method/condition별로 실행해 import baseline, input-load high-water, common 준비 후 peak, core peak와 차이, packet/evidence/lookup/candidate/cover 배열 bytes를 기록했다. Darwin/Linux RSS를 bytes로 통일했다. Native Qhull/allocator overhead는 RSS에 포함되며 별도 분리는 불가하다. High-water 차이는 순간 allocation 합계가 아니다. 분석을 포함한 전체 process peak는 JSON 별도 항목이다.

## 7. 검증 및 산출물

[tests.log](../../experiments/{run_id}/tests.log): {summary['tests']['count']} tests PASS. 기존 Task05·K8·Feature·v1/v2 receiver 회귀와 신규 DE/EDE 계약을 실행했다. 독립 Fraction raster oracle, 방향/불균등 간격/행 permutation chain, source 불변, Qhull 오류/누락, Edge 독립성, 정상 normal 변화 허용, GT-independent, 보존·집합 계약을 포함한다.

재현 command는 repository root에서 다음과 같다. 기존 저장 입력과 root `.venv`를 사용하며 dependency 설치는 포함하지 않는다. 새 평가는 Logger가 새 UUID를 생성한다. 저장 수치만으로 그림/보고서를 다시 만들 때는 `--reports-only experiments/{run_id}`를 사용한다. 실제 계산 소스와 보고서 보완 소스는 각각 `source_snapshot/`, `report_source_snapshot/` 및 source_hashes.json으로 구분한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task05b-mpl .venv/bin/python -m visual.runtime.test.run_extended_delaunay_edges
```

JSON/NPZ/packet readback, 모든 PNG decode, 상대 링크, 원본 해시 검사는 verification.json에 각각 기록한다. PNG 실제 시각 검토: `{verify.get('png_visual_review', {}).get('status', 'PENDING')}`. 브라우저: `{verify.get('browser', {}).get('status', 'NOT_EXECUTED')}`. 미실행 검사를 PASS로 처리하지 않는다. **Contract PASS와 물리적 연결 정확성은 구분한다.**

![Fallback rules](../../experiments/{run_id}/plots/fallback_rules.png)

![Distributions](../../experiments/{run_id}/plots/length_degree_components.png)

모든 조건의 동일 Point K8/TMD/DE/EDE overlay, 후보 출처, 복구/잔존 고립, edge-wise witness, normal 예시는 정적 HTML에 연결했다. JSON/NPZ에는 결과를 사후 선택하지 않고 전체 조건과 object를 저장했다. 추가 threshold/sweep/학습/최종 정책 변경 없이 이번 구현·실험·보고서 제출에서 중단한다.
'''
    REPORT.write_text(md)
    html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task05-B EDE | {run_id}</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#18212c;background:#f6f8fa}}h1,h2{{color:#123f54}}p{{line-height:1.65}}table{{border-collapse:collapse;background:white;margin:18px 0;width:100%;font-size:12px}}td,th{{border:1px solid #ccd7de;padding:7px;text-align:right}}th{{background:#e2edf2}}img{{max-width:100%;height:auto}}figure{{margin:30px 0;background:white;padding:12px;border:1px solid #d7e0e4}}a{{color:#006a91}}.note{{padding:16px;background:#fff2d2}}code{{overflow-wrap:anywhere}}</style>
<h1>Task05-B — Extended Delaunay + Edge-wise Validation</h1><p><code>{run_id}</code> · EDE candidate / DE analytical control · final policy undecided</p>
<p><a href="../../docs/visual-graph-integration/05b_extended_delaunay_edgewise.md">결과 Markdown</a> · <a href="../../docs/visual-graph-integration/05b_extended_delaunay_edgewise_plan.md">계획</a> · <a href="extended_delaunay_edgewise.json">전체 JSON</a> · <a href="extended_delaunay_edgewise.npz">전체 NPZ</a> · <a href="verification.json">검증</a> · <a href="input_gate.json">입력 gate</a> · <a href="source_hashes.json">source hashes</a> · <a href="tests.log">tests.log</a> · <a href="implementation.diff">implementation diff</a> · <a href="config.yaml">config</a></p>
<p>{escape(findings)}</p>{html_table(overview)}<p class="note">{escape(limitations)}</p><p>{escape(scope)}</p>
<h2>All frame / ratio / object slices</h2>{html_table(slices)}
<h2>Isolated recovery</h2>{html_table(recovery_table)}<h2>Every EDE candidate result by object</h2>{html_table(reason_table)}
<p>{escape(fallback_detail)}</p>
<h2>Frame0 remote plateau</h2>{html_table(remote_table)}<p>Actual chain/source/row/UV/XYZ and every validation witness: JSON remote. Reconnection of jointly wrong mask/depth is a failure possibility.</p>
<h2>CPU and memory</h2><p>{escape(cost)}</p><p>{escape(memory_text)}</p>{html_table(timing_table)}
<h2>Contracts and readback</h2><p>{summary['tests']['count']} tests PASS; input gate PASS; TMD subset DE subset EDE: row/source pairs PASS for 16/16 conditions. Contract PASS does not establish physical connection accuracy. Browser: {escape(verify.get('browser', {}).get('status', 'NOT_EXECUTED'))}; visual PNG review: {escape(verify.get('png_visual_review', {}).get('status', 'PENDING'))}.</p>
<h2>Per-condition records and packets</h2><ul>'''+''.join(f'<li>{escape(key)}: <a href="records/{key}.json">record</a> · <a href="packets/{key}_DE.npz">DE packet</a> · <a href="packets/{key}_EDE.npz">EDE packet</a></li>' for key in summary['records'])+'</ul><h2>All figures</h2>'+''.join(gallery)+'</html>'
    (out/'extended_delaunay_edgewise.html').write_text(html)


def verify_outputs(out, summary, initial):
    import re
    from PIL import Image
    for path, expected in summary['input_hashes'].items():
        require(sha(REPO/path) == expected, 'original input hash changed after evaluation: '+path)
    npz = out/'extended_delaunay_edgewise.npz'
    with np.load(npz, allow_pickle=False) as a:
        require(len(a.files) > 0, 'NPZ empty')
        for key in summary['records']:
            original = read_packet(BASE/f'packets/{key}.npz')
            for method in ('DE', 'EDE'):
                packet = read_packet(out/f'packets/{key}_{method}.npz')
                preserved(original, packet); receive_vision_packet(packet)
                np.testing.assert_array_equal(packet['intra_object_edge_index'], a[key+'/'+method+'_edges'])
            for name, value in flatten_arrays(original, prefix=key+'/original').items():
                require(value.dtype == a[name].dtype and value.tobytes() == a[name].tobytes(), 'original aggregate bytes')
    parsed = json.loads((out/'extended_delaunay_edgewise.json').read_text())
    require(parsed['run_id'] == out.name and len(parsed['records']) == 16, 'JSON readback identity/count')
    require(parsed == serial(summary), 'JSON full content readback')
    pngs = sorted((out/'plots').glob('*.png'))
    for png in pngs:
        with Image.open(png) as im:im.verify()
        with Image.open(png) as im:im.load(); require(min(im.size) > 0, 'PNG empty')
    class Links(HTMLParser):
        def __init__(self):super().__init__(); self.links=[]
        def handle_starttag(self, tag, attrs):
            self.links.extend(value for name, value in attrs if name in ('href', 'src'))
    parser = Links(); parser.feed((out/'extended_delaunay_edgewise.html').read_text())
    links = [(out, value) for value in parser.links]
    links += [(REPORT.parent, value) for value in re.findall(r'\]\(([^)]+)\)', REPORT.read_text())]
    for root, link in links:
        require(not link.startswith(('http:', 'https:', '//')), 'external report dependency')
        require((root/link.split('#')[0]).exists(), 'missing relative link: '+link)
    before = summary.get('verification', {})
    result = dict(status='PASS', input_gate='PASS', tests=summary['tests'], row_source_subset_conditions=16,
        input_artifact_hashes=dict(status='PASS', checked=len(summary['input_hashes'])),
        nondegenerate_candidate_exact_conditions=16, original_arrays='EXACT_BYTES_PRESERVED',
        aggregate_npz_readback='PASS', packet_receiver_readback='PASS', json_readback='PASS',
        png_decode_count=len(pngs), relative_links_checked=len(links), protected_files=protected_check(initial),
        png_visual_review=before.get('png_visual_review', dict(status='PENDING', files=[])),
        browser=before.get('browser', dict(status='NOT_EXECUTED')),
        physical_connection_accuracy='NOT_ESTABLISHED', final_policy_adopted=False,
        source_extraction_calls=0, sampling_calls=0, ee_transform_calls=0,
        simulator='NOT_EXECUTED', GPU='NOT_EXECUTED', PCL='NOT_EXECUTED', training='NOT_EXECUTED')
    write(out/'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exact-optimization', action='store_true',
                        help='Task05-C isolated snapshot vs explicit single-SAT opt-in evaluation')
    parser.add_argument('--exact-memory-worker', nargs=2, metavar=('KEY', 'VARIANT'))
    parser.add_argument('--memory-worker', nargs=2, metavar=('KEY', 'METHOD'))
    parser.add_argument('--reports-only', type=Path)
    args = parser.parse_args()
    if args.exact_memory_worker:
        from visual.runtime.test.ede_exact_optimization_eval import memory_worker as exact_memory
        return exact_memory(*args.exact_memory_worker)
    if args.exact_optimization:
        from visual.runtime.test.ede_exact_optimization_eval import main as exact_main
        return exact_main()
    if args.memory_worker:return memory_worker(*args.memory_worker)
    if args.reports_only:
        out = args.reports_only.resolve(); summary = json.loads((out/'extended_delaunay_edgewise.json').read_text())
        hashes = json.loads((out/'source_hashes.json').read_text()); initial = hashes['protected_before']
        immutable = sorted(list(out.glob('packets/*.npz'))+list(out.glob('records/*.json'))+
                           [out/'extended_delaunay_edgewise.npz'])
        before = {str(p.relative_to(out)):sha(p) for p in immutable}
        changes, diff = {}, []
        for rel, old_hash in hashes['evaluated_sources'].items():
            if sha(REPO/rel) == old_hash:continue
            require(rel in NEW_SOURCES[1:3], 'non-report/core source changed after measurement: '+rel)
            dest = out/'report_source_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/rel, dest)
            changes[rel] = dict(computation_sha256=old_hash, report_sha256=sha(REPO/rel))
            diff.extend(difflib.unified_diff((out/'source_snapshot'/rel).read_text().splitlines(True),
                (REPO/rel).read_text().splitlines(True), fromfile='computation/'+rel, tofile='report/'+rel))
        hashes['report_source_changes'] = changes; write(out/'source_hashes.json', hashes)
        (out/'report_refresh.diff').write_text(''.join(diff))
        figures(out, summary)
        summary['report_process_peak_rss_bytes'] = rss()
        summary['report_process_memory_scope'] = 'fresh report-only worker including loaded numerical JSON/NPZ, plotting and diagnostics; not added to numerical worker high-water'
        after = {str(p.relative_to(out)):sha(p) for p in immutable}
        require(before == after, 'report refresh changed numerical artifacts')
        summary['report_refresh'] = dict(new_uuid=False, new_edges=False, new_benchmark=False,
            numerical_artifact_hashes_before=before, numerical_artifact_hashes_after=after,
            reason='diagram layout, independent depth witness, actual edge-independence example and review status',
            source_changes=changes)
        reports(out, summary)
        write(out/'extended_delaunay_edgewise.json', summary)
        summary['verification'] = verify_outputs(out, summary, initial)
        write(out/'extended_delaunay_edgewise.json', summary); reports(out, summary)
        from utils.experiment_artifacts import register_existing_output
        for path in sorted(out.rglob('*'))+[REPORT]:
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json', 'manifest.json', 'metrics.json', 'config.yaml'):
                kind = 'trajectory' if path.suffix == '.npz' else 'metrics' if path.suffix == '.json' else 'reports'
                register_existing_output(out, path, kind=kind)
        return
    os.environ['WANDB_MODE'] = 'disabled'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        require(os.getenv(key) == '1', 'launch with '+key+'=1 before NumPy import')
    initial_path = Path('/tmp/task05b_initial.json')
    if initial_path.exists():initial = json.loads(initial_path.read_text())
    else:
        initial = {str(p.relative_to(REPO)):sha(p) for run in (BASE, SELECTION, OLD) for p in run.rglob('*') if p.is_file()}
        initial.update({str(p.relative_to(REPO)):sha(p) for p in (REPO/'visual/runtime').rglob('*.py') if str(p.relative_to(REPO)) not in NEW_SOURCES})
    config = dict(task='05-B', frames=FRAMES, keep_ratios=RATIOS, phase=0,
        source_runs=[SELECTION.name, BASE.name, OLD.name], methods=METHODS,
        new_policy_candidate='EDE', analysis_control='DE', warmups=3, measurements=5,
        grouping='stored observed_component_id; no resegmentation',
        DE_correctness_candidates='stored T0', benchmark_candidates='fresh independent generation each invocation',
        thread_env={k:os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    with ExperimentLogger(team='visual', job_type='evaluate', model='task05b_extended_delaunay_edgewise', config=config, seed=0) as logger:
        out = logger.directory
        print('RUN_ID', logger.run_id, flush=True)
        for folder in ('records', 'packets', 'plots'): (out/folder).mkdir()
        snapshot(out, initial)
        packets, evidences, baseline, old, gate = load_inputs(out)
        test = subprocess.run([sys.executable, '-m', 'unittest', *TESTS, '-v'], cwd=REPO, capture_output=True, text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr)
        require(test.returncode == 0, 'contract tests failed; see tests.log')
        import re
        count = int(re.search(r'Ran (\d+) tests', test.stdout+test.stderr).group(1))
        print('input gate PASS; tests', count, flush=True)
        result = numerics(out, packets, evidences, baseline, old)
        summary = dict(run_id=logger.run_id, config=config, environment=environment(),
                       tests=dict(status='PASS', count=count, modules=TESTS), input_hashes=gate['input_hashes'],
                       browser='NOT_EXECUTED', physical_connection_accuracy='NOT_ESTABLISHED', final_policy_adopted=False, **result)
        write(out/'extended_delaunay_edgewise.json', summary)
        tick = perf_counter(); figures(out, summary)
        summary['render_wall_ms'] = (perf_counter()-tick)*1000
        # Placeholder exists before relative-link validation.
        write(out/'verification.json', dict(status='PENDING'))
        reports(out, summary); write(out/'extended_delaunay_edgewise.json', summary)
        summary['verification'] = verify_outputs(out, summary, initial)
        write(out/'extended_delaunay_edgewise.json', summary); reports(out, summary)
        for method, total in summary['totals'].items():logger.log({method:total})
        logger.record(source_runs=config['source_runs'], input_gate='PASS', contracts='PASS',
                      physical_connection_accuracy='NOT_ESTABLISHED', final_policy_adopted=False)
        for path in list(out.rglob('*'))+[REPORT]:
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json', 'metrics.json', 'manifest.json', 'config.yaml'):
                kind = 'reports' if path.suffix in ('.md', '.html', '.png', '.log', '.diff', '.py', '.txt') else 'trajectory' if path.suffix == '.npz' else 'metrics'
                logger.link_output(path, kind=kind)
        print('DONE', logger.run_id, json.dumps(serial(summary['totals'])), flush=True)


if __name__ == '__main__':main()
