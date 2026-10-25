"""Task05 fixed-input CPU geometry evaluation; never resamples or re-extracts.

Run from repo root with WANDB_MODE=disabled and writable MPLCONFIGDIR.
--reports-only DIR only reads saved numerical results after a report failure.
"""
import argparse
from collections import Counter
import copy
import difflib
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from graph.vision_input import receive_vision_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.intra_object_edges import observed_components, build_intra_object_edges, EDGE_POLICY
from visual.runtime.triangle_intra_object_edges import (
    TRIANGLE_POLICY, SUPPORT_REASONS, DEPTH_REASONS, prepare_validation,
    supercover, validate_cover, build_triangle_intra_object_edges, triangles_to_edges,
)
# Reuse the current branch's existing lossless packet/evidence IO and digest.
from visual.runtime.test.run_vision_feature_edge import read_packet, save_packet, array_bytes
from visual.runtime.test.run_adaptive_node_budget import load_evidence
from visual.runtime.test.run_variable_rate_shape_first import packet_digest

REPO = Path(__file__).resolve().parents[3]
BASE = REPO/'experiments/9f90293b227e4dbfaaa6290afad9cc3b'
SELECTION = REPO/'experiments/9185355a32b8433abc70f1738f983377'
FRAMES = (0, 18, 27, 45, 56, 66, 80, 120)
RATIOS = (.5, 1.)
STAGES = ('K8', 'T0', 'TM', 'TMD')
REPORT = REPO/'docs/visual-graph-integration/05_triangle_intra_object_edges.md'
PLAN = REPO/'docs/visual-graph-integration/05_triangle_intra_object_edges_plan.md'
SOURCES = ('visual/runtime/triangle_intra_object_edges.py',
           'visual/runtime/test/test_triangle_intra_object_edges.py',
           'visual/runtime/test/run_triangle_intra_object_edges.py', 'graph/vision_input.py',
           'docs/visual-graph-integration/00_interface.md',
           'visual/runtime/intra_object_edges.py', 'visual/runtime/vision_packet.py',
           'visual/runtime/edge_evidence.py', 'utils/experiment_logger.py',
           'visual/runtime/test/run_vision_feature_edge.py')
TESTS = ('visual.runtime.test.test_triangle_intra_object_edges',
         'visual.runtime.test.test_intra_object_edges', 'visual.runtime.test.test_vision_features',
         'visual.runtime.test.test_variable_rate_graph_contract')


def serial(value):
    if isinstance(value, np.ndarray):return serial(value.tolist())
    if isinstance(value, np.generic):return serial(value.item())
    if isinstance(value, dict):return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):return None
    if isinstance(value, Path):return str(value)
    return value


def write(path, value):
    Path(path).write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def stats(a):
    a = np.asarray(a, float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                max=float(a.max()) if len(a) else None,
                min=float(a.min()) if len(a) else None)


def rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == 'darwin' else r*1024)


def selected_component_rows(packet, maps):
    result = np.empty(len(packet['points']), np.int64)
    for obj in (0, 1):
        rows = packet['object_id'] == obj
        result[rows] = maps[obj].ravel()[packet['provenance']['source_index'][rows]]
    return result


def check_preserved(original, candidate):
    changed = {'edge_policy', 'intra_object_edge_index', 'accepted_triangles', 'edge_evidence_sha256'}
    for key in original:
        if key not in changed and packet_digest(original[key]) != packet_digest(candidate[key]):
            raise AssertionError('immutable packet field changed: '+key)


def load_inputs(out):
    """Exact baseline gate for ALL conditions precedes any source triangulation."""
    start = perf_counter()
    cfg_path = REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json'
    cfg = json.loads(cfg_path.read_text())['config']
    input_hashes = {str(cfg_path.relative_to(REPO)): sha(cfg_path)}
    for name in ('input', 'masks'):
        path = Path(cfg[name])
        if sha(path) != cfg[name+'_sha256']:raise AssertionError('original observation hash mismatch')
        input_hashes[str(path.relative_to(REPO))] = sha(path)
    freeze = json.loads((SELECTION/'selection_before_gt.json').read_text())
    if sha(SELECTION/'selection_before_gt.npz') != freeze['sha256']:
        raise AssertionError('K selection freeze mismatch')
    for name in ('selection_before_gt.json', 'selection_before_gt.npz'):
        input_hashes[str((SELECTION/name).relative_to(REPO))] = sha(SELECTION/name)
    evidence, packets, gates = {}, {}, {}
    # Read only named observation arrays; never read any object GT fields.
    with np.load(cfg['input'], allow_pickle=False) as raw, np.load(cfg['masks'], allow_pickle=False) as masks:
        depth = raw['base_camera/depth']; intrinsic = raw['base_camera/intrinsic']
        c2w = raw['base_camera/cam2world']; tcp = raw['tcp_pose']
        frame_idx, timestamp = raw['frame_idx'], raw['timestamp']
        ms = masks['source_frame']; raw_masks = (masks['peg'], masks['box'])
        for fid in FRAMES:
            ep = BASE/f'runtime_evidence/f{fid:04d}.npz'; input_hashes[str(ep.relative_to(REPO))] = sha(ep)
            e = evidence[fid] = load_evidence(ep); g = e.frame.geometry
            if e.frame.source_frame != fid or e.frame.camera != 'base_camera':raise AssertionError('evidence identity')
            np.testing.assert_array_equal(g.valid_depth, np.isfinite(depth[fid]) & (depth[fid] > 0))
            np.testing.assert_array_equal(g.xyz[..., 2][g.valid_depth], depth[fid][g.valid_depth])
            mi = np.flatnonzero(ms == fid)
            if len(mi) != 1:raise AssertionError('mask source identity')
            for obj in (0, 1):
                np.testing.assert_array_equal(g.object_masks[obj], raw_masks[obj][mi[0]])
                np.testing.assert_array_equal(g.observed_mask(obj), e.frame.objects[obj].valid_points)
            np.testing.assert_array_equal(g.overlap_mask, g.object_masks[0] & g.object_masks[1])
            maps, _ = observed_components(e)
            # Validate original cached depth gate numerically, without estimating any feature.
            for pairs in e.pairs.values():
                z = g.xyz[..., 2].ravel()[pairs.pixels]
                valid = g.valid_depth.ravel()[pairs.pixels].all(1)
                delta = z[valid, 1]-z[valid, 0]
                np.testing.assert_array_equal(pairs.signed_depth_delta_m[valid], delta)
                gate = np.zeros(len(z), bool)
                gate[valid] = np.abs(delta) > (.01+.02*z[valid].min(1))
                np.testing.assert_array_equal(pairs.depth_jump, gate)
            for ratio in RATIOS:
                key = f'f{fid:04d}_r{ratio:g}'
                path = BASE/f'packets/{key}.npz'; kp = SELECTION/f'selection/{key}.npz'
                input_hashes[str(path.relative_to(REPO))] = sha(path)
                input_hashes[str(kp.relative_to(REPO))] = sha(kp)
                p = packets[key] = read_packet(path)
                original_hash = packet_digest(p)
                receive_vision_packet(p)
                if p['edge_policy'] != EDGE_POLICY:raise AssertionError('baseline policy')
                if (p['source_frame'], p['camera'], p['frame_idx'], p['timestamp']) != (fid, 'base_camera', int(frame_idx[fid]), float(timestamp[fid])):
                    raise AssertionError('packet observation identity')
                for name, expected in (('camera_intrinsic', intrinsic[fid]), ('camera_to_world', c2w[fid]), ('observation_ee_pose', tcp[fid])):
                    np.testing.assert_array_equal(p[name], expected)
                with np.load(kp, allow_pickle=False) as k:
                    np.testing.assert_array_equal(p['camera_points'], k['packet/points'])
                    np.testing.assert_array_equal(p['object_id'], k['packet/object_id'])
                    for name, value in p['provenance'].items():
                        np.testing.assert_array_equal(value, k['packet/provenance/'+name])
                src = p['provenance']['source_index']
                np.testing.assert_array_equal(p['camera_points'], g.xyz.reshape(-1, 3)[src])
                edge, comp = build_intra_object_edges(p['camera_points'], p['object_id'], src, maps)
                np.testing.assert_array_equal(edge, p['intra_object_edge_index'])
                np.testing.assert_array_equal(comp, p['observed_component_id'])
                if original_hash != packet_digest(p):raise AssertionError('input mutated during baseline gate')
                gates[key] = dict(status='PASS', N=len(p['points']), E=edge.shape[1], packet_digest=original_hash,
                                  source_and_provenance_exact=True, baseline_edges_components_exact=True)
    write(out/'input_gate.json', dict(status='PASS', conditions=gates, input_hashes=input_hashes,
                                    load_verify_wall_ms=(perf_counter()-start)*1000))
    return packets, evidence, input_hashes


def execute_method(method, original, evidence, maps):
    begin = perf_counter()
    p = copy.deepcopy(original)
    timing = {}
    if method == 'K8':
        tick = perf_counter()
        edge, comp = build_intra_object_edges(p['camera_points'], p['object_id'],
                                             p['provenance']['source_index'], maps)
        timing['knn_dedup_ms'] = (perf_counter()-tick)*1000
        audit = None
    else:
        edge, comp, audit = build_triangle_intra_object_edges(p['camera_points'], p['object_id'],
                                                             p['provenance']['source_index'], maps, evidence)
        timing.update(audit['timing_ms'])
    tick = perf_counter()
    p.update(intra_object_edge_index=edge, observed_component_id=comp)
    if audit is not None:
        p.update(edge_policy=copy.deepcopy(TRIANGLE_POLICY),
                 accepted_triangles=audit['triangles'][audit['accepted']].copy())
    timing['packet_packaging_ms'] = (perf_counter()-tick)*1000
    tick = perf_counter(); received = receive_vision_packet(p)
    timing['receiver_ms'] = (perf_counter()-tick)*1000
    timing['method_wall_ms'] = (perf_counter()-begin)*1000
    # Hashes/diagnostics/readback are intentionally outside the measured core.
    check_preserved(original, p)
    if packet_digest(received.packet) != packet_digest(p):raise AssertionError('receiver mutation')
    if audit is not None:
        failures = [c for c in audit['components'] if c['status'] in ('qhull_error', 'omitted_vertices')]
        if failures:raise AssertionError('triangulation computation failure: '+json.dumps(failures))
    return p, audit, timing


def memory_worker(key, method):
    before = rss()
    fid = int(key[1:5])
    p = read_packet(BASE/f'packets/{key}.npz')
    evidence = load_evidence(BASE/f'runtime_evidence/f{fid:04d}.npz')
    loaded = rss()
    maps, _ = observed_components(evidence)
    p, audit, _ = execute_method(method, p, evidence, maps)
    print(json.dumps(serial(dict(import_baseline_peak_bytes=before, input_loaded_peak_bytes=loaded,
        total_peak_bytes=rss(), operation_peak_increase_bytes=max(0, rss()-loaded),
        input_array_bytes=array_bytes(dict(packet=p, evidence=evidence, maps=maps)),
        triangle_audit_array_bytes=array_bytes(audit) if audit else None,
        raster_work=audit['raster_work'] if audit else None,
        native_qhull_overhead='included_in_RSS_not_separately_identifiable',
        rss_units='bytes; Darwin ru_maxrss bytes, Linux KiB converted',
        scope='fresh process imports + input + components + method + receiver + preservation audit'))), flush=True)


def graph_metrics(packet, edges):
    n = len(packet['points']); oid = packet['object_id']
    degree = np.bincount(edges[0], minlength=n)
    adjacency = coo_array((np.ones(edges.shape[1], bool), (edges[0], edges[1])), shape=(n, n)).tocsr()
    if n:_, labels = connected_components(adjacency, directed=False)
    else:labels = np.empty(0, np.int64)
    unique = edges[:, edges[0] < edges[1]]
    xyz = packet['camera_points'].astype(np.float64)
    src = packet['provenance']['source_index']; w = int(packet['image_shape'][1])
    length = np.linalg.norm(xyz[unique[0]]-xyz[unique[1]], axis=1)
    objects = {}
    for obj in (0, 1):
        rows = np.flatnonzero(oid == obj); keep = oid[unique[0]] == obj
        chosen = np.flatnonzero(keep)
        top = chosen[np.argsort(-length[chosen], kind='stable')[:10]]
        sizes = np.unique(labels[rows], return_counts=True)[1]
        objects[obj] = dict(N=len(rows), U=int(keep.sum()), E=int(2*keep.sum()),
            degree=degree[rows], degree_distribution=stats(degree[rows]),
            isolated_rows=rows[degree[rows] == 0], isolated_count=int((degree[rows] == 0).sum()),
            components=len(sizes), component_sizes=sorted(sizes.tolist(), reverse=True),
            largest_component_fraction=float(sizes.max()/len(rows)) if len(rows) else None,
            length_mm=stats(length[keep]*1000),
            top10=[dict(rows=unique[:, i], source=src[unique[:, i]],
                        uv=np.column_stack((src[unique[:, i]] % w, src[unique[:, i]]//w)),
                        length_mm=float(length[i]*1000), camera_xyz=xyz[unique[:, i]],
                        ee_xyz=packet['points'][unique[:, i]]) for i in top])
    return dict(N=n, U=unique.shape[1], E=edges.shape[1], objects=objects,
                degree=degree, graph_labels=labels)


def diagnose(packet, evidence, audit):
    contexts = prepare_validation(evidence)
    n = len(packet['points']); src = packet['provenance']['source_index']
    w = int(packet['image_shape'][1]); oid = packet['object_id']
    uv = np.column_stack((src % w, src//w))
    edges = dict(K8=packet['intra_object_edge_index'], **audit['stage_edges'])
    cache, stage_metrics = {}, {}
    for name, edge in edges.items():
        m = graph_metrics(packet, edge); reasons = Counter(); invalid, high_normal = [], []
        unique = edge[:, edge[0] < edge[1]]
        for a, b in unique.T:
            identity = (int(a), int(b))
            if identity not in cache:
                r = validate_cover(supercover(uv[[a, b]]), contexts[int(oid[a])])
                normal = packet['geometric_features'][[a, b], :3]
                angle = None
                if packet['feature_field_valid'][[a, b], 0].all():
                    angle = float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(*normal)), np.dot(*normal))))
                cache[identity] = dict(rows=identity, object_id=int(oid[a]), uv=uv[[a, b]],
                    normal_angle_deg=angle, length_mm=float(np.linalg.norm(
                        packet['camera_points'][a].astype(float)-packet['camera_points'][b])*1000), **r)
            r = cache[identity]; reasons.update(r['reasons'])
            if r['reasons']:invalid.append(identity)
            if r['normal_angle_deg'] is not None and r['normal_angle_deg'] > 20:high_normal.append(identity)
        m.update(edge_audit_reason_counts=dict(reasons), invalid_edge_rows=invalid,
                 invalid_edge_count=len(invalid), audited_unique_edges=unique.shape[1],
                 high_normal_rows=high_normal, high_normal_count=len(high_normal))
        stage_metrics[name] = m
    if stage_metrics['TMD']['invalid_edge_count']:raise AssertionError('accepted edge observation audit violation')
    baseline, final = stage_metrics['K8'], stage_metrics['TMD']
    new_isolated = np.flatnonzero((baseline['degree'] > 0) & (final['degree'] == 0))
    splits = []
    for obj in (0, 1):
        for cid in np.unique(baseline['graph_labels'][oid == obj]):
            rows = np.flatnonzero((oid == obj) & (baseline['graph_labels'] == cid))
            sizes = np.unique(final['graph_labels'][rows], return_counts=True)[1]
            if len(sizes) > 1:splits.append(dict(object_id=obj, baseline_graph_component=int(cid),
                rows=rows, sizes=sorted(sizes.tolist(), reverse=True)))
    rejection = Counter(reason for c in audit['triangle_checks'] for reason in c['reasons'])
    candidate_edges = edges['T0'][:, edges['T0'][0] < edges['T0'][1]]
    final_pairs = set(map(tuple, edges['TMD'].T.tolist()))
    absent = [tuple(map(int, ab)) for ab in candidate_edges.T if tuple(ab) not in final_pairs]
    removal = [dict(rows=ab, local_reasons=cache[ab]['reasons'],
                    reason='no_accepted_incident_triangle') for ab in absent]
    return dict(stages=stage_metrics, new_isolated_rows=new_isolated, new_isolated_source=src[new_isolated],
                baseline_components_split=splits, triangle_reasons=dict(rejection),
                removed_candidate_edges=removal, edge_audit=list(cache.values()))


def run_numerics(out, packets, evidences):
    records, raw, common = {}, {}, {}
    for fid in FRAMES:
        e = evidences[fid]; timings = {f'f{fid:04d}_r{r:g}': {'K8': [], 'TMD': []} for r in RATIOS}
        saved, audits, cc_samples = {}, {}, []
        for repeat in range(8):
            tick = perf_counter(); maps, ccstats = observed_components(e)
            common_ms = (perf_counter()-tick)*1000
            order = ('K8', 'TMD') if repeat % 2 == 0 else ('TMD', 'K8')
            for ratio in RATIOS:
                key = f'f{fid:04d}_r{ratio:g}'
                for method in order:
                    p, audit, timing = execute_method(method, packets[key], e, maps)
                    timing['common_component_ms'] = common_ms
                    timing['core_wall_with_common_ms'] = common_ms+timing['method_wall_ms']
                    if method == 'K8':
                        np.testing.assert_array_equal(p['intra_object_edge_index'], packets[key]['intra_object_edge_index'])
                    else:
                        if key in audits:
                            for name in ('triangles', 'accepted', 'support_pass', 'depth_pass'):
                                np.testing.assert_array_equal(audit[name], audits[key][name])
                        audits[key] = audit; saved[key] = p
                    if repeat >= 3:timings[key][method].append(timing)
            if repeat >= 3:cc_samples.append(common_ms)
        common[fid] = dict(component_ms=stats(cc_samples), raw_ms=cc_samples,
                            observed_components=ccstats, scope='once per frame repeat shared by both ratios and methods')
        for ratio in RATIOS:
            key = f'f{fid:04d}_r{ratio:g}'; p = packets[key]; audit = audits[key]
            print('diagnostics', key, 'N', len(p['points']), 'T', len(audit['triangles']),
                  'accepted', int(audit['accepted'].sum()), flush=True)
            tick = perf_counter(); diag = diagnose(p, e, audit)
            memory = {}
            for method in ('K8', 'TMD'):
                worker = subprocess.run([sys.executable, '-m', 'visual.runtime.test.run_triangle_intra_object_edges',
                    '--memory-worker', key, method], cwd=REPO, capture_output=True, text=True,
                    env=dict(os.environ, MPLCONFIGDIR='/tmp/task05-mpl', WANDB_MODE='disabled'), check=True)
                memory[method] = json.loads(worker.stdout.strip().splitlines()[-1])
            rec = dict(frame=fid, ratio=ratio, N=len(p['points']),
                triangles=dict(T0=len(audit['triangles']), TM=int(audit['support_pass'].sum()),
                               TMD=int(audit['accepted'].sum())),
                rejected_at_support=int((~audit['support_pass']).sum()),
                rejected_at_depth=int((audit['support_pass'] & ~audit['depth_pass']).sum()),
                triangle_checks=audit['triangle_checks'], component_status=audit['components'],
                raster_work=audit['raster_work'], timings_raw=timings[key],
                timing_ms={m: {stage: stats([t[stage] for t in samples]) for stage in samples[0]}
                           for m, samples in timings[key].items()}, memory=memory,
                diagnostics_and_memory_workers_ms=(perf_counter()-tick)*1000, **diag)
            records[key] = rec
            save_packet(out/'packets'/f'{key}.npz', saved[key])
            # Aggregate preserves every original packet array plus triangle stages.
            from visual.runtime.test.run_variable_rate_shape_first import flatten_arrays
            raw.update(flatten_arrays(p, prefix=key+'/original'))
            for name in ('triangles', 'accepted', 'support_pass', 'depth_pass'):
                raw[key+'/'+name] = audit[name]
            for name, edge in audit['stage_edges'].items():raw[key+'/'+name+'_edges'] = edge
            for name, m in diag['stages'].items():
                raw[key+'/'+name+'_degree'] = m['degree']; raw[key+'/'+name+'_graph_labels'] = m['graph_labels']
            raw[key+'/new_isolated_rows'] = diag['new_isolated_rows']
            raw[key+'/uv'] = np.column_stack((p['provenance']['source_index'] % p['image_shape'][1],
                                            p['provenance']['source_index']//p['image_shape'][1]))
            write(out/'records'/f'{key}.json', rec)
        write(out/'progress.json', dict(last_completed_frame=fid, completed_conditions=len(records)))
    np.savez_compressed(out/'triangle_intra_object_edges.npz', **raw)
    with np.load(out/'triangle_intra_object_edges.npz', allow_pickle=False) as restored:
        if set(raw) != set(restored.files):raise AssertionError('NPZ keys mismatch')
        for key, value in raw.items():np.testing.assert_array_equal(value, restored[key])
    return records, common


def track_suspect(out, records):
    """Post-selection diagnostic only; these UVs never enter the edge helper."""
    result = {}
    with np.load(out/'triangle_intra_object_edges.npz', allow_pickle=False) as a:
        for ratio in RATIOS:
            key = f'f0000_r{ratio:g}'; prefix = key+'/original/'
            p = read_packet(BASE/f'packets/{key}.npz')
            src = p['provenance']['source_index']; w = int(p['image_shape'][1])
            rows = [np.flatnonzero((p['object_id'] == 1) & (src == 206*w+u)).tolist() for u in (251, 259)]
            if not all(rows):
                result[key] = dict(status='endpoint_not_selected', rows=rows); continue
            pair = np.sort(np.array([r[0] for r in rows])); triangles = a[key+'/triangles']
            incident = np.flatnonzero(np.isin(triangles, pair).sum(1) == 2)
            result[key] = dict(rows=pair, uv=[[251, 206], [259, 206]],
                candidate_exists=bool(len(incident)), incident_triangle_ids=incident,
                incident_triangles=triangles[incident],
                incident_checks=[records[key]['triangle_checks'][i] for i in incident],
                support_pass=a[key+'/support_pass'][incident], depth_pass=a[key+'/depth_pass'][incident],
                final_exists=tuple(pair) in set(map(tuple, a[key+'/TMD_edges'].T.tolist())),
                baseline_length_mm=float(np.linalg.norm(p['camera_points'][pair[0]].astype(float)
                                                        -p['camera_points'][pair[1]])*1000))
    return result


def build_figures(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from scipy.ndimage import binary_fill_holes
    plt.rcParams.update({'font.size': 9})
    cfg = json.loads((REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json').read_text())['config']
    with np.load(cfg['input'], allow_pickle=False) as a:
        rgb = {f:a['base_camera/rgb'][f] for f in FRAMES}
    def edge2(ax, uv, edge, color='#00b6d6', width=.4):
        unique = edge[:, edge[0] < edge[1]]
        if unique.shape[1]:ax.add_collection(LineCollection(uv[unique.T], colors=color, linewidths=width, alpha=.65))
    def edge3(ax, xyz, edge):
        unique = edge[:, edge[0] < edge[1]]
        ax.scatter(*xyz.T, s=2, c='#263e50')
        if unique.shape[1]:ax.add_collection3d(Line3DCollection(xyz[unique.T], colors='#00a8c4', linewidths=.35, alpha=.6))
        ax.set(xlabel='camera x (m)', ylabel='camera y (m)', zlabel='camera z (m)')
    def save(fig, name):
        fig.savefig(out/'plots'/f'{name}.png', dpi=135, bbox_inches='tight'); plt.close(fig)
    normal_examples = []
    visual_witnesses = {}
    with np.load(out/'triangle_intra_object_edges.npz', allow_pickle=False) as raw:
        for key, record in summary['records'].items():
            fid = record['frame']; p = read_packet(BASE/f'packets/{key}.npz')
            uv = raw[key+'/uv']; triangles = raw[key+'/triangles']; accepted = raw[key+'/accepted']
            support_pass = raw[key+'/support_pass']; xyz = p['camera_points'].astype(float)
            edges = dict(K8=p['intra_object_edge_index'], **{s:raw[key+'/'+s+'_edges'] for s in STAGES[1:]})
            e = load_evidence(BASE/f'runtime_evidence/f{fid:04d}.npz'); g = e.frame.geometry
            fig, axs = plt.subplots(1, 4, figsize=(18, 4.7))
            axs[0].imshow(rgb[fid]); axs[0].set_title('Original RGB / mask / observed support')
            masked = np.ma.masked_where(~(g.object_masks[0] | g.object_masks[1]), g.object_masks[1].astype(float))
            axs[0].imshow(masked, cmap='cool', vmin=0, vmax=1, alpha=.25)
            axs[0].scatter(uv[:, 0], uv[:, 1], c=p['object_id'], cmap='cool', s=3)
            for ax, stage, keep in zip(axs[1:], STAGES[1:], (np.ones(len(triangles), bool), support_pass, accepted)):
                ax.imshow(rgb[fid]); ax.add_collection(PolyCollection(uv[triangles[keep]],
                    facecolors='#00d0bc', edgecolors='#007e83', linewidths=.35, alpha=.28))
                ax.scatter(uv[:, 0], uv[:, 1], s=2, c='#d22b5b')
                ax.set_title(f'{stage}: T={int(keep.sum())}; directed E={edges[stage].shape[1]}')
            for ax in axs:ax.set(xlim=(0, 511), ylim=(511, 0), xlabel='u', ylabel='v')
            fig.suptitle(key+' · unchanged nodes; whole-triangle rejection; closed-cell supercover')
            save(fig, key+'_triangles')
            fig = plt.figure(figsize=(13, 10))
            for i, stage in enumerate(('K8', 'TMD')):
                ax = fig.add_subplot(2, 2, i+1); ax.imshow(rgb[fid]); edge2(ax, uv, edges[stage])
                ax.scatter(uv[:, 0], uv[:, 1], s=3, c='#d52d5b')
                ax.set(xlim=(0, 511), ylim=(511, 0), title=stage+' · same Point rows', xlabel='u', ylabel='v')
                ax = fig.add_subplot(2, 2, i+3, projection='3d'); edge3(ax, xyz, edges[stage])
                ax.set_title(stage+' · full finite extent (no point clipping)')
            fig.suptitle(key+' · 2D/3D edge comparison; full extent includes remote observations')
            save(fig, key+'_edges')
            # Witnesses are chosen AFTER edge generation and cannot affect it.
            support_ids = [i for i, c in enumerate(record['triangle_checks']) if set(c['reasons']) & SUPPORT_REASONS]
            # Prefer an actual enclosed mask/support gap, using observations only
            # AFTER all numerical results have been frozen. No UV/shape prior.
            holes = {obj:binary_fill_holes(g.observed_mask(obj)) & ~g.observed_mask(obj) for obj in (0, 1)}
            def hole_witness(index):
                obj = int(p['object_id'][triangles[index, 0]])
                check = record['triangle_checks'][index]['interior']
                sources = np.asarray(check['witnesses'].get('mask_outside', []), np.int64)
                return bool(len(sources) and holes[obj].ravel()[sources].any())
            support_ids.sort(key=lambda i:not hole_witness(i))
            # pair_unsupported is NOT a depth-discontinuity example. Show actual
            # stored jump=true with all support checks passing in this column.
            depth_ids = [i for i, c in enumerate(record['triangle_checks'])
                         if support_pass[i] and 'depth_jump' in c['reasons']]
            visual_witnesses[key] = dict(support_triangle=support_ids[0] if support_ids else None,
                enclosed_observation_gap=hole_witness(support_ids[0]) if support_ids else False,
                depth_triangle=depth_ids[0] if depth_ids else None)
            fig, axs = plt.subplots(2, 2, figsize=(11, 9))
            for col, (title, ids) in enumerate((('Mask hole / support gap reject', support_ids), ('Supported local-depth jump reject', depth_ids))):
                axs[0, col].imshow(rgb[fid]); im = axs[1, col].imshow(g.xyz[..., 2], cmap='viridis')
                fig.colorbar(im, ax=axs[1, col], label='camera z (m)')
                if ids:
                    index = ids[0]; tri = triangles[index]; vertices = uv[tri]
                    info = record['triangle_checks'][index]
                    for ax in axs[:, col]:
                        ax.plot(*np.vstack((vertices, vertices[0])).T, color='#ef3d50')
                        cover = supercover(vertices)
                        ax.scatter(cover[:, 0], cover[:, 1], s=7, c='#ffb300', alpha=.4)
                        for check in [info['interior'], *info['edges']]:
                            for reason, values in check['witnesses'].items():
                                if reason == 'image_outside':pts = np.asarray(values)
                                else:
                                    s = np.asarray(values).ravel(); pts = np.column_stack((s % 512, s//512))
                                if len(pts):ax.scatter(pts[:, 0], pts[:, 1], s=20, c='#d50048', marker='x')
                        low, high = vertices.min(0)-4, vertices.max(0)+4
                        ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]))
                    crop = g.xyz[max(0,low[1]):min(512,high[1]+1), max(0,low[0]):min(512,high[0]+1), 2]
                    finite = crop[np.isfinite(crop)]
                    if len(finite) and finite.max() > finite.min():im.set_clim(float(finite.min()), float(finite.max()))
                    axs[0, col].set_title(f'{title} · triangle {index}\n'+', '.join(info['reasons']))
                else:axs[0, col].set_title(title+' · no example in this condition')
            fig.suptitle(key+' · raster cells / original depth / reject witnesses; no GT')
            save(fig, key+'_witnesses')
            # High-normal accepted edge nearest to a 90-degree diagnostic, then length.
            high = record['stages']['TMD']['high_normal_rows']
            details = {tuple(a['rows']):a for a in record['edge_audit']}
            if high:
                choices = sorted(high, key=lambda ab:(abs(details[tuple(ab)]['normal_angle_deg']-90), details[tuple(ab)]['length_mm']))
                ab = np.asarray(choices[0], np.int64); info = details[tuple(ab)]; vertices = uv[ab]
                normal_examples.append(dict(condition=key, **info))
                fig = plt.figure(figsize=(12, 4.7))
                ax = fig.add_subplot(131); ax.imshow(rgb[fid]); edge2(ax, uv, edges['TMD'], width=.25)
                ax.plot(*vertices.T, c='#fd334a', lw=2); low, highuv = vertices.min(0)-12, vertices.max(0)+12
                ax.set(xlim=(low[0], highuv[0]), ylim=(highuv[1], low[1]), title='Observed RGB + accepted connection')
                ax = fig.add_subplot(132)
                crop = g.xyz[max(0,low[1]):min(512,highuv[1]+1), max(0,low[0]):min(512,highuv[0]+1), 2]
                finite = crop[np.isfinite(crop)]
                ax.imshow(g.xyz[..., 2], cmap='viridis', vmin=float(finite.min()), vmax=float(finite.max()))
                cover = supercover(vertices); ax.scatter(cover[:, 0], cover[:, 1], s=13, c='#ffab00')
                ax.plot(*vertices.T, c='#fd334a', lw=2)
                ax.set(xlim=(low[0], highuv[0]), ylim=(highuv[1], low[1]), title='Path: support / local depth pass')
                ax = fig.add_subplot(133, projection='3d')
                local = np.all((uv >= low) & (uv <= highuv), axis=1) & (p['object_id'] == p['object_id'][ab[0]])
                ax.scatter(*xyz[local].T, s=5, c='#587187'); ax.plot(*xyz[ab].T, c='#fd334a', lw=2)
                normals = e.frame.objects[int(p['object_id'][ab[0]])].normals.reshape(-1, 3)[src_for(p, ab)]
                for point, normal in zip(xyz[ab], normals):ax.quiver(*point, *normal, length=.01, color='#f28c00')
                ax.set(xlabel='camera x (m)', ylabel='camera y (m)', zlabel='camera z (m)')
                fig.suptitle(f'{key} · normal difference {info["normal_angle_deg"]:.1f}° allowed; rows {ab}\n'
                             'Observation continuity demonstrated; physical face labels NOT_ESTABLISHED')
                save(fig, key+'_normal_connection')
        # The known remote pair, both candidate membership and triangle outcomes.
        for key, track in summary['suspect'].items():
            if 'candidate_exists' not in track:continue
            p = read_packet(BASE/f'packets/{key}.npz'); uv = raw[key+'/uv']; xyz = p['camera_points'].astype(float)
            tri = raw[key+'/triangles']; rows = track['rows']; incident = track['incident_triangle_ids']
            fig = plt.figure(figsize=(14, 8))
            for col, stage in enumerate(('K8', 'T0', 'TMD')):
                edge = p['intra_object_edge_index'] if stage == 'K8' else raw[key+'/'+stage+'_edges']
                ax = fig.add_subplot(2, 3, col+1); ax.imshow(rgb[0]); edge2(ax, uv, edge)
                if len(incident):ax.add_collection(PolyCollection(uv[tri[incident]], facecolors='#f2a23b', alpha=.25))
                ax.plot(*uv[rows].T, c='#f21b45', lw=2)
                ax.set(xlim=(243, 270), ylim=(222, 197), title=stage+' · diagnostic UV zoom')
                ax = fig.add_subplot(2, 3, col+4, projection='3d'); edge3(ax, xyz, edge)
                ax.plot(*xyz[rows].T, c='#f21b45', lw=2); ax.set_title(stage+' · all finite points')
            fig.suptitle(f'{key} · 263mm pair: candidate={track["candidate_exists"]}, final={track["final_exists"]}\n'
                         'No coordinate-specific edge removal; red segment is a diagnostic reference')
            save(fig, key+'_suspect')
    summary['normal_examples'] = normal_examples
    summary['visual_witnesses'] = visual_witnesses
    keys = list(summary['records']); records = summary['records']; x = np.arange(len(keys))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.7))
    for stage in STAGES:
        for ax, metric in zip(axs, ('mean', 'p95', 'max')):
            values = []
            for key in keys:
                objs = records[key]['stages'][stage]['objects']
                vals = [obj['length_mm'][metric] for obj in objs.values() if obj['length_mm'][metric] is not None]
                values.append(max(vals) if vals else np.nan)
            ax.plot(x, values, marker='.', label=stage); ax.set_title('Max object '+metric+' edge length (mm)')
    for ax in axs:ax.set_xticks(x, keys, rotation=90); ax.legend()
    save(fig, 'edge_lengths')
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.7))
    for stage in STAGES:
        degree = np.concatenate([r['stages'][stage]['degree'] for r in records.values()])
        axs[0].hist(degree, bins=np.arange(int(degree.max())+2)-.5, histtype='step', label=stage)
        axs[1].plot(x, [sum(o['isolated_count'] for o in r['stages'][stage]['objects'].values()) for r in records.values()], label=stage)
        axs[2].plot(x, [sum(o['components'] for o in r['stages'][stage]['objects'].values()) for r in records.values()], label=stage)
    axs[0].set(title='Degree histogram (all fixed conditions)', xlabel='degree', ylabel='node count')
    axs[1].set_title('Isolated nodes'); axs[2].set_title('Actual Graph components (incl. isolated)')
    for ax in axs:ax.legend()
    for ax in axs[1:]:ax.set_xticks(x, keys, rotation=90)
    save(fig, 'degree_components')
    fig, axs = plt.subplots(2, 1, figsize=(13, 9))
    for method in ('K8', 'TMD'):
        vals = [r['timing_ms'][method]['core_wall_with_common_ms']['mean'] for r in records.values()]
        axs[0].plot(x, vals, marker='o', label=method)
    axs[0].axhline(50, c='#b24545', ls='--', label='50ms reference (not total Vision latency)')
    axs[0].set(ylabel='CPU ms', title='Core method wall + common component cost'); axs[0].legend()
    stages = ('input_lookup_ms', 'delaunay_ms', 'supercover_ms', 'support_validation_ms',
              'depth_validation_ms', 'edge_packaging_ms', 'packet_packaging_ms', 'receiver_ms')
    bottom = np.zeros(len(keys))
    for stage in stages:
        vals = np.array([r['timing_ms']['TMD'][stage]['mean'] for r in records.values()])
        axs[1].bar(x, vals, bottom=bottom, label=stage); bottom += vals
    axs[1].set(ylabel='CPU ms', title='TMD stage timers (wall also includes copy / loop overhead)'); axs[1].legend(fontsize=7, ncol=3)
    for ax in axs:ax.set_xticks(x, keys, rotation=90)
    fig.tight_layout(); save(fig, 'latency')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for method in ('K8', 'TMD'):
        axs[0].plot(x, [r['memory'][method]['total_peak_bytes']/2**20 for r in records.values()], marker='.', label=method)
        axs[1].plot(x, [r['memory'][method]['operation_peak_increase_bytes']/2**20 for r in records.values()], marker='.', label=method)
    axs[0].set_title('Fresh worker peak RSS (MiB)'); axs[1].set_title('Peak increase after input load (MiB)')
    for ax in axs:ax.legend(); ax.set_xticks(x, keys, rotation=90)
    save(fig, 'memory')


def src_for(packet, rows):
    return packet['provenance']['source_index'][np.asarray(rows, np.int64)]


def enrich_saved_metrics(out, summary):
    """Report-only counts derived from saved triangles; no edge computation."""
    with np.load(out/'triangle_intra_object_edges.npz', allow_pickle=False) as a:
        for key, rec in summary['records'].items():
            oid = a[key+'/original/object_id']; t = a[key+'/triangles']
            for stage in STAGES:
                keep = (np.ones(len(t), bool) if stage == 'T0' else
                        a[key+'/support_pass'] if stage == 'TM' else
                        a[key+'/accepted'] if stage == 'TMD' else None)
                for obj, m in rec['stages'][stage]['objects'].items():
                    m['T'] = None if stage == 'K8' else int((oid[t[keep, 0]] == int(obj)).sum())
        for key, track in summary.get('suspect', {}).items():
            if not track.get('rows') is None and 'candidate_exists' in track:
                comp = a[key+'/original/observed_component_id']
                cid = int(comp[track['rows'][0]])
                track['observed_component_id'] = cid
                track['component_status'] = [c for c in summary['records'][key]['component_status']
                    if c['object_id'] == 1 and c['component_id'] == cid]
                track['incident_validation_status'] = ('EVALUATED' if track['candidate_exists']
                                                        else 'NOT_APPLICABLE_NO_CANDIDATE')
    # Read version bytes from the installed extension; no Qhull call is made.
    import re
    import scipy.spatial._qhull as qhull
    binary = Path(qhull.__file__)
    strings = re.findall(rb'[ -~]{4,}', binary.read_bytes())
    versions = [s.decode() for s in strings if s.startswith(b'qhull_r ') and b'2020' in s]
    summary['environment']['qhull_bundled_version'] = versions[0] if versions else 'NOT_EXPOSED'
    summary['environment']['qhull_extension_sha256'] = sha(binary)


def overview(summary):
    records = summary['records']
    total = {}
    for stage in STAGES:
        total[stage] = dict(
            T=None if stage == 'K8' else sum(r['triangles'][stage] for r in records.values()),
            E=sum(r['stages'][stage]['E'] for r in records.values()),
            U=sum(r['stages'][stage]['U'] for r in records.values()),
            isolated=sum(o['isolated_count'] for r in records.values() for o in r['stages'][stage]['objects'].values()),
            graph_components=sum(o['components'] for r in records.values() for o in r['stages'][stage]['objects'].values()),
            invalid_edges=sum(r['stages'][stage]['invalid_edge_count'] for r in records.values()),
            high_normal_edges=sum(r['stages'][stage]['high_normal_count'] for r in records.values()))
    return dict(stages=total, new_isolated=sum(len(r['new_isolated_rows']) for r in records.values()),
                split_baseline_components=sum(len(r['baseline_components_split']) for r in records.values()),
                support_rejections=sum(r['rejected_at_support'] for r in records.values()),
                depth_rejections=sum(r['rejected_at_depth'] for r in records.values()),
                cpu_mean_ms={m: float(np.mean([r['timing_ms'][m]['core_wall_with_common_ms']['mean'] for r in records.values()]))
                             for m in ('K8', 'TMD')})


def render_reports(out, summary):
    o = summary['overview']; totals = o['stages']; run = summary['run_id']
    isolation = o['new_isolated']; speed = o['cpu_mean_ms']
    status = '채택 보류'
    lines = [
        '# Task05 — Observation-Constrained Triangular Graph Edges', '',
        f'Run UUID: `{run}`. 계약 **PASS**, 고정 입력 16개 조건의 CPU offline 비교 완료. 최종 정책은 **{status}**다.', '',
        f'[정적 HTML](../../experiments/{run}/triangle_intra_object_edges.html) · '
        f'[JSON](../../experiments/{run}/triangle_intra_object_edges.json) · '
        f'[NPZ](../../experiments/{run}/triangle_intra_object_edges.npz) · '
        f'[검증](../../experiments/{run}/verification.json) · '
        f'[테스트 로그](../../experiments/{run}/tests.log) · '
        f'[Source/input hashes](../../experiments/{run}/source_hashes.json) · '
        f'[Implementation diff](../../experiments/{run}/implementation.diff)', '',
        '## 결과와 판단', '',
        f'TMD는 관측 규칙상 invalid unique Edge를 {totals["K8"]["invalid_edges"]}→{totals["TMD"]["invalid_edges"]}개로 줄였다. '
        f'동시에 K8에서 비고립이던 {isolation}개 node가 고립되고, 기존 Graph component '
        f'{o["split_baseline_components"]}개가 추가로 분리됐다. 이는 16개 조건을 합한 수이며 물리적 동일 node의 episode 통계가 아니다.', '',
        f'공통 component 비용을 포함한 조건별 CPU mean의 평균은 K8 {speed["K8"]:.3f}ms, '
        f'TMD {speed["TMD"]:.3f}ms다. '
        + ('이번 구현에서 속도 개선은 없었다.' if speed['TMD'] >= speed['K8'] else '이번 고정 조건의 평균에서 TMD 비용이 작았다.')
        + ' 검증 비용을 포함한 실제 값이며 전체 Vision→Graph latency나 20Hz 결과가 아니다.', '',
        '| stage | Triangle T | unique U | directed E | isolated | Graph components | 관측 감사 invalid U | normal >20° U |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage, value in totals.items():
        lines.append('| '+stage+' | '+' | '.join(str(value[k]) if value[k] is not None else 'N/A'
            for k in ('T', 'U', 'E', 'isolated', 'graph_components', 'invalid_edges', 'high_normal_edges'))+' |')
    lines += ['', f'Support 단계에서 triangle {o["support_rejections"]}개, 이후 depth 단계에서 '
              f'{o["depth_rejections"]}개를 거부했다. T0/TM은 동일 Delaunay의 중간 진단이며 새 후보는 TMD 하나다.', '',
        '## 입력·규칙·수신 계약', '',
        'Task04 `9f90293b227e4dbfaaa6290afad9cc3b`와 Task03-K `9185355a32b8433abc70f1738f983377`를 사용했다. '
        'Frame 0/18/27/45/56/66/80/120 × keep_ratio .5/1, phase 0이다. 전체 입력의 k-NN(k=8)과 '
        'observed component exact 재현을 신규 비교 전에 완료했다. Frame18/56의 빈 Box도 유지했다.', '',
        '저장 Point/camera XYZ/EE XYZ/F=5/validity/provenance/행 순서와 변환 metadata를 그대로 읽었다. '
        'Sampling, PCA5/Feature extraction, EE 변환을 재실행하지 않았다. Receiver의 기존 좌표 일관성 검사는 '
        '유지했으며 새로운 좌표를 생성하거나 덮어쓰지 않았다.', '',
        f'동일 object/observed component의 UV에 2D Delaunay (`{summary["environment"]["scipy"]}`, '
        '`Qbb Qc Qz Q12`)를 적용했다. Integer pixel centre의 닫힌 ±.5 cell과 접촉하는 모든 cell을 '
        'int64 SAT로 포함했다. Boundary 세 경로와 triangle 전체를 각각 검사하고, **해당 cover에 양 끝 '
        'cell이 모두 속하는 right/down pair만** 검사했다. Bbox 외부 pair는 검사하지 않는다. Corner-touch는 '
        '모든 접촉 cell을 포함하므로 보수적인 경계 거부 가능성이 있다.', '',
        'Mask 밖/invalid depth/overlap/image 밖과 pair missing/unsupported/invalid delta/depth jump를 구분했다. '
        '기존 adjacent gate `abs(delta_z)>0.01+0.02*min(z)`를 유지했다. Endpoint 전체 depth 차이, '
        'normal angle, face label을 veto로 사용하지 않았다. 모든 거부 사유를 기록하기 위해 support에서 '
        '이미 거부된 triangle도 depth 진단을 수행했으며 그 비용은 core에 포함했다.', '',
        'Triangle 하나가 실패하면 그 세 변을 모두 제거하고 다른 accepted triangle의 공유 변만 유지했다. '
        'Receiver는 accepted triangle index/소속/양의 UV 면적과 Edge 합집합의 정확한 일치를 검증한다. '
        'Dense RGB-D 검증은 helper의 책임이다. 기존 k-NN/v1 의미와 실제 공급된 robot에서만 +3 규칙을 보존했다.', '',
        '## Frame 0 원거리 연결', '']
    for key, track in summary['suspect'].items():
        if 'candidate_exists' not in track:
            lines.append(f'- `{key}`: 해당 diagnostic endpoint가 선택되지 않음.'); continue
        lines.append(f'- `{key}`: 약 {track["baseline_length_mm"]:.3f}mm pair의 Delaunay 후보 포함 '
                     f'**{track["candidate_exists"]}**, incident triangle {len(track["incident_triangle_ids"])}개, '
                     f'최종 Edge 유지 **{track["final_exists"]}**. Observed component {track.get("observed_component_id")}의 '
                     '선택점 12개가 UV collinear여서 triangle을 만들지 못했다. 따라서 이 pair의 support/depth '
                     'triangle 검증은 N/A다. 상세 상태와 다른 top-10 원거리 Edge는 HTML/JSON에 있다.')
    lines += ['', '해당 pixel 경로는 약 8.431m의 동일 depth와 jump=false를 갖는 기존 관측이다. '
        'Mask/depth가 함께 오염된 plateau를 이 규칙으로 물리적 outlier라고 판정할 수 없다. '
        '대상 pair가 없어져도 물리적 오류 해결을 의미하지 않는다. 좌표/Box hole/GT에 따른 선택 예외는 없다.', '',
        'Frame 0 Box의 최종 최대 Edge는 ratio .5에서 104.559mm, ratio 1에서 136.428mm다. '
        '각각 endpoint camera-z가 약 1.290↔1.389m, 1.260↔1.389m인 다른 연결이 남았다. '
        '전체 원거리 Point는 삭제하지 않았으며 모든 길이 통계와 top-10에 포함했다.', '',
        '## 연결성·모서리 진단', '',
        f'Normal 차이가 20°를 넘는 TMD Edge {totals["TMD"]["high_normal_edges"]}개가 유지됐다. '
        '조건별 실제 RGB-D 경로와 normal 화살표 PNG를 제공한다. 관측의 굽은 영역을 잇는 사례와 '
        'normal-only veto가 없다는 계약 근거이며, 물리적 face/corner 정답 라벨은 **NOT_ESTABLISHED**다. '
        'Synthetic corner 테스트는 동일 support/depth에서 normal angle을 90°로 바꿔도 연결이 불변임을 확인했다.', '',
        '새 고립 row/source, K8 Graph component별 TMD 분할 크기, 모든 degree와 최종 Graph component label, '
        'triangle·edge 거부 원인/witness를 JSON/NPZ에 보존했다. 모든 Point를 유지하고 2점/collinear '
        '성분에 fallback Edge를 넣지 않았다. 정상 연결 손실과 관측 공백 제거 사이의 trade-off가 있어 '
        '채택을 자동화하지 않는다.', '',
        '## 조건별 비용·크기', '',
        '| condition | N | T0→TM→TMD | K8→TMD E | 새 고립 N | K8 core mean ms | TMD core mean ms | K8/TMD peak RSS MiB |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | --- |']
    for key, r in summary['records'].items():
        lines.append(f'| {key} | {r["N"]} | '+ '→'.join(str(r['triangles'][s]) for s in STAGES[1:])
            + f' | {r["stages"]["K8"]["E"]}→{r["stages"]["TMD"]["E"]} | {len(r["new_isolated_rows"])}'
            + f' | {r["timing_ms"]["K8"]["core_wall_with_common_ms"]["mean"]:.3f}'
            + f' | {r["timing_ms"]["TMD"]["core_wall_with_common_ms"]["mean"]:.3f}'
            + f' | {r["memory"]["K8"]["total_peak_bytes"]/2**20:.1f}/{r["memory"]["TMD"]["total_peak_bytes"]/2**20:.1f} |')
    lines += ['', '같은 CPU/interpreter에서 3회 warmup + 5회 측정하고 반복마다 방법 순서를 교대했다.']
    lines += ['공통 component는 frame-repeat마다 한 번 생성해 두 ratio/방법이 공유한다. 조건별 core에는 '
        '그 공통 비용을 한 번 더한 standalone subtotal을 표시하며 조건 합산에서 중복 없는 전체 wall과 구분한다. '
        '양쪽 모두 packet copy/receiver를 포함하고 TMD lookup·rasterization·support/depth·dedup도 계측했다. '
        'Hash audit/진단/NPZ/그림/logging은 core 밖이다. 5개 표본의 p95는 기술통계다.', '',
        '계측 대상은 현재 evaluation helper다. 이 helper는 T0/TM/TMD의 변 합집합 세 개와 모든 '
        'triangle 거부 사유를 함께 만든다. `edge_packaging_ms`에도 세 단계 포장이 포함된다. '
        '따라서 이 수치를 분석 출력을 생략한 production TMD-only 구현의 최소 latency로 해석하지 않는다. '
        '이번 구현의 실제 비용을 비교하며 별도 fast path/최적화는 실행하지 않았다.', '',
        '메모리는 fresh worker의 import baseline/input-load high-water/전체 peak와 배열 byte, lookup·scratch 추정치를 '
        '기록했다. Native Qhull/Python allocator overhead는 RSS에 포함되며 별도로 정확히 분리 측정하지 못했다. '
        'K8의 기존 Task04 시간은 재사용하지 않고 같은 실행에서 재계측했다.', '',
        f'![길이](../../experiments/{run}/plots/edge_lengths.png)', '',
        f'![연결성](../../experiments/{run}/plots/degree_components.png)', '',
        f'![CPU](../../experiments/{run}/plots/latency.png)', '',
        f'![메모리](../../experiments/{run}/plots/memory.png)', '',
        '## 검증과 범위', '',
        f'계약 테스트 {summary["tests"]["count"]}개 PASS. Fraction 기반 독립 raster oracle, 퇴화/누락/QhullError, '
        'shared edge, ramp/interior seam, GT 독립성, normal veto 없음, accepted-triangle union, deep copy/offset을 검사했다. '
        'JSON/NPZ readback, PNG decode·상대 링크·원본 보호 hash는 verification.json에 별도 기록한다.', '',
        '실제 브라우저: **NOT_EXECUTED**. PNG 시각 검토 상태는 verification.json에 실제 검사한 파일만 기록한다. '
        '계약 PASS와 TMD validation 위반 0은 물리적 mesh 정확성·최종 정책 채택·GNN 성능을 증명하지 않는다.', '',
        '공통 ExperimentLogger와 로컬 artifact 연결을 재사용했다. Dependency 설치·외부 업로드·Git commit, '
        '추가 episode/simulator/renderer, CUDA/MPS/PCL, GNN/GRU/BC 학습·rollout, feature/sampling/threshold '
        '변경, k-NN fallback은 수행하지 않았다. 결과 검토 후 다음 정책을 결정한다.']
    if summary.get('report_only_recovery'):
        lines += ['', '최초 수치 계산·계측 완료 후 normal PNG 렌더러의 row 인덱싱 오류가 발생했다. '
            '`failure.json`과 최초 계산 소스·수치를 보존하고 **동일 UUID의 저장 수치만으로 보고서를 복구**했다. '
            '새 Edge/benchmark/extraction은 실행하지 않았다. `report_recovery.diff`, '
            '`numerical_results_before_report_recovery.json` 및 JSON의 전후 numerical hash를 제공한다. '
            '최초 failed 이력은 metadata의 `initial_report_failure`에 보존하고 '
            '`report_recovery_completed=true`, `numerical_evaluation_completed=true`, `status=finished`로 완료 범위를 구분했다.']
    REPORT.write_text('\n'.join(lines)+'\n')
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task05 Triangle edges</title>',
        '<style>body{font:15px/1.55 system-ui;margin:32px auto;max-width:1500px;padding:0 20px;color:#243642} '
        'table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #c9d3d8;padding:5px} '
        'img{max-width:100%;height:auto}pre{white-space:pre-wrap;max-height:480px;overflow:auto;font-size:12px} '
        'details{border:1px solid #ccd7dd;padding:10px;margin:12px 0}.muted{color:#5b6970}</style>',
        '<h1>Task05 · Observation-constrained triangular graph edges</h1>',
        f'<p>Run <code>{run}</code> · contract PASS · evaluation only · policy adoption DEFERRED</p>',
        '<p><a href="../../docs/visual-graph-integration/05_triangle_intra_object_edges.md">결과 Markdown</a> · '
        '<a href="../../docs/visual-graph-integration/05_triangle_intra_object_edges_plan.md">Plan</a> · '
        '<a href="triangle_intra_object_edges.json">JSON</a> · <a href="triangle_intra_object_edges.npz">NPZ</a> · '
        '<a href="verification.json">Verification</a> · <a href="tests.log">Tests</a> · '
        '<a href="input_gate.json">Exact baseline gate</a> · <a href="source_hashes.json">Hashes</a> · '
        '<a href="implementation.diff">Implementation diff</a></p>',
        f'<p>Unchanged nodes/features; 8 source frames × 2 ratios from one episode. New isolated nodes: {isolation}; '
        f'baseline Graph components split: {o["split_baseline_components"]}. K8/TMD mean core: '
        f'{speed["K8"]:.3f}/{speed["TMD"]:.3f} ms. Physical mesh correctness NOT_ESTABLISHED.</p>',
        '<p>Normal angle is diagnostic only. Boundary + interior closed-cell covers are independently checked. '
        'Only right/down pairs with both cells in each cover are examined. No bbox-only external pairs. '
        'Missing supported evidence differs from unsupported observations. Triangle rejection removes all its sides; '
        'another accepted incident triangle can keep a shared side. No k-NN fallback.</p>',
        '<pre>'+escape(json.dumps(serial(o), ensure_ascii=False, indent=2))+'</pre>']
    for name in ('edge_lengths', 'degree_components', 'latency', 'memory'):
        page.append(f'<a href="plots/{name}.png"><img src="plots/{name}.png" alt="{name}"></a>')
    page += ['<h2>Known remote pair · candidate / triangle rejection / final Edge</h2>',
             '<pre>'+escape(json.dumps(serial(summary['suspect']), ensure_ascii=False, indent=2))+'</pre>']
    for key in summary['suspect']:
        if (out/'plots'/f'{key}_suspect.png').exists():page.append(f'<img src="plots/{key}_suspect.png" alt="Known remote pair diagnostic">')
    for key, r in summary['records'].items():
        page += [f'<h2>{key} · N={r["N"]} · T={r["triangles"]}</h2>',
            f'<p><a href="records/{key}.json">Full triangle/edge reason and witness JSON</a> · '
            f'<a href="packets/{key}.npz">TMD v2 packet (unchanged Point/Feature/provenance)</a></p>',
            '<table><tr><th>stage</th><th>object</th><th>N</th><th>T</th><th>U</th><th>E</th><th>length mean/p95/max mm</th>'
            '<th>degree median/p95/max</th><th>isolated</th><th>Graph component sizes</th></tr>']
        for stage in STAGES:
            for obj, m in r['stages'][stage]['objects'].items():
                val = lambda a, names:'/'.join('N/A' if a[n] is None else f'{a[n]:.2f}' for n in names)
                page.append(f'<tr><td>{stage}</td><td>{obj}</td><td>{m["N"]}</td><td>{m["T"] if m.get("T") is not None else "N/A"}</td><td>{m["U"]}</td><td>{m["E"]}</td>'
                    f'<td>{val(m["length_mm"], ("mean","p95","max"))}</td>'
                    f'<td>{val(m["degree_distribution"], ("median","p95","max"))}</td><td>{m["isolated_count"]}</td>'
                    f'<td>{m["component_sizes"]}</td></tr>')
        page += ['</table>', '<p>RGB masks / T0 raw / TM support / TMD support+depth:</p>',
            f'<img loading="lazy" src="plots/{key}_triangles.png" alt="Triangle stage overlays">',
            f'<img loading="lazy" src="plots/{key}_edges.png" alt="Identical Point K8 versus TMD 2D and 3D">',
            f'<img loading="lazy" src="plots/{key}_witnesses.png" alt="Support and depth rejection witnesses">']
        if (out/'plots'/f'{key}_normal_connection.png').exists():
            page += [f'<img loading="lazy" src="plots/{key}_normal_connection.png" alt="High normal difference allowed connection">',
                     '<p>Continuous observed support/local depth, normal difference allowed. Physical face labels NOT_ESTABLISHED.</p>']
        for label, payload in (('Triangle rejection and triangulation status', dict(reasons=r['triangle_reasons'], status=r['component_status'])),
            ('New isolated nodes and baseline Graph splits', dict(rows=r['new_isolated_rows'], source=r['new_isolated_source'], splits=r['baseline_components_split'])),
            ('Timing raw samples, memory and raster work', dict(timing=r['timings_raw'], memory=r['memory'], raster=r['raster_work'])),
            ('Per-stage long edges and invalid-path counts', {s:dict(objects=r['stages'][s]['objects'], reasons=r['stages'][s]['edge_audit_reason_counts']) for s in STAGES})):
            page.append('<details><summary>'+label+'</summary><pre>'+escape(json.dumps(serial(payload), ensure_ascii=False, indent=2))+'</pre></details>')
    page += ['<h2>Scope and verification</h2><p>3 warmups + 5 measured repetitions, alternating K8/TMD order. '
        'Core includes common components, packet copy and receiver; lookup/raster/validation/dedup included for TMD. '
        'The current evaluation helper also packages T0/TM/TMD side unions and all rejection reasons; '
        'this is not an isolated production TMD-only minimum latency. '
        'Diagnostics, file IO and rendering are outside core. Fresh worker RSS includes native Qhull/Python overhead, '
        'not separately identifiable. Browser NOT_EXECUTED. No GPU, simulator, new episode, training, upload, dependency install or commit.</p>',
        '<p>Validation violations=0 proves rule compliance only. Coherent far-depth contamination and true physical '
        'mesh topology remain unresolved. No algorithm/threshold changes were made in response to measured results.</p>',
        '<p>Report-only recovery retained the original UUID, numerical NPZ/records/packets/timing hashes and '
        '<a href="failure.json">original rendering failure</a>. '
        '<a href="report_recovery.diff">Report-only source diff</a>; '
        '<a href="numerical_results_before_report_recovery.json">Numerical results before recovery</a>. '
        'No new edges or benchmarks. Original failed process history is preserved as initial_report_failure; numerical evaluation and report recovery completed.</p></html>'
        if summary.get('report_only_recovery') else '</html>']
    (out/'triangle_intra_object_edges.html').write_text('\n'.join(page))


def verify_outputs(out, summary):
    from PIL import Image
    files = list((out/'plots').glob('*.png'))
    for path in files:
        with Image.open(path) as im:im.verify()
        with Image.open(path) as im:im.load()
    class Links(HTMLParser):
        def __init__(self):super().__init__(); self.paths = []
        def handle_starttag(self, tag, attrs):
            for k, v in attrs:
                if k in ('href', 'src') and not v.startswith(('http:', 'https:', '#')):self.paths.append(v)
    parser = Links(); parser.feed((out/'triangle_intra_object_edges.html').read_text())
    missing = [link for link in parser.paths if not (out/link).exists()]
    if missing:raise AssertionError('HTML links missing: '+str(missing))
    relative = f'../../experiments/{out.name}/triangle_intra_object_edges.html'
    if relative not in REPORT.read_text():raise AssertionError('Markdown missing backlink')
    import re
    md_links = re.findall(r'\]\(([^)]+)\)', REPORT.read_text())
    if any(not (REPORT.parent/link).exists() for link in md_links):raise AssertionError('Markdown relative link missing')
    loaded = json.loads((out/'triangle_intra_object_edges.json').read_text())
    if loaded != serial(summary):raise AssertionError('summary JSON readback')
    protected = json.loads((out/'source_hashes.json').read_text())['protected_before']
    recovery_changes = summary.get('report_only_recovery', {}).get('report_source_changes', {})
    changed = [p for p, h in protected.items() if not (REPO/p).is_file() or
               sha(REPO/p) != recovery_changes.get(p, {}).get('after', h)]
    if changed:raise AssertionError('protected files changed: '+str(changed))
    expected = summary['input_hashes']
    if any(sha(REPO/p) != h for p, h in expected.items()):raise AssertionError('input hash changed')
    return dict(status='PASS', baseline_exact_conditions=16, contracts='PASS',
        original_point_feature_provenance='EXACT_PRESERVED', aggregate_npz_readback='PASS',
        json_readback='PASS', png_decode_count=len(files), relative_links_checked=len(parser.paths)+len(md_links),
        protected_file_count=len(protected), protected_files_unchanged_except_report_recovery=True,
        report_source_changes_exempted=list(recovery_changes),
        original_task04_and_selection_files_unchanged=True,
        browser='NOT_EXECUTED', png_visual_review=summary.get('visual_review', 'NOT_EXECUTED'),
        physical_mesh_accuracy='NOT_ESTABLISHED', policy_adopted=False,
        source_extraction_calls=0, sampling_calls=0, new_ee_transform_calls=0,
        simulator='NOT_EXECUTED', gpu='NOT_EXECUTED')


def snapshot(out):
    paths = [p for root in (BASE, SELECTION) for p in root.rglob('*') if p.is_file()]
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    paths += [REPO/line[3:] for line in status.splitlines() if (REPO/line[3:]).is_file()]
    protected = {str(p.relative_to(REPO)):sha(p) for p in paths}
    current = {s:sha(REPO/s) for s in SOURCES}
    for name in SOURCES:
        dest = out/'source_snapshot'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, dest)
    diff = []
    for name in SOURCES[:5]:
        before = Path('/tmp/task05_start')/(Path(name).name+'.before')
        original = before.read_text().splitlines(True) if before.exists() else []
        diff.extend(difflib.unified_diff(original, (REPO/name).read_text().splitlines(True), fromfile='before/'+name, tofile='after/'+name))
    (out/'implementation.diff').write_text(''.join(diff))
    initial = Path('/tmp/task05_start/initial.diff')
    if initial.exists():shutil.copyfile(initial, out/'preexisting_worktree.diff')
    (out/'worktree_status_before.txt').write_text(status)
    return dict(protected_before=protected, evaluated_source_hashes=current)


def finish_recovered_run(out, summary):
    """Existing shared report-recovery pattern, retaining identity and failure."""
    from datetime import datetime, timezone
    from utils.experiment_logger import write as shared_write, scalars
    from utils.experiment_artifacts import save_manifest
    metadata = json.loads((out/'metadata.json').read_text())
    metadata.setdefault('initial_report_failure', dict(status=metadata['status'],
        exception_type=metadata.get('exception_type'), finished_at=metadata.get('finished_at'), evidence='failure.json'))
    metadata.update(status='finished', finished_at=datetime.now(timezone.utc).isoformat(),
        report_recovery_completed=True, numerical_evaluation_completed=True,
        contract='PASS', policy_adopted=False, browser='NOT_EXECUTED')
    metadata.pop('exception_type', None)
    shared_write(out/'metadata.json', metadata)
    metrics = json.loads((out/'metrics.json').read_text())
    if not metrics:
        shared_write(out/'metrics.json', [dict(run_id=out.name, step=0, metrics=scalars(dict(
            contract_pass=1, conditions=16, new_isolated=summary['overview']['new_isolated'],
            k8_core_mean_ms=summary['overview']['cpu_mean_ms']['K8'],
            tmd_core_mean_ms=summary['overview']['cpu_mean_ms']['TMD'])))])
    save_manifest(out, metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-worker', nargs=2, metavar=('CONDITION', 'METHOD'))
    parser.add_argument('--reports-only', type=Path)
    args = parser.parse_args()
    if args.memory_worker:return memory_worker(*args.memory_worker)
    if args.reports_only:
        out = args.reports_only.resolve()
        summary = json.loads((out/'triangle_intra_object_edges.json').read_text())
        immutable = [out/'triangle_intra_object_edges.npz', out/'tests.log', out/'input_gate.json',
                     *sorted((out/'packets').glob('*.npz')), *sorted((out/'records').glob('*.json'))]
        before = {str(p.relative_to(out)):sha(p) for p in immutable}
        if not (out/'numerical_results_before_report_recovery.json').exists():
            shutil.copyfile(out/'triangle_intra_object_edges.json', out/'numerical_results_before_report_recovery.json')
        changes = {}
        evaluated = json.loads((out/'source_hashes.json').read_text())['evaluated_source_hashes']
        # Only this new report runner may change for report recovery. All other
        # evaluated source files, including the edge algorithm, must stay exact.
        for name, old_hash in evaluated.items():
            new_hash = sha(REPO/name)
            if new_hash != old_hash:
                if name != 'visual/runtime/test/run_triangle_intra_object_edges.py':
                    raise AssertionError('non-report source changed during recovery: '+name)
                changes[name] = dict(before=old_hash, after=new_hash)
        summary['report_only_recovery'] = dict(original_failure='failure.json', new_uuid=False,
            new_edges=False, new_benchmark=False, new_sampling=False, new_extraction=False,
            report_source_changes=changes, numerical_hashes_before=before)
        dest = out/'report_source_snapshot/visual/runtime/test/run_triangle_intra_object_edges.py'
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO/SOURCES[2], dest)
        original = (out/'source_snapshot'/SOURCES[2]).read_text().splitlines(True)
        (out/'report_recovery.diff').write_text(''.join(difflib.unified_diff(original,
            (REPO/SOURCES[2]).read_text().splitlines(True), fromfile='evaluated/'+SOURCES[2], tofile='report/'+SOURCES[2])))
        enrich_saved_metrics(out, summary)
        tick = perf_counter(); build_figures(out, summary)
        summary['render_wall_ms'] = (perf_counter()-tick)*1000
        summary['report_only_recovery']['numerical_hashes_after'] = {str(p.relative_to(out)):sha(p) for p in immutable}
        if before != summary['report_only_recovery']['numerical_hashes_after']:
            raise AssertionError('report recovery changed numerical artifacts')
        render_reports(out, summary)
        write(out/'triangle_intra_object_edges.json', summary)
        write(out/'verification.json', dict(status='RUNNING', browser='NOT_EXECUTED'))
        write(out/'verification.json', verify_outputs(out, summary))
        from utils.experiment_artifacts import register_existing_output
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
                register_existing_output(out, path, kind='reports' if path.suffix in ('.png','.html','.diff','.log') else 'metrics')
        register_existing_output(out, REPORT, kind='reports')
        finish_recovered_run(out, summary)
        print('REPORT_RECOVERY_COMPLETE', out.name, flush=True); return
    os.environ['WANDB_MODE'] = 'disabled'
    config = dict(task='05', baseline_run_id=BASE.name, selection_run_id=SELECTION.name,
        frames=FRAMES, ratios=RATIOS, phase=0, triangle_policy=TRIANGLE_POLICY,
        warmup=3, repetitions=5, execution_order='alternating K8/TMD each repeat',
        no_sampling=True, no_feature_extraction=True, no_transform=True,
        interpreter=sys.executable, numpy=np.__version__, scipy=scipy.__version__)
    with ExperimentLogger(team='visual', config=config, seed=0,
                          model='triangle_intra_object_edges_geometry_only', job_type='evaluation') as logger:
        out = logger.directory; print('RUN_DIRECTORY', str(out), flush=True)
        for name in ('packets', 'records', 'plots'):(out/name).mkdir()
        try:
            hashes = snapshot(out); write(out/'source_hashes.json', hashes)
            tests_cmd = [sys.executable, '-m', 'unittest', *TESTS, '-v']
            with (out/'tests.log').open('w') as log:
                log.write('COMMAND '+' '.join(tests_cmd)+'\n'); log.flush()
                result = subprocess.run(tests_cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, MPLCONFIGDIR='/tmp/task05-mpl', WANDB_MODE='disabled'))
            if result.returncode:raise AssertionError('contract tests failed; stop before new source comparisons')
            import re
            count = int(re.search(r'Ran (\d+) tests', (out/'tests.log').read_text()).group(1))
            packets, evidences, inputs = load_inputs(out)
            hashes['input_hashes'] = inputs; write(out/'source_hashes.json', hashes)
            logger.record(source_run_id=BASE.name, selection_run_id=SELECTION.name,
                          baseline_gate='PASS', source_extraction_calls=0)
            tick = perf_counter(); records, common = run_numerics(out, packets, evidences)
            summary = dict(run_id=logger.run_id, frames=FRAMES, ratios=RATIOS, phase=0,
                environment=dict(python=platform.python_version(), interpreter=sys.executable,
                    platform=platform.platform(), cpu=platform.processor(), numpy=np.__version__,
                    scipy=scipy.__version__, thread_env={n:os.environ.get(n) for n in
                        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
                policy=TRIANGLE_POLICY, tests=dict(status='PASS', count=count), input_hashes=inputs,
                records=records, common=common, numerical_pipeline_wall_ms=(perf_counter()-tick)*1000,
                browser='NOT_EXECUTED', physical_mesh_accuracy='NOT_ESTABLISHED', policy_adopted=False)
            summary['suspect'] = track_suspect(out, records); summary['overview'] = overview(summary)
            enrich_saved_metrics(out, summary)
            write(out/'triangle_intra_object_edges.json', summary)
            tick = perf_counter(); build_figures(out, summary)
            summary['render_wall_ms'] = (perf_counter()-tick)*1000
            render_reports(out, summary); write(out/'triangle_intra_object_edges.json', summary)
            # Verification file exists before link validation; replace with full checks below.
            write(out/'verification.json', dict(status='RUNNING', browser='NOT_EXECUTED'))
            verification = verify_outputs(out, summary); write(out/'verification.json', verification)
            for path in sorted(out.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
                    logger.link_output(path, kind='reports' if path.suffix in ('.html','.png','.md','.diff','.log') else 'metrics')
            logger.link_output(REPORT, kind='reports')
            logger.log(dict(contract_pass=1, conditions=16, new_isolated=summary['overview']['new_isolated'],
                            k8_core_mean_ms=summary['overview']['cpu_mean_ms']['K8'],
                            tmd_core_mean_ms=summary['overview']['cpu_mean_ms']['TMD']))
            logger.record(contract='PASS', policy_adopted=False, browser='NOT_EXECUTED')
            print('COMPLETE', logger.run_id, json.dumps(summary['overview']), flush=True)
        except Exception as exc:
            write(out/'failure.json', dict(type=type(exc).__name__, message=str(exc)))
            logger.link_output(out/'failure.json', kind='metrics')
            raise


if __name__ == '__main__':main()
