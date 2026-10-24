"""Task03-AB: gated CPU offline total-budget ablation; no production changes.

WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03ab-mpl .venv/bin/python -m \
    visual.runtime.test.run_adaptive_node_budget --preservation-before PATH
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
from numbers import Integral
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.edge_evidence import PairEvidence
from visual.runtime.geometry import GeometricFeatures, GeometryFrame, OrganizedGeometry
from visual.runtime.rgbd import PEG, BOX, iter_saved_rgbd
from visual.runtime.sampling import (
    SamplingConfig, SamplingEvidence, allocate_budget, extract_candidates,
    extract_sampling_evidence, sample_points,
)
from visual.runtime.test.normal_diagnostics import stats
from visual.runtime.test.run_rim_bottleneck import source_contract, occupancy
from visual.runtime.test.run_sampling import validate_result
from visual.runtime.test.sampling_evaluation import (
    CoverageMatcher, TOLERANCES, evaluation_context, score_selection,
    staged_reference_coverage, summarize,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f'
KEY = 'S3_n256_r0.25_t20_s0'
CONDITIONS = ('Fixed256', 'Adaptive', 'FixedMatched')
OBJECTS = (('peg', PEG), ('box', BOX))
REPRESENTATIVE = (0, 15, 18, 45, 80, 120)
FIELDS = ('points', 'pixels_vu', 'flat_index', 'object_id',
          'priority_selected', 'depth_priority_selected')
TIMERS = ('candidate_selection', 'budget_allocation', 'structural_sampling', 'surface_fps')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def distribution(values):
    a = np.asarray([x for x in values if x is not None], float)
    a = a[np.isfinite(a)]
    return dict(stats(a), min=float(a.min()) if len(a) else None)


def ratio(n, d):
    return n / d if d else None


def support_counts(counts):
    if len(counts) != 2 or any(not isinstance(x, Integral) or isinstance(x, (bool, np.bool_))
                               or x < 0 for x in counts):
        raise ValueError('two nonnegative integer support counts required')
    return tuple(int(x) for x in counts)


def requested_budget(counts, reference):
    """Only current support counts and an immutable past/current reference."""
    counts = support_counts(counts)
    area = sum(counts)
    minimums = tuple(min(16, x) for x in counts)
    if reference is None:
        if area:
            raise ValueError('positive support requires initialized reference')
        return 0, minimums
    if not isinstance(reference, Integral) or reference <= 0:
        raise ValueError('reference must be a positive integer')
    rounded = (512 * area + int(reference)) // (2 * int(reference))
    return min(256, max(sum(minimums), rounded)), minimums


@dataclass
class AreaBudget:
    reference: int | None = None
    reference_source: int | None = None

    def step(self, counts, source):
        counts = support_counts(counts)
        if self.reference is None and sum(counts):
            self.reference, self.reference_source = sum(counts), int(source)
        return requested_budget(counts, self.reference)


def matched_budget(areas, requests):
    """One offline matched-count control, selected without geometry scores."""
    if len(areas) != len(requests) or not len(areas):
        raise ValueError('aligned nonempty counts required')
    target = sum(min(int(a), int(n)) for a, n in zip(areas, requests))
    b = min(range(32, 257), key=lambda n: (abs(sum(min(n, int(a)) for a in areas) - target), n))
    actual = sum(min(b, int(a)) for a in areas)
    return dict(budget=b, adaptive_selected_sum=target, fixed_selected_sum=actual,
                mean_error_nodes=(actual-target)/len(areas), causal_online=False,
                selection='count-only absolute total difference; smaller integer on ties')


def measure_areas(frame, geometry):
    """Validity count cost and diagnostic raw-mask scans are separate."""
    t0 = perf_counter()
    counts = tuple(int(np.count_nonzero(geometry.observed_mask(oid))) for _, oid in OBJECTS)
    count_ms = (perf_counter()-t0)*1000
    t0 = perf_counter()
    result = {}
    for (name, oid), valid in zip(OBJECTS, counts):
        mask = frame.masks[oid]
        raw = int(np.count_nonzero(mask))
        invalid = int(np.count_nonzero(mask & ~geometry.valid_depth))
        overlap = int(np.count_nonzero(mask & geometry.valid_depth & geometry.overlap_mask))
        if raw != valid + invalid + overlap:
            raise AssertionError('raw/valid exclusion partition')
        result[name] = dict(raw=raw, valid=valid, invalid_depth=invalid,
                            positive_depth_overlap=overlap, minimum=min(16, valid))
    raw_ms = (perf_counter()-t0)*1000
    return result, count_ms, raw_ms


def exact_result(first, second):
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field), err_msg=field)
    if set(first.metadata) != set(second.metadata):
        raise AssertionError('feature field mismatch')
    for field in first.metadata:
        np.testing.assert_array_equal(first.metadata[field], second.metadata[field], err_msg=field)
    if first.counts != second.counts or first.object_budgets != second.object_budgets:
        raise AssertionError('selection counts mismatch')


def exact_saved(result, archive, key=KEY):
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(result, field), archive[f'{key}/{field}'], err_msg=field)
    saved = {n.split('/metadata/')[1] for n in archive.files if n.startswith(key+'/metadata/')}
    if saved != set(result.metadata):
        raise AssertionError('available feature fields mismatch')
    for field in result.metadata:
        np.testing.assert_array_equal(result.metadata[field], archive[f'{key}/metadata/{field}'], err_msg=field)


def validate_selection(result, evidence, config, candidates):
    validate_result(result, evidence, config)
    counts = [len(candidates[i].points) for _, i in OBJECTS]
    expected = allocate_budget(config.num_points, counts, config.min_points, counts)
    np.testing.assert_array_equal([result.object_budgets[i] for _, i in OBJECTS], expected)
    if len(result.points) != min(config.num_points, sum(counts)):
        raise AssertionError('selected total differs from capacity-limited request')
    if len(np.unique(result.points, axis=0)) != len(result.points):
        raise AssertionError('duplicate XYZ')
    for _, oid in OBJECTS:
        keep = result.object_id == oid
        rows = np.searchsorted(candidates[oid].flat_index, result.flat_index[keep])
        for field, values in result.metadata.items():
            np.testing.assert_array_equal(values[keep], candidates[oid].metadata[field][rows], err_msg=field)
        c = result.counts[oid]
        if c['selected'] != c['priority'] + c['surface_fps'] or c['depth_priority']:
            raise AssertionError('S3 role accounting')


def save_evidence(path, evidence):
    """Lossless temporary observation cache: global gate before any ablation."""
    f = evidence.frame
    g = f.geometry
    a = dict(xyz=g.xyz, valid_depth=g.valid_depth, overlap=g.overlap_mask)
    for _, oid in OBJECTS:
        a[f'mask/{oid}'] = g.object_masks[oid]
        for field in GeometricFeatures.__dataclass_fields__:
            a[f'feature/{oid}/{field}'] = getattr(f.objects[oid], field)
        for field in PairEvidence.__dataclass_fields__:
            a[f'pair/{oid}/{field}'] = np.asarray(getattr(evidence.pairs[oid], field))
    a['provenance'] = np.array(json.dumps(dict(source_frame=f.source_frame, camera=f.camera,
        frame_idx=f.frame_idx, timestamp=f.timestamp, timings_ms=evidence.timings_ms)))
    np.savez_compressed(path, **a)


def load_evidence(path):
    with np.load(path, allow_pickle=False) as a:
        p = json.loads(str(a['provenance']))
        g = OrganizedGeometry(a['xyz'], a['valid_depth'], {i:a[f'mask/{i}'] for _, i in OBJECTS}, a['overlap'])
        features = {i:GeometricFeatures(**{k:a[f'feature/{i}/{k}'] for k in GeometricFeatures.__dataclass_fields__})
                    for _, i in OBJECTS}
        pairs = {i:PairEvidence(**{k:tuple(a[f'pair/{i}/{k}']) if k == 'shape' else a[f'pair/{i}/{k}']
                                   for k in PairEvidence.__dataclass_fields__}) for _, i in OBJECTS}
        f = GeometryFrame(p['source_frame'], p['camera'], p['frame_idx'], p['timestamp'], g, features, p['timings_ms'])
        return SamplingEvidence(f, pairs, 'pca5', p['timings_ms'])


def nearest(source, target):
    if not len(source):
        return np.empty(0)
    return cKDTree(target).query(source)[0]*1000 if len(target) else np.full(len(source), np.nan)


def distance_summary(distances):
    finite = np.isfinite(distances)
    return dict(distance_mm=distribution(distances), source_count=len(distances),
                missing_count=int((~finite).sum()), within_3mm=int((distances[finite] <= 3).sum()),
                coverage_3mm=ratio(int((distances[finite] <= 3).sum()), len(distances)))


def general_geometry(frame, evidence, candidates, sample, context):
    report, arrays = {}, {}
    g = evidence.frame.geometry
    for name, oid in OBJECTS:
        obj = candidates[oid]
        x = obj.flat_index
        c = x[obj.metadata['structural_candidate']]
        chosen = sample.object_id == oid
        f = sample.flat_index[chosen]
        p = sample.flat_index[chosen & sample.priority_selected]
        s = sample.flat_index[chosen & ~sample.priority_selected]
        xyz = g.xyz.reshape(-1, 3)
        xf, cp = nearest(xyz[x], xyz[f]), nearest(xyz[c], xyz[p])
        components, count = label(g.observed_mask(oid))
        component_rows = []
        component_index = components.ravel()
        for k in range(1, count+1):
            keep = component_index[x] == k
            nc, np_, nf = (int((component_index[v] == k).sum()) for v in (c, p, f))
            component_rows.append(dict(id=k, support=int(keep.sum()), candidate=nc, priority=np_, final=nf,
                small=bool(keep.sum() <= 64), candidate_without_priority=bool(nc and not np_),
                candidate_without_final=bool(nc and not nf), **distance_summary(xf[keep])))
        paths = dict(X=x, C=c, P=p, S=s, F=f)
        stages = staged_reference_coverage(context['matchers'][name], c, p, s, f)
        depth = occupancy(frame, paths, context['negatives'][name])
        report[name] = dict(x_to_f=distance_summary(xf), c_to_p=distance_summary(cp), components=component_rows,
            stages=stages, occupancy=depth,
            camera_range_m=distribution(np.linalg.norm(xyz[x].astype(float), axis=1)))
        arrays[name+'/c_to_p'] = cp
        arrays[name+'/x_to_f'] = xf
    return report, arrays


def sampling_timing(sample):
    t = {k:float(sample.timings_ms[k]) for k in (*TIMERS, 'sampling_total')}
    t['gather_packaging_and_other_overhead'] = t['sampling_total']-sum(t[k] for k in TIMERS)
    if t['gather_packaging_and_other_overhead'] < -1e-9:
        raise AssertionError('negative timer residual')
    return t


def latency_summary(records):
    result = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r['condition'] == condition]
        per_frame, effective, executed = {}, {}, {}
        for r in rows:
            owner = next(x for x in records if x['source_frame'] == r['source_frame']
                         and x['repeat'] == r['repeat'] and x['condition'] == (r['alias_of'] or condition))
            for key, value in owner['timings_ms'].items():
                effective.setdefault(key, []).append(value)
                per_frame.setdefault(key, {}).setdefault(r['source_frame'], []).append(value)
                if r['executed']:
                    executed.setdefault(key, []).append(value)
        result[condition] = dict(executed_calls=sum(r['executed'] for r in rows),
            alias_calls=sum(not r['executed'] for r in rows),
            frame_median_then_distribution={k:distribution([np.median(v) for v in frames.values()])
                                            for k, frames in per_frame.items()},
            all_repetitions_effective_including_explicit_alias={k:distribution(v) for k,v in effective.items()},
            executed_only={k:distribution(v) for k,v in executed.items()})
    return result


def preserve_check(before):
    changed = [name for name, expected in before['sha256'].items()
               if not (REPO/name).is_file() or digests(REPO/name)[1] != expected]
    return dict(checked_files=len(before['sha256']), changed_or_missing=changed, unchanged=not changed)


def add_temporal_and_pairs(rows):
    for index, row in enumerate(rows):
        if row['source_frame'] != index:
            raise AssertionError('source frame continuity')
        for condition, data in row['conditions'].items():
            previous = rows[index-1]['conditions'][condition] if index else None
            for field in ('requested', 'selected'):
                delta = data[field]-previous[field] if previous else None
                data['delta_'+field] = delta
                data['abs_delta_'+field] = abs(delta) if delta is not None else None
            fixed = row['conditions']['Fixed256']['selected']
            data['node_reduction_vs_fixed256'] = 1-data['selected']/fixed if fixed else None
            for name, _ in OBJECTS:
                base = row['conditions']['Fixed256']['geometry']['objects'][name]
                current = data['geometry']['objects'][name]
                f, a = (v['structural']['tolerance']['2px_3mm']['sampling_retention'] for v in (base, current))
                data['objects'][name]['retention_difference_pp_vs_fixed256'] = (a-f)*100 if a is not None and f is not None else None
                for field in ('mean', 'p95'):
                    f, a = (v['surface']['all']['distance_mm'][field] for v in (base, current))
                    data['objects'][name]['surface_'+field+'_difference_mm_vs_fixed256'] = a-f if a is not None and f is not None else None


def make_summary(rows, pools, distance_pools, matched, policy):
    summary = {}
    fixed_sum = sum(r['conditions']['Fixed256']['selected'] for r in rows)
    for condition in CONDITIONS:
        geometry = summarize(pools[condition], distance_pools[condition])
        geometry.pop('timings_ms', None)  # timings are paired separately, with explicit aliases
        counts = {field:distribution([r['conditions'][condition][field] for r in rows])
                  for field in ('requested', 'selected', 'priority', 'surface', 'shortfall',
                                'delta_requested', 'delta_selected', 'abs_delta_selected')}
        selected = sum(r['conditions'][condition]['selected'] for r in rows)
        objects = {}
        for name, _ in OBJECTS:
            objects[name] = dict(
                allocated=distribution([r['conditions'][condition]['objects'][name]['allocated'] for r in rows]),
                selected=distribution([r['conditions'][condition]['objects'][name]['selected'] for r in rows]),
                density_per_valid_pixel=distribution([r['conditions'][condition]['objects'][name]['density_per_valid_pixel'] for r in rows]),
                retention_difference_pp=distribution([r['conditions'][condition]['objects'][name]['retention_difference_pp_vs_fixed256'] for r in rows]),
                c_to_p_mm=distribution(np.concatenate(distance_pools[condition][name+'/c_to_p'])))
        summary[condition] = dict(counts=counts, selected_sum=selected, node_reduction_fraction=1-selected/fixed_sum,
            geometry=geometry, objects=objects,
            requested_max_frames=[r['source_frame'] for r in rows if r['conditions'][condition]['requested'] == 256],
            requested_min_frames=[r['source_frame'] for r in rows if r['conditions'][condition]['requested'] == r['minimum_total']])
    area = [sum(r['areas'][name]['valid'] for name, _ in OBJECTS) for r in rows]
    median_area = float(np.median(area))
    strata = {}
    for title, ids in [('area_le_median',[i for i,a in enumerate(area) if a <= median_area]),
                       ('area_gt_median',[i for i,a in enumerate(area) if a > median_area]),
                       ('box_support_zero',[i for i,r in enumerate(rows) if not r['areas']['box']['valid']])]:
        strata[title] = dict(source_frames=ids, objects={name:distribution([
            rows[i]['conditions']['Adaptive']['objects'][name]['retention_difference_pp_vs_fixed256'] for i in ids])
            for name, _ in OBJECTS})
    return dict(conditions=summary, matched=matched,
        policy=dict(alpha=1, reference=policy.reference, reference_source=policy.reference_source,
                    rounding='integer half-up', maximum=256, minimum='sum min(16, current object support)'),
        area_strata=dict(median_valid_area=median_area, strata=strata))


def plot_results(output, rows, latency):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(Fixed256='#64748b', Adaptive='#e76f51', FixedMatched='#247ba0')
    x = np.arange(len(rows))
    plotdir = output/'plots'
    plotdir.mkdir(exist_ok=True)
    def finish(fig, axes, filename):
        for ax in np.asarray(axes).ravel():
            for lo, hi in ((18,26),(56,65)):
                if ax.get_xlabel() == 'Source frame':
                    ax.axvspan(lo-.5,hi+.5,color='#cbd5e1',alpha=.4)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc='best')
        fig.savefig(plotdir/filename, dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(4,1,figsize=(15,12),layout='constrained')
    for name, _ in OBJECTS:
        for field, ls in [('raw','--'),('valid','-')]:
            axes[0].plot(x,[r['areas'][name][field] for r in rows],ls,label=name+' '+field)
    for condition in CONDITIONS:
        axes[1].plot(x,[r['conditions'][condition]['selected'] for r in rows],label=condition,color=colors[condition])
        for name, ls in [('peg','-'),('box','--')]:
            axes[2].plot(x,[r['conditions'][condition]['objects'][name]['allocated'] for r in rows],ls,
                         label=condition+' '+name,color=colors[condition])
        for field, ls in [('priority','-'),('surface','--')]:
            axes[3].plot(x,[r['conditions'][condition][field] for r in rows],ls,label=condition+' '+field,color=colors[condition])
    for ax,title in zip(axes,['Raw / valid projected pixel areas','Actual point nodes','Object allocated budgets','Priority / Surface roles']):
        ax.set(title=title,xlabel='Source frame')
    finish(fig,axes,'budget_timeseries.png')
    fig, axes = plt.subplots(2,3,figsize=(17,9),layout='constrained')
    area = [sum(r['areas'][n]['valid'] for n,_ in OBJECTS) for r in rows]
    reduction = [r['conditions']['Adaptive']['node_reduction_vs_fixed256']*100 for r in rows]
    for col,(name,_) in enumerate(OBJECTS):
        for condition in CONDITIONS:
            retention = [r['conditions'][condition]['geometry']['objects'][name]['structural']['tolerance']['2px_3mm']['sampling_retention'] for r in rows]
            axes[0,col].scatter(reduction,retention,s=10,label=condition,color=colors[condition],alpha=.7)
            axes[1,col].plot(x,[r['conditions'][condition]['geometry']['objects'][name]['surface']['all']['distance_mm']['p95'] for r in rows],
                             label=condition,color=colors[condition])
        axes[0,col].set(title=name+' retention vs Adaptive node reduction',xlabel='Adaptive node reduction (%)',ylabel='Observable reference retention')
        axes[1,col].set(title=name+' X to F p95 distance',xlabel='Source frame',ylabel='mm')
    for condition in CONDITIONS:
        axes[0,2].scatter(area,[r['conditions'][condition]['selected'] for r in rows],s=12,label=condition,color=colors[condition])
    axes[0,2].set(title='Area / node relationship (cap 256)',xlabel='Valid projected pixels',ylabel='Actual nodes',ylim=(0,270))
    axes[1,2].plot(x,[r['conditions']['Adaptive']['delta_selected'] for r in rows],label='Adaptive signed delta')
    axes[1,2].plot(x,[r['conditions']['Adaptive']['abs_delta_selected'] for r in rows],label='Adaptive abs delta')
    axes[1,2].set(title='Temporal node changes',xlabel='Source frame',ylabel='Nodes / frame')
    finish(fig,axes,'preservation_temporal.png')
    fig, axes = plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for condition in CONDITIONS:
        ys = [np.median([next(t for t in latency if t['source_frame']==r['source_frame'] and t['repeat']==rep and
                        t['condition']==(next(z for z in latency if z['source_frame']==r['source_frame'] and z['repeat']==rep and z['condition']==condition)['alias_of'] or condition))['timings_ms']['sampling_total']
                        for rep in range(3)]) for r in rows]
        axes[0,0].plot(x,ys,label=condition,color=colors[condition])
        axes[0,1].scatter([r['conditions'][condition]['selected'] for r in rows],ys,s=12,label=condition,color=colors[condition])
    axes[0,0].set(title='Paired sampling (frame median; aliases explicit in JSON)',xlabel='Source frame',ylabel='ms')
    axes[0,1].set(title='Selected nodes / actual measured selection latency',xlabel='Actual nodes',ylabel='ms')
    for name,_ in OBJECTS:
        axes[1,0].plot(x,[r['conditions']['Adaptive']['general'][name]['occupancy']['depth_m']['X']['median'] for r in rows],label=name+' median z')
        axes[1,1].scatter([r['areas'][name]['valid'] for r in rows],
            [r['conditions']['Adaptive']['objects'][name]['retention_difference_pp_vs_fixed256'] for r in rows],s=15,label=name)
    axes[1,0].set(title='Observed depth; contamination retained',xlabel='Source frame',ylabel='Camera z (m)')
    axes[1,1].set(title='Small masks / sampling retention loss',xlabel='Valid projected pixels',ylabel='Adaptive - Fixed256 (pp)')
    finish(fig,axes,'latency_depth.png')


def overlay(output, frame, candidates, samples, context):
    import matplotlib.pyplot as plt
    h,w = frame.depth.shape
    fig,axes = plt.subplots(3,3,figsize=(15,13),layout='constrained')
    for row, oid in enumerate((None, PEG, BOX)):
        union = frame.masks[PEG] | frame.masks[BOX] if oid is None else frame.masks[oid]
        v,u = np.nonzero(union)
        limits = (max(0,int(u.min())-10),min(w-1,int(u.max())+10),min(h-1,int(v.max())+10),max(0,int(v.min())-10)) if len(u) else (0,w-1,h-1,0)
        for col,condition in enumerate(CONDITIONS):
            ax=axes[row,col]; ax.imshow(frame.rgb)
            for name,i in OBJECTS:
                if oid is not None and oid != i:
                    continue
                mask=frame.masks[i]
                if mask.any() and not mask.all():
                    ax.contour(mask,[.5],colors=['#69ff72'],linewidths=.5)
                flat=candidates[i].flat_index[candidates[i].metadata['structural_candidate']]
                cv,cu=np.unravel_index(flat,(h,w));ax.scatter(cu,cv,s=2,c='#ff5d41',alpha=.45)
                uv=context['references'][name][0];ax.scatter(uv[:,0],uv[:,1],s=12,c='#ff65cc',marker='+',linewidths=.4)
            s=samples[condition]
            keep=np.ones(len(s.points),bool) if oid is None else s.object_id==oid
            for priority,color,marker,size in [(False,'#00d9ff','o',9),(True,'#ffff00','*',24)]:
                pv,pu=s.pixels_vu[keep & (s.priority_selected==priority)].T
                ax.scatter(pu,pv,s=size,c=color,marker=marker,edgecolors='black',linewidths=.2)
            ax.set(xlim=limits[:2],ylim=limits[2:],title=f'{condition} | {"All" if oid is None else ("Peg" if oid==PEG else "Box")} N={int(keep.sum())}')
            ax.axis('off')
    fig.suptitle(f'Source {frame.source_frame} | same crop per row | yellow=Priority, cyan=Surface, red=candidate, pink=GT eval\nMasks may be contaminated; absent object remains empty. No padding.')
    fig.savefig(output/'frames'/f'f{frame.source_frame:04d}_overlay.png',dpi=140);plt.close(fig)


def html_report(output, data, rows):
    def value(v):
        return 'N/A' if v is None else f'{v:.3f}'
    summary=[]
    for condition,s in data['conditions'].items():
        g=s['geometry']['objects']
        summary.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [condition,
            value(s['counts']['selected']['mean']), value(s['counts']['selected']['p50']),
            value(100*s['node_reduction_fraction']),
            value(100*g['peg']['structural']['2px_3mm']['sampling_retention']),
            value(100*g['box']['structural']['2px_3mm']['sampling_retention']),
            value(g['peg']['surface']['all']['distance_mm']['p95']),
            value(g['box']['surface']['all']['distance_mm']['p95'])])+'</tr>')
    table=[]
    for r in rows:
        source=r['source_frame']
        links=f'<a href="frames/f{source:04d}_samples.npz">NPZ</a>'
        if source in REPRESENTATIVE:
            links+=f' <a href="frames/f{source:04d}_overlay.png">overlay</a>'
        table.append('<tr>'+''.join(f'<td>{v}</td>' for v in [source,
            r['areas']['peg']['raw'],r['areas']['peg']['valid'],r['areas']['box']['raw'],r['areas']['box']['valid'],
            *[r['conditions'][c]['selected'] for c in CONDITIONS],links])+'</tr>')
    plots=''.join(f'<a href="plots/{p.name}"><img src="plots/{p.name}" alt="{p.stem}"></a>' for p in sorted((output/'plots').glob('*.png')))
    images=''.join(f'<a href="frames/f{s:04d}_overlay.png"><img src="frames/f{s:04d}_overlay.png" alt="Source {s}"></a>' for s in REPRESENTATIVE)
    dependencies=''.join(f'<li><a href="{html.escape(v)}">{html.escape(k)}</a></li>' for k,v in data['dependency_links'].items())
    (output/'adaptive_node_budget.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Task03-AB adaptive total node budget</title><style>body{font:15px system-ui;max-width:1500px;margin:28px auto;padding:0 16px;color:#17212b}img{width:100%;margin:15px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:7px;border:1px solid #ccd3da;text-align:right}th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap}a{color:#146cb2}</style>
<h1>Task03-AB — Adaptive Total Node Budget</h1>'''+f'<p>Run {data["run_id"]}. CPU only; ep000 source 0–120. All 121-frame gates passed before adaptive geometry evaluation.</p>'
        +'<p>First positive valid support fixes the causal reference. Pixel area is projected area, not physical 3D surface area. FixedMatched uses episode counts offline. Same S3 / 20° / .25 / seed0 / 3D FPS; Priority cap changes with requested budget.</p>'
        +'<p>Joint matcher: the same pair within 2px AND 3mm. Retention denominator is tracker-observable references. N/A is preserved. Gray bands mark recorded Box support loss, not independently proven occlusion. Sampling latency excludes PCA5 extraction, GT and I/O; aliases have no invented FPS calls. No GNN/BC, stale-cache integration, GPU or end-to-end 20Hz claim.</p>'
        +'<p><a href="adaptive_node_budget.json">Summary/config</a> · <a href="frame_metrics.json">All frame metrics</a> · <a href="latency.json">Raw/aggregated timing</a> · <a href="verification.json">Gates/preservation</a> · <a href="tests.log">Tests</a> · <a href="implementation.patch">Implementation diff</a></p>'
        +'<table><tr><th>Condition</th><th>Mean N</th><th>Median N</th><th>Reduction %</th><th>Peg retention %</th><th>Box retention %</th><th>Peg surface p95 mm</th><th>Box surface p95 mm</th></tr>'+''.join(summary)+'</table>'
        +plots+'<h2>Representative frames</h2>'+images+'<h2>All frame artifacts</h2><table><tr><th>Source</th><th>Peg raw</th><th>Peg valid</th><th>Box raw</th><th>Box valid</th><th>Fixed256</th><th>Adaptive</th><th>FixedMatched</th><th>Files</th></tr>'+''.join(table)+'</table>'
        +'<h2>Original read-only dependencies</h2><ul>'+dependencies+'</ul><details><summary>Policy / reference / matched control</summary><pre>'
        +html.escape(json.dumps(dict(policy=data['policy'],reference=data['reference_camera'],matched=data['matched']),indent=2))+'</pre></details></html>',encoding='utf-8')


def evaluate(preservation_before, tests_log=None):
    source=SOURCE
    original=json.loads((source/'comparison.json').read_text())
    cfg=original['config']
    episode=REPO/'visual/data/step2/ep000.npz'
    masks=REPO/'visual/results/runtime_tracking/ep000/masks.npz'
    camera='base_camera'
    saved=[json.loads((source/f'f{i:04d}_metrics.json').read_text())[KEY] for i in range(121)]
    counts=[tuple(r['objects'][n]['counts']['observed'] for n,_ in OBJECTS) for r in saved]
    policy=AreaBudget()
    planned=[policy.step(c,i)[0] for i,c in enumerate(counts)]
    matched=matched_budget([sum(c) for c in counts],planned)
    config=dict(task='03-AB', source_run_id=original['run_id'], input=str(episode), masks=str(masks),
        camera=camera, alpha=1, maximum=256, minimum_per_object=16, matched=matched,
        reference=policy.reference, reference_source=policy.reference_source,
        baseline=KEY, tolerances=TOLERANCES, seed=0, device='cpu', independent_episode=False,
        timing_repetitions=3, extraction='once per frame; lossless cache to enforce all-frame gates first',
        policy_inputs='current observed support counts and fixed first-positive reference only',
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03ab_adaptive_node_budget',job_type='evaluate') as logger:
        output=logger.directory
        print('Output:',output,flush=True)
        for p in ('cache','frames','evaluated_source'):
            (output/p).mkdir()
        before=json.loads(Path(preservation_before).read_text())
        write_json(output/'preservation_before.json',before)
        verification=dict(gates_passed=False,adaptive_geometry_executed=False,frame_gates=[],
                          primary_tolerance='same pair 2px AND 3mm')
        try:
            for path, expected in ((episode,cfg['input_sha256']),(masks,cfg['masks_sha256'])):
                if digests(path)[1] != expected:
                    raise AssertionError('input hash mismatch: '+str(path))
            verification['input_hashes']={str(p.relative_to(REPO)):digests(p)[1] for p in (episode,masks)}
            verification['source_contract']=source_contract(source,cfg)
            with np.load(masks,allow_pickle=False) as ar:
                np.testing.assert_array_equal(ar['source_frame'],np.arange(121))
            tracking=json.loads(masks.with_name('tracking.json').read_text())
            if tracking['camera'] != camera:
                raise AssertionError('tracker camera mismatch')
            if digests(Path(tracking['input']))[1] != cfg['input_sha256']:
                raise AssertionError('tracker input provenance mismatch')
            if TOLERANCES != ((1.,.002),(2.,.003),(3.,.005)):
                raise AssertionError('joint tolerances changed')
            for relative in (*cfg['source_sha256'], 'visual/runtime/test/run_rim_bottleneck.py',
                             'visual/runtime/test/run_adaptive_node_budget.py',
                             'visual/runtime/test/test_adaptive_node_budget.py'):
                dest=output/'evaluated_source'/relative
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(REPO/relative,dest)
            patch=''
            for relative in ('visual/runtime/test/run_adaptive_node_budget.py','visual/runtime/test/test_adaptive_node_budget.py'):
                proc=subprocess.run(['git','diff','--no-index','--','/dev/null',relative],cwd=REPO,capture_output=True,text=True)
                patch+=proc.stdout
            (output/'implementation.patch').write_text(patch)
            if tests_log:
                shutil.copyfile(tests_log,output/'tests.log')
            required=[f'{camera}/seg','peg_id','box_id','peg_pose','box_pose','box_hole_pose','peg_half_sizes','box_hole_radii','seed']
            with np.load(episode,allow_pickle=False) as ar:
                if len(ar[f'{camera}/depth']) != 121:
                    raise AssertionError('episode frame count')
                gt={k:ar[k] for k in required}  # never supplied to the budget helper or sampler
            robot_ids=cfg['robot_ids']
            pools={c:[] for c in CONDITIONS}
            distance_pools={c:{} for c in CONDITIONS}
            gate_rows=[]
            reference_camera=None
            for index,frame in enumerate(iter_saved_rgbd(episode,masks,camera=camera)):
                if frame.source_frame != index:
                    raise AssertionError('source mapping')
                evidence=extract_sampling_evidence(frame,'pca5')
                candidates=extract_candidates(evidence,SamplingConfig(),'S3')
                areas,count_ms,raw_ms=measure_areas(frame,evidence.frame.geometry)
                actual=tuple(areas[n]['valid'] for n,_ in OBJECTS)
                if actual != counts[index]:
                    raise AssertionError('observed count mismatch')
                fixed=sample_points(evidence,SamplingConfig(),method='S3')
                validate_selection(fixed,evidence,SamplingConfig(),candidates)
                with np.load(source/f'f{index:04d}_samples.npz',allow_pickle=False) as archive:
                    if int(archive['source_frame']) != index:
                        raise AssertionError('archive source mapping')
                    np.testing.assert_array_equal(archive['shape'],frame.depth.shape)
                    for _,oid in OBJECTS:
                        c=candidates[oid]
                        np.testing.assert_array_equal(c.flat_index[c.metadata['structural_candidate']],
                            archive[f'evidence/pca5/{oid}/20/orientation_flat'])
                    exact_saved(fixed,archive)
                    context=evaluation_context(frame,evidence,gt,camera,robot_ids)
                    for name,oid in OBJECTS:
                        reference=tuple(archive[f'evaluation/{name}/reference_{s}'] for s in ('uv','xyz'))
                        for a,b in zip(reference,context['references'][name]):
                            np.testing.assert_array_equal(a,b)
                        context['references'][name]=reference
                        context['matchers'][name]=CoverageMatcher(evidence.frame.geometry,oid,reference)
                    score,dist=score_selection(fixed,context)
                    np.testing.assert_equal(score['objects'],saved[index]['objects'])
                score['source_frame']=index
                pools['Fixed256'].append(score)
                for k,v in dist.items():
                    distance_pools['Fixed256'].setdefault(k,[]).append(v.astype(np.float32))
                save_evidence(output/'cache'/f'f{index:04d}_evidence.npz',evidence)
                gate_rows.append(dict(source_frame=index,areas=areas,support_count_ms=count_ms,
                    raw_area_diagnostics_ms=raw_ms,extraction_ms=evidence.timings_ms))
                if index == policy.reference_source:
                    reference_camera=dict(source_frame=index,frame_idx=frame.frame_idx,timestamp=frame.timestamp,
                        shape=list(frame.depth.shape),camera=camera,intrinsic=frame.intrinsic.tolist(),
                        cam2world=frame.cam2world.tolist(),areas=areas,unit='projected pixel count')
                verification['frame_gates'].append(dict(source_frame=index,support_exact=True,candidate_exact=True,
                    fixed256_all_fields_exact=True,reference_exact=True,fixed256_original_metrics_exact=True))
                if index%15 == 0:
                    print(f'Gate {index+1}/121: support/candidates/selection/features/reference/metrics exact',flush=True)
            if len(verification['frame_gates']) != 121:
                raise AssertionError('all-frame gate incomplete')
            verification['gates_passed']=True
            write_json(output/'verification.json',verification)
            print('All 121 gates passed. Starting paired ablation.',flush=True)
        except Exception as exc:
            verification['failure']=dict(type=type(exc).__name__,message=str(exc))
            write_json(output/'verification.json',verification)
            (output/'adaptive_node_budget.html').write_text('<!doctype html><meta charset="utf-8"><h1>Gate failed; adaptive geometry NOT RUN</h1><a href="verification.json">Verification</a>')
            raise

        rows=[];timings=[];actual_policy=AreaBudget();warmups=[]
        for index,frame in enumerate(iter_saved_rgbd(episode,masks,camera=camera)):
            evidence=load_evidence(output/'cache'/f'f{index:04d}_evidence.npz')
            candidates=extract_candidates(evidence,SamplingConfig(),'S3')
            area=gate_rows[index]['areas']
            current_counts=tuple(area[n]['valid'] for n,_ in OBJECTS)
            request,minimums=actual_policy.step(current_counts,index)
            if request != planned[index]:
                raise AssertionError('causal policy differs from count-only planning')
            configs=dict(Fixed256=SamplingConfig(),Adaptive=SamplingConfig(num_points=request,min_points=minimums),
                         FixedMatched=SamplingConfig(num_points=matched['budget']))
            owner={};unique={}
            for condition,c in configs.items():
                signature=(c.num_points,tuple(min(m,a) for m,a in zip(c.min_points,current_counts)))
                owner[condition]=unique.setdefault(signature,condition)
            names=list(dict.fromkeys(owner.values()))
            if index == 0:
                for condition in names:
                    sample_points(evidence,configs[condition],method='S3')
                    warmups.append(dict(condition=condition,source_frame=0,measured=False))
            selected={}
            budget_times=[]
            for repeat in range(3):
                start=perf_counter();check=requested_budget(current_counts,actual_policy.reference)
                budget_times.append((perf_counter()-start)*1000)
                if check != (request,minimums):
                    raise AssertionError('deterministic budget')
                order=names[(index+repeat)%len(names):]+names[:(index+repeat)%len(names)]
                for condition in order:
                    sample=sample_points(evidence,configs[condition],method='S3')
                    measured=sampling_timing(sample)
                    validate_selection(sample,evidence,configs[condition],candidates)
                    if repeat:
                        exact_result(sample,selected[condition])
                    else:
                        selected[condition]=sample
                    timings.append(dict(source_frame=index,repeat=repeat,condition=condition,executed=True,
                        alias_of=None,order=order,timings_ms=measured))
                for condition in CONDITIONS:
                    if owner[condition] != condition:
                        timings.append(dict(source_frame=index,repeat=repeat,condition=condition,executed=False,
                            alias_of=owner[condition],timings_ms=None))
            results={c:selected[owner[c]] for c in CONDITIONS}
            with np.load(source/f'f{index:04d}_samples.npz',allow_pickle=False) as ar:
                exact_saved(results['Fixed256'],ar)
                if request == 256:
                    exact_saved(results['Adaptive'],ar)
            # No GT is supplied until all selections and deterministic checks are complete.
            context=evaluation_context(frame,evidence,gt,camera,robot_ids)
            with np.load(source/f'f{index:04d}_samples.npz',allow_pickle=False) as ar:
                for name,oid in OBJECTS:
                    reference=tuple(ar[f'evaluation/{name}/reference_{s}'] for s in ('uv','xyz'))
                    for a,b in zip(reference,context['references'][name]):
                        np.testing.assert_array_equal(a,b)
                    context['references'][name]=reference
                    context['matchers'][name]=CoverageMatcher(evidence.frame.geometry,oid,reference)
            row=dict(gate_rows[index],minimum_total=sum(minimums),adaptive_budget_ms=budget_times,
                     intrinsic=frame.intrinsic.tolist(),cam2world=frame.cam2world.tolist(),conditions={})
            archive=dict(source_frame=np.array(index),shape=np.array(frame.depth.shape))
            for name,oid in OBJECTS:
                c=candidates[oid]
                archive[f'support/{name}/flat_index']=c.flat_index
                archive[f'support/{name}/xyz']=c.points
                archive[f'candidate/{name}/flat_index']=c.flat_index[c.metadata['structural_candidate']]
                archive[f'candidate/{name}/xyz']=c.points[c.metadata['structural_candidate']]
                for suffix,a in zip(('uv','xyz'),context['references'][name]):
                    archive[f'evaluation/{name}/reference_{suffix}']=a
            for condition,sample in results.items():
                if condition == 'Fixed256':
                    score=pools['Fixed256'][index]
                else:
                    score,dist=score_selection(sample,context);score['source_frame']=index
                    pools[condition].append(score)
                    for k,v in dist.items():
                        distance_pools[condition].setdefault(k,[]).append(v.astype(np.float32))
                general,dist=general_geometry(frame,evidence,candidates,sample,context)
                for k,v in dist.items():
                    distance_pools[condition].setdefault(k,[]).append(v.astype(np.float32))
                object_rows={}
                for name,oid in OBJECTS:
                    c=sample.counts[oid]
                    object_rows[name]=dict(allocated=c['budget'],selected=c['selected'],priority=c['priority'],
                        surface=c['surface_fps'],candidates=c['candidates'],selected_candidates=c['selected_candidates'],
                        shortfall=c['budget']-c['selected'],minimum_support_shortfall=max(0,16-area[name]['valid']),
                        density_per_valid_pixel=ratio(c['selected'],area[name]['valid']),
                        density_per_raw_pixel=ratio(c['selected'],area[name]['raw']))
                n=configs[condition].num_points
                row['conditions'][condition]=dict(requested=n,selected=len(sample.points),
                    allocated_total=sum(sample.object_budgets.values()),priority=int(sample.priority_selected.sum()),
                    surface=int((~sample.priority_selected).sum()),priority_cap=n//4,
                    shortfall=n-len(sample.points),maximum_saturated=n==256,minimum_saturated=n==sum(minimums),
                    alias_of=owner[condition] if owner[condition]!=condition else None,
                    objects=object_rows,geometry=score,general=general)
                for field in FIELDS:
                    archive[f'{condition}/{field}']=getattr(sample,field)
                for field,a in sample.metadata.items():
                    archive[f'{condition}/metadata/{field}']=a
            rows.append(row)
            np.savez_compressed(output/'frames'/f'f{index:04d}_samples.npz',**archive)
            if index in REPRESENTATIVE:
                overlay(output,frame,candidates,results,context)
            if index%15 == 0:
                print(f'Ablation {index+1}/121: adaptive request={request}, matched={matched["budget"]}',flush=True)
        add_temporal_and_pairs(rows)
        latency=dict(raw=timings,summary=latency_summary(timings),warmups=warmups,
            adaptive_budget_ms=distribution([np.median(r['adaptive_budget_ms']) for r in rows]),
            support_count_ms=distribution([r['support_count_ms'] for r in rows]),
            raw_area_diagnostics_ms=distribution([r['raw_area_diagnostics_ms'] for r in rows]),
            extraction_ms={k:distribution([r['extraction_ms'][k] for r in rows]) for k in rows[0]['extraction_ms']},
            scope='CPU same process, 3 rotated repetitions; per-frame median then across frames. Raw diagnostic scan excluded; support count shared once. Aliases have timings_ms=null; effective statistics reference an actual owner call. Gather residual includes overhead. GT/I/O/plots/extraction excluded from sampling_total.')
        data=make_summary(rows,pools,distance_pools,matched,actual_policy)
        import os
        data.update(run_id=logger.run_id,source_run_id=original['run_id'],config=config,
            reference_camera=reference_camera,latency=latency['summary'],
            input_hashes=verification['input_hashes'],source_contract=verification['source_contract'],
            dependency_links={k:os.path.relpath(v,output) for k,v in dict(
                original_comparison=source/'comparison.json', original_html=source/'comparison.html',
                original_samples_frame0=source/'f0000_samples.npz', rgbd=episode, tracker_masks=masks).items()})
        write_json(output/'adaptive_node_budget.json',data)
        write_json(output/'frame_metrics.json',rows)
        write_json(output/'latency.json',latency)
        plot_results(output,rows,timings)
        html_report(output,data,rows)
        verification.update(adaptive_geometry_executed=True,deterministic_repetitions=True,
            adaptive_request256_exact_frames=[r['source_frame'] for r in rows if r['conditions']['Adaptive']['requested']==256],
            preservation=preserve_check(before),evaluated_frames=len(rows),gpu_executed=False,
            source_snapshot_sha256={str(p.relative_to(output/'evaluated_source')):digests(p)[1]
                                    for p in (output/'evaluated_source').rglob('*') if p.is_file()})
        if not verification['preservation']['unchanged']:
            raise AssertionError('pre-existing source/artifact changed')
        write_json(output/'verification.json',verification)
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name not in ('config.yaml','metadata.json','metrics.json','manifest.json') and not p.name.endswith('.tracking.json'):
                logger.link_output(p,kind='metrics' if p.suffix=='.json' else 'trajectory' if p.suffix=='.npz' else 'reports')
        logger.log({c:dict(mean_nodes=data['conditions'][c]['counts']['selected']['mean'],
                          reduction=data['conditions'][c]['node_reduction_fraction']) for c in CONDITIONS},prefix='eval/task03ab')
        logger.record(evaluated_frames=121,evaluation_device='cpu',gpu_executed=False,source_run_id=original['run_id'])
        print('Completed:',output,flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preservation-before',type=Path,required=True)
    parser.add_argument('--tests-log',type=Path)
    args=parser.parse_args()
    evaluate(args.preservation_before,args.tests_log)


if __name__ == '__main__':
    main()
