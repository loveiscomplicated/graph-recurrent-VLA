"""PCA5 optimization: paired fidelity, fixed GT evaluation and CPU latency.

Keeps existing outputs and estimators. Excludes initial grounding by using the
same saved masks; tracker-inclusive timing is a separate explicit benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import platform
from time import perf_counter
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_logger import ExperimentLogger
from utils.experiment_artifacts import digests
from visual.runtime.geometry import GeometryFrame, GeometryParameters, build_organized_geometry, compute_geometric_features
from visual.runtime.rgbd import iter_saved_rgbd, PEG
from visual.runtime.test.normal_estimators import local_plane
from visual.runtime.test.normal_diagnostics import angular_error, quantize, rays, stats, synthetic_case
from visual.runtime.test.pca_optimized import process_optimized
from visual.runtime.test.run_normal_comparison import Pool, real_truth, save_json

METHODS = ('reference_pca5', 'optimized_eigh', 'optimized_analytic', 'optimized_parallel')


def reference(frame):
    start = perf_counter(); geometry = build_organized_geometry(frame); t1 = perf_counter()
    p = GeometryParameters()
    normals = {i: local_plane(geometry, i, 5, p) for i in geometry.object_masks}; t2 = perf_counter()
    features = {i: compute_geometric_features(geometry, i, *normals[i], p) for i in geometry.object_masks}; end = perf_counter()
    return GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx, frame.timestamp, geometry, features,
                         dict(geometry=(t1-start)*1000, normals=(t2-t1)*1000,
                              discontinuity_features=(end-t2)*1000, total=(end-start)*1000))


def run_method(frame, method, profile=None):
    if method=='optimized_parallel':return process_optimized(frame,solver='analytic',profile=profile,parallel=True)
    return reference(frame) if method=='reference_pca5' else process_optimized(frame, solver=method.removeprefix('optimized_'), profile=profile)


def fidelity(expected, actual):
    row = dict(validity_differences=0, other_field_differences=0, normal_max_deg=0., variation_max_deg=0.,
               threshold_flips={str(t): 0 for t in (5, 10, 20, 30)})
    np.testing.assert_array_equal(expected.geometry.xyz, actual.geometry.xyz)
    for oid, a in expected.objects.items():
        b = actual.objects[oid]
        row['validity_differences'] += int(np.count_nonzero(a.normal_valid != b.normal_valid))
        row['validity_differences'] += int(np.count_nonzero(a.normal_variation_valid != b.normal_variation_valid))
        common = a.normal_valid & b.normal_valid
        if common.any(): row['normal_max_deg'] = max(row['normal_max_deg'], float(angular_error(a.normals[common], b.normals[common]).max()))
        vv = a.normal_variation_valid & b.normal_variation_valid
        if vv.any(): row['variation_max_deg'] = max(row['variation_max_deg'], float(np.rad2deg(np.abs(a.normal_variation_rad[vv]-b.normal_variation_rad[vv])).max()))
        for key in a.__dataclass_fields__:
            if key in ('normals', 'normal_variation_rad', 'normal_valid', 'normal_variation_valid'): continue
            aa, bb = getattr(a, key), getattr(b, key)
            equal = (aa == bb) | (np.isnan(aa) & np.isnan(bb)) if np.issubdtype(aa.dtype, np.floating) else aa==bb
            row['other_field_differences'] += int(np.count_nonzero(~equal))
        for t in (5, 10, 20, 30):
            row['threshold_flips'][str(t)] += int(np.count_nonzero(vv & ((np.rad2deg(a.normal_variation_rad)>t) != (np.rad2deg(b.normal_variation_rad)>t))))
    return row


def combine_fidelity(rows):
    return dict(samples=len(rows), validity_differences=sum(r['validity_differences'] for r in rows),
                other_field_differences=sum(r['other_field_differences'] for r in rows),
                normal_max_deg=max((r['normal_max_deg'] for r in rows), default=0.),
                variation_max_deg=max((r['variation_max_deg'] for r in rows), default=0.),
                threshold_flips={str(t):sum(r['threshold_flips'][str(t)] for r in rows) for t in (5, 10, 20, 30)})


def synthetic_observations(k):
    for kind in ('front', 'tilted', 'crease', 'step'):
        for z in (.2, .25, .35, .6, 1.):
            for phase in (0., .00025, .0005, .00075):
                f, truth, regions = synthetic_case(kind, k, z0=z, phase_m=phase)
                for mode in ('float', 'floor', 'nearest'):
                    yield dict(geometry=kind, z0_m=z, phase_m=phase, quantization=mode), f if mode=='float' else replace(f, depth=quantize(f.depth, mode)), truth, regions
    for z in (.25, .6):
        for slope in (.2, .8):
            for azimuth in (0, 45, 90, 135):
                f, _, regions = synthetic_case('tilted', k, z0=z, phase_m=.00025)
                a, b = slope*np.cos(np.deg2rad(azimuth)), slope*np.sin(np.deg2rad(azimuth))
                ray = rays(f.depth.shape, k)
                depth = ((z+.00025)/(1-a*ray[..., 0]-b*ray[..., 1])).astype(np.float32)
                f = replace(f, depth=depth)
                truth = np.broadcast_to(np.array([a, b, -1.])/np.sqrt(a*a+b*b+1), (*depth.shape, 3))
                for mode in ('float', 'floor', 'nearest'):
                    yield dict(geometry='direction', z0_m=z, slope=slope, azimuth_deg=azimuth, quantization=mode), f if mode=='float' else replace(f, depth=quantize(f.depth, mode)), truth, regions


def plot(output, frame, results):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(METHODS)+1, figsize=(4*(len(METHODS)+1), 9), constrained_layout=True)
    axes[0, 0].imshow(frame.rgb); axes[0, 0].set_title('Same observation / saved tracking masks')
    axes[1, 0].imshow(frame.rgb); axes[1, 0].set_title('Normal valid=yellow / invalid=dark')
    for col, method in enumerate(METHODS, 1):
        var = np.full(frame.depth.shape, np.nan); valid = var.copy()
        for f in results[method].objects.values():
            var[f.normal_variation_valid] = np.rad2deg(f.normal_variation_rad[f.normal_variation_valid])
            valid[f.valid_points] = f.normal_valid[f.valid_points]
        axes[0, col].imshow(var, cmap=plt.get_cmap('turbo').with_extremes(bad='#cccccc'), vmin=0, vmax=45)
        axes[1, col].imshow(valid, cmap=plt.get_cmap('viridis').with_extremes(bad='#cccccc'), vmin=0, vmax=1)
        axes[0, col].set_title(method+' variation (0-45 deg)')
        axes[1, col].set_title(method+' normal validity')
    for ax in axes.ravel(): ax.axis('off')
    fig.suptitle(f'Source {frame.source_frame}: same PCA5 support; implementation optimization only')
    fig.savefig(output/f'f{frame.source_frame:04d}.png', dpi=120); plt.close(fig)


def evaluate(input_path, masks_path, output_root, *, repeats=5, synthetic=True):
    if repeats < 1: raise ValueError('repeats must be positive')
    input_path, masks_path = Path(input_path).resolve(), Path(masks_path).resolve()
    frames = list(iter_saved_rgbd(input_path, masks_path))
    with np.load(input_path, allow_pickle=False) as a:
        gt = {k:a[k] for k in ['base_camera/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose', 'box_hole_pose', 'peg_half_sizes', 'box_hole_radii', 'seed']}
    sources = [Path(__file__), Path(__file__).with_name('pca_optimized.py'), Path(__file__).with_name('pca_torch.py'), Path(__file__).with_name('normal_estimators.py')]
    cfg = dict(task='pca5_optimization', input=str(input_path), masks=str(masks_path), repeats=repeats,
               methods=list(METHODS), input_sha256=digests(input_path)[1], masks_sha256=digests(masks_path)[1],
               source_sha256={str(p.relative_to(REPO)):digests(p)[1] for p in sources},
               initial_grounding_excluded=True, tracking_included=False, final_graph_handoff_implemented=False,
               cpu_normal_target_p95_ms=10, cpu_geometry_target_p95_ms=25, final_target_period_ms=50,
               same_neighbourhood=True, pca_dtype='float64', synthetic=synthetic,
               python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
               timing='one warmup each, round-robin order, full per-method build/normal/features, no I/O/GT/metrics')
    with ExperimentLogger(team='visual', config=cfg, seed=int(gt['seed']), model='pca5_optimization', job_type='evaluate') as logger:
        output = Path(output_root)/logger.run_id; output.mkdir(parents=True, exist_ok=False)
        print('Output:',output,flush=True)
        source_dir=output/'source'; source_dir.mkdir()
        for p in sources: (source_dir/p.name).write_bytes(p.read_bytes())
        for method in METHODS: run_method(frames[0], method)
        pool, records = Pool(), []
        all_fidelity = {m:[] for m in METHODS[1:]}
        timings = {m:[] for m in METHODS}
        profiles = []
        for idx, frame in enumerate(frames):
            results = {}
            for repeat in range(repeats):
                offset = (idx+repeat)%len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:
                    p = {} if method=='optimized_analytic' and repeat==0 else None
                    result = run_method(frame, method, profile=p)
                    timings[method].append(dict(source_frame=frame.source_frame, repeat=repeat, **result.timings_ms))
                    if repeat==0: results[method]=result
                    if p is not None: profiles.append(dict(source_frame=frame.source_frame, objects=p))
            truths, regions, _ = real_truth(frame, results['reference_pca5'], gt, 'base_camera')
            record = dict(source_frame=frame.source_frame, fidelity={}, metrics={})
            for method in METHODS[1:]:
                f = fidelity(results['reference_pca5'], results[method]); record['fidelity'][method]=f; all_fidelity[method].append(f)
            for method in METHODS:
                rows = {}
                for name, oid in [('peg',0),('box',1)]:
                    for key, region in regions[name].items():
                        rows[name+'/'+key] = pool.add(method+'/'+name+'/'+key, results[method].objects[oid], region, truths[name]['normals'] if key.startswith('plane_') else None)
                record['metrics'][method]=rows
            if frame.source_frame in (0, 15, 45, 80, 120): plot(output, frame, results)
            records.append(record)
            if idx==0 or (idx+1)%20==0: print('Real frames',idx+1,'/',len(frames),flush=True)
        summary = pool.summary()
        syn_rows=[]; syn_fidelity={m:[] for m in METHODS[1:]}
        if synthetic:
            from visual.runtime.test.run_normal_comparison import response_metrics
            for idx, (condition, frame, truth, regions) in enumerate(synthetic_observations(frames[0].intrinsic)):
                results = {m:run_method(frame,m) for m in METHODS}
                row=dict(condition=condition, methods={}, fidelity={})
                for method in METHODS[1:]:
                    f=fidelity(results['reference_pca5'],results[method]);syn_fidelity[method].append(f);row['fidelity'][method]=f
                for method in METHODS:
                    row['methods'][method]={key:response_metrics(results[method].objects[PEG],region,truth) for key,region in regions.items()}
                syn_rows.append(row)
                if (idx+1)%48==0: print('Synthetic conditions',idx+1,'/288',flush=True)
        latency = {m:{k:stats([r[k] for r in timings[m]]) for k in ['geometry','normals','discontinuity_features','total']} for m in METHODS}
        fidelity_summary={m:combine_fidelity(all_fidelity[m]) for m in METHODS[1:]}
        syn_summary={m:combine_fidelity(syn_fidelity[m]) for m in METHODS[1:]}
        for part in [fidelity_summary, syn_summary] if synthetic else [fidelity_summary]:
            for r in part.values():
                if r['validity_differences'] or r['other_field_differences'] or r['normal_max_deg']>.001 or r['variation_max_deg']>.001:
                    raise AssertionError('same-estimator fidelity tolerance failed')
        data=dict(run_id=logger.run_id,config=cfg,evaluated_frames=len(frames),latency_ms=latency,
                  fidelity=fidelity_summary,summary=summary,frames=records,profile=profiles,timing_samples=timings,
                  synthetic=dict(conditions=len(syn_rows),fidelity=syn_summary,cases=syn_rows))
        save_json(output/'optimization.json',data)
        body=''.join(f'<p><img style="width:100%" src="{p.name}"></p>' for p in sorted(output.glob('*.png')))
        (output/'optimization.html').write_text(f'<!doctype html><meta charset="utf-8"><title>PCA5 optimization</title><h1>PCA5 optimization</h1><p>Run {logger.run_id}. Initial grounding excluded; saved tracking masks. Final Graph handoff is not implemented. <a href="optimization.json">Metrics, fidelity, latency, source hashes</a>.</p>'+body)
        for path in output.iterdir():
            if path.is_file(): logger.link_output(path,kind='metrics' if path.suffix=='.json' else 'reports')
        logger.log(latency,prefix='eval/pca5_optimization/latency_ms');logger.log(fidelity_summary,prefix='eval/pca5_optimization/fidelity')
        logger.record(evaluation_device='cpu',gpu_executed=False,initial_grounding_excluded=True,evaluated_frames=len(frames))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=REPO/'visual/data/step2/ep000.npz')
    p.add_argument('--masks',type=Path,default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    p.add_argument('--out',type=Path,default=REPO/'visual/results/pca_optimization')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--no-synthetic',action='store_true')
    a=p.parse_args();print(evaluate(a.input,a.masks,a.out,repeats=a.repeats,synthetic=not a.no_synthetic))


if __name__=='__main__': main()
