"""Task03 offline observation sampling / GT preservation evaluation (CPU only).

Run from repository root with WANDB_MODE=disabled. No simulator/tracker rerun,
no remote publication, final classification, graph output or stale cache.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import html
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from utils.experiment_logger import ExperimentLogger
from utils.experiment_artifacts import digests
from visual.runtime.rgbd import PEG, BOX, iter_saved_rgbd
from visual.runtime.geometry import build_organized_geometry, compute_geometric_features, GeometryParameters
from visual.runtime.normal_estimators import local_plane
from visual.runtime.sampling import (METHODS, SamplingConfig, extract_sampling_evidence, sample_points)
from visual.runtime.test.normal_diagnostics import stats, angular_error
from visual.runtime.test.run_normal_diagnostics import save_json
from visual.runtime.test.sampling_evaluation import evaluation_context, score_selection, summarize, TOLERANCES
from visual.runtime.test.sampling_plots import overlays, clouds, quantitative


def conditions(seeds=(0,1,2)):
    """Staged sweeps rather than a full factorial. Every condition uses all frames."""
    out={}
    for method in METHODS:
        rows=[(n,.25,20,0) for n in (64,128,256,512)]
        if method!='S0':
            rows += [(256,r,20,0) for r in (0.,.125,.5)]
            rows += [(256,.25,t,0) for t in (10,30,40)]
        rows += [(256,.25,20,s) for s in seeds if s!=0]
        for n,r,t,s in rows:
            r=0. if method=='S0' else r
            key=f'{method}_n{n}_r{r:g}_t{t}_s{s}'
            out[key]=dict(method=method,num_points=n,structural_ratio=r,orientation_threshold_deg=t,seed=s)
    return out


def check_fidelity(evidence, frame):
    """Full real-frame PCA5 candidate/validity alignment against unchanged reference."""
    g=build_organized_geometry(frame)
    np.testing.assert_array_equal(g.xyz,evidence.frame.geometry.xyz)
    row=dict(max_normal_error_deg=0.,max_variation_error_deg=0.,validity_differences=0,candidate_differences=0)
    for oid in (PEG,BOX):
        expected=compute_geometric_features(g,oid,*local_plane(g,oid,5),GeometryParameters())
        actual=evidence.frame.objects[oid]
        for field in expected.__dataclass_fields__:
            a,b=getattr(actual,field),getattr(expected,field)
            if field=='normals':
                if actual.normal_valid.any(): row['max_normal_error_deg']=max(row['max_normal_error_deg'],float(angular_error(a[actual.normal_valid],b[actual.normal_valid]).max()))
                np.testing.assert_allclose(a,b,atol=2e-5,rtol=0)
            elif field=='normal_variation_rad':
                ok=actual.normal_variation_valid
                if ok.any():row['max_variation_error_deg']=max(row['max_variation_error_deg'],float(np.rad2deg(np.abs(a[ok]-b[ok])).max()))
                np.testing.assert_allclose(a,b,atol=2e-5,rtol=0)
            else:np.testing.assert_array_equal(a,b,err_msg=field)
        for t in (10,20,30,40):
            expected_map=expected.valid_points & expected.normal_variation_valid & (np.rad2deg(expected.normal_variation_rad)>t)
            np.testing.assert_array_equal(evidence.pairs[oid].candidates(t)['orientation'],expected_map)
    return row


def validate_result(result,evidence,cfg):
    if len(result.points)>cfg.num_points:raise AssertionError('total node budget')
    if result.priority_selected.sum()>int(cfg.num_points*cfg.structural_ratio):raise AssertionError('structural ratio')
    if result.depth_priority_selected.sum()>int(int(cfg.num_points*cfg.structural_ratio)*cfg.depth_priority_fraction):raise AssertionError('depth reservation')
    if len(np.unique(result.flat_index))!=len(result.points):raise AssertionError('duplicate source pixels')
    v,u=result.pixels_vu.T
    np.testing.assert_array_equal(result.points,evidence.frame.geometry.xyz[v,u])
    if not np.isfinite(result.points).all():raise AssertionError('invalid point')
    for oid in (PEG,BOX):
        keep=result.object_id==oid
        if not evidence.frame.geometry.observed_mask(oid)[v[keep],u[keep]].all():raise AssertionError('object support')
        if keep.sum()!=result.object_budgets[oid]:raise AssertionError('object budget')
    for arr in result.metadata.values():
        if len(arr)!=len(result.points):raise AssertionError('metadata alignment')


def evaluate(episode_path,masks_path,output_root,camera='base_camera',selected=(0,15,18,45,80,120),seeds=(0,1,2),make_plots=True):
    episode_path,masks_path=Path(episode_path).resolve(),Path(masks_path).resolve()
    with np.load(episode_path,allow_pickle=False) as ar:
        required=[f'{camera}/seg','peg_id','box_id','peg_pose','box_pose','box_hole_pose','peg_half_sizes','box_hole_radii','seed']
        if set(required)-set(ar.files):raise ValueError('GT evaluation fields missing')
        gt={k:ar[k] for k in required}
    with np.load(masks_path,allow_pickle=False) as ar:sources=ar['source_frame']
    if not np.array_equal(sources,np.arange(len(gt[f'{camera}/seg']))):raise ValueError('every source frame exactly once in order is required')
    if set(selected)-set(sources.tolist()):raise ValueError('representative source frame missing')
    track_path=masks_path.with_name('tracking.json')
    if track_path.exists():
        tracking=json.loads(track_path.read_text())
        if tracking.get('camera',camera)!=camera:raise ValueError('tracker camera mismatch')
        original=Path(tracking.get('input') or episode_path)
        if original.is_file() and digests(original)[1]!=digests(episode_path)[1]:raise ValueError('tracker input mismatch')
    segmap_path=episode_path.with_name('segmentation_id_map.json')
    segmap=json.loads(segmap_path.read_text()) if segmap_path.exists() else {}
    robot_ids=[int(k) for k,v in segmap.items() if v.startswith('panda_')]
    configs=conditions(seeds)
    source_paths=[REPO/'visual/runtime'/p for p in ('rgbd.py','geometry.py','normal_estimators.py','pca.py','edge_evidence.py','sampling.py',
        'test/run_sampling.py','test/sampling_evaluation.py','test/sampling_plots.py','test/test_sampling.py',
        'test/test_sampling_evaluation.py','test/test_sampling_geometry.py',
        'test/normal_diagnostics.py','test/run_normal_comparison.py','test/run_structural_edges.py','test/run_normal_diagnostics.py')]
    config=dict(task='03',input=str(episode_path),masks=str(masks_path),camera=camera,
        input_sha256=digests(episode_path)[1],masks_sha256=digests(masks_path)[1],conditions=configs,
        defaults=asdict(SamplingConfig()),tolerances_px_m=TOLERANCES,selected_sources=selected,
        robot_ids=robot_ids,segmentation_map_sha256=digests(segmap_path)[1] if segmap else None,
        device='cpu',gpu_executed=False,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        gt_sampling_input=False,independent_episode=False,source_sha256={str(p.relative_to(REPO)):digests(p)[1] for p in source_paths},
        timing='one warmup per estimator, rotated order; no GT/I/O/metrics/plots; S0 charged PCA diagnostic evidence too; CPU arrays, no H2D/D2H',
        fps='source raster order, uniform first row per SeedSequence([seed,object_id]); argmax first-row ties; surface distances seeded by priority rows',
        sweep='four budgets at .25/20/seed0; four ratios at 256/20/seed0; four angles at 256/.25/seed0; three seeds at 256/.25/20, all frames',
        single_episode_note='Only this saved RGB-D episode has matching tracker masks; no independent RGB-D episode in visual/data')
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03_sampling',job_type='evaluate') as logger:
        output=Path(output_root).resolve()/logger.run_id;output.mkdir(parents=True,exist_ok=False)
        print('Output:',output,flush=True)
        for source in source_paths:
            dest=output/'evaluated_source'/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        pools={k:[] for k in configs};arrays={k:{} for k in configs};fidelity=[];extraction=[]
        for index,frame in enumerate(iter_saved_rgbd(episode_path,masks_path,camera=camera)):
            order=['baseline','fd2','pca5'];order=order[index%3:]+order[:index%3]
            if index==0:
                for method in order:sample_points(extract_sampling_evidence(frame,method),method={'baseline':'S1','fd2':'S2','pca5':'S3'}[method])
            evidence={m:extract_sampling_evidence(frame,m) for m in order}
            fidelity.append(dict(source_frame=frame.source_frame,**check_fidelity(evidence['pca5'],frame)))
            extraction.append({m:e.timings_ms for m,e in evidence.items()})
            # Everything above accepts current observations only. GT begins here.
            context=evaluation_context(frame,evidence['pca5'],gt,camera,robot_ids)
            mapping=dict(S0=evidence['pca5'],S1=evidence['baseline'],S2=evidence['fd2'],S3=evidence['pca5'],S4=evidence['pca5'])
            archive=dict(source_frame=np.array(frame.source_frame),shape=np.array(frame.depth.shape))
            results={};frame_rows={}
            for key,condition in configs.items():
                method=condition['method'];cfg=SamplingConfig(**{k:v for k,v in condition.items() if k!='method'})
                result=sample_points(mapping[method],cfg,method=method);validate_result(result,mapping[method],cfg)
                results[key]=result
                row,dist=score_selection(result,context);row['source_frame']=frame.source_frame
                pools[key].append(row);frame_rows[key]=row
                for k,v in dist.items():arrays[key].setdefault(k,[]).append(v.astype(np.float32))
                for field in ('points','pixels_vu','flat_index','object_id','priority_selected','depth_priority_selected'):
                    archive[f'{key}/{field}']=getattr(result,field)
                for name,values in result.metadata.items():archive[f'{key}/metadata/{name}']=values
            for method in METHODS[1:]:
                np.testing.assert_array_equal(results[f'{method}_n256_r0_t20_s0'].flat_index,results['S0_n256_r0_t20_s0'].flat_index)
            for method,e in evidence.items():
                for oid in (PEG,BOX):
                    pair=e.pairs[oid]
                    for t in (10,20,30,40):
                        c=pair.candidates(t)
                        for channel in ('orientation','depth','mask','invalid','occluding','occluded'):
                            archive[f'evidence/{method}/{oid}/{t}/{channel}_flat']=np.flatnonzero(c[channel])
            for name,ref in context['references'].items():
                archive[f'evaluation/{name}/reference_uv']=ref[0];archive[f'evaluation/{name}/reference_xyz']=ref[1]
            np.savez_compressed(output/f'f{frame.source_frame:04d}_samples.npz',**archive)
            save_json(output/f'f{frame.source_frame:04d}_metrics.json',frame_rows)
            if frame.source_frame in selected and make_plots:
                primary={m:results[f'{m}_n256_r{0 if m=="S0" else .25:g}_t20_s0'] for m in METHODS}
                overlays(output,frame,context,primary,mapping);clouds(output,frame,context,primary,mapping)
            if index%10==0:print(f'Frames {index+1}/{len(sources)}; {len(configs)} conditions each',flush=True)
        summary={key:summarize(rows,arrays[key]) for key,rows in pools.items()}
        latency={m:{k:stats([v[m][k] for v in extraction]) for k in extraction[0][m]} for m in ('baseline','fd2','pca5')}
        data=dict(run_id=logger.run_id,config=config,git_commit=logger.metadata['git_commit'],git_dirty=logger.metadata['git_dirty'],
                  evaluated_frames=len(sources),conditions=len(configs),summary=summary,extraction_latency_ms=latency,fidelity=fidelity)
        save_json(output/'comparison.json',data)
        if make_plots:quantitative(output,summary,configs)
        image_names=['tradeoffs.png','sensitivity.png']+[f'f{s:04d}_{kind}.png' for s in selected for kind in ('overlay','clouds')]
        body=''.join(f'<p><a href="{p}"><img src="{p}" alt="{p}"></a></p>' for p in image_names if (output/p).exists())
        (output/'comparison.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Task03 sampling</title>
<style>body{{font:15px system-ui;max-width:1800px;margin:24px auto}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Task03 — current-observation point sampling</h1><p>Run {logger.run_id}; {len(sources)} frames; {len(configs)} conditions; CPU only.
<a href="comparison.json">Full summary, source hashes, fidelity, configuration and latency</a>.</p>
<p>GT is evaluation only. No final structural class, node feature, graph, cache or end-to-end 20Hz validation.
Joint matching uses the same pair within 2px AND 3mm; sensitivity 1px/2mm and 3px/5mm. Visible GT, tracker-observed references and sampled references
are separate denominators. Retention excludes structures already lost to tracking. Surface distances retain all observed points, including contamination.
Known-negative ROIs overlap; union is counted once. Priority role differs from detector candidate flag. S4 depth-only maximum is 25% of priority slots,
not a raw OR policy. S0 includes PCA diagnostic extraction in its reported total even though its selection ignores candidates.</p>
<p>Single saved ep000; frames are temporally dependent. Thresholds are diagnostics, no generalization claim or adopted final setting.
Every frame has a metrics JSON and selected coordinates/provenance/evidence NPZ; pixels are (v,u), coordinates CV-camera metres.
Absent predictions have zero coverage and missing distance counts. 3D viewports clip display only; raw support is not filtered.</p>{body}
<details><summary>Configuration</summary><pre>{html.escape(json.dumps(config,indent=2))}</pre></details></html>''',encoding='utf-8')
        for p in sorted(p for p in output.rglob('*') if p.is_file()):
            logger.link_output(p,kind='metrics' if p.suffix=='.json' else 'trajectory' if p.suffix=='.npz' else 'reports')
        logger.log({m:summary[f'{m}_n256_r{0 if m=="S0" else .25:g}_t20_s0']['objects'] for m in METHODS},prefix='eval/task03')
        logger.record(evaluated_frames=len(sources),evaluation_device='cpu',gpu_executed=False,dataset_sha256=config['input_sha256'],masks_sha256=config['masks_sha256'])
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=REPO/'visual/data/step2/ep000.npz')
    parser.add_argument('--masks',type=Path,default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    parser.add_argument('--out',type=Path,default=REPO/'visual/results/sampling')
    parser.add_argument('--camera',default='base_camera')
    args=parser.parse_args();print(evaluate(args.input,args.masks,args.out,args.camera))


if __name__=='__main__':main()
