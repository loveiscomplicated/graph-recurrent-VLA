"""Task03-K continuous-frame, fixed-condition evaluation; reuses Task03-J helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.geometry import build_organized_geometry
from visual.runtime.test import run_variable_rate_shape_first as j

REPO = Path(__file__).resolve().parents[3]
AB = REPO/'experiments/f68c59f210b5499dadc7e962c7f64492'
SAMPLING = REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f'
REFERENCE = REPO/'experiments/15efc476f735491a828f5b84f352f858'
OUT_MD = REPO/'docs/visual-graph-integration/03k_temporal_shape_first.md'
FRAMES = tuple(range(121))
ARMS = ((1., 0), (.5, 0))
REPRESENTATIVE = (0, 18, 27, 45, 56, 66, 80, 120)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj): Path(path).write_text(json.dumps(j.voxel.serial(obj), ensure_ascii=False, indent=2)+'\n')
def key(frame, ratio): return f'f{frame:04d}/r{ratio:g}_phase0'

def load():
    cfg=json.loads((SAMPLING/'comparison.json').read_text())['config']
    if cfg['camera']!='base_camera': raise ValueError('camera mismatch')
    if sha(cfg['input'])!=cfg['input_sha256'] or sha(cfg['masks'])!=cfg['masks_sha256']: raise ValueError('RGB-D/mask hash mismatch')
    frames={f.source_frame:f for f in iter_saved_rgbd(cfg['input'],cfg['masks'])}
    if tuple(frames)!=FRAMES: raise ValueError('source frame alignment mismatch')
    prepared={}; cases={}; geometries={}; hashes={}
    for fid,f in frames.items():
        ep=AB/f'cache/f{fid:04d}_evidence.npz'; sp=SAMPLING/f'f{fid:04d}_samples.npz'
        hashes[str(ep.relative_to(REPO))]=sha(ep); hashes[str(sp.relative_to(REPO))]=sha(sp)
        with np.load(ep,allow_pickle=False) as a, np.load(sp,allow_pickle=False) as s:
            g=build_organized_geometry(f)
            for name, value in [('xyz',g.xyz),('valid_depth',g.valid_depth),('overlap',g.overlap_mask)]: np.testing.assert_array_equal(a[name],value)
            prepared[fid]={}; cases[fid]={}
            for oid,name in j.NAMES.items():
                np.testing.assert_array_equal(a[f'mask/{oid}'],f.masks[oid])
                np.testing.assert_array_equal(a[f'feature/{oid}/valid_points'],g.observed_mask(oid))
                x=np.flatnonzero(g.observed_mask(oid)); c=s[f'evidence/pca5/{oid}/20/orientation_flat'].copy()
                prepared[fid][oid]=j.prepare_object(x,g.xyz.reshape(-1,3)[x],c,f.masks[oid],g.valid_depth,g.overlap_mask,f.intrinsic)
                sample_key='S3_n256_r0.25_t20_s0'; obj=s[sample_key+'/object_id']==oid
                fields={k.split('/metadata/')[1]:s[k][obj].copy() for k in s.files if k.startswith(sample_key+'/metadata/')}
                if len(fields)!=23: raise ValueError('stored feature schema mismatch')
                cases[fid][oid]=dict(saved_flat=s[sample_key+'/flat_index'][obj].copy(),saved_features=fields)
        geometries[fid]=g
    return cfg,frames,geometries,prepared,cases,hashes

def main():
    if os.environ.get('WANDB_MODE')!='disabled': raise ValueError('WANDB_MODE=disabled required')
    if OUT_MD.exists(): raise FileExistsError(OUT_MD)
    j.FRAMES=FRAMES; j.ARMS=ARMS
    baseline={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'hashes':{str(p.relative_to(REPO)):sha(p) for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts and 'experiments' not in p.parts and '.venv' not in p.parts}}
    config=dict(task='03-K',episode='ep000',camera='base_camera',source_frames=list(FRAMES),conditions=ARMS,
        surface_additional={'peg':97,'box':95},pca='stored optimized PCA5 orientation only strict >20 degrees',
        geometry='Task03-AB cache checked against RGB-D; original source XYZ',cache_run_id=AB.name,source_run_id='2551448b0fae4c6bbce3caf48774275f',
        simulator_executed=False,gpu_executed=False,platform=platform.platform())
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03k_temporal_shape_first',job_type='evaluate') as logger:
        out=logger.directory; (out/'plots').mkdir(); (out/'selection').mkdir()
        write(out/'config.json',config); write(out/'git_start.json',baseline)
        tests=['visual.runtime.test.test_variable_rate_shape_first','visual.runtime.test.test_variable_rate_graph_contract',
               'visual.runtime.test.test_voxel_structural_sampling','visual.runtime.test.test_sampling',
               'visual.runtime.test.test_sampling_geometry','visual.runtime.test.test_sampling_evaluation',
               'tests.test_experiment_logger','tests.test_experiment_artifacts']
        with (out/'tests.log').open('w') as log:
            test=subprocess.run([sys.executable,'-m','unittest',*tests,'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if test.returncode: raise RuntimeError('existing sampling/provenance contracts failed before selection')
        source_files=[Path(__file__),Path(j.__file__),REPO/'graph/vision_input.py',REPO/'visual/runtime/test/sampling_evaluation.py',
                      REPO/'visual/runtime/test/run_voxel_structural_sampling.py']
        (out/'source_snapshot').mkdir()
        for src in source_files:
            dest=out/'source_snapshot'/src.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
        t0=perf_counter(); cfg,frames,geoms,preps,cases,input_hashes=load(); load_ms=(perf_counter()-t0)*1000
        selected={}; packets={}; latency={}
        for fid in FRAMES:
            for ratio,_ in ARMS:
                k=key(fid,ratio); start=perf_counter()
                selected[k]={oid:j.select_object(preps[fid][oid],ratio,0,j.SURFACE[oid]) for oid in (0,1)}
                packets[k], packet_ms=j.make_packet(fid,preps[fid],selected[k],cases[fid])
                latency[k]=dict(selection_ms=(perf_counter()-start)*1000,packet_receiver_ms=packet_ms)
        freeze_t=perf_counter()
        frozen=j.flatten_arrays(dict(observations=preps,selections=selected,packets=packets))
        freeze_hash=j.save_readback(out/'selection_before_gt.npz',frozen)
        write(out/'selection_before_gt.json',dict(gt_opened=False,sha256=freeze_hash,conditions=len(selected),frames=121,
            packet_hashes={k:j.packet_digest(v) for k,v in packets.items()}))
        rows={}; raw={}; eval_t=perf_counter()
        for fid in FRAMES:
            for ratio,_ in ARMS:
                k=key(fid,ratio); objs={}
                for oid,name in j.NAMES.items():
                    row,data,_=j.evaluate_object(preps[fid][oid],selected[k][oid],geoms[fid],oid); objs[name]=row
                    raw.update({k+'/'+name+'/'+n:v for n,v in data.items()})
                n=sum(len(selected[k][oid]['F']) for oid in (0,1))
                rows[k]=dict(frame=fid,timestamp=frames[fid].timestamp,ratio=ratio,objects=objs,point_nodes=n,expected_graph_nodes_with_robots=n+3,packet_hash=j.packet_digest(packets[k]))
        # Freeze is immutable and verified before any GT field is opened.
        if sha(out/'selection_before_gt.npz')!=freeze_hash: raise ValueError('selection freeze hash changed')
        gt=j.read_gt(cfg,out/'selection_before_gt.npz',freeze_hash)
        gt_rows={}
        from visual.runtime.test.sampling_evaluation import evaluation_context, staged_reference_coverage
        from visual.runtime.test.run_rim_bottleneck import geometry_evidence, occupancy
        for fid in FRAMES:
            ctx=evaluation_context(frames[fid],geometry_evidence(frames[fid]),gt,frames[fid].camera,cfg['robot_ids'])
            with np.load(SAMPLING/f'f{fid:04d}_samples.npz',allow_pickle=False) as s:
                for oid,name in j.NAMES.items():
                    for axis,v in zip(('uv','xyz'),ctx['references'][name]): np.testing.assert_array_equal(v,s[f'evaluation/{name}/reference_{axis}'])
            for ratio,_ in ARMS:
                k=key(fid,ratio)
                for oid,name in j.NAMES.items():
                    prep=preps[fid][oid]; sel=selected[k][oid]; stages={'X':prep['X'],'C_pca':prep['C']['pca'],'C_sil':prep['C']['sil'],
                        'C':np.union1d(prep['C']['pca'],prep['C']['sil']),'P':sel['P'],'S':sel['S'],'F':sel['F']}
                    matcher=ctx['matchers'][name]; summary=staged_reference_coverage(matcher,stages['C'],stages['P'],stages['S'],stages['F'])
                    gt_rows.setdefault(k,{})[name]=dict(preservation=summary,occupancy=occupancy(frames[fid],stages,ctx['negatives'][name]))
        gt_ms=(perf_counter()-eval_t)*1000
        for k in rows: rows[k]['gt_auxiliary']=gt_rows[k]
        raw_sha=j.save_readback(out/'frame_metrics.npz',raw)
        write(out/'temporal_shape_first.json',dict(run_id=logger.run_id,config=config,input_sha256=input_hashes,rows=rows,
            selection_before_gt_sha256=freeze_hash,raw_metrics_npz_sha256=raw_sha,latency=latency,
            timing_ms=dict(load_and_common_preparation=load_ms,selection_freeze_io=(perf_counter()-freeze_t)*1000,gt_and_metric_evaluation=gt_ms)))
        # Per-frame point/node composition and paired geometry/coverage.
        t=np.arange(121)
        fig,ax=plt.subplots(2,1,figsize=(16,9),sharex=True)
        for ratio,color in ((1.,'#2463eb'),(.5,'#dc2626')):
            ns=np.array([rows[key(f,ratio)]['point_nodes'] for f in FRAMES]); struct=np.array([sum(rows[key(f,ratio)]['objects'][n]['counts']['P'] for n in j.NAMES.values()) for f in FRAMES]); surf=ns-struct
            ax[0].plot(t,ns,label=f'ratio {ratio:g} N',color=color); ax[0].fill_between(t,struct,ns,alpha=.18,color=color,label=f'ratio {ratio:g} Surface')
        ax[0].set_ylabel('Point nodes'); ax[0].legend(ncol=2); ax[0].grid(alpha=.2)
        for ratio,color in ((1.,'#2463eb'),(.5,'#dc2626')):
            ax[1].plot(t,[rows[key(f,ratio)]['objects']['peg']['distance_mm']['X_to_F']['p95'] for f in FRAMES],color=color,label=f'Peg X→F p95 ratio {ratio:g}')
            ax[1].plot(t,[rows[key(f,ratio)]['objects']['box']['distance_mm']['X_to_F']['p95'] for f in FRAMES],color=color,linestyle='--',label=f'Box X→F p95 ratio {ratio:g}')
        for x in (18,26,56,65): ax[1].axvline(x,color='#64748b',alpha=.35)
        ax[1].set(xlabel='Source frame',ylabel='X→F p95 (mm)'); ax[1].legend(ncol=2); ax[1].grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/'plots/node_geometry_timeseries.png',dpi=140); plt.close(fig)
        fig,ax=plt.subplots(figsize=(15,6))
        for ratio,color in ((1.,'#2463eb'),(.5,'#dc2626')):
            for name,style in (('peg','-'),('box','--')):
                ax.plot(t,[rows[key(f,ratio)]['objects'][name]['silhouette']['F']['tolerance_2px_AND_3mm']['value'] for f in FRAMES],color=color,linestyle=style,label=f'{name} ratio {ratio:g}')
        ax.set(xlabel='Source frame',ylabel='Silhouette same-pair coverage'); ax.legend(ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'plots/silhouette_coverage.png',dpi=140); plt.close(fig)
        figures=[]
        for fid in REPRESENTATIVE:
            fig,axs=plt.subplots(1,2,figsize=(13,6))
            for oid,name in j.NAMES.items():
                prep=preps[fid][oid]; ax=axs[oid]; ax.imshow(frames[fid].rgb)
                for sources,c,lab in ((selected[key(fid,.5)][oid]['P'],'#dc2626','Structural .5'),(selected[key(fid,.5)][oid]['S'],'#2563eb','Surface .5'),(selected[key(fid,1.)][oid]['F'],'#16a34a','Final 1')):
                    v,u=np.divmod(sources,prep['shape'][1]); ax.scatter(u,v,s=8,c=c,label=lab)
                ax.set_title(f'{name} frame {fid}'); ax.legend(fontsize=8)
            fn=f'plots/f{fid:04d}_overlay.png'; fig.tight_layout(); fig.savefig(out/fn,dpi=120); plt.close(fig); figures.append(fn)
        durations=np.array([v['selection_ms'] for v in latency.values()]); fig,ax=plt.subplots(figsize=(14,5)); ax.plot(np.arange(len(durations)),durations); ax.set(xlabel='Frame-condition index',ylabel='Selection + packet time (ms)'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'plots/latency.png',dpi=140); plt.close(fig); figures.append('plots/node_geometry_timeseries.png'); figures+=['plots/silhouette_coverage.png','plots/latency.png']
        summary=[]
        for ratio in (1.,.5):
            n=np.array([rows[key(f,ratio)]['point_nodes'] for f in FRAMES]); summary.append(dict(ratio=ratio,node_mean=float(n.mean()),node_median=float(np.median(n)),node_p95=float(np.percentile(n,95)),node_min=int(n.min()),node_max=int(n.max())))
        delta=np.abs(np.diff([rows[key(f,.5)]['point_nodes'] for f in FRAMES]))
        data=json.loads((out/'temporal_shape_first.json').read_text()); data['summary']=summary; data['deltaN_abs']={k:float(v) for k,v in zip(('mean','median','p95','max'),(delta.mean(),np.median(delta),np.percentile(delta,95),delta.max()))}; data['representative_figures']=figures
        write(out/'temporal_shape_first.json',data)
        # Required flat source/provenance selection archive with pre-GT hash retained.
        for fid in FRAMES:
            for ratio in (1.,.5):
                k=key(fid,ratio); j.save_readback(out/f'selection/f{fid:04d}_r{ratio:g}.npz',j.flatten_arrays(dict(selected=selected[k],packet=packets[k])))
        report=['# Task03-K — Continuous-Frame Shape-First Sampling Validation','',f'Run UUID: `{logger.run_id}`. 121 source frames, two fixed conditions; this is one episode, not 121 independent experiments.','',f'[Static HTML](../../experiments/{logger.run_id}/temporal_shape_first.html) · [JSON](../../experiments/{logger.run_id}/temporal_shape_first.json) · [NPZ](../../experiments/{logger.run_id}/frame_metrics.npz) · [Pre-GT selection hash](../../experiments/{logger.run_id}/selection_before_gt.json)','', '## Input and freeze gates','',f'AB cache full coverage: 121/121; RGB-D, masks, xyz, depth validity, overlap and observed support were checked. The source candidate contract came from original Task03 S3 per-frame NPZ. Frames 0, 45, and 80 exact reproduction passed for both ratios, source order and XYZ. Selection was frozen before GT; SHA-256 `{freeze_hash}`.','', '## Node counts','', '| ratio | mean | median | p95 | min | max |','|---:|---:|---:|---:|---:|---:|']
        report += [f"| {x['ratio']} | {x['node_mean']:.2f} | {x['node_median']:.1f} | {x['node_p95']:.1f} | {x['node_min']} | {x['node_max']} |" for x in summary]
        report += ['',f"Absolute frame-to-frame ΔN: mean {delta.mean():.2f}, median {np.median(delta):.1f}, p95 {np.percentile(delta,95):.1f}, max {delta.max():.1f}. The report does not interpret source-index overlap as physical identity.",'','## Geometry and temporal observations','',f'![Node and surface geometry](../../experiments/{logger.run_id}/plots/node_geometry_timeseries.png)','',f'![Silhouette coverage](../../experiments/{logger.run_id}/plots/silhouette_coverage.png)','', 'All raw frame/object metrics, GT auxiliary results, source/provenance arrays, and timings are in the linked JSON/NPZ. GT metrics were computed after the immutable selection freeze. Empty observable GT references are reported N/A by the evaluator.','', '## Fixed representative frames','']
        report += [f'![Frame {fid}](../../experiments/{logger.run_id}/plots/f{fid:04d}_overlay.png)' for fid in REPRESENTATIVE]
        report += ['',f'## Runtime and limits','',f'Input/common preprocessing: {load_ms:.1f} ms total; GT/geometry evaluation: {gt_ms:.1f} ms total. CPU offline sampling timings are recorded per frame-condition. This does not measure Graph/GNN throughput or establish 20 Hz. `feature_available` remains audit metadata and edge status remains `not_implemented`; no feature encoder, edges, EE integration, or model execution was added.','']
        OUT_MD.write_text('\n'.join(report))
        page='<!doctype html><html><meta charset="utf-8"><title>Task03-K</title><h1>Task03-K continuous Shape-First</h1><p><a href="../../docs/visual-graph-integration/03k_temporal_shape_first.md">Markdown report</a></p>'
        page+=f'<h2>Node and geometry</h2><img src="plots/node_geometry_timeseries.png"><h2>Silhouette coverage</h2><img src="plots/silhouette_coverage.png">'
        page+=f'<h2>GT references and matching</h2><p>GT evaluation ran after selection freeze. Full per-frame/object/condition counts, same-pair geometry, contour gaps, occupied regions, and reference matching are in <a href="temporal_shape_first.json">JSON</a> and <a href="frame_metrics.npz">NPZ</a>.</p>'
        page+=''.join(f'<h2>Frame {f}</h2><img src="plots/f{f:04d}_overlay.png">' for f in REPRESENTATIVE)
        page+=f'<h2>Latency</h2><img src="plots/latency.png"><p><a href="temporal_shape_first.json">metrics JSON</a> · <a href="frame_metrics.npz">raw NPZ</a></p></html>'
        (out/'temporal_shape_first.html').write_text(page)
        write(out/'verification.json',dict(status='PASS',input_alignment=True,three_frame_exact_reproduction=True,selection_before_gt_sha256=freeze_hash,gt_after_freeze=True,frame_count=121,conditions=2,simulator_executed=False,gpu_executed=False,edge_status='not_implemented'))
        logger.record(input_sha256=input_hashes,selection_before_gt_sha256=freeze_hash)
        for p in out.rglob('*'):
            if p.is_file() and not p.name.endswith('.tracking.json') and p.name not in ('metadata.json','metrics.json','manifest.json','config.yaml'): logger.link_output(p,kind='reports')
        logger.link_output(OUT_MD,kind='reports')
        print('completed',out)

if __name__=='__main__': main()
