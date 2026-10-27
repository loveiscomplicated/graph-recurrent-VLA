"""FP32 GPU feasibility probe, same recorded masks and CPU output boundaries.

Compares S3 and current Shape-First/F5/EE packet scopes separately. No SAM2,
Raw EDE, policy, training or GT. Numerical/source-choice differences are measured
against an explicit experimental envelope, not a bitwise adoption gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import scipy
import cv2
from scipy.spatial import cKDTree

from utils.experiment_logger import ExperimentLogger
from visual.runtime.cuda_sampling import CudaSampling
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence, extract_candidates, sample_points, SamplingConfig
from visual.runtime.vision_packet import complete_vision_packet
from visual.runtime.test import run_variable_rate_shape_first as shape

MODES = ('cpu', 'gpu_pca', 'gpu_pca_fps')
ENVELOPE = dict(normal_p95_deg=2., normal_valid_disagreement_fraction=.02,
                orientation_candidate_flip_fraction=.05, point_count_relative_change=.05,
                coverage_p95_ratio=1.15,
                scope='provisional geometry screen, not task-success or trained-policy equivalence')


def stats(values):
    a=np.asarray(values,float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))


def selection_packet(frame, evidence, fps_fn):
    candidates=extract_candidates(evidence,SamplingConfig(),method='S3')
    prepared,selected,cases={},{},{}
    g=evidence.frame.geometry
    for oid in (0,1):
        obj=candidates[oid]
        orientation=evidence.pairs[oid].candidates(20)['orientation']
        pca=obj.flat_index[orientation.ravel()[obj.flat_index]]
        prepared[oid]=shape.prepare_object(obj.flat_index,obj.points,pca,frame.masks[oid],g.valid_depth,g.overlap_mask,frame.intrinsic)
        selected[oid]=shape.select_object(prepared[oid],.5,0,shape.SURFACE[oid],fps_fn=fps_fn)
        cases[oid]=dict(saved_flat=obj.flat_index,saved_features=obj.metadata)
    selection,_=shape.make_packet(frame.source_frame,prepared,selected,cases)
    selection.update(episode='gpu_probe',camera=frame.camera)
    return complete_vision_packet(selection,evidence,frame)


def run_case(scope, mode, frame, gpu):
    ev=extract_sampling_evidence(frame) if mode=='cpu' else gpu.evidence(frame)
    fps=gpu.fps_indices if mode=='gpu_pca_fps' else None
    out=sample_points(ev,SamplingConfig(),method='S3',fps_fn=fps) if scope=='s3' else selection_packet(frame,ev,fps)
    return ev,out


def selected(scope, out):
    if scope=='s3':return out.flat_index,out.object_id
    return out['provenance']['source_index'],out['object_id']


def compare(scope, reference, candidate):
    a,ao=reference;b,bo=candidate
    sa,ia=selected(scope,ao);sb,ib=selected(scope,bo)
    g=a.frame.geometry
    metrics=[]
    for oid in (0,1):
        support=g.observed_mask(oid);den=max(1,int(support.sum()))
        x=a.frame.objects[oid];y=b.frame.objects[oid]
        common=x.normal_valid&y.normal_valid&support
        n=x.normals[common].astype(float);m=y.normals[common].astype(float)
        angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n,m),axis=1),np.einsum('ij,ij->i',n,m)))
        ca=a.pairs[oid].candidates(20)['orientation'];cb=b.pairs[oid].candidates(20)['orientation']
        aa=sa[ia==oid];bb=sb[ib==oid]
        if len(np.unique(bb))!=len(bb) or not support.ravel()[bb].all():
            raise AssertionError('invalid/duplicate/unobserved GPU selection')
        full=g.xyz[support]
        cover=[]
        for s in (aa,bb):
            if len(full) and len(s): cover.append(float(np.percentile(cKDTree(g.xyz.reshape(-1,3)[s]).query(full)[0],95)))
            elif not len(full):cover.append(0.)
            else:cover.append(None)
        ratio=(cover[1]/cover[0] if cover[0] else (1. if cover[1]==cover[0] else None)) if None not in cover else None
        f5=None
        if scope=='shape_packet':
            _,ra,rb=np.intersect1d(aa,bb,return_indices=True)
            fa=ao['geometric_features'][np.flatnonzero(ia==oid)[ra]]
            fb=bo['geometric_features'][np.flatnonzero(ib==oid)[rb]]
            same=np.isfinite(fa)&np.isfinite(fb)
            f5=dict(common_source_points=len(ra),validity_disagreements=int(np.count_nonzero(np.isfinite(fa)^np.isfinite(fb))),
                    absolute_error_by_column=[stats(np.abs(fa[same[:,j],j]-fb[same[:,j],j])) for j in range(5)])
        metrics.append(dict(object_id=oid,normal_angle_deg=stats(angle),normal_common_count=int(common.sum()),
            normal_valid_disagreement=int(np.count_nonzero((x.normal_valid^y.normal_valid)&support)),
            observed_count=int(support.sum()),
            orientation_candidate_flip=int(np.count_nonzero((ca^cb)&support)),
            normal_valid_disagreement_fraction=float(np.count_nonzero((x.normal_valid^y.normal_valid)&support)/den),
            orientation_candidate_flip_fraction=float(np.count_nonzero((ca^cb)&support)/den),
            reference_points=len(aa),gpu_points=len(bb),point_count_relative_change=abs(len(bb)-len(aa))/max(1,len(aa)),
            source_jaccard=(len(np.intersect1d(aa,bb))/len(np.union1d(aa,bb)) if len(aa) or len(bb) else 1.),
            coverage_p95_m=cover,coverage_p95_ratio=ratio,f5_common_source=f5))
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--provenance',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--output-root',type=Path,default=Path('experiments'))
    p.add_argument('--instance-id',default='57676962');args=p.parse_args()
    proof=json.loads(args.provenance.read_text())
    with np.load(args.inputs,allow_pickle=False) as z:
        datasets={name:{k.split('/',1)[1]:z[k] for k in z.files if k.startswith(name+'/')}
                  for name in sorted({k.split('/')[0] for k in z.files})}
    frames=[]
    for name,a in datasets.items():
        for i in range(0,len(a['depth']),args.stride):
            h,w=a['depth'][i].shape;t=int(a['timestep'][i])
            frame=TrackedRGBDFrame(t,'base_camera',np.zeros((h,w,3),np.uint8),a['depth'][i],a['intrinsic'][i],
                a['cam2world'][i],a['tcp_pose'][i],{o:a['masks'][i,o] for o in (0,1)},t,t*.05)
            frames.append((name,t,frame))
    config=dict(instance_id=args.instance_id,repeats=args.repeats,stride=args.stride,frames=len(frames),
        inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),provenance=proof,
        modes=MODES,envelope=ENVELOPE,precision='FP32 GPU Jacobi PCA5/FPS; existing CPU FP64 reference',
        boundary='CPU depth/masks/calibration -> CPU sampling or F5/EE packet; transfers included',
        rgb='unused by geometry; zero-filled placeholder',
        scopes=['PCA5 evidence + S3 sampling','PCA5 evidence + Shape-First + F5/EE packet; Raw EDE excluded'],
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                     '--format=csv,noheader'],text=True).strip(),
        omp_threads=os.environ.get('OMP_NUM_THREADS'),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),optimizer_updates=0,
        cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()))
    start=perf_counter();gpu=CudaSampling();config['cuda_setup_compile_ms']=(perf_counter()-start)*1000
    config['compute_capability']=gpu.driver.compute_capability
    with ExperimentLogger(team='visual',model='cuda_sampling_fp32_probe',config=config,seed=0,job_type='latency',root=args.output_root) as log:
        root=log.directory.resolve();print('RUN_DIR='+str(root),flush=True)
        def save(name,value):
            target=root/name;target.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');log.link_output(target,kind='metrics')
        source_root=Path(__file__).resolve().parents[3]
        inventory=['visual/runtime/cuda_driver.py','visual/runtime/cuda_sampling.py','visual/runtime/cuda_sampling.cu',
            'visual/runtime/sampling.py','visual/runtime/pca.py','visual/runtime/edge_evidence.py',
            'visual/runtime/geometry.py','visual/runtime/normal_estimators.py','visual/runtime/rgbd.py',
            'visual/runtime/vision_packet.py','graph/vision_input.py','utils/experiment_logger.py',
            'visual/runtime/test/run_variable_rate_shape_first.py','visual/runtime/test/run_voxel_structural_sampling.py',
            'visual/runtime/test/run_priority_fps_metric_control.py','visual/runtime/test/run_rim_bottleneck.py',
            'visual/runtime/test/sampling_evaluation.py','visual/runtime/test/run_cuda_sampling_probe.py']
        hashes={}
        for rel in inventory:
            src=source_root/rel;dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
            hashes[rel]=hashlib.sha256(src.read_bytes()).hexdigest()
        save('source_hashes.json',hashes);save('input_provenance.json',proof)
        rows=[];quality=[]
        try:
            for scope in ('s3','shape_packet'):
                for mode in MODES:
                    for _,_,frame in frames:run_case(scope,mode,frame,gpu)
                print('WARMUP_COMPLETE='+scope,flush=True)
                for repeat in range(args.repeats):
                    for fidx,(dataset,t,frame) in enumerate(frames):
                        outputs={}
                        order=MODES[(repeat+fidx)%3:]+MODES[:(repeat+fidx)%3]
                        for mode in order:
                            gpu.driver.sync();tick=perf_counter();ev,out=run_case(scope,mode,frame,gpu);gpu.driver.sync()
                            elapsed=(perf_counter()-tick)*1000
                            sa,ids=selected(scope,out)
                            if len(np.unique(np.column_stack((ids,sa)),axis=0))!=len(sa):raise AssertionError('duplicate source rows')
                            rows.append(dict(scope=scope,mode=mode,dataset=dataset,timestep=t,repeat=repeat,elapsed_ms=elapsed,
                                points=len(sa),evidence_ms=ev.timings_ms))
                            if repeat==0:outputs[mode]=(ev,out)
                        if repeat==0:
                            for mode in MODES[1:]:
                                for metric in compare(scope,outputs['cpu'],outputs[mode]):
                                    quality.append(dict(scope=scope,mode=mode,dataset=dataset,timestep=t,**metric))
                    print(f'REPEAT_COMPLETE={scope}:{repeat}',flush=True)
            summary={}
            for scope in ('s3','shape_packet'):
                summary[scope]={}
                for dataset in datasets:
                    group={}
                    for mode in MODES:
                        samples=[r['elapsed_ms'] for r in rows if r['scope']==scope and r['dataset']==dataset and r['mode']==mode]
                        metrics=[q for q in quality if q['scope']==scope and q['dataset']==dataset and q['mode']==mode]
                        normal_p95=max((q['normal_angle_deg'].get('p95',0) for q in metrics),default=0)
                        tests=dict(normal_p95_deg=normal_p95,
                            normal_valid_disagreement_fraction=max((q['normal_valid_disagreement_fraction'] for q in metrics),default=0),
                            orientation_candidate_flip_fraction=max((q['orientation_candidate_flip_fraction'] for q in metrics),default=0),
                            point_count_relative_change=max((q['point_count_relative_change'] for q in metrics),default=0),
                            coverage_p95_ratio=max((q['coverage_p95_ratio'] for q in metrics if q['coverage_p95_ratio'] is not None),default=1))
                        passed=all(v<=ENVELOPE[k] for k,v in tests.items()) and all(q['coverage_p95_ratio'] is not None for q in metrics)
                        group[mode]=dict(latency_ms=stats(samples),difference_maxima=tests,provisional_geometry_pass=passed)
                    for mode in MODES[1:]:group[mode]['median_speedup']=group['cpu']['latency_ms']['median']/group[mode]['latency_ms']['median']
                    summary[scope][dataset]=group
            save('frames.json',rows);save('quality.json',quality);save('summary.json',dict(status='COMPLETE',run_id=log.run_id,config=config,results=summary))
            lines=['# FP32 CUDA sampling probe','',f'Run: `{log.run_id}`','',config['boundary'],'',
                '| Scope | Dataset | Mode | p50 ms | p95 ms | CPU / GPU p50 | Geometry screen |',
                '|---|---|---|---:|---:|---:|---|']
            for scope,dsets in summary.items():
                for dataset,modes in dsets.items():
                    for mode,r in modes.items():
                        s=r['latency_ms'];lines.append(f'| {scope} | {dataset} | {mode} | {s["median"]:.3f} | {s["p95"]:.3f} | {r.get("median_speedup",1):.2f} | {r["provisional_geometry_pass"]} |')
            lines+=['','Numerical screen is provisional, not trained-policy/task-success equivalence.','CPU feature/pair/contour/voxel/packet stages remain; GPU kernels cover PCA5 normals and FPS.','Startup, compile and full-input warmup excluded; host/device copies included.']
            report=root/'report.md';report.write_text('\n'.join(lines)+'\n');log.link_output(report,kind='reports')
            log.record(validation_status='COMPLETE',optimizer_updates=0,published=False)
            print('\n'.join(lines),flush=True)
        finally:gpu.close()


if __name__=='__main__':main()
