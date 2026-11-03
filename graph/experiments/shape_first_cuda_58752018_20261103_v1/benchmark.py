"""Synchronized post-feature -> selected CPU rows benchmark on Vast58752018."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import process_cuda, to_cpu_evidence
from visual.runtime.rgbd import TrackedRGBDFrame
from graph.integration_baseline import vision


def cpu_tail(frame, prepared, features):
    return cpu_shape(frame,*cpu_info(frame,prepared,features))


def cpu_info(frame,prepared,features):
    evidence=to_cpu_evidence(frame, prepared, features)
    objects=vision.extract_candidates(evidence,vision.SamplingConfig(),method='S3')
    return evidence,objects


def cpu_shape(frame,evidence,objects):
    prep={};selection={}
    g=evidence.frame.geometry
    for oid in (0,1):
        obj=objects[oid]
        orient=evidence.pairs[oid].candidates(20)['orientation']
        pca=obj.flat_index[orient.ravel()[obj.flat_index]]
        prep[oid]=vision.shape_first.prepare_object(obj.flat_index,obj.points,pca,frame.masks[oid],
            g.valid_depth,g.overlap_mask,frame.intrinsic)
        selection[oid]=vision.shape_first.select_object(prep[oid],.5,0,vision.shape_first.SURFACE[oid])
    return prep,selection


def load_frame(inputs):
    with np.load(inputs/'hand_camera.npz',allow_pickle=False) as a:
        depth=a['depth_raw_mm'].astype(np.float32)/1000
        depth[depth<=0]=np.nan
        rgb=a['rgb'].copy();k=a['intrinsic_cv'].copy()
    with np.load(inputs/'masks.npz',allow_pickle=False) as a:masks=a['masks'].copy()
    return TrackedRGBDFrame(80,'hand_camera',rgb,depth,k,np.eye(4),
        np.array([0.,0.,0.,1.,0.,0.,0.]),dict(enumerate(masks)),80,4.)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('cpu','gpu','paired'),default='cpu')
    parser.add_argument('--warmups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--phase-repeats',type=int,default=20)
    args=parser.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018'
    assert torch.cuda.is_available()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:raise RuntimeError('another GPU process is active; do not alter it')
    args.output.mkdir(parents=True,exist_ok=False)
    frame=load_frame(args.inputs)
    depth=torch.as_tensor(frame.depth,device='cuda')
    masks=torch.as_tensor(np.stack(list(frame.masks.values())),device='cuda')
    k=torch.as_tensor(frame.intrinsic,device='cuda')
    prepared,features=process_cuda(depth,masks,k)
    torch.cuda.synchronize()
    functions={'cpu':lambda:cpu_tail(frame,prepared,features)}
    if args.variant!='cpu':
        from visual.runtime.cuda_sampling import select_cuda,to_cpu_selection
        functions['gpu']=lambda:to_cpu_selection(select_cuda(prepared,features,depth,masks,k))
    variants=list(functions) if args.variant=='paired' else [args.variant]
    config=dict(task='10i_shape_first_cuda_58752018_20261103_v1',variant=args.variant,
        warmups=args.warmups,repeats=args.repeats,phase_repeats=args.phase_repeats,
        scope='GPU feature-ready -> selected CPU rows; no projection/PCA/packet/filter',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.inputs.glob('*.npz'))},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',config=config,seed=42,root=args.output/'experiments')
    try:
        timing={v:[] for v in variants}
        for _ in range(args.warmups):
            for v in variants:functions[v]()
        for repeat in range(args.repeats):
            for v in (variants if repeat%2==0 else variants[::-1]):
                torch.cuda.synchronize();start=perf_counter()
                result=functions[v]()
                torch.cuda.synchronize();timing[v].append((perf_counter()-start)*1000)
        # Keep correctness outside the timed region.
        prep,selected=functions['cpu']()
        arrays={f'{oid}_{key}':selected[oid][key] for oid in (0,1) for key in ('F','P','S','xyz')}
        for oid in (0,1):
            for key in ('pca','sil'):arrays[f'{oid}_C_{key}']=prep[oid]['C'][key];arrays[f'{oid}_R_{key}']=prep[oid]['R'][key]
        np.savez_compressed(args.output/'baseline_selection.npz',**arrays)
        if args.variant!='cpu':
            actual=to_cpu_selection(select_cuda(prepared,features,depth,masks,k),diagnostic=True)
            for oid in (0,1):
                for key in ('F','P','S','xyz'):np.testing.assert_array_equal(actual['selected'][oid][key],selected[oid][key],err_msg=f'{oid}:{key}')
                for key in ('pca','sil'):
                    np.testing.assert_array_equal(actual['prepared'][oid]['C'][key],prep[oid]['C'][key])
                    np.testing.assert_array_equal(actual['prepared'][oid]['R'][key],prep[oid]['R'][key])
        phases={v:dict(info=[],shape_and_handoff=[]) for v in variants}
        for _ in range(args.phase_repeats):
            for v in variants:
                torch.cuda.synchronize();start=perf_counter()
                if v=='cpu':info=cpu_info(frame,prepared,features)
                else:
                    from visual.runtime.cuda_sampling import build_sampling_info_cuda,shape_first_cuda
                    info=build_sampling_info_cuda(prepared,features,depth,masks,k)
                torch.cuda.synchronize();middle=perf_counter()
                if v=='cpu':result=cpu_shape(frame,*info)
                else:result=to_cpu_selection(shape_first_cuda(info))
                torch.cuda.synchronize();end=perf_counter()
                phases[v]['info'].append((middle-start)*1000)
                phases[v]['shape_and_handoff'].append((end-middle)*1000)
        summary=dict(run_id=logger.run_id,resolution=list(frame.depth.shape),gpu=torch.cuda.get_device_name(),
            mask_pixels=[int(m.sum()) for m in frame.masks.values()],
            selected_counts=[len(selected[i]['F']) for i in (0,1)],
            timings={v:dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),min_ms=min(t),max_ms=max(t)) for v,t in timing.items()},
            diagnostic_phase_medians_ms={v:{name:float(np.median(t)) for name,t in stages.items()} for v,stages in phases.items()},
            verification='baseline saved' if args.variant=='cpu' else 'all candidate/representative/selected source arrays exact')
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{v:np.array(t) for v,t in timing.items()})
        np.savez(args.output/'phase_timings.npz',**{v+'_'+name:np.array(t) for v,stages in phases.items() for name,t in stages.items()})
        logger.record(summary=summary)
        for p in args.output.glob('*.*'):logger.link_output(p)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
