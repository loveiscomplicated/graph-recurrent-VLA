"""Actual Vast58752018 CPU/full vs CUDA/compact-mask projection comparison."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ['WANDB_MODE'] = 'disabled'
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
try:
    from .masked_torch import project
except ImportError:
    from masked_torch import project


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def gpu_status():
    return dict(gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip(),
        processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True).strip())


def stats(samples):
    return dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),
                min_ms=min(samples),max_ms=max(samples),samples_ms=samples)


def compare(actual, expected):
    for field in ('pixels', 'object_id'):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
    np.testing.assert_allclose(actual['points'], expected['points'], atol=1e-6, rtol=0)
    return dict(source_pixels_exact=True,object_ids_exact=True,ordering_exact=True,
        xyz_bitwise_equal=bool(np.array_equal(actual['points'],expected['points'])),
        xyz_max_abs_m=float(np.max(np.abs(actual['points']-expected['points']),initial=0)),
        xyz_atol_m=1e-6,point_count=len(expected['pixels']))


@torch.inference_mode()
def main(args):
    if os.getenv('CONTAINER_ID') != '58752018':raise RuntimeError('requested instance58752018 only')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA execution required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.root).resolve();output=root/args.output
    output.mkdir(parents=True,exist_ok=False)
    status_before=gpu_status()
    if status_before['processes']:raise RuntimeError('GPU busy before benchmark; no existing job is stopped')
    spec=importlib.util.spec_from_file_location('original_pca',root/'inputs/original_pca.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    with np.load(root/'inputs/input.npz',allow_pickle=False) as z:
        arrays={name:z[name].copy() for name in ('depth','intrinsic','masks')}
    if arrays['depth'].shape != (512,512):raise ValueError('native512 required')
    frame=SimpleNamespace(depth=arrays['depth'],intrinsic=arrays['intrinsic'],
                          masks=dict(enumerate(arrays['masks'])))
    def cpu_full():
        geometry=original.build_geometry(frame)
        points=[];pixels=[];objects=[]
        for oid in (0,1):
            index=np.flatnonzero(geometry.observed_mask(oid))
            points.append(geometry.xyz.reshape(-1,3)[index]);pixels.append(index)
            objects.append(np.full(len(index),oid,np.int64))
        return dict(points=np.concatenate(points),pixels=np.concatenate(pixels),object_id=np.concatenate(objects))
    cpu_tensors={k:torch.from_numpy(a) for k,a in arrays.items()}
    def upload():return {k:v.to('cuda') for k,v in cpu_tensors.items()}
    resident=upload();torch.cuda.synchronize()
    def gpu_resident():return project(resident['depth'],resident['masks'],resident['intrinsic'])
    def gpu_roundtrip():
        inputs=upload()
        result=project(inputs['depth'],inputs['masks'],inputs['intrinsic'])
        return {k:v.cpu().numpy() for k,v in result.items()}
    expected=cpu_full()
    result=gpu_resident();torch.cuda.synchronize()
    checks=compare({k:v.cpu().numpy() for k,v in result.items()},expected)
    compare(gpu_roundtrip(),expected)
    input_hashes={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}
    source_files={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('masked_torch.py')):sha(Path(__file__).with_name('masked_torch.py')),
        str(root/'inputs/original_pca.py'):sha(root/'inputs/original_pca.py')}
    dependencies={}
    for name in ('visual.runtime.geometry','visual.runtime.rgbd','visual.runtime.normal_estimators','utils.experiment_logger','utils.experiment_artifacts'):
        module=sys.modules.get(name)
        if module is not None and getattr(module,'__file__',None):
            dependencies[module.__file__]=sha(module.__file__)
    config=dict(task='10i_backprojection_cuda_58752018_20261103',instance=os.getenv('CONTAINER_ID'),
        host=platform.platform(),cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        gpu=torch.cuda.get_device_name(0),python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        warmups=args.warmups,repeats=args.repeats,gc_enabled=gc.isenabled(),torch_threads=torch.get_num_threads(),
        cpu_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        cgroup={p:Path('/sys/fs/cgroup',p).read_text().strip() for p in ('cpu.max','memory.max') if Path('/sys/fs/cgroup',p).exists()},
        input_file_sha256=sha(root/'inputs/input.npz'),input_arrays_sha256=input_hashes,
        mask_pixels={str(i):int(arrays['masks'][i].sum()) for i in (0,1)},
        mask_union_fraction=float(arrays['masks'].any(0).mean()),
        cpu_source='byte-preserved original pca.build_geometry + object point/source gathering',
        gpu_source='eager Torch; mask selection is recalculated each call; float64 projection -> float32 compact XYZ; no dense XYZ, no cached pixel list',
        input_dtype={k:str(a.dtype) for k,a in arrays.items()},
        outputs='camera XYZ float32[N,3], pixel index int64[N], object_id int64[N]; no downsampling',
        output_bytes=sum(a.nbytes for a in expected.values()),
        gpu_input_bytes=sum(a.nbytes for a in arrays.values()),
        timing_boundaries={
            'cpu_numpy_full':'CPU depth/masks/K -> dense XYZ -> compact CPU object points + source pixels/IDs; includes original validation/mask copies/allocation',
            'cuda_mask_resident':'GPU depth/masks/K -> compact GPU points + pixels/IDs; includes dynamic nonzero, calibration/finite checks and host synchronizations; completion-synchronized wall time',
            'cuda_mask_roundtrip':'pageable CPU depth/masks/K -> H2D -> same CUDA operation -> D2H compact outputs; completion-synchronized wall time'},
        exclusions=['file loading/decoding','depth-mm conversion','normals/features','sampling','graph assembly','robot filter','simulator','models'],
        verification=checks,source_sha256=source_files,dependency_sha256=dependencies,gpu_status_before=status_before)
    write(output/'config.json',config)
    sources=output/'source';sources.mkdir()
    for name in ('benchmark.py','masked_torch.py','test_projection.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    functions={'cpu_numpy_full':cpu_full,'cuda_mask_resident':gpu_resident,'cuda_mask_roundtrip':gpu_roundtrip}
    samples={k:[] for k in functions}
    with ExperimentLogger(team='graph',model='cpu_full_vs_cuda_mask_512',job_type='evaluate',
                          seed=0,config=config,root=output/'experiments') as log:
        for _ in range(args.warmups):
            for fn in functions.values():value=fn();torch.cuda.synchronize();del value
        for repeat in range(args.repeats):
            names=list(functions);shift=repeat%len(names);names=names[shift:]+names[:shift]
            if repeat%2:names.reverse()
            for name in names:
                torch.cuda.synchronize()
                start=time.perf_counter_ns();value=functions[name]()
                if name.startswith('cuda'):torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-start)/1e6
                samples[name].append(elapsed);del value
            if (repeat+1)%250==0:print(json.dumps(dict(repeats_complete=repeat+1)),flush=True)
        # Additional event spans are diagnostics, not sums of pure kernel time:
        # calibration decisions/nonzero can introduce host launch gaps.
        event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True)
        events=[]
        for _ in range(100):
            torch.cuda.synchronize();event_start.record();value=gpu_resident();event_end.record()
            event_end.synchronize();events.append(event_start.elapsed_time(event_end));del value
        torch.cuda.synchronize();baseline_allocated=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats();value=gpu_resident();torch.cuda.synchronize()
        extra_gpu_peak=torch.cuda.max_memory_allocated()-baseline_allocated;del value
        final=compare({k:v.cpu().numpy() for k,v in gpu_resident().items()},expected)
        compare(gpu_roundtrip(),expected)
        assert input_hashes=={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()}
        assert all(sha(p)==h for p,h in {**source_files,**dependencies}.items())
        output_file=output/'outputs.npz'
        np.savez_compressed(output_file,**{'cpu_'+k:v for k,v in expected.items()},
            **{'gpu_'+k:v.cpu().numpy() for k,v in gpu_resident().items()})
        summary={k:stats(v) for k,v in samples.items()}
        results=dict(status='PASS',run_id=log.run_id,instance=os.getenv('CONTAINER_ID'),verification=final,
            input_and_sources_unchanged=True,timings=summary,cuda_event_span=stats(events),
            extra_gpu_peak_allocated_bytes=int(extra_gpu_peak),gpu_status_after=gpu_status(),
            speedup_resident=summary['cpu_numpy_full']['median_ms']/summary['cuda_mask_resident']['median_ms'],
            speedup_roundtrip=summary['cpu_numpy_full']['median_ms']/summary['cuda_mask_roundtrip']['median_ms'])
        write(output/'results.json',results)
        log.log({k:row['median_ms'] for k,row in summary.items()})
        for path in (output/'config.json',output/'results.json',output_file):log.link_output(path)
        print(json.dumps(dict(status='PASS',run_id=log.run_id,verification=final,
            medians={k:v['median_ms'] for k,v in summary.items()},
            speedup_resident=results['speedup_resident'],speedup_roundtrip=results['speedup_roundtrip']),indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',default='paired_v1');parser.add_argument('--warmups',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=1000)
    main(parser.parse_args())
