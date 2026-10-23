"""CUDA PCA5 resident/transfer timing and same-estimator fidelity.

Uses real CUDA execution, paired CPU reference truth, full point support and
float64 covariance. Does not claim tracking or final Graph handoff timing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from utils.experiment_logger import ExperimentLogger
from utils.experiment_artifacts import digests
from visual.runtime.geometry import GeometryFrame,GeometricFeatures,build_organized_geometry
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.test.pca_torch import TorchPCA5
from visual.runtime.test.run_pca_optimization import reference,fidelity,combine_fidelity,synthetic_observations
from visual.runtime.test.normal_diagnostics import stats
from visual.runtime.test.run_normal_diagnostics import save_json


def benchmark(input_path,masks_path,output_root,*,repeats=3,backend='eigh',synthetic=False):
    frames=list(iter_saved_rgbd(input_path,masks_path))
    cfg=dict(task='pca5_cuda',repeats=repeats,backend=backend,synthetic=synthetic,pca_dtype='float64',initial_grounding_excluded=True,tracking_included=False,
             input_sha256=digests(Path(input_path))[1],masks_sha256=digests(Path(masks_path))[1],torch=torch.__version__,platform=platform.platform(),
             source_sha256={str(p.relative_to(REPO)):digests(p)[1] for p in [Path(__file__),Path(__file__).with_name('pca_torch.py'),Path(__file__).with_name('pca_triton.py')]})
    with ExperimentLogger(team='visual',config=cfg,seed=0,model='pca5_cuda',job_type='evaluate') as logger:
        out=Path(output_root)/logger.run_id;out.mkdir(parents=True,exist_ok=False);print('Output:',out,flush=True)
        if not torch.cuda.is_available():
            save_json(out/'cuda.json',dict(run_id=logger.run_id,config=cfg,status='not_executed',reason='CUDA unavailable'))
            logger.link_output(out/'cuda.json',kind='metrics');logger.record(gpu_executed=False);return out
        torch.set_num_threads(4)
        cfg['gpu']=torch.cuda.get_device_name();logger.record(evaluation_device='cuda',gpu_executed=True,gpu=cfg['gpu'])
        processor=TorchPCA5(frames[0].intrinsic,frames[0].depth.shape,device='cuda',backend=backend)
        resident=[(torch.as_tensor(f.depth,device='cuda'),{i:torch.as_tensor(m,device='cuda') for i,m in f.masks.items()}) for f in frames]
        for _ in range(3): processor(*resident[0])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        rows=[];quality=[]
        for idx,frame in enumerate(frames):
            expected=reference(frame)
            for repeat in range(repeats):
                for mode in (('resident','cpu_input') if (idx+repeat)%2==0 else ('cpu_input','resident')):
                    torch.cuda.synchronize();start=perf_counter()
                    if mode=='cpu_input':
                        data=(torch.as_tensor(frame.depth,device='cuda'),{i:torch.as_tensor(m,device='cuda') for i,m in frame.masks.items()})
                    else:data=resident[idx]
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();result=processor(*data);end.record();torch.cuda.synchronize();wall=(perf_counter()-start)*1000
                    rows.append(dict(source_frame=frame.source_frame,repeat=repeat,mode=mode,wall_ms=wall,device_ms=begin.elapsed_time(end)))
                    if repeat==0 and mode=='resident':
                        objects={i:GeometricFeatures(**{k:v.cpu().numpy() for k,v in f.items()}) for i,f in result['objects'].items()}
                        actual=GeometryFrame(frame.source_frame,frame.camera,frame.frame_idx,frame.timestamp,build_organized_geometry(frame),objects,{})
                        np.testing.assert_array_equal(result['xyz'].cpu().numpy(),expected.geometry.xyz)
                        q=fidelity(expected,actual);quality.append(q)
            if idx==0 or (idx+1)%20==0:print('CUDA frames',idx+1,'/',len(frames),flush=True)
        synthetic_rows=[]
        if synthetic:
            for idx,(condition,frame,truth,regions) in enumerate(synthetic_observations(frames[0].intrinsic)):
                expected=reference(frame)
                result=processor(torch.as_tensor(frame.depth,device='cuda'),{i:torch.as_tensor(m,device='cuda') for i,m in frame.masks.items()})
                objects={i:GeometricFeatures(**{k:v.cpu().numpy() for k,v in f.items()}) for i,f in result['objects'].items()}
                actual=GeometryFrame(frame.source_frame,frame.camera,frame.frame_idx,frame.timestamp,build_organized_geometry(frame),objects,{})
                np.testing.assert_array_equal(result['xyz'].cpu().numpy(),expected.geometry.xyz)
                synthetic_rows.append(dict(condition=condition,fidelity=fidelity(expected,actual)))
                if (idx+1)%48==0:print('CUDA synthetic conditions',idx+1,'/288',flush=True)
        fs=combine_fidelity(quality)
        sf=combine_fidelity([r['fidelity'] for r in synthetic_rows])
        summary={mode:{k:stats([r[k] for r in rows if r['mode']==mode]) for k in ('wall_ms','device_ms')} for mode in ('resident','cpu_input')}
        passed=all(v['validity_differences']==0 and v['other_field_differences']==0 and v['normal_max_deg']<=.001 and v['variation_max_deg']<=.001 for v in (fs,sf))
        save_json(out/'cuda.json',dict(run_id=logger.run_id,config=cfg,status='executed',quality_passed=passed,fidelity=fs,summary=summary,frames=rows,
                                     synthetic=dict(fidelity=sf,cases=synthetic_rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        logger.link_output(out/'cuda.json',kind='metrics');logger.log(summary,prefix='eval/cuda');logger.log(fs,prefix='eval/cuda/fidelity')
        if not passed:raise AssertionError('CUDA fidelity tolerance failed')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=REPO/'visual/data/step2/ep000.npz')
    p.add_argument('--masks',type=Path,default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    p.add_argument('--out',type=Path,default=REPO/'visual/results/pca_optimization')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--backend',choices=['eigh','triton','triton_fused'],default='eigh')
    p.add_argument('--synthetic',action='store_true')
    a=p.parse_args();print(benchmark(a.input,a.masks,a.out,repeats=a.repeats,backend=a.backend,synthetic=a.synthetic))


if __name__=='__main__':main()
