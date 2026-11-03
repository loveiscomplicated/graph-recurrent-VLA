"""Same-case filter-only 3-warmup/15-repeat wall benchmark; no model/Graph calls."""
from pathlib import Path
import os,sys,time,json,hashlib,resource,shutil,inspect
OLD=Path('/workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4')
OVERLAY=Path('/workspace/task10i_robot_filter_gpu_io_20261103_v1/code')
sys.path.insert(0,str(OLD))
from run import C,FROZEN,geometry,load_inputs,masks_for,Candidate,RayCache,ExperimentLogger,np,torch,identity,exact
from filter_benchmark import pair
sys.path.insert(0,str(OVERLAY))
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter


def cuda_pair(inputs,masks,gpu_masks,return_cpu=False):
    with Candidate(9) as ctx:
        ctx.cache={};filt=CudaRobotFilter()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();result={}
        for camera in inputs[0]:
            raw=(torch.as_tensor(np.ascontiguousarray(masks[camera.camera]),device='cuda')
                 if return_cpu else gpu_masks[camera.camera])
            clean=filt(camera,inputs[2],raw)
            result[camera.camera]=clean.cpu().numpy() if return_cpu else clean
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        return dict(ms=elapsed,clean=result,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())


def main():
    C.assert_source();torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    output=Path(os.environ['TASK10I_GPU_IO_BENCH_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
    scene='10i_s1_0921_t080';size=512;warmups=3;repeats=15
    spec=next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id']==scene)
    geo=geometry();inputs=load_inputs(scene,size,geo);masks=masks_for(spec,size,0,'actual')
    # SAM2 would already own these CUDA bool masks. Saved-mask upload is outside
    # the GPU->GPU timer. A separate CPU->CPU variant includes this upload.
    gpu_masks={k:torch.as_tensor(v,device='cuda') for k,v in masks.items()};torch.cuda.synchronize()
    expected=pair(inputs,masks,'baseline',RayCache())['clean']
    variants=['baseline_cpu_cpu','strict_cpu_cpu','new_cuda_cpu_cpu','new_cuda_gpu_gpu']
    rays=RayCache();rows=[]
    boundaries=dict(baseline_cpu_cpu='CPU masks -> CPU clean masks',strict_cpu_cpu='CPU masks -> CPU clean masks',
        new_cuda_cpu_cpu='CPU masks -> CPU clean masks; raw mask upload and clean readback inside timer',
        new_cuda_gpu_gpu='resident CUDA masks -> CUDA clean masks; CPU sensor/calibration/FK surfaces, hull/ray preparation and uploads still inside timer')
    def run(v):
        if v=='baseline_cpu_cpu':return pair(inputs,masks,'baseline',RayCache())
        if v=='strict_cpu_cpu':return pair(inputs,masks,'strict',rays)
        return cuda_pair(inputs,masks,gpu_masks,return_cpu=v=='new_cuda_cpu_cpu')
    def check(out):
        clean={k:(v.cpu().numpy() if isinstance(v,torch.Tensor) else v) for k,v in out['clean'].items()}
        exact(clean,expected,'clean masks')
    config=dict(scene=scene,size=size,mask_repeat=0,cameras=list(masks),warmups=warmups,repeats=repeats,
        variants=variants,boundaries=boundaries,identity=identity(),no_sam2_inference=True,no_graph_execution=True,
        sources={str(p):C.sha(p) for p in (Path(__file__),Path(inspect.getfile(CudaRobotFilter)),OLD/'filter_benchmark.py',OLD/'roi_filter.py')},
        input_sha256=dict(masks={k:C.array_sha(v) for k,v in masks.items()},
            cameras={c.camera:{k:C.array_sha(getattr(c,k)) for k in ('depth_raw_mm','intrinsic','extrinsic_cv')} for c in inputs[0]},
            surfaces=[dict(link=s.link,vertices=C.array_sha(s.world_vertices)) for s in inputs[2]]),
        exclusions='file/decode/join/FK/SAM2/grounding/gate/Shape-First/Graph/action/verification/logging; GPU-only variant additionally excludes raw mask upload and clean mask readback',
        order='same prior alternating forward/reverse order; validation/copies/hash/logger writes outside the timer')
    C.write(output/'config.json',config)
    snapshot=output/'source';snapshot.mkdir()
    for p in (Path(__file__),Path(inspect.getfile(CudaRobotFilter)),OLD/'filter_benchmark.py',OLD/'roi_filter.py'):shutil.copy2(p,snapshot/p.name)
    with ExperimentLogger(team='integration',model='task10i_gpu_io_filter_512_latency',job_type='evaluate',seed=42,root=output/'experiments',config=config) as logger:
        # Includes each variant's first kernel invocation/JIT outside final timing.
        for _ in range(warmups):
            for v in variants:check(run(v))
        for repeat in range(repeats):
            for v in variants if repeat%2==0 else variants[::-1]:
                out=run(v);check(out)
                row=dict(variant=v,repeat=repeat,ms=out['ms'],allocated=out['allocated'],reserved=out['reserved'],clean_exact=True)
                rows.append(row)
        summary={}
        for v in variants:
            t=np.array([r['ms'] for r in rows if r['variant']==v])
            summary[v]=dict(n=len(t),median_ms=float(np.median(t)),q25_ms=float(np.quantile(t,.25)),q75_ms=float(np.quantile(t,.75)),min_ms=float(t.min()),max_ms=float(t.max()))
        original={}
        for pid in (327662,327663,327664,327665,327666,357110,357111):
            state=next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:'))
            assert '\tT' in state;original[str(pid)]=state
        result=dict(status='PASS',rows=rows,summary=summary,original_processes=original)
        C.write(output/'results.json',result)
        logger.log({v:s['median_ms'] for v,s in summary.items()});logger.link_output(output/'results.json');logger.link_output(output/'config.json')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
