"""Compare the frozen 10.9ms CUDA filter with a further filter-only candidate."""
from pathlib import Path
import sys, os, argparse, json, inspect, shutil, time, resource, importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_optimized as B

PREVIOUS=Path('/workspace/task10i_robot_filter_gpu_optimization_20261103_v1/final_512/source/gpu_filter.py')


def previous_class():
    spec=importlib.util.spec_from_file_location('previous_10_9_filter',PREVIOUS)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.CudaRobotFilter


def main():
    global PREVIOUS
    parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true')
    parser.add_argument('--reference',type=Path,default=PREVIOUS)
    parser.add_argument('--clip-pixels',type=int)
    parser.add_argument('--no-compact',action='store_true')
    parser.add_argument('--full-mask',action='store_true')
    parser.add_argument('--cpu-depth',action='store_true')
    parser.add_argument('--repeats',type=int,default=15);args=parser.parse_args()
    PREVIOUS=args.reference
    B.C.assert_source();B.torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    output=Path(os.environ['TASK10I_GPU_REFINE_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
    scene='10i_s1_0921_t080';spec=next(s for s in B.C.read(B.FROZEN/'protocol.json')['scenes'] if s['id']==scene)
    inputs=B.load_inputs(scene,512,B.geometry());masks=B.masks_for(spec,512,0,'actual')
    if args.full_mask:masks={k:B.np.stack((B.np.ones_like(v[0]),B.np.zeros_like(v[1]))) for k,v in masks.items()}
    raw={k:B.torch.as_tensor(v,device='cuda') for k,v in masks.items()};B.torch.cuda.synchronize()
    previous=previous_class();expected=B.pair(inputs,masks,'baseline',B.RayCache())['clean']
    variants=['previous_gpu_gpu'] if args.profile else ['previous_gpu_gpu','refined_gpu_gpu','refined_cpu_cpu']
    candidate_kwargs={}
    if args.clip_pixels is not None:candidate_kwargs['clip_pixels']=args.clip_pixels
    if args.no_compact:candidate_kwargs['compact']=False
    if args.cpu_depth:candidate_kwargs['depth_from_mm']=False
    filters={v:(previous() if v.startswith('previous') else B.CudaRobotFilter(**candidate_kwargs)) for v in variants}
    def run(v):return B.cuda_pair(inputs,masks,raw,lambda:filters[v],batch=True,return_cpu=v.endswith('cpu_cpu'))
    def check(out):B.exact({k:v.cpu().numpy() if isinstance(v,B.torch.Tensor) else v for k,v in out['clean'].items()},expected,'clean masks')
    files=[Path(__file__),Path(inspect.getfile(B)),Path(inspect.getfile(B.CudaRobotFilter)),PREVIOUS,B.OLD/'filter_benchmark.py']
    config=dict(scene=scene,size=512,mask_repeat=0,mask_mode='full_stress' if args.full_mask else 'actual',cameras=list(masks),warmups=3,repeats=args.repeats,profile=args.profile,candidate_kwargs=candidate_kwargs,
        variants=variants,identity=B.identity(),sources={str(p):B.C.sha(p) for p in files},
        boundary='resident CUDA masks -> CUDA clean masks, two cameras; CPU hull/bounds preparation, current sensor/calibration uploads and completion inside timer; cpu_cpu also includes masks upload/readback',
        cache='persistent filter per variant; static calibration rays and, when implemented, universal int16 unit-conversion table; no current-frame depth/pose/hulls reused across calls',
        order='alternating forward/reverse; warmup3 per variant, timed15 unless explicit development override; perf_counter, CUDA synchronize before and after',
        exclusions='file/decode/join/FK/SAM2/grounding/gate/Shape-First/Graph/action/verification/logging',
        input_sha256=dict(masks={k:B.C.array_sha(v) for k,v in masks.items()},
            cameras={c.camera:{k:B.C.array_sha(getattr(c,k)) for k in ('depth_raw_mm','intrinsic','extrinsic_cv')} for c in inputs[0]},
            surfaces=[dict(link=s.link,vertices=B.C.array_sha(s.world_vertices)) for s in inputs[2]]))
    B.C.write(output/'config.json',config);(output/'source').mkdir()
    for p in files:shutil.copy2(p,output/'source'/('previous_gpu_filter.py' if p==PREVIOUS else p.name))
    rows=[]
    with B.ExperimentLogger(team='integration',model='task10i_gpu_filter_refinement_512',job_type='evaluate',seed=42,root=output/'experiments',config=config) as log:
        for _ in range(3):
            for v in variants:check(run(v))
        if args.profile:
            import cProfile,pstats
            profiler=cProfile.Profile();profiler.enable()
            for _ in range(20):run('previous_gpu_gpu')
            profiler.disable();profiler.dump_stats(str(output/'filter.prof'))
            with (output/'profile.txt').open('w') as f:pstats.Stats(profiler,stream=f).strip_dirs().sort_stats('cumtime').print_stats(40)
            module=sys.modules[previous.__module__]
            prepared=previous.prepare_surfaces(inputs[2])
            visible={c.camera:[i for i,(v,_) in enumerate(prepared.hulls) if (bb:=module.bounds(c,v))[2]>=bb[0] and bb[3]>=bb[1]] for c in inputs[0]}
            B.C.write(output/'visibility.json',dict(total=len(prepared.hulls),visible=visible))
            log.link_output(output/'profile.txt');log.link_output(output/'visibility.json')
            print((output/'profile.txt').read_text(),flush=True);print(json.dumps(visible),flush=True);return
        for repeat in range(args.repeats):
            for v in variants if repeat%2==0 else variants[::-1]:
                out=run(v);check(out)
                rows.append(dict(variant=v,repeat=repeat,ms=out['ms'],allocated=out['allocated'],reserved=out['reserved'],clean_exact=True))
        summary={}
        for v in variants:
            t=B.np.array([r['ms'] for r in rows if r['variant']==v])
            summary[v]=dict(n=len(t),median_ms=float(B.np.median(t)),q25_ms=float(B.np.quantile(t,.25)),q75_ms=float(B.np.quantile(t,.75)),min_ms=float(t.min()),max_ms=float(t.max()))
        original={}
        for pid in (327662,327663,327664,327665,327666,357110,357111):
            state=next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:'));assert '\tT' in state;original[str(pid)]=state
        B.C.write(output/'results.json',dict(status='PASS',summary=summary,rows=rows,original_processes=original))
        log.log({v:s['median_ms'] for v,s in summary.items()});log.link_output(output/'results.json');log.link_output(output/'config.json')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
