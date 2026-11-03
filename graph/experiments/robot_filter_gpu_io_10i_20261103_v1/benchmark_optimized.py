"""Frozen 512 two-camera, filter-only paired comparison; never runs a model."""
from pathlib import Path
import os, sys, time, json, inspect, resource, shutil, importlib.util, argparse

OLD = Path('/workspace/task10i_robot_filter_optimization_20261103_v1/code_final_v4')
OVERLAY = Path(__file__).resolve().parents[3]
PREVIOUS = Path('/workspace/task10i_robot_filter_gpu_io_20261103_v1/paired_512_latency_v1/source/gpu_filter.py')
sys.path.insert(0, str(OLD))
from run import C, FROZEN, geometry, load_inputs, masks_for, Candidate, RayCache, ExperimentLogger, np, torch, identity, exact
from filter_benchmark import pair
sys.path.insert(0, str(OVERLAY))
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter


def previous_class():
    spec = importlib.util.spec_from_file_location('previous_gpu_filter', PREVIOUS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CudaRobotFilter


def cuda_pair(inputs, masks, gpu_masks, cls, *, batch=False, return_cpu=False):
    with Candidate(9) as ctx:
        ctx.cache = {}
        filt = cls()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        raw = ({k: torch.as_tensor(np.ascontiguousarray(v), device='cuda') for k, v in masks.items()}
               if return_cpu else gpu_masks)
        if batch:
            result = filt.filter_cameras(inputs[0], inputs[2], raw)
        else:
            result = {c.camera: filt(c, inputs[2], raw[c.camera]) for c in inputs[0]}
        if return_cpu:
            result = {k: v.cpu().numpy() for k, v in result.items()}
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        return dict(ms=elapsed, clean=result, allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', action='store_true')
    p.add_argument('--repeats', type=int, default=15)
    args = p.parse_args()
    C.assert_source()
    torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_DATA, (80*1024**3, 80*1024**3))
    output = Path(os.environ['TASK10I_GPU_OPT_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    scene = '10i_s1_0921_t080'
    spec = next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id'] == scene)
    inputs = load_inputs(scene, 512, geometry())
    masks = masks_for(spec, 512, 0, 'actual')
    gpu_masks = {k: torch.as_tensor(v, device='cuda') for k, v in masks.items()}
    torch.cuda.synchronize()
    previous = previous_class()
    expected = pair(inputs, masks, 'baseline', RayCache())['clean']
    variants = ['baseline_cpu_cpu', 'strict_cpu_cpu', 'previous_gpu_gpu', 'optimized_gpu_gpu', 'optimized_cpu_cpu']
    if args.profile:
        variants = ['previous_gpu_gpu']
    rays = RayCache()
    optimized_filters = {v: CudaRobotFilter() for v in variants if v.startswith('optimized')}
    def run(v):
        if v == 'baseline_cpu_cpu': return pair(inputs, masks, 'baseline', RayCache())
        if v == 'strict_cpu_cpu': return pair(inputs, masks, 'strict', rays)
        return cuda_pair(inputs, masks, gpu_masks, previous if v == 'previous_gpu_gpu' else lambda: optimized_filters[v],
                         batch=v.startswith('optimized'), return_cpu=v == 'optimized_cpu_cpu')
    def check(out):
        exact({k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in out['clean'].items()}, expected, 'clean masks')
    sources = [Path(__file__), Path(inspect.getfile(CudaRobotFilter)), PREVIOUS, OLD/'filter_benchmark.py', OLD/'roi_filter.py']
    config = dict(scene=scene, size=512, mask_repeat=0, cameras=list(masks), warmups=3, repeats=args.repeats,
        variants=variants, profile=args.profile, identity=identity(), no_sam2_inference=True, no_graph_execution=True,
        boundary='two cameras, current CPU sensor/calibration/FK surfaces; all hull preparation inside timer; GPU variants resident CUDA masks -> CUDA clean masks',
        cpu_variant='raw mask upload and clean readback also inside timer',
        optimized_cache='persistent filter instance per variant, calibration-only full CV ray cache, 8 entries / 64MiB; warmed with 3 calls; hull/FK snapshot preparation and world ray rotation recomputed inside every timed batch',
        input_sha256=dict(masks={k: C.array_sha(v) for k, v in masks.items()},
            cameras={c.camera: {k: C.array_sha(getattr(c, k)) for k in ('depth_raw_mm', 'intrinsic', 'extrinsic_cv')} for c in inputs[0]},
            surfaces=[dict(link=s.link, vertices=C.array_sha(s.world_vertices)) for s in inputs[2]]),
        sources={str(p): C.sha(p) for p in sources},
        order='alternate forward/reverse; warmup3 each, perf_counter with CUDA synchronization before/after; checks outside timer',
        exclusions='file/decode/join/FK/SAM2/grounding/gate/Shape-First/Graph/action/verification/logging')
    C.write(output/'config.json', config)
    (output/'source').mkdir()
    for p in sources:
        shutil.copy2(p, output/'source'/('previous_gpu_filter.py' if p == PREVIOUS else p.name))
    rows = []
    with ExperimentLogger(team='integration', model='task10i_gpu_filter_optimization_512', job_type='evaluate', seed=42, root=output/'experiments', config=config) as logger:
        for _ in range(3):
            for v in variants: check(run(v))
        if args.profile:
            import cProfile, pstats
            profiler = cProfile.Profile()
            profiler.enable()
            for _ in range(20): run('previous_gpu_gpu')
            profiler.disable()
            profiler.dump_stats(str(output/'filter.prof'))
            with (output/'profile.txt').open('w') as f:
                pstats.Stats(profiler, stream=f).strip_dirs().sort_stats('cumtime').print_stats(40)
            logger.link_output(output/'profile.txt')
            print((output/'profile.txt').read_text(), flush=True)
            return
        for repeat in range(args.repeats):
            for v in variants if repeat % 2 == 0 else variants[::-1]:
                out = run(v); check(out)
                rows.append(dict(variant=v, repeat=repeat, ms=out['ms'], allocated=out['allocated'], reserved=out['reserved'], clean_exact=True))
        summary = {}
        for v in variants:
            t = np.array([r['ms'] for r in rows if r['variant'] == v])
            summary[v] = dict(n=len(t), median_ms=float(np.median(t)), q25_ms=float(np.quantile(t,.25)), q75_ms=float(np.quantile(t,.75)), min_ms=float(t.min()), max_ms=float(t.max()))
        original = {}
        for pid in (327662,327663,327664,327665,327666,357110,357111):
            state = next(x for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('State:'))
            assert '\tT' in state
            original[str(pid)] = state
        C.write(output/'results.json', dict(status='PASS', rows=rows, summary=summary, original_processes=original))
        logger.log({v: s['median_ms'] for v, s in summary.items()})
        logger.link_output(output/'results.json'); logger.link_output(output/'config.json')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
